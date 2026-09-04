"""Efficient 5-seed matched-control sweep for MetaSTC-J V14.

Frozen static MetaSTC expert outputs are cached once per dataset, so repeated
adapter seeds/controls do not rerun the LSTM experts. Full per-window dynamic
mechanism diagnostics are computed only for V14 seed 42 on each dataset.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dynamic_residual_meta_adapter_v14 import (
    DynamicResidualMetaAdapter,
    _cluster_tensors,
    _device,
    _load_experts,
    _load_labels,
    _loaders,
)
from dynamic_residual_mechanism_v14 import (
    ConstantGateAdapter,
    LinearClusterAdapter,
    _count_params,
    diagnostics,
)
from optimized_runner import _load_flow, _metrics, _set_seed, _windows

ROOT = Path('param/journal/dynamic_residual_mechanism_v14_matched')
SEEDS = [42, 43, 44, 45, 46]
VARIANTS = ['v14', 'constant_gate', 'shared_no_task', 'linear_cluster']
DATASETS = {
    'beijing': Path('param/4090_tuned/lstm/beijing'),
    'shanghai': Path('param/4090_tuned/epoch60/lstm/shanghai'),
    'largest': Path('param/4090_tuned/epoch60/lstm/largest'),
}
DEVICE = 'cuda:0'
CLUSTERS = 5
CACHE_BATCH = 32768
ADAPTER_BATCH = 16384
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAE_WEIGHT = 0.02
SAFETY_WEIGHT = 0.02
CORRECTION_WEIGHT = 1e-4
CORRECTION_LIMIT = 0.8
GATE_BIAS = -1.5


class SharedNoTaskAdapter(DynamicResidualMetaAdapter):
    """V14-capacity control with task identity removed from FiLM conditioning."""
    def __init__(self, clusters: int, correction_limit=0.8, gate_bias=-1.5):
        super().__init__(clusters, correction_limit=correction_limit, gate_bias=gate_bias)
        self.cluster_embedding.weight.requires_grad_(False)
        self.global_embedding = nn.Parameter(torch.zeros(self.cluster_embedding.embedding_dim))

    def forward(self, x, base, task):
        history = x.squeeze(-1)
        base2 = base.squeeze(-1)
        state, shared_input, local_scale = self._context(history, base2)
        z = self.state_encoder(state)
        global_emb = self.global_embedding.unsqueeze(0).expand(x.shape[0], -1)
        cond = torch.cat([z, global_emb], dim=1)
        shared = self.shared_residual(shared_input)
        gamma, beta = self.modulator(cond).chunk(2, dim=1)
        hidden = torch.nn.functional.gelu(
            shared * (1.0 + 0.2 * torch.tanh(gamma)) + 0.2 * beta
        )
        delta_norm = self.correction_limit * torch.tanh(self.delta_head(hidden))
        gate = torch.sigmoid(self.gate_head(cond))
        correction = local_scale * gate * delta_norm
        return (base2 + correction).unsqueeze(-1), gate, correction


def build_adapter(variant: str):
    if variant == 'v14':
        return DynamicResidualMetaAdapter(CLUSTERS, correction_limit=CORRECTION_LIMIT, gate_bias=GATE_BIAS)
    if variant == 'constant_gate':
        return ConstantGateAdapter(CLUSTERS, CORRECTION_LIMIT, GATE_BIAS)
    if variant == 'shared_no_task':
        return SharedNoTaskAdapter(CLUSTERS, CORRECTION_LIMIT, GATE_BIAS)
    if variant == 'linear_cluster':
        return LinearClusterAdapter(CLUSTERS, CORRECTION_LIMIT, GATE_BIAS)
    raise ValueError(variant)


def cache_static_outputs(experts, raw_loaders, device):
    cached = {}
    for c, loader in raw_loaders.items():
        xs, ys, bs = [], [], []
        expert = experts[c]
        with torch.inference_mode():
            for x, y in loader:
                x_gpu = x.to(device, non_blocking=True)
                base = expert(x_gpu)
                xs.append(x.cpu())
                ys.append(y.cpu())
                bs.append(base.cpu())
        cached[c] = TensorDataset(
            torch.cat(xs, dim=0), torch.cat(ys, dim=0), torch.cat(bs, dim=0)
        )
    return cached


def cached_loaders(cached, shuffle: bool):
    return {
        c: DataLoader(
            ds,
            batch_size=ADAPTER_BATCH,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        for c, ds in cached.items()
    }


def evaluate_cached(adapter, loaders, device, scale):
    adapter.eval()
    static_all, dynamic_all, y_all, gates = [], [], [], []
    with torch.inference_mode():
        for c, loader in loaders.items():
            for x, y, base in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                base = base.to(device, non_blocking=True)
                task = torch.full((x.shape[0],), c, device=device, dtype=torch.long)
                pred, gate, _ = adapter(x, base, task)
                static_all.append((base * scale).cpu().numpy())
                dynamic_all.append((pred * scale).cpu().numpy())
                y_all.append((y * scale).cpu().numpy())
                gates.append(gate.cpu().numpy().reshape(-1))
    s = np.concatenate(static_all).reshape(-1)
    d = np.concatenate(dynamic_all).reshape(-1)
    y = np.concatenate(y_all).reshape(-1)
    g = np.concatenate(gates)
    sm, dm = _metrics(s, y), _metrics(d, y)
    return {
        'static': sm,
        'dynamic': dm,
        'relative_mae_vs_static_pct': 100.0 * (dm['MAE'] / sm['MAE'] - 1.0),
        'relative_mse_vs_static_pct': 100.0 * (dm['MSE'] / sm['MSE'] - 1.0),
        'mean_gate': float(g.mean()),
        'std_gate': float(g.std()),
    }


def train_one(dataset, variant, seed, cached_train, cached_val, cached_test,
              experts, labels, fit_x, test_x, test_y, device, scale):
    out = ROOT / dataset / variant / f'seed_{seed}'
    metrics_path = out / 'metrics.json'
    if metrics_path.exists():
        result = json.loads(metrics_path.read_text(encoding='utf-8'))
        print('RESUME', dataset, variant, seed, 'MAE', round(result['test']['dynamic']['MAE'], 6), flush=True)
        return result
    _set_seed(seed)
    train_loaders = cached_loaders(cached_train, True)
    val_loaders = cached_loaders(cached_val, False)
    test_loaders = cached_loaders(cached_test, False)
    adapter = build_adapter(variant).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in adapter.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    best_score = float('inf')
    best_state = None
    history = []
    for epoch in range(EPOCHS):
        adapter.train()
        total_loss, seen = 0.0, 0
        for c, loader in train_loaders.items():
            for x, y, base in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                base = base.to(device, non_blocking=True)
                task = torch.full((x.shape[0],), c, device=device, dtype=torch.long)
                pred, gate, correction = adapter(x, base, task)
                mse = torch.mean((pred - y) ** 2)
                mae = torch.mean(torch.abs(pred - y))
                dyn = torch.mean((pred - y) ** 2, dim=(1, 2))
                sta = torch.mean((base - y) ** 2, dim=(1, 2))
                safety = torch.relu(dyn - sta).mean()
                corr_reg = torch.mean(correction ** 2)
                loss = (
                    mse
                    + MAE_WEIGHT * mae
                    + SAFETY_WEIGHT * safety
                    + CORRECTION_WEIGHT * corr_reg
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * x.shape[0]
                seen += x.shape[0]
        val = evaluate_cached(adapter, val_loaders, device, scale)
        score = val['dynamic']['MSE'] + 0.1 * val['dynamic']['MAE']
        history.append({
            'epoch': epoch + 1,
            'train_loss': total_loss / max(seen, 1),
            'val_mae_pct': val['relative_mae_vs_static_pct'],
            'val_mse_pct': val['relative_mse_vs_static_pct'],
            'mean_gate': val['mean_gate'],
        })
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    assert best_state is not None
    adapter.load_state_dict(best_state)
    validation = evaluate_cached(adapter, val_loaders, device, scale)
    test = evaluate_cached(adapter, test_loaders, device, scale)
    diag = {'per_window': [], 'correlations': {}}
    if variant == 'v14' and seed == 42:
        diag = diagnostics(
            adapter, experts, labels, fit_x, test_x, test_y, device, scale, CLUSTERS
        )
    result = {
        'experiment': 'dynamic_residual_mechanism_v14_fast',
        'dataset': dataset,
        'variant': variant,
        'seed': seed,
        'trainable_params': _count_params(adapter),
        'validation': validation,
        'test': test,
        'diagnostics': diag,
        'history': history,
    }
    out = ROOT / dataset / variant / f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out / 'adapter_best.pt')
    (out / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(
        'DONE', dataset, variant, seed,
        'MAE', round(test['dynamic']['MAE'], 6),
        'REL%', round(test['relative_mae_vs_static_pct'], 4),
        'PARAMS', result['trainable_params'],
        flush=True,
    )
    return result


def aggregate(all_results):
    summary = {
        'protocol': 'Official L=12/P=6 split; frozen tuned static MetaSTC anchors; cached static outputs; 5 seeds; 10 epochs; matched objectives.',
        'datasets': {},
    }
    lines = [
        '# V14 Mechanism Validation — Efficient 5-seed Matched Controls',
        '', summary['protocol'], '',
        '| Dataset | Variant | Params | Test MAE mean±std | Relative MAE vs static | V14 wins paired |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for ds in DATASETS:
        summary['datasets'][ds] = {}
        by_variant = {v: sorted(all_results[ds][v], key=lambda r: r['seed']) for v in VARIANTS}
        v14_mae = np.array([r['test']['dynamic']['MAE'] for r in by_variant['v14']], float)
        for v in VARIANTS:
            rows = by_variant[v]
            mae = np.array([r['test']['dynamic']['MAE'] for r in rows], float)
            rel = np.array([r['test']['relative_mae_vs_static_pct'] for r in rows], float)
            gates = np.array([r['test']['mean_gate'] for r in rows], float)
            rec = {
                'n': len(rows),
                'trainable_params': rows[0]['trainable_params'],
                'test_mae_mean': float(mae.mean()),
                'test_mae_std': float(mae.std(ddof=1)),
                'relative_mae_pct_mean': float(rel.mean()),
                'relative_mae_pct_std': float(rel.std(ddof=1)),
                'mean_gate_mean': float(gates.mean()),
                'mean_gate_std': float(gates.std(ddof=1)),
            }
            if v == 'v14':
                paired = {'v14_wins': None, 'control_wins': None, 'ties': None, 'mean_control_minus_v14_mae': 0.0}
            else:
                diff = mae - v14_mae
                paired = {
                    'v14_wins': int((diff > 1e-9).sum()),
                    'control_wins': int((diff < -1e-9).sum()),
                    'ties': int((np.abs(diff) <= 1e-9).sum()),
                    'mean_control_minus_v14_mae': float(diff.mean()),
                }
            rec['paired_vs_v14'] = paired
            summary['datasets'][ds][v] = rec
            wins = '—' if v == 'v14' else f"{paired['v14_wins']}/5"
            lines.append(
                f"| {ds} | {v} | {rec['trainable_params']} | "
                f"{rec['test_mae_mean']:.4f} ± {rec['test_mae_std']:.4f} | "
                f"{rec['relative_mae_pct_mean']:.2f}% ± {rec['relative_mae_pct_std']:.2f} | {wins} |"
            )
        seed42 = by_variant['v14'][0]
        summary['datasets'][ds]['v14_seed42_dynamic_evidence'] = seed42['diagnostics']['correlations']
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (ROOT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print('DYNAMIC_EVIDENCE', flush=True)
    for ds in DATASETS:
        print(ds, summary['datasets'][ds]['v14_seed42_dynamic_evidence'], flush=True)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    device = _device(DEVICE)
    all_results = {ds: {v: [] for v in VARIANTS} for ds in DATASETS}
    for dataset, checkpoint in DATASETS.items():
        print('CACHE_START', dataset, flush=True)
        flow, _, scale = _load_flow(dataset)
        labels = _load_labels(checkpoint, flow.shape[0], CLUSTERS)
        fit_x, fit_y, val_x, val_y, test_x, test_y = _windows(flow, 12, 6)
        experts = _load_experts(checkpoint, CLUSTERS, device)
        train_raw = _loaders(
            _cluster_tensors(fit_x, fit_y, labels, CLUSTERS), CACHE_BATCH, device, False
        )
        val_raw = _loaders(
            _cluster_tensors(val_x, val_y, labels, CLUSTERS), CACHE_BATCH, device, False
        )
        test_raw = _loaders(
            _cluster_tensors(test_x, test_y, labels, CLUSTERS), CACHE_BATCH, device, False
        )
        cached_train = cache_static_outputs(experts, train_raw, device)
        cached_val = cache_static_outputs(experts, val_raw, device)
        cached_test = cache_static_outputs(experts, test_raw, device)
        print('CACHE_DONE', dataset, flush=True)
        for seed in SEEDS:
            for variant in VARIANTS:
                result = train_one(
                    dataset, variant, seed,
                    cached_train, cached_val, cached_test,
                    experts, labels, fit_x, test_x, test_y, device, scale,
                )
                all_results[dataset][variant].append(result)
        del cached_train, cached_val, cached_test, train_raw, val_raw, test_raw, experts
        torch.cuda.empty_cache()
    aggregate(all_results)


if __name__ == '__main__':
    main()
