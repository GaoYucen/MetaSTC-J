"""Diagnose when V9 dynamic corrections help or hurt across chronological splits.

This is an analysis-only script.  It compares the reproduced static MetaSTC
prediction with the frozen V9 continuous generator at road-window granularity,
and measures how the dynamic gain relates to causal recent-history features.
No model or selector is fit on the test segment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from optimized_runner import LSTMModel, _cluster_features, _device, _load_flow, _set_seed
from dynamic_support_routing_v4 import _episodic_windows, _load_static_experts, _static_prediction
from dynamic_continuous_generator_v9 import ContinuousTaskGenerator, _make_loader


def _corr(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3: return float('nan')
    x = x[good]; y = y[good]
    if x.std() < 1e-12 or y.std() < 1e-12: return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _analyze(model, experts, loader, labels_tensor, device, scale, max_batches=0):
    names = [
        'level_mean', 'level_std', 'trend_raw', 'recent_shift_raw', 'diff_std_raw',
        'trend_norm', 'recent_shift_norm', 'diff_std_norm',
        'support_level_mean', 'support_level_std',
        'global_support_mse', 'global_support_mae',
        'support_resid_mean_norm', 'support_resid_std_norm',
    ]
    features = {n: [] for n in names}
    gains, times, roads = [], [], []
    dyn_errs, sta_errs = [], []
    model.eval()
    with torch.no_grad():
        for bi, (sx, sy, qx, qy, road, time_id) in enumerate(loader):
            if max_batches > 0 and bi >= max_batches: break
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            road_d = road.to(device)
            dynamic, _, _, _, _, _ = model(sx, sy, qx, 1.0)
            static = _static_prediction(experts, qx, labels_tensor[road_d])
            de = (dynamic.squeeze(-1) - qy.squeeze(-1)).abs().mean(dim=1)
            se = (static.squeeze(-1) - qy.squeeze(-1)).abs().mean(dim=1)
            gain = se - de

            q = qx.squeeze(-1); s = sy.squeeze(-1)
            qmean = q.mean(dim=1); qstd = q.std(dim=1, unbiased=False).clamp_min(1e-4)
            trend = q[:, -1] - q[:, 0]
            shift = q[:, 6:].mean(dim=1) - q[:, :6].mean(dim=1)
            dstd = torch.diff(q, dim=1).std(dim=1, unbiased=False)
            _, sbase = model._features(sx)
            resid = s - sbase
            scale_local = s.std(dim=1, unbiased=False).clamp_min(1e-4)

            vals = {
                'level_mean': qmean,
                'level_std': qstd,
                'trend_raw': trend,
                'recent_shift_raw': shift,
                'diff_std_raw': dstd,
                'trend_norm': trend / qstd,
                'recent_shift_norm': shift / qstd,
                'diff_std_norm': dstd / qstd,
                'support_level_mean': s.mean(dim=1),
                'support_level_std': s.std(dim=1, unbiased=False),
                'global_support_mse': (resid ** 2).mean(dim=1),
                'global_support_mae': resid.abs().mean(dim=1),
                'support_resid_mean_norm': resid.mean(dim=1) / scale_local,
                'support_resid_std_norm': resid.std(dim=1, unbiased=False) / scale_local,
            }
            for n, v in vals.items(): features[n].append(v.cpu().numpy())
            gains.append(gain.cpu().numpy()); dyn_errs.append(de.cpu().numpy()); sta_errs.append(se.cpu().numpy())
            times.append(time_id.numpy()); roads.append(road.numpy())

    f = {n: np.concatenate(v) for n, v in features.items()}
    gain = np.concatenate(gains); de = np.concatenate(dyn_errs); se = np.concatenate(sta_errs)
    t = np.concatenate(times); r = np.concatenate(roads)
    corr = {n: _corr(v, gain) for n, v in f.items()}
    # Aggregate by forecast origin to expose chronological regime changes.
    time_rows = []
    for ti in np.unique(t):
        idx = t == ti
        time_rows.append({
            'time_id': int(ti),
            'mean_gain_raw': float(gain[idx].mean() * scale),
            'dynamic_better_rate': float((gain[idx] > 0).mean()),
            'static_mae': float(se[idx].mean() * scale),
            'dynamic_mae': float(de[idx].mean() * scale),
            'mean_level_raw': float(f['level_mean'][idx].mean() * scale),
            'mean_recent_shift_norm': float(f['recent_shift_norm'][idx].mean()),
            'mean_diff_std_norm': float(f['diff_std_norm'][idx].mean()),
        })
    # Feature summaries among samples where dynamic clearly helps/hurts.
    q20, q80 = np.quantile(gain, [0.2, 0.8])
    help_idx = gain >= q80; hurt_idx = gain <= q20
    contrast = {}
    for n, v in f.items():
        contrast[n] = {
            'help_top20_mean': float(v[help_idx].mean()),
            'hurt_bottom20_mean': float(v[hurt_idx].mean()),
            'standardized_gap': float((v[help_idx].mean() - v[hurt_idx].mean()) / max(v.std(), 1e-8)),
        }
    return {
        'samples': int(gain.size),
        'static_mae': float(se.mean() * scale),
        'dynamic_mae': float(de.mean() * scale),
        'mean_dynamic_gain': float(gain.mean() * scale),
        'dynamic_better_rate': float((gain > 0).mean()),
        'gain_std': float(gain.std() * scale),
        'feature_gain_correlation': corr,
        'feature_contrast': contrast,
        'per_time': time_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='beijing')
    ap.add_argument('--clusters', type=int, default=5)
    ap.add_argument('--checkpoint-dir', default='param/4090_tuned/lstm/beijing')
    ap.add_argument('--v9-checkpoint', default='param/journal/dynamic_continuous_v9/beijing_lstm_full/model_best.pt')
    ap.add_argument('--output-dir', default='param/journal/dynamic_shift_diagnostic_v11/beijing_lstm')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=8192)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed); device = _device(args.device)
    flow, ids, scale = _load_flow(args.dataset)
    labels = _cluster_features(args.dataset, 'lstm', flow, ids, args.clusters, args.seed)
    _, val, test = _episodic_windows(flow)
    vl = _make_loader(val, args.batch_size, device, False)
    tl = _make_loader(test, args.batch_size, device, False)
    ckpt = Path(args.checkpoint_dir)
    base = LSTMModel(12, 6).to(device)
    base.load_state_dict(torch.load(ckpt/'global_best.pt', map_location=device, weights_only=True))
    model = ContinuousTaskGenerator(base, num_bases=8, rank=4, latent_dim=16, hidden_dim=48).to(device)
    model.load_state_dict(torch.load(args.v9_checkpoint, map_location=device, weights_only=True))
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]
    experts = _load_static_experts(ckpt, args.clusters, device)
    labels_tensor = torch.from_numpy(labels).long().to(device)

    result = {
        'experiment': 'dynamic_shift_diagnostic_v11',
        'validation': _analyze(model, experts, vl, labels_tensor, device, scale),
        'exploratory_future_segment': _analyze(model, experts, tl, labels_tensor, device, scale),
    }
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out/'diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('SUMMARY')
    for split, x in result.items():
        if not isinstance(x, dict) or 'static_mae' not in x: continue
        top = sorted(x['feature_gain_correlation'].items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
        print(split, json.dumps({
            'static_mae': x['static_mae'], 'dynamic_mae': x['dynamic_mae'],
            'mean_gain': x['mean_dynamic_gain'], 'dynamic_better_rate': x['dynamic_better_rate'],
            'top_abs_correlations': top,
            'per_time': x['per_time'],
        }))


if __name__ == '__main__': main()
