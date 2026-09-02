"""V4 support-conditioned dynamic soft task discovery.

Instead of asking a learned router to extrapolate task identity into an unseen
future time segment, infer the current task from *recently observed* behavior.
For each forecast episode we use 18 observed steps plus 6 future query steps:

  support input : t-18 .. t-7   (12 steps)
  support target: t-6  .. t-1   (6 already-observed steps)
  query input   : t-12 .. t-1   (12 latest observed steps)
  query target  : t    .. t+5   (6 future steps; evaluation only)

The K task adapters are scored on the support pair. Their support losses produce
soft responsibilities, which are then used to mix task-conditioned forecasts on
the query input. Therefore test-time task discovery uses only information that
is already available at prediction time and can react to temporal distribution
shift without future-label leakage.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from optimized_runner import LSTMModel, _cluster_features, _device, _load_flow, _metrics, _set_seed


def _episodic_windows(flow: np.ndarray):
    train_size = int(flow.shape[1] * 0.8)
    train_part = flow[:, :train_size].T
    test_part = flow[:, train_size:].T

    def build(part: np.ndarray):
        count = part.shape[0] - 24
        if count <= 0:
            raise ValueError("not enough time steps for 18-step support/query history + 6-step target")
        support_x = np.stack([part[i:i + 12] for i in range(count)]).astype(np.float32)
        support_y = np.stack([part[i + 12:i + 18] for i in range(count)]).astype(np.float32)
        query_x = np.stack([part[i + 6:i + 18] for i in range(count)]).astype(np.float32)
        query_y = np.stack([part[i + 18:i + 24] for i in range(count)]).astype(np.float32)
        return support_x, support_y, query_x, query_y

    train = build(train_part)
    test = build(test_part)
    split = max(1, int(train[0].shape[0] * 0.9))
    fit = tuple(array[:split] for array in train)
    val = tuple(array[split:] for array in train)
    return fit, val, test


def _all_samples(episode_arrays):
    support_x, support_y, query_x, query_y = episode_arrays
    roads = support_x.shape[2]
    windows = support_x.shape[0]

    def flatten_x(x):
        return x.transpose(2, 0, 1).reshape(-1, x.shape[1], 1).astype(np.float32)

    def flatten_y(y):
        return y.transpose(2, 0, 1).reshape(-1, y.shape[1], 1).astype(np.float32)

    road_ids = np.repeat(np.arange(roads, dtype=np.int64), windows)
    time_ids = np.tile(np.arange(windows, dtype=np.int64), roads)
    return (
        flatten_x(support_x), flatten_y(support_y),
        flatten_x(query_x), flatten_y(query_y),
        road_ids, time_ids,
    )


def _limited(loader: Iterable, max_batches: int):
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        yield batch


class TaskAdapterBank(nn.Module):
    def __init__(self, base: LSTMModel, num_tasks: int, rank: int = 6):
        super().__init__()
        self.num_tasks = num_tasks
        self.feature_dim = base.linear.in_features
        self.output_dim = base.linear.out_features
        self.rank = min(rank, self.feature_dim, self.output_dim)
        self.base_lstm = base.lstm
        self.base_linear = base.linear
        for p in self.base_lstm.parameters(): p.requires_grad_(False)
        for p in self.base_linear.parameters(): p.requires_grad_(False)
        self.adapter_a = nn.Parameter(torch.empty(num_tasks, self.feature_dim, self.rank))
        self.adapter_b = nn.Parameter(torch.zeros(num_tasks, self.rank, self.output_dim))
        self.adapter_bias = nn.Parameter(torch.zeros(num_tasks, self.output_dim))
        nn.init.normal_(self.adapter_a, std=0.02)

    @torch.no_grad()
    def warmstart_from_static_heads(self, cluster_states: List[Dict[str, torch.Tensor]]):
        base_w = self.base_linear.weight.detach().cpu()
        base_b = self.base_linear.bias.detach().cpu()
        for task, state in enumerate(cluster_states):
            delta = (state["linear.weight"].detach().cpu() - base_w).t().float()
            u, s, vh = torch.linalg.svd(delta, full_matrices=False)
            r = min(self.rank, s.numel())
            root = torch.sqrt(torch.clamp(s[:r], min=0.0))
            self.adapter_a[task].zero_(); self.adapter_b[task].zero_()
            self.adapter_a[task, :, :r].copy_((u[:, :r] * root.unsqueeze(0)).to(self.adapter_a.device))
            self.adapter_b[task, :r, :].copy_((root.unsqueeze(1) * vh[:r, :]).to(self.adapter_b.device))
            self.adapter_bias[task].copy_((state["linear.bias"].detach().cpu() - base_b).to(self.adapter_bias.device))

    def forward(self, x: torch.Tensor):
        self.base_lstm.flatten_parameters()
        sequence, _ = self.base_lstm(x)
        flat = sequence.reshape(sequence.shape[0], -1)
        base = self.base_linear(flat)
        z = torch.einsum("bf,kfr->bkr", flat, self.adapter_a)
        delta = torch.einsum("bkr,kro->bko", z, self.adapter_b) + self.adapter_bias.unsqueeze(0)
        task_pred = base.unsqueeze(1) + delta
        return task_pred.unsqueeze(-1), base.unsqueeze(-1), delta


def _support_weights(errors, static_task, k, temperature, prior_strength):
    best = errors.min(dim=1, keepdim=True).values
    spread = errors.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
    logits = -(errors - best) / (spread * temperature)
    if prior_strength != 0.0:
        prior = torch.nn.functional.one_hot(static_task, k).to(errors.dtype)
        logits = logits + prior_strength * prior
    return torch.softmax(logits, dim=1)


def _load_static_experts(ckpt: Path, k: int, device):
    experts = []
    for task in range(k):
        m = LSTMModel(12, 6).to(device)
        m.load_state_dict(torch.load(ckpt / f"cluster_{task}.pt", map_location=device, weights_only=True))
        m.eval(); m.lstm.flatten_parameters()
        for p in m.parameters(): p.requires_grad_(False)
        experts.append(m)
    return experts


def _static_prediction(experts, x, static_task):
    out = torch.empty((x.shape[0], 6, 1), device=x.device, dtype=x.dtype)
    with torch.no_grad():
        for task, expert in enumerate(experts):
            idx = torch.nonzero(static_task == task, as_tuple=False).squeeze(1)
            if idx.numel(): out[idx] = expert(x[idx])
    return out


def _validation_loss(bank, loader, labels_tensor, device, max_batches, temperature, prior_strength):
    bank.eval(); losses = []
    with torch.no_grad():
        for sx, sy, qx, qy, road, _ in _limited(loader, max_batches):
            sx, sy = sx.to(device), sy.to(device)
            qx, qy = qx.to(device), qy.to(device)
            road = road.to(device)
            static_task = labels_tensor[road]
            support_pred, _, _ = bank(sx)
            support_errors = ((support_pred - sy[:, None]) ** 2).mean(dim=(2, 3))
            weights = _support_weights(support_errors, static_task, bank.num_tasks, temperature, prior_strength)
            query_pred, _, _ = bank(qx)
            mixed = (query_pred * weights[:, :, None, None]).sum(dim=1)
            losses.append(float(nn.functional.mse_loss(mixed, qy).cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def _evaluate(bank, experts, loader, labels, labels_tensor, scale, device, max_batches, temperature, prior_strength):
    bank.eval()
    chunks = {n: [] for n in ["target", "global", "old_static", "adapter_static", "dynamic_soft", "dynamic_hard", "oracle"]}
    weights_all, roads_all, times_all = [], [], []
    with torch.no_grad():
        for sx, sy, qx, qy, road, time_id in _limited(loader, max_batches):
            sx, sy = sx.to(device), sy.to(device)
            qx, qy = qx.to(device), qy.to(device)
            road = road.to(device)
            static_task = labels_tensor[road]
            support_pred, _, _ = bank(sx)
            support_errors = ((support_pred - sy[:, None]) ** 2).mean(dim=(2, 3))
            weights = _support_weights(support_errors, static_task, bank.num_tasks, temperature, prior_strength)
            query_pred, base, _ = bank(qx)
            row = torch.arange(qx.shape[0], device=device)
            dynamic_soft = (query_pred * weights[:, :, None, None]).sum(dim=1)
            dynamic_task = weights.argmax(dim=1)
            dynamic_hard = query_pred[row, dynamic_task]
            adapter_static = query_pred[row, static_task]
            old_static = _static_prediction(experts, qx, static_task)
            query_errors = ((query_pred - qy[:, None]) ** 2).mean(dim=(2, 3))
            oracle = query_pred[row, query_errors.argmin(dim=1)]
            values = {"target": qy, "global": base, "old_static": old_static, "adapter_static": adapter_static,
                      "dynamic_soft": dynamic_soft, "dynamic_hard": dynamic_hard, "oracle": oracle}
            for name, value in values.items(): chunks[name].append((value * scale).cpu().numpy())
            weights_all.append(weights.cpu().numpy()); roads_all.append(road.cpu().numpy()); times_all.append(time_id.numpy())

    target = np.concatenate(chunks["target"]).reshape(-1)
    pred = {n: np.concatenate(chunks[n]).reshape(-1) for n in chunks if n != "target"}
    weights = np.concatenate(weights_all); roads = np.concatenate(roads_all); times = np.concatenate(times_all)
    static_task = labels[roads]; dynamic_task = weights.argmax(axis=1)
    changes = transitions = roads_changed = 0
    for road in np.unique(roads):
        idx = np.flatnonzero(roads == road); idx = idx[np.argsort(times[idx])]
        seq = dynamic_task[idx]
        if len(seq) > 1:
            diff = seq[1:] != seq[:-1]
            changes += int(diff.sum()); transitions += int(diff.size); roads_changed += int(diff.any())
    entropy = -(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum(axis=1)
    result = {
        "global_base": _metrics(pred["global"], target),
        "reproduced_static_hard": _metrics(pred["old_static"], target),
        "adapter_static": _metrics(pred["adapter_static"], target),
        "support_dynamic_soft": _metrics(pred["dynamic_soft"], target),
        "support_dynamic_hard": _metrics(pred["dynamic_hard"], target),
        "oracle_best_task_diagnostic": _metrics(pred["oracle"], target),
        "routing": {
            "mean_entropy": float(entropy.mean()), "mean_max_weight": float(weights.max(axis=1).mean()),
            "dynamic_vs_static_argmax_disagreement": float((dynamic_task != static_task).mean()),
            "temporal_task_change_rate": float(changes / max(transitions, 1)),
            "roads_with_temporal_task_change": float(roads_changed / max(len(np.unique(roads)), 1)),
            "task_usage": weights.mean(axis=0).tolist(),
        },
        "num_window_road_samples": int(weights.shape[0]),
    }
    result["dynamic_vs_reproduced_static_delta_mae"] = result["support_dynamic_soft"]["MAE"] - result["reproduced_static_hard"]["MAE"]
    result["dynamic_vs_reproduced_static_relative_mae_pct"] = 100.0 * (result["support_dynamic_soft"]["MAE"] / result["reproduced_static_hard"]["MAE"] - 1.0)
    result["dynamic_vs_adapter_static_delta_mae"] = result["support_dynamic_soft"]["MAE"] - result["adapter_static"]["MAE"]
    return result


def main():
    ap = argparse.ArgumentParser(description="Support-conditioned dynamic task routing")
    ap.add_argument("--dataset", choices=["beijing", "shanghai", "largest"], default="beijing")
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--checkpoint-dir", default="param/4090_tuned/lstm/beijing")
    ap.add_argument("--output-dir", default="param/journal/dynamic_support_v4/beijing_lstm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--prior-strength", type=float, default=0.25)
    ap.add_argument("--support-loss-weight", type=float, default=0.25)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--no-head-warmstart", action="store_true")
    ap.add_argument("--train-max-batches", type=int, default=50)
    ap.add_argument("--val-max-batches", type=int, default=0)
    ap.add_argument("--test-max-batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed); device = _device(args.device); k = args.clusters
    flow, ids, scale = _load_flow(args.dataset)
    labels = _cluster_features(args.dataset, "lstm", flow, ids, k, args.seed)
    fit, val, test = _episodic_windows(flow)

    def make_loader(arrays, shuffle):
        sx, sy, qx, qy, roads, times = _all_samples(arrays)
        ds = TensorDataset(torch.from_numpy(sx), torch.from_numpy(sy), torch.from_numpy(qx), torch.from_numpy(qy),
                           torch.from_numpy(roads), torch.from_numpy(times))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0, pin_memory=device.type == "cuda")

    train_loader = make_loader(fit, True); val_loader = make_loader(val, False); test_loader = make_loader(test, False)
    ckpt = Path(args.checkpoint_dir)
    base = LSTMModel(12, 6).to(device)
    base.load_state_dict(torch.load(ckpt / "global_best.pt", map_location=device, weights_only=True)); base.eval(); base.lstm.flatten_parameters()
    states = [torch.load(ckpt / f"cluster_{task}.pt", map_location="cpu", weights_only=True) for task in range(k)]
    bank = TaskAdapterBank(base, k, rank=args.rank).to(device)
    if not args.no_head_warmstart: bank.warmstart_from_static_heads(states)
    experts = _load_static_experts(ckpt, k, device)
    labels_tensor = torch.from_numpy(labels).long().to(device)
    optimizer = torch.optim.AdamW([bank.adapter_a, bank.adapter_b, bank.adapter_bias], lr=args.lr, weight_decay=args.weight_decay)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    best_val = float("inf"); best_state = copy.deepcopy(bank.state_dict()); history = []
    for epoch in range(args.epochs):
        bank.train(); losses = []; route_entropy = []; route_max = []
        for sx, sy, qx, qy, road, _ in _limited(train_loader, args.train_max_batches):
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device); road = road.to(device)
            static_task = labels_tensor[road]
            support_pred, _, _ = bank(sx)
            support_errors = ((support_pred - sy[:, None]) ** 2).mean(dim=(2, 3))
            weights = _support_weights(support_errors.detach(), static_task, k, args.temperature, args.prior_strength)
            query_pred, _, _ = bank(qx)
            mixed = (query_pred * weights[:, :, None, None]).sum(dim=1)
            query_loss = nn.functional.mse_loss(mixed, qy)
            support_loss = (weights * support_errors).sum(dim=1).mean()
            loss = query_loss + args.support_loss_weight * support_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(bank.parameters(), 5.0); optimizer.step()
            losses.append((float(loss.detach().cpu()), float(query_loss.detach().cpu()), float(support_loss.detach().cpu())))
            route_entropy.append(float((-(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=1).mean()).cpu()))
            route_max.append(float(weights.max(dim=1).values.mean().cpu()))
        val_loss = _validation_loss(bank, val_loader, labels_tensor, device, args.val_max_batches, args.temperature, args.prior_strength)
        arr = np.asarray(losses)
        row = {"epoch": epoch + 1, "loss": float(arr[:, 0].mean()), "query_loss": float(arr[:, 1].mean()),
               "support_loss": float(arr[:, 2].mean()), "route_entropy": float(np.mean(route_entropy)),
               "route_mean_max": float(np.mean(route_max)), "val_mse": val_loss}
        history.append(row); print(json.dumps(row))
        if val_loss < best_val:
            best_val = val_loss; best_state = copy.deepcopy(bank.state_dict()); torch.save(best_state, out / "model_best.pt")

    bank.load_state_dict(best_state)
    validation = _evaluate(bank, experts, val_loader, labels, labels_tensor, scale, device, args.val_max_batches, args.temperature, args.prior_strength)
    test_result = _evaluate(bank, experts, test_loader, labels, labels_tensor, scale, device, args.test_max_batches, args.temperature, args.prior_strength)
    result = {"experiment": "dynamic_support_routing_v4", "dataset": args.dataset, "clusters": k,
              "best_val_normalized_mse": best_val, "history": history, "validation": validation, "test": test_result}
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
