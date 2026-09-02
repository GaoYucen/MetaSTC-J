"""V1.1 learned residual gate for dynamic soft task assignment.

The frozen reproduction experts are never updated. A tiny gate learns when the
current 12-step context should stay with the road's original static task and
when it should borrow predictions from other task experts.

Design:
  static task (KMeans road label) -> one-hot prior
  current 12-step context + simple statistics + static prior -> residual MLP
  w = (1 - alpha) * static_one_hot + alpha * softmax(residual_logits)

`alpha` is sample-dependent and initialized close to zero, so training starts
near the conference baseline rather than destroying it. The gate is trained on
the fit split, selected by validation MSE, and evaluated once on the held-out
test split. This avoids selecting routing hyperparameters on the test set.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from optimized_runner import (
    FiLMModel,
    LSTMModel,
    _cluster_features,
    _device,
    _load_flow,
    _metrics,
    _set_seed,
    _windows,
)


def _model(family: str, look_back: int, look_forward: int) -> nn.Module:
    return LSTMModel(look_back, look_forward) if family == "lstm" else FiLMModel(look_back, look_forward)


def _all_samples(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_roads = x.shape[2]
    num_windows = x.shape[0]
    sx = x.transpose(2, 0, 1).reshape(-1, x.shape[1], 1).astype(np.float32)
    sy = y.transpose(2, 0, 1).reshape(-1, y.shape[1], 1).astype(np.float32)
    road_ids = np.repeat(np.arange(num_roads, dtype=np.int64), num_windows)
    time_ids = np.tile(np.arange(num_windows, dtype=np.int64), num_roads)
    return sx, sy, road_ids, time_ids


class ResidualTaskGate(nn.Module):
    def __init__(self, look_back: int, num_tasks: int, hidden: int = 64, init_alpha: float = 0.05):
        super().__init__()
        if not 0 < init_alpha < 1:
            raise ValueError("init_alpha must be in (0,1)")
        self.num_tasks = num_tasks
        # Raw context + mean/std/min/max/trend + static one-hot.
        input_dim = look_back + 5 + num_tasks
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.task_head = nn.Linear(hidden, num_tasks)
        self.alpha_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.task_head.weight)
        nn.init.zeros_(self.task_head.bias)
        nn.init.zeros_(self.alpha_head.weight)
        self.alpha_head.bias.data.fill_(math.log(init_alpha / (1.0 - init_alpha)))

    def forward(self, x: torch.Tensor, static_task: torch.Tensor):
        context = x.squeeze(-1)
        mean = context.mean(dim=1, keepdim=True)
        std = context.std(dim=1, keepdim=True, unbiased=False)
        minimum = context.min(dim=1, keepdim=True).values
        maximum = context.max(dim=1, keepdim=True).values
        trend = context[:, -1:] - context[:, :1]
        prior = torch.nn.functional.one_hot(static_task, self.num_tasks).to(context.dtype)
        features = torch.cat([context, mean, std, minimum, maximum, trend, prior], dim=1)
        h = self.encoder(features)
        residual = torch.softmax(self.task_head(h), dim=1)
        alpha = torch.sigmoid(self.alpha_head(h))
        weights = (1.0 - alpha) * prior + alpha * residual
        return weights, alpha.squeeze(1)


def _limited(loader: Iterable, max_batches: int):
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        yield batch


def _expert_stack(experts: List[nn.Module], data: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack([expert(data) for expert in experts], dim=1)  # [B,K,H,1]


def _eval_loss(
    gate: ResidualTaskGate,
    experts: List[nn.Module],
    loader: DataLoader,
    labels_tensor: torch.Tensor,
    device: torch.device,
    max_batches: int,
) -> float:
    gate.eval()
    losses = []
    with torch.no_grad():
        for data, target, road in _limited(loader, max_batches):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            road = road.to(device, non_blocking=True)
            static_task = labels_tensor[road]
            preds = _expert_stack(experts, data)
            weights, _ = gate(data, static_task)
            mixed = (preds * weights[:, :, None, None]).sum(dim=1)
            losses.append(float(nn.functional.mse_loss(mixed, target).cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def _evaluate(
    gate: ResidualTaskGate,
    experts: List[nn.Module],
    loader: DataLoader,
    labels: np.ndarray,
    labels_tensor: torch.Tensor,
    scale: float,
    device: torch.device,
    max_batches: int,
) -> Dict:
    gate.eval()
    target_chunks, static_chunks, soft_chunks = [], [], []
    alpha_chunks, weight_chunks, road_chunks, time_chunks = [], [], [], []
    next_time = {}
    with torch.no_grad():
        for data, target, road, time_id in _limited(loader, max_batches):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            road = road.to(device, non_blocking=True)
            static_task = labels_tensor[road]
            preds = _expert_stack(experts, data)
            row = torch.arange(data.shape[0], device=device)
            static_pred = preds[row, static_task]
            weights, alpha = gate(data, static_task)
            soft_pred = (preds * weights[:, :, None, None]).sum(dim=1)
            target_chunks.append((target * scale).cpu().numpy())
            static_chunks.append((static_pred * scale).cpu().numpy())
            soft_chunks.append((soft_pred * scale).cpu().numpy())
            alpha_chunks.append(alpha.cpu().numpy())
            weight_chunks.append(weights.cpu().numpy())
            road_chunks.append(road.cpu().numpy())
            time_chunks.append(time_id.numpy())

    target = np.concatenate(target_chunks).reshape(-1)
    static_pred = np.concatenate(static_chunks).reshape(-1)
    soft_pred = np.concatenate(soft_chunks).reshape(-1)
    alpha = np.concatenate(alpha_chunks)
    weights = np.concatenate(weight_chunks, axis=0)
    roads = np.concatenate(road_chunks)
    times = np.concatenate(time_chunks)
    dynamic_task = weights.argmax(axis=1)
    static_task = labels[roads]

    changes = transitions = roads_with_change = 0
    for road in np.unique(roads):
        idx = np.flatnonzero(roads == road)
        idx = idx[np.argsort(times[idx])]
        seq = dynamic_task[idx]
        if len(seq) > 1:
            diff = seq[1:] != seq[:-1]
            changes += int(diff.sum())
            transitions += int(diff.size)
            roads_with_change += int(diff.any())

    entropy = -(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum(axis=1)
    return {
        "static_hard": _metrics(static_pred, target),
        "dynamic_soft": _metrics(soft_pred, target),
        "routing": {
            "mean_alpha": float(alpha.mean()),
            "median_alpha": float(np.median(alpha)),
            "p90_alpha": float(np.quantile(alpha, 0.9)),
            "mean_entropy": float(entropy.mean()),
            "mean_max_weight": float(weights.max(axis=1).mean()),
            "dynamic_vs_static_argmax_disagreement": float((dynamic_task != static_task).mean()),
            "temporal_task_change_rate": float(changes / max(transitions, 1)),
            "roads_with_temporal_task_change": float(roads_with_change / max(len(np.unique(roads)), 1)),
            "task_usage": weights.mean(axis=0).tolist(),
        },
        "num_window_road_samples": int(weights.shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Learned residual dynamic soft routing")
    ap.add_argument("--family", choices=["lstm", "film"], default="lstm")
    ap.add_argument("--dataset", choices=["beijing", "shanghai", "largest"], default="beijing")
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--checkpoint-dir", default="param/4090_tuned/lstm/beijing")
    ap.add_argument("--output-dir", default="param/journal/dynamic_gate_v1/beijing_lstm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--init-alpha", type=float, default=0.05)
    ap.add_argument("--train-max-batches", type=int, default=50)
    ap.add_argument("--val-max-batches", type=int, default=0)
    ap.add_argument("--test-max-batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed)
    device = _device(args.device)
    look_back, look_forward, k = 12, 6, args.clusters
    flow, ids, scale = _load_flow(args.dataset)
    labels = _cluster_features(args.dataset, args.family, flow, ids, k, args.seed)
    fit_x, fit_y, val_x, val_y, test_x, test_y = _windows(flow, look_back, look_forward)

    def make_loader(x, y, shuffle, batch_size):
        sx, sy, roads, times = _all_samples(x, y)
        ds = TensorDataset(
            torch.from_numpy(sx), torch.from_numpy(sy),
            torch.from_numpy(roads), torch.from_numpy(times),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                          pin_memory=device.type == "cuda")

    train_loader = make_loader(fit_x, fit_y, True, args.batch_size)
    val_loader = make_loader(val_x, val_y, False, args.batch_size)
    test_loader = make_loader(test_x, test_y, False, args.batch_size)

    ckpt_dir = Path(args.checkpoint_dir)
    experts: List[nn.Module] = []
    for task in range(k):
        model = _model(args.family, look_back, look_forward).to(device)
        model.load_state_dict(torch.load(ckpt_dir / f"cluster_{task}.pt", map_location=device, weights_only=True))
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if args.family == "lstm":
            model.lstm.flatten_parameters()
        experts.append(model)

    labels_tensor = torch.from_numpy(labels).long().to(device)
    gate = ResidualTaskGate(look_back, k, hidden=args.hidden, init_alpha=args.init_alpha).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_state = copy.deepcopy(gate.state_dict())
    best_val = float("inf")
    history = []

    for epoch in range(args.epochs):
        gate.train()
        losses = []
        alphas = []
        for data, target, road, _ in _limited(train_loader, args.train_max_batches):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            road = road.to(device, non_blocking=True)
            static_task = labels_tensor[road]
            preds = _expert_stack(experts, data)
            weights, alpha = gate(data, static_task)
            mixed = (preds * weights[:, :, None, None]).sum(dim=1)
            loss = nn.functional.mse_loss(mixed, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            alphas.append(float(alpha.detach().mean().cpu()))

        val_loss = _eval_loss(gate, experts, val_loader, labels_tensor, device, args.val_max_batches)
        row = {
            "epoch": epoch + 1,
            "train_mse": float(np.mean(losses)),
            "val_mse": val_loss,
            "mean_train_alpha": float(np.mean(alphas)),
        }
        history.append(row)
        print(json.dumps(row))
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(gate.state_dict())
            torch.save(best_state, out / "gate_best.pt")

    gate.load_state_dict(best_state)
    val_result = _evaluate(gate, experts, val_loader, labels, labels_tensor, scale, device, args.val_max_batches)
    test_result = _evaluate(gate, experts, test_loader, labels, labels_tensor, scale, device, args.test_max_batches)
    result = {
        "experiment": "learned_residual_dynamic_gate_v1",
        "family": args.family,
        "dataset": args.dataset,
        "clusters": k,
        "checkpoint_dir": str(ckpt_dir),
        "best_val_normalized_mse": best_val,
        "history": history,
        "validation": val_result,
        "test": test_result,
    }
    result["test"]["delta_mae"] = (
        result["test"]["dynamic_soft"]["MAE"] - result["test"]["static_hard"]["MAE"]
    )
    result["test"]["relative_mae_pct"] = 100.0 * (
        result["test"]["dynamic_soft"]["MAE"] / result["test"]["static_hard"]["MAE"] - 1.0
    )
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
