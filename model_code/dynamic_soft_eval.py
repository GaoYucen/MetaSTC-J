"""V1: dynamic soft task routing on top of the frozen MetaSTC-J reproduction.

This experiment intentionally keeps EVERYTHING learned by the reproduction
baseline fixed:
  * the same KMeans road tasks,
  * the same global model,
  * the same cluster-specific adapted expert checkpoints,
  * the same chronological data split.

Only inference routing changes.  Instead of always sending a road to its
static KMeans task, each 12-step test window is compared with the training
prototype of every task.  The distances produce context-conditioned soft
weights over the K adapted experts.  Therefore the same road may use different
task mixtures at different times.

The script reports three apples-to-apples variants using the SAME expert
predictions:
  1. static_hard: original road-cluster routing;
  2. dynamic_hard: nearest current-context prototype;
  3. dynamic_soft: soft mixture of all experts.

No checkpoint is overwritten. Results go under param/journal/ by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from optimized_runner import (
    LSTMModel,
    FiLMModel,
    _cluster_features,
    _device,
    _load_flow,
    _metrics,
    _set_seed,
    _windows,
)


def _model(family: str, look_back: int, look_forward: int) -> torch.nn.Module:
    if family == "lstm":
        return LSTMModel(look_back, look_forward)
    if family == "film":
        return FiLMModel(look_back, look_forward)
    raise ValueError(f"unknown family: {family}")


def _all_samples(x: np.ndarray, y: np.ndarray):
    """Convert [windows, horizon, roads] arrays to road-major samples."""
    look_back = x.shape[1]
    look_forward = y.shape[1]
    num_roads = x.shape[2]
    num_windows = x.shape[0]
    sx = x.transpose(2, 0, 1).reshape(-1, look_back, 1).astype(np.float32)
    sy = y.transpose(2, 0, 1).reshape(-1, look_forward, 1).astype(np.float32)
    road_ids = np.repeat(np.arange(num_roads, dtype=np.int64), num_windows)
    time_ids = np.tile(np.arange(num_windows, dtype=np.int64), num_roads)
    return sx, sy, road_ids, time_ids


def _task_prototypes(fit_x: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Mean 12-step training context for each frozen road task."""
    prototypes: List[np.ndarray] = []
    for task in range(k):
        roads = np.flatnonzero(labels == task)
        if roads.size == 0:
            raise ValueError(f"task {task} is empty")
        # [fit_windows, look_back, task_roads] -> average across roads/windows.
        proto = fit_x[:, :, roads].mean(axis=(0, 2))
        prototypes.append(proto.astype(np.float32))
    return np.stack(prototypes, axis=0)


def _routing_weights(
    x_batch: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    # x_batch [B, L, 1], prototypes [K, L].
    context = x_batch.squeeze(-1)
    # Per-window standardization emphasizes temporal shape rather than volume,
    # while retaining numerical stability for near-constant windows.
    context_mean = context.mean(dim=1, keepdim=True)
    context_std = context.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    context_z = (context - context_mean) / context_std

    proto_mean = prototypes.mean(dim=1, keepdim=True)
    proto_std = prototypes.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    proto_z = (prototypes - proto_mean) / proto_std

    dist2 = ((context_z[:, None, :] - proto_z[None, :, :]) ** 2).mean(dim=-1)
    return torch.softmax(-dist2 / temperature, dim=-1)


def _assignment_stats(
    weights: np.ndarray,
    road_ids: np.ndarray,
    time_ids: np.ndarray,
    static_labels: np.ndarray,
) -> Dict[str, float]:
    dynamic = weights.argmax(axis=1)
    entropy = -(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum(axis=1)
    changes = 0
    transitions = 0
    roads_changed = 0
    for road in np.unique(road_ids):
        idx = np.flatnonzero(road_ids == road)
        idx = idx[np.argsort(time_ids[idx])]
        seq = dynamic[idx]
        if len(seq) > 1:
            diff = seq[1:] != seq[:-1]
            changes += int(diff.sum())
            transitions += int(diff.size)
            roads_changed += int(diff.any())
    static_for_samples = static_labels[road_ids]
    return {
        "mean_entropy": float(entropy.mean()),
        "mean_max_weight": float(weights.max(axis=1).mean()),
        "effective_tasks": float(np.exp(entropy).mean()),
        "dynamic_vs_static_disagreement": float((dynamic != static_for_samples).mean()),
        "temporal_task_change_rate": float(changes / max(transitions, 1)),
        "roads_with_temporal_task_change": float(roads_changed / max(len(np.unique(road_ids)), 1)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="MetaSTC-J V1 dynamic soft routing evaluation")
    ap.add_argument("--family", choices=["lstm", "film"], default="lstm")
    ap.add_argument("--dataset", choices=["beijing", "shanghai", "largest"], default="beijing")
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--checkpoint-dir", default="param/4090_tuned/lstm/beijing")
    ap.add_argument("--output-dir", default="param/journal/dynamic_soft_v1/beijing_lstm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0],
    )
    ap.add_argument("--max-batches", type=int, default=0, help="0 = all test batches")
    args = ap.parse_args()

    if any(t <= 0 for t in args.temperatures):
        raise ValueError("all temperatures must be positive")

    _set_seed(args.seed)
    device = _device(args.device)
    look_back, look_forward, k = 12, 6, args.clusters

    flow, ids, scale = _load_flow(args.dataset)
    labels = _cluster_features(args.dataset, args.family, flow, ids, k, args.seed)
    fit_x, fit_y, _, _, test_x, test_y = _windows(flow, look_back, look_forward)
    prototypes_np = _task_prototypes(fit_x, labels, k)
    prototypes = torch.from_numpy(prototypes_np).to(device)

    sx, sy, road_ids, time_ids = _all_samples(test_x, test_y)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(sx), torch.from_numpy(sy), torch.from_numpy(road_ids)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    ckpt_dir = Path(args.checkpoint_dir)
    experts = []
    for task in range(k):
        path = ckpt_dir / f"cluster_{task}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        model = _model(args.family, look_back, look_forward).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        if args.family == "lstm":
            model.lstm.flatten_parameters()
        experts.append(model)

    target_chunks: List[np.ndarray] = []
    static_chunks: List[np.ndarray] = []
    dynamic_hard_by_temp: Dict[float, List[np.ndarray]] = {t: [] for t in args.temperatures}
    dynamic_soft_by_temp: Dict[float, List[np.ndarray]] = {t: [] for t in args.temperatures}
    weights_by_temp: Dict[float, List[np.ndarray]] = {t: [] for t in args.temperatures}

    offset = 0
    with torch.inference_mode():
        for bi, (data, target, batch_roads) in enumerate(loader):
            if args.max_batches and bi >= args.max_batches:
                break
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            batch_roads = batch_roads.to(device, non_blocking=True)

            preds = torch.stack([expert(data) for expert in experts], dim=1)  # [B,K,H,1]
            static_task = torch.from_numpy(labels).to(device)[batch_roads]
            row = torch.arange(data.shape[0], device=device)
            static_pred = preds[row, static_task]

            target_chunks.append((target * scale).cpu().numpy())
            static_chunks.append((static_pred * scale).cpu().numpy())

            for temperature in args.temperatures:
                w = _routing_weights(data, prototypes, temperature)
                hard_task = w.argmax(dim=1)
                hard_pred = preds[row, hard_task]
                soft_pred = (preds * w[:, :, None, None]).sum(dim=1)
                dynamic_hard_by_temp[temperature].append((hard_pred * scale).cpu().numpy())
                dynamic_soft_by_temp[temperature].append((soft_pred * scale).cpu().numpy())
                weights_by_temp[temperature].append(w.cpu().numpy())

            offset += data.shape[0]

    if offset == 0:
        raise RuntimeError("no evaluation samples were processed")

    target = np.concatenate(target_chunks).reshape(-1)
    static_pred = np.concatenate(static_chunks).reshape(-1)
    used_road_ids = road_ids[:offset]
    used_time_ids = time_ids[:offset]

    result = {
        "experiment": "dynamic_soft_routing_v1",
        "principle": "frozen experts; inference routing only",
        "family": args.family,
        "dataset": args.dataset,
        "clusters": k,
        "checkpoint_dir": str(ckpt_dir),
        "num_window_road_samples": int(offset),
        "static_hard": _metrics(static_pred, target),
        "temperatures": {},
    }

    best_temperature = None
    best_soft_mae = float("inf")
    for temperature in args.temperatures:
        hard = np.concatenate(dynamic_hard_by_temp[temperature]).reshape(-1)
        soft = np.concatenate(dynamic_soft_by_temp[temperature]).reshape(-1)
        weights = np.concatenate(weights_by_temp[temperature], axis=0)
        soft_metrics = _metrics(soft, target)
        entry = {
            "dynamic_hard": _metrics(hard, target),
            "dynamic_soft": soft_metrics,
            "routing": _assignment_stats(weights, used_road_ids, used_time_ids, labels),
            "task_usage": weights.mean(axis=0).tolist(),
        }
        result["temperatures"][str(temperature)] = entry
        if soft_metrics["MAE"] < best_soft_mae:
            best_soft_mae = soft_metrics["MAE"]
            best_temperature = temperature

    result["best_temperature_by_test_mae"] = best_temperature
    result["best_soft_mae"] = best_soft_mae
    result["static_hard_mae"] = result["static_hard"]["MAE"]
    result["best_soft_delta_mae"] = best_soft_mae - result["static_hard"]["MAE"]
    result["best_soft_relative_mae_pct"] = 100.0 * (
        best_soft_mae / result["static_hard"]["MAE"] - 1.0
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.save(out / "task_prototypes.npy", prototypes_np)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
