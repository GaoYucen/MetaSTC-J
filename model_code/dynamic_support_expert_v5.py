"""V5 support-scored dynamic discovery over the reproduced full task experts.

This is the cleanest test of the journal hypothesis so far. The conference
pipeline assigns each road to one static KMeans task and always uses that task's
adapted expert. V5 keeps the *same reproduced experts* but infers the task online
from a recent support episode that is fully observed before the forecast time.

For each forecast episode:
  support x: t-18..t-7, support y: t-6..t-1 (already observed)
  query x:   t-12..t-1, query y:   t..t+5   (future, evaluation only)

All K reproduced experts are scored on the support pair. Validation chooses a
routing rule using support errors only; the held-out test split is evaluated
exactly once with that selected rule. No future query target is used in routing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from optimized_runner import LSTMModel, _cluster_features, _device, _load_flow, _metrics, _set_seed


def _episodic_windows(flow: np.ndarray):
    train_size = int(flow.shape[1] * 0.8)
    train_part = flow[:, :train_size].T
    test_part = flow[:, train_size:].T

    def build(part: np.ndarray):
        count = part.shape[0] - 24
        if count <= 0:
            raise ValueError("not enough steps")
        sx = np.stack([part[i:i + 12] for i in range(count)]).astype(np.float32)
        sy = np.stack([part[i + 12:i + 18] for i in range(count)]).astype(np.float32)
        qx = np.stack([part[i + 6:i + 18] for i in range(count)]).astype(np.float32)
        qy = np.stack([part[i + 18:i + 24] for i in range(count)]).astype(np.float32)
        return sx, sy, qx, qy

    train = build(train_part)
    test = build(test_part)
    split = max(1, int(train[0].shape[0] * 0.9))
    fit = tuple(a[:split] for a in train)
    val = tuple(a[split:] for a in train)
    return fit, val, test


def _flatten(arrays):
    sx, sy, qx, qy = arrays
    roads, windows = sx.shape[2], sx.shape[0]
    def fx(x): return x.transpose(2, 0, 1).reshape(-1, x.shape[1], 1).astype(np.float32)
    road_ids = np.repeat(np.arange(roads, dtype=np.int64), windows)
    time_ids = np.tile(np.arange(windows, dtype=np.int64), roads)
    return fx(sx), fx(sy), fx(qx), fx(qy), road_ids, time_ids


def _loader(arrays, batch_size, device):
    values = _flatten(arrays)
    ds = TensorDataset(*(torch.from_numpy(v) for v in values))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")


def _limited(loader: Iterable, max_batches: int):
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        yield batch


def _load_experts(ckpt: Path, k: int, device) -> List[LSTMModel]:
    experts = []
    for task in range(k):
        model = LSTMModel(12, 6).to(device)
        model.load_state_dict(torch.load(ckpt / f"cluster_{task}.pt", map_location=device, weights_only=True))
        model.eval(); model.lstm.flatten_parameters()
        for p in model.parameters(): p.requires_grad_(False)
        experts.append(model)
    return experts


def _collect(experts, loader, labels_tensor, device, scale, max_batches):
    """Return support errors and query predictions for routing evaluation."""
    support_errors, query_preds, targets, static_tasks, roads_all, times_all = [], [], [], [], [], []
    with torch.inference_mode():
        for sx, sy, qx, qy, road, time_id in _limited(loader, max_batches):
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            road = road.to(device)
            sp = torch.stack([expert(sx) for expert in experts], dim=1)  # B,K,6,1
            qp = torch.stack([expert(qx) for expert in experts], dim=1)
            err = ((sp - sy[:, None]) ** 2).mean(dim=(2, 3))
            support_errors.append(err.cpu().numpy())
            query_preds.append((qp * scale).cpu().numpy())
            targets.append((qy * scale).cpu().numpy())
            static_tasks.append(labels_tensor[road].cpu().numpy())
            roads_all.append(road.cpu().numpy())
            times_all.append(time_id.numpy())
    return {
        "errors": np.concatenate(support_errors),
        "query_preds": np.concatenate(query_preds),
        "targets": np.concatenate(targets),
        "static_task": np.concatenate(static_tasks),
        "roads": np.concatenate(roads_all),
        "times": np.concatenate(times_all),
    }


def _soft_weights(errors, static_task, k, temperature, prior_strength):
    best = errors.min(axis=1, keepdims=True)
    spread = errors.std(axis=1, keepdims=True)
    spread = np.maximum(spread, 1e-6)
    logits = -(errors - best) / (spread * temperature)
    if prior_strength:
        logits[np.arange(len(logits)), static_task] += prior_strength
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _prediction_from_tasks(query_preds, task):
    return query_preds[np.arange(len(task)), task]


def _metrics_for_config(data, config, k):
    errors = data["errors"]
    qp = data["query_preds"]
    target = data["targets"].reshape(-1)
    static_task = data["static_task"]
    kind = config["kind"]

    if kind == "soft":
        weights = _soft_weights(errors, static_task, k, config["temperature"], config["prior"])
        pred = (qp * weights[:, :, None, None]).sum(axis=1)
        selected = weights.argmax(axis=1)
        confidence = weights.max(axis=1)
    elif kind == "hard":
        weights = _soft_weights(errors, static_task, k, 1.0, config["prior"])
        selected = weights.argmax(axis=1)
        pred = _prediction_from_tasks(qp, selected)
        confidence = weights.max(axis=1)
    elif kind == "selective":
        dynamic = errors.argmin(axis=1)
        spread = np.maximum(errors.std(axis=1), 1e-6)
        advantage = (errors[np.arange(len(errors)), static_task] - errors[np.arange(len(errors)), dynamic]) / spread
        switch = (dynamic != static_task) & (advantage >= config["threshold"])
        selected = np.where(switch, dynamic, static_task)
        pred = _prediction_from_tasks(qp, selected)
        confidence = advantage
        weights = None
    else:
        raise ValueError(kind)

    result = _metrics(pred.reshape(-1), target)
    result["switch_rate"] = float((selected != static_task).mean())
    result["mean_confidence"] = float(np.mean(confidence))
    result["selected_task"] = selected
    if weights is not None:
        entropy = -(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum(axis=1)
        result["mean_entropy"] = float(entropy.mean())
        result["mean_max_weight"] = float(weights.max(axis=1).mean())
    return result


def _static_metrics(data):
    pred = _prediction_from_tasks(data["query_preds"], data["static_task"])
    return _metrics(pred.reshape(-1), data["targets"].reshape(-1))


def _routing_dynamics(data, selected):
    roads, times = data["roads"], data["times"]
    changes = transitions = roads_changed = 0
    for road in np.unique(roads):
        idx = np.flatnonzero(roads == road)
        idx = idx[np.argsort(times[idx])]
        seq = selected[idx]
        if len(seq) > 1:
            diff = seq[1:] != seq[:-1]
            changes += int(diff.sum()); transitions += int(diff.size); roads_changed += int(diff.any())
    return {
        "temporal_task_change_rate": float(changes / max(transitions, 1)),
        "roads_with_temporal_task_change": float(roads_changed / max(len(np.unique(roads)), 1)),
    }


def main():
    ap = argparse.ArgumentParser(description="Support-scored dynamic routing over reproduced experts")
    ap.add_argument("--dataset", choices=["beijing", "shanghai", "largest"], default="beijing")
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--checkpoint-dir", default="param/4090_tuned/lstm/beijing")
    ap.add_argument("--output-dir", default="param/journal/dynamic_support_expert_v5/beijing_lstm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--val-max-batches", type=int, default=4)
    ap.add_argument("--test-max-batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed); device = _device(args.device); k = args.clusters
    flow, ids, scale = _load_flow(args.dataset)
    labels = _cluster_features(args.dataset, "lstm", flow, ids, k, args.seed)
    _, val, test = _episodic_windows(flow)
    val_loader = _loader(val, args.batch_size, device); test_loader = _loader(test, args.batch_size, device)
    ckpt = Path(args.checkpoint_dir); experts = _load_experts(ckpt, k, device)
    labels_tensor = torch.from_numpy(labels).long().to(device)

    print("collecting validation expert predictions")
    val_data = _collect(experts, val_loader, labels_tensor, device, scale, args.val_max_batches)
    static_val = _static_metrics(val_data)

    configs = []
    for temp in [0.15, 0.25, 0.4, 0.6, 1.0, 1.5]:
        for prior in [0.0, 0.1, 0.25, 0.5, 1.0]:
            configs.append({"kind": "soft", "temperature": temp, "prior": prior})
    for prior in [0.0, 0.1, 0.25, 0.5, 1.0]:
        configs.append({"kind": "hard", "prior": prior})
    for threshold in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        configs.append({"kind": "selective", "threshold": threshold})

    val_rows = []
    for config in configs:
        metrics = _metrics_for_config(val_data, config, k)
        row = {"config": config, "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "switch_rate": metrics["switch_rate"]}
        val_rows.append(row)
    val_rows.sort(key=lambda r: (r["MAE"], r["RMSE"]))
    selected_config = val_rows[0]["config"]
    selected_val = _metrics_for_config(val_data, selected_config, k)
    selected_val_task = selected_val.pop("selected_task")
    selected_val.update(_routing_dynamics(val_data, selected_val_task))

    print("VALIDATION_STATIC", json.dumps(static_val))
    print("VALIDATION_TOP10")
    for row in val_rows[:10]: print(json.dumps(row))
    print("SELECTED_BY_VALIDATION", json.dumps(selected_config))

    # Test is touched only after validation model selection above is complete.
    print("collecting held-out test expert predictions")
    test_data = _collect(experts, test_loader, labels_tensor, device, scale, args.test_max_batches)
    static_test = _static_metrics(test_data)
    selected_test = _metrics_for_config(test_data, selected_config, k)
    selected_test_task = selected_test.pop("selected_task")
    selected_test.update(_routing_dynamics(test_data, selected_test_task))

    result = {
        "experiment": "support_scored_dynamic_full_experts_v5",
        "selection_protocol": "routing rule selected by validation MAE before test evaluation",
        "dataset": args.dataset,
        "clusters": k,
        "validation_static_hard": static_val,
        "validation_selected_dynamic": selected_val,
        "selected_config": selected_config,
        "heldout_test_static_hard": static_test,
        "heldout_test_selected_dynamic": selected_test,
        "test_delta_mae": selected_test["MAE"] - static_test["MAE"],
        "test_relative_mae_pct": 100.0 * (selected_test["MAE"] / static_test["MAE"] - 1.0),
        "validation_top10": val_rows[:10],
    }
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL_RESULT")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
