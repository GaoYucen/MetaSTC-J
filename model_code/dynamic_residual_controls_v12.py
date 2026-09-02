"""V12 diagnostic controls for V11 dynamic residual adaptation.

Fit simple residual-calibration baselines using only the causal fit split and
compare them on validation / future holdout:
  - global horizon bias
  - static-cluster horizon bias
  - support residual mean persistence
  - support residual vector persistence
  - global linear support-residual -> future-residual map
  - static-cluster-specific linear residual map

The purpose is diagnostic: determine whether V11 gains require nonlinear
current-context dynamics, or can be explained by simple calibration of the
frozen Static MetaSTC experts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from optimized_runner import _cluster_features, _device, _load_flow, _metrics, _set_seed
from dynamic_support_routing_v4 import (
    _all_samples, _episodic_windows, _limited, _load_static_experts, _static_prediction,
)


def _loader(arrays, batch_size, device, shuffle=False):
    sx, sy, qx, qy, roads, times = _all_samples(arrays)
    ds = TensorDataset(
        torch.from_numpy(sx), torch.from_numpy(sy), torch.from_numpy(qx), torch.from_numpy(qy),
        torch.from_numpy(roads), torch.from_numpy(times),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=device.type == "cuda")


@torch.no_grad()
def _fit_controls(experts, loader, labels_tensor, k, device, max_batches, ridge):
    h = 6
    global_sum = torch.zeros(h, device=device)
    cluster_sum = torch.zeros(k, h, device=device)
    global_n = 0
    cluster_n = torch.zeros(k, device=device)

    xtx = torch.zeros(1 + h, 1 + h, device=device, dtype=torch.float64)
    xty = torch.zeros(1 + h, h, device=device, dtype=torch.float64)
    c_xtx = torch.zeros(k, 1 + h, 1 + h, device=device, dtype=torch.float64)
    c_xty = torch.zeros(k, 1 + h, h, device=device, dtype=torch.float64)

    for sx, sy, qx, qy, road, _ in _limited(loader, max_batches):
        sx, sy, qx, qy, road = sx.to(device), sy.to(device), qx.to(device), qy.to(device), road.to(device)
        task = labels_tensor[road]
        ss = _static_prediction(experts, sx, task)
        sq = _static_prediction(experts, qx, task)
        sr = (sy - ss).squeeze(-1)
        qr = (qy - sq).squeeze(-1)

        global_sum += qr.sum(dim=0)
        global_n += qr.shape[0]
        for c in range(k):
            idx = task == c
            if idx.any():
                cluster_sum[c] += qr[idx].sum(dim=0)
                cluster_n[c] += int(idx.sum())

        x = torch.cat([torch.ones(sr.shape[0], 1, device=device), sr], dim=1).double()
        y = qr.double()
        xtx += x.T @ x
        xty += x.T @ y
        for c in range(k):
            idx = task == c
            if idx.any():
                xc = x[idx]; yc = y[idx]
                c_xtx[c] += xc.T @ xc
                c_xty[c] += xc.T @ yc

    if global_n == 0:
        raise RuntimeError("No fit samples")
    global_bias = global_sum / float(global_n)
    cluster_bias = cluster_sum / cluster_n.clamp_min(1)[:, None]

    reg = torch.eye(1 + h, device=device, dtype=torch.float64) * ridge
    reg[0, 0] = 0.0
    w = torch.linalg.solve(xtx + reg, xty).float()
    cw = []
    for c in range(k):
        cw.append(torch.linalg.solve(c_xtx[c] + reg, c_xty[c]).float())
    cluster_w = torch.stack(cw, dim=0)
    return {
        "global_bias": global_bias,
        "cluster_bias": cluster_bias,
        "ridge_global": w,
        "ridge_cluster": cluster_w,
        "fit_samples": int(global_n),
        "cluster_samples": [int(v) for v in cluster_n.cpu().tolist()],
    }


@torch.no_grad()
def _evaluate(controls, experts, loader, labels_tensor, scale_data, device, max_batches):
    names = [
        "static", "global_bias", "cluster_bias", "support_mean", "support_vector",
        "ridge_global", "ridge_cluster",
    ]
    target = []
    chunks = {name: [] for name in names}
    for sx, sy, qx, qy, road, _ in _limited(loader, max_batches):
        sx, sy, qx, qy, road = sx.to(device), sy.to(device), qx.to(device), qy.to(device), road.to(device)
        task = labels_tensor[road]
        ss = _static_prediction(experts, sx, task)
        sq = _static_prediction(experts, qx, task)
        sr = (sy - ss).squeeze(-1)
        base = sq.squeeze(-1)
        x = torch.cat([torch.ones(sr.shape[0], 1, device=device), sr], dim=1)

        pred = {
            "static": base,
            "global_bias": base + controls["global_bias"][None, :],
            "cluster_bias": base + controls["cluster_bias"][task],
            "support_mean": base + sr.mean(dim=1, keepdim=True),
            "support_vector": base + sr,
            "ridge_global": base + x @ controls["ridge_global"],
            "ridge_cluster": torch.empty_like(base),
        }
        for c in range(controls["ridge_cluster"].shape[0]):
            idx = task == c
            if idx.any():
                pred["ridge_cluster"][idx] = base[idx] + x[idx] @ controls["ridge_cluster"][c]

        target.append((qy.squeeze(-1) * scale_data).cpu().numpy())
        for name in names:
            chunks[name].append((pred[name] * scale_data).cpu().numpy())

    y = np.concatenate(target).reshape(-1)
    out = {}
    static_mae = None
    for name in names:
        p = np.concatenate(chunks[name]).reshape(-1)
        met = _metrics(p, y)
        if name == "static":
            static_mae = met["MAE"]
        met["relative_mae_vs_static_pct"] = 0.0 if name == "static" else 100.0 * (met["MAE"] / static_mae - 1.0)
        out[name] = met
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beijing", choices=["beijing", "shanghai", "largest"])
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--v11-metrics", default="")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--fit-max-batches", type=int, default=0)
    ap.add_argument("--val-max-batches", type=int, default=0)
    ap.add_argument("--test-max-batches", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed)
    device = _device(args.device)
    flow, ids, scale_data = _load_flow(args.dataset)
    labels = _cluster_features(args.dataset, "lstm", flow, ids, args.clusters, args.seed)
    fit, val, test = _episodic_windows(flow)
    fl = _loader(fit, args.batch_size, device, False)
    vl = _loader(val, args.batch_size, device, False)
    tl = _loader(test, args.batch_size, device, False)
    experts = _load_static_experts(Path(args.checkpoint_dir), args.clusters, device)
    labels_tensor = torch.from_numpy(labels).long().to(device)

    controls = _fit_controls(experts, fl, labels_tensor, args.clusters, device, args.fit_max_batches, args.ridge)
    validation = _evaluate(controls, experts, vl, labels_tensor, scale_data, device, args.val_max_batches)
    future = _evaluate(controls, experts, tl, labels_tensor, scale_data, device, args.test_max_batches)

    v11 = None
    if args.v11_metrics and Path(args.v11_metrics).exists():
        m = json.loads(Path(args.v11_metrics).read_text())
        v11 = {
            "validation": m["validation"]["static_anchored_dynamic"],
            "validation_relative_mae_vs_static_pct": m["validation"]["dynamic_vs_static_relative_mae_pct"],
            "future": m["exploratory_future_segment"]["static_anchored_dynamic"],
            "future_relative_mae_vs_static_pct": m["exploratory_future_segment"]["dynamic_vs_static_relative_mae_pct"],
        }

    result = {
        "experiment": "dynamic_residual_controls_v12",
        "dataset": args.dataset,
        "protocol": "simple residual controls fit on causal fit split only; validation/future untouched",
        "config": vars(args),
        "fit_samples": controls["fit_samples"],
        "cluster_samples": controls["cluster_samples"],
        "validation": validation,
        "future": future,
        "v11_c": v11,
    }
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("V12_RESULT")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
