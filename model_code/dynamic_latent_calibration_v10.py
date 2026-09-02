"""V10 support-calibrated continuous latent adaptation.

V9 shows that a continuous task-conditioned parameter generator can improve the
validation segment strongly, but its amortized encoder can still extrapolate
poorly to a later temporal segment.  V10 keeps the trained V9 generator frozen
and performs a small inner-loop update only on the per-sample adapter mixture
logits, using the already-observed support pair.  No model weights or future
query labels are updated at inference time.

The V9 encoder supplies an amortized initialization c_0.  Before forecasting,
we refine logits l around log(c_0) by minimizing support prediction error plus
a proximal penalty.  Hyperparameters are selected on validation only, then the
selected rule is applied once to the held-out test segment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from optimized_runner import LSTMModel, _cluster_features, _device, _load_flow, _metrics, _set_seed
from dynamic_support_routing_v4 import _episodic_windows, _load_static_experts, _static_prediction
from dynamic_continuous_generator_v9 import ContinuousTaskGenerator, _make_loader


def _collect_cache(model, experts, loader, labels_tensor, device, max_batches=0):
    cache = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (sx, sy, qx, qy, road, time_id) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            road = road.to(device)
            _, coeff = model.infer_coefficients(sx, sy, qx, 1.0)
            sflat, sbase = model._features(sx)
            _, sbasis = model._apply_coeff(sflat, sbase, coeff)
            qflat, qbase = model._features(qx)
            _, qbasis = model._apply_coeff(qflat, qbase, coeff)
            static = _static_prediction(experts, qx, labels_tensor[road])
            cache.append({
                "init_logits": torch.log(coeff.clamp_min(1e-8)).cpu(),
                "sbase": sbase.cpu(), "sbasis": sbasis.cpu(), "sy": sy.squeeze(-1).cpu(),
                "qbase": qbase.cpu(), "qbasis": qbasis.cpu(), "qy": qy.squeeze(-1).cpu(),
                "static": static.squeeze(-1).cpu(),
                "road": road.cpu(), "time": time_id.cpu(),
            })
    return cache


def _calibrate_logits(init_logits, sbase, sbasis, sy, steps, lr, prox_weight, max_delta):
    if steps <= 0 or lr <= 0:
        return init_logits
    logits = init_logits.detach().clone().requires_grad_(True)
    init = init_logits.detach()
    for _ in range(steps):
        coeff = torch.softmax(logits, dim=1)
        pred = sbase + (sbasis * coeff[:, :, None]).sum(dim=1)
        per_sample = ((pred - sy) ** 2).mean(dim=1)
        prox = ((logits - init) ** 2).mean(dim=1)
        loss = (per_sample + prox_weight * prox).sum()
        grad = torch.autograd.grad(loss, logits, only_inputs=True)[0]
        logits = (logits - lr * grad).detach()
        if max_delta > 0:
            logits = torch.maximum(torch.minimum(logits, init + max_delta), init - max_delta)
        logits.requires_grad_(True)
    return logits.detach()


def _replay(cache, scale, device, steps, lr, prox_weight, max_delta, collect_details=False):
    preds, targets, statics = [], [], []
    init_preds = []
    coeff_before, coeff_after = [], []
    support_before, support_after = [], []
    roads, times = [], []
    for item in cache:
        init_logits = item["init_logits"].to(device)
        sbase = item["sbase"].to(device); sbasis = item["sbasis"].to(device); sy = item["sy"].to(device)
        qbase = item["qbase"].to(device); qbasis = item["qbasis"].to(device); qy = item["qy"].to(device)
        c0 = torch.softmax(init_logits, dim=1)
        init_q = qbase + (qbasis * c0[:, :, None]).sum(dim=1)
        init_s = sbase + (sbasis * c0[:, :, None]).sum(dim=1)
        logits = _calibrate_logits(init_logits, sbase, sbasis, sy, steps, lr, prox_weight, max_delta)
        c = torch.softmax(logits, dim=1)
        pred = qbase + (qbasis * c[:, :, None]).sum(dim=1)
        cal_s = sbase + (sbasis * c[:, :, None]).sum(dim=1)
        preds.append(pred.cpu().numpy()); targets.append(qy.cpu().numpy())
        statics.append(item["static"].numpy()); init_preds.append(init_q.cpu().numpy())
        coeff_before.append(c0.cpu().numpy()); coeff_after.append(c.cpu().numpy())
        support_before.append(((init_s - sy) ** 2).mean(dim=1).cpu().numpy())
        support_after.append(((cal_s - sy) ** 2).mean(dim=1).cpu().numpy())
        roads.append(item["road"].numpy()); times.append(item["time"].numpy())

    pred = np.concatenate(preds); target = np.concatenate(targets)
    static = np.concatenate(statics); init_pred = np.concatenate(init_preds)
    c0 = np.concatenate(coeff_before); c = np.concatenate(coeff_after)
    sb = np.concatenate(support_before); sa = np.concatenate(support_after)
    result = _metrics((pred * scale).reshape(-1), (target * scale).reshape(-1))
    result["static_metrics"] = _metrics((static * scale).reshape(-1), (target * scale).reshape(-1))
    result["v9_amortized_metrics"] = _metrics((init_pred * scale).reshape(-1), (target * scale).reshape(-1))
    result["support_mse_before"] = float(sb.mean())
    result["support_mse_after"] = float(sa.mean())
    result["mean_coefficient_l1_shift"] = float(np.abs(c - c0).sum(axis=1).mean())
    result["mean_max_coefficient_before"] = float(c0.max(axis=1).mean())
    result["mean_max_coefficient_after"] = float(c.max(axis=1).mean())
    if collect_details:
        roads_arr = np.concatenate(roads); times_arr = np.concatenate(times)
        arg0 = c0.argmax(axis=1); arg1 = c.argmax(axis=1)
        result["basis_argmax_change_from_v9"] = float((arg0 != arg1).mean())
        changes = total = roads_changed = 0
        for r in np.unique(roads_arr):
            idx = np.flatnonzero(roads_arr == r); idx = idx[np.argsort(times_arr[idx])]
            if idx.size > 1:
                d = arg1[idx][1:] != arg1[idx][:-1]
                changes += int(d.sum()); total += int(d.size); roads_changed += int(d.any())
        result["temporal_argmax_change_rate"] = float(changes / max(total, 1))
        result["roads_with_temporal_change"] = float(roads_changed / max(len(np.unique(roads_arr)), 1))
    return result


def main():
    ap = argparse.ArgumentParser(description="V10 support-calibrated continuous latent adaptation")
    ap.add_argument("--dataset", default="beijing", choices=["beijing", "shanghai", "largest"])
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--checkpoint-dir", default="param/4090_tuned/lstm/beijing")
    ap.add_argument("--v9-checkpoint", default="param/journal/dynamic_continuous_v9/beijing_lstm_full/model_best.pt")
    ap.add_argument("--output-dir", default="param/journal/dynamic_latent_calibration_v10/beijing_lstm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--steps-grid", type=int, nargs="+", default=[0, 1, 3, 5])
    ap.add_argument("--lr-grid", type=float, nargs="+", default=[0.03, 0.1, 0.3])
    ap.add_argument("--prox-grid", type=float, nargs="+", default=[0.0, 0.05])
    ap.add_argument("--max-delta", type=float, default=2.0)
    ap.add_argument("--val-max-batches", type=int, default=0)
    ap.add_argument("--test-max-batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed); device = _device(args.device)
    flow, ids, scale = _load_flow(args.dataset)
    labels = _cluster_features(args.dataset, "lstm", flow, ids, args.clusters, args.seed)
    _, val_arrays, test_arrays = _episodic_windows(flow)
    val_loader = _make_loader(val_arrays, args.batch_size, device, False)
    test_loader = _make_loader(test_arrays, args.batch_size, device, False)
    ckpt = Path(args.checkpoint_dir)
    base = LSTMModel(12, 6).to(device)
    base.load_state_dict(torch.load(ckpt / "global_best.pt", map_location=device, weights_only=True))
    model = ContinuousTaskGenerator(base, num_bases=8, rank=4, latent_dim=16, hidden_dim=48).to(device)
    model.load_state_dict(torch.load(args.v9_checkpoint, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    experts = _load_static_experts(ckpt, args.clusters, device)
    labels_tensor = torch.from_numpy(labels).long().to(device)

    print("collecting validation cache")
    val_cache = _collect_cache(model, experts, val_loader, labels_tensor, device, args.val_max_batches)
    grid = []
    seen = set()
    for steps in args.steps_grid:
        if steps == 0:
            cfgs = [(0, 0.0, 0.0)]
        else:
            cfgs = [(steps, lr, prox) for lr in args.lr_grid for prox in args.prox_grid]
        for st, lr, prox in cfgs:
            key = (st, lr, prox)
            if key in seen: continue
            seen.add(key)
            m = _replay(val_cache, scale, device, st, lr, prox, args.max_delta, False)
            row = {"steps": st, "lr": lr, "prox": prox, "MAE": m["MAE"], "RMSE": m["RMSE"],
                   "support_before": m["support_mse_before"], "support_after": m["support_mse_after"],
                   "coeff_l1_shift": m["mean_coefficient_l1_shift"]}
            grid.append(row); print("VAL_CANDIDATE", json.dumps(row))
    grid.sort(key=lambda x: (x["MAE"], x["RMSE"]))
    best = grid[0]
    selected = {k: best[k] for k in ["steps", "lr", "prox"]}
    print("SELECTED_BY_VALIDATION", json.dumps(selected))
    val_result = _replay(val_cache, scale, device, selected["steps"], selected["lr"], selected["prox"], args.max_delta, True)

    print("collecting held-out test cache")
    test_cache = _collect_cache(model, experts, test_loader, labels_tensor, device, args.test_max_batches)
    test_result = _replay(test_cache, scale, device, selected["steps"], selected["lr"], selected["prox"], args.max_delta, True)
    result = {
        "experiment": "dynamic_latent_calibration_v10",
        "protocol": "V9 amortized latent initialization + causal support-only inner-loop coefficient calibration; config selected on validation only",
        "selected_config": selected,
        "validation_grid": grid,
        "validation": val_result,
        "heldout_test": test_result,
    }
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL_RESULT")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
