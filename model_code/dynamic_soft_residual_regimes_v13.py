"""V13: dynamic soft residual regimes beyond linear calibration.

A strong static-cluster-specific ridge residual calibrator is fitted on the causal
fit split and frozen.  The learned model is only allowed to add a bounded
correction on top of that strong anchor.

Two learned controls are trained with the same features/losses:
  1) single residual transition map (no latent regime discovery)
  2) soft mixture of M residual transition maps, dynamically routed from the
     current normalized spatio-temporal context.

This makes the key diagnostic explicit: dynamic task discovery is useful only
if the soft-regime model consistently beats both ridge-cluster calibration and
the single-map learned control on untouched future windows.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from optimized_runner import _cluster_features, _device, _load_flow, _metrics, _set_seed
from dynamic_support_routing_v4 import (
    _all_samples, _episodic_windows, _limited, _load_static_experts, _static_prediction,
)
from dynamic_residual_controls_v12 import _fit_controls


def _loader(arrays, batch_size, device, shuffle):
    sx, sy, qx, qy, roads, times = _all_samples(arrays)
    ds = TensorDataset(
        torch.from_numpy(sx), torch.from_numpy(sy), torch.from_numpy(qx), torch.from_numpy(qy),
        torch.from_numpy(roads), torch.from_numpy(times),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=device.type == "cuda")


def _ridge_query(experts, sx, sy, qx, task, ridge_cluster):
    with torch.no_grad():
        ss = _static_prediction(experts, sx, task)
        sq = _static_prediction(experts, qx, task)
        sr = (sy - ss).squeeze(-1)
        x = torch.cat([torch.ones(sr.shape[0], 1, device=sr.device), sr], dim=1)
        pred = torch.empty_like(sq.squeeze(-1))
        for c in range(ridge_cluster.shape[0]):
            idx = task == c
            if idx.any():
                pred[idx] = sq.squeeze(-1)[idx] + x[idx] @ ridge_cluster[c]
    return ss, sq, sr, pred.unsqueeze(-1)


def _features(sy, qx, static_support, ridge_query):
    q = qx.squeeze(-1)
    s = sy.squeeze(-1)
    ss = static_support.squeeze(-1)
    rq = ridge_query.squeeze(-1)
    mean = q.mean(dim=1, keepdim=True)
    std = q.std(dim=1, unbiased=False, keepdim=True)
    scale = (std + 0.05 * mean.abs() + 1e-3).clamp_min(1e-3)
    qn = (q - mean) / scale
    sn = (s - mean) / scale
    srn = (s - ss) / scale
    rqn = (rq - mean) / scale
    trend = (q[:, -1:] - q[:, :1]) / scale
    shift = (q[:, 6:].mean(dim=1, keepdim=True) - q[:, :6].mean(dim=1, keepdim=True)) / scale
    diffstd = torch.diff(q, dim=1).std(dim=1, unbiased=False, keepdim=True) / scale
    rmean = srn.mean(dim=1, keepdim=True)
    rstd = srn.std(dim=1, unbiased=False, keepdim=True)
    feat = torch.cat([qn, sn, srn, rqn, trend, shift, diffstd, rmean, rstd], dim=1)
    return torch.nan_to_num(feat), torch.nan_to_num(srn), scale


class ContextEncoder(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        d = 12 + 6 + 6 + 6 + 5
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class SingleResidualMap(nn.Module):
    """One support-residual transition map + context-dependent safety gate."""
    def __init__(self, hidden=64, correction_limit=1.0, gate_bias=-2.0):
        super().__init__()
        self.encoder = ContextEncoder(hidden)
        self.map = nn.Linear(7, 6, bias=False)  # [1, six support residuals] -> six future residuals
        self.gate = nn.Linear(hidden // 2, 6)
        self.correction_limit = float(correction_limit)
        nn.init.zeros_(self.map.weight)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, sy, qx, static_support, ridge_query):
        feat, srn, scale = _features(sy, qx, static_support, ridge_query)
        h = self.encoder(feat)
        u = torch.cat([torch.ones(srn.shape[0], 1, device=srn.device), srn], dim=1)
        raw = self.map(u)
        gate = torch.sigmoid(self.gate(h))
        ndelta = self.correction_limit * gate * torch.tanh(raw)
        pred = ridge_query.squeeze(-1) + scale * ndelta
        return pred.unsqueeze(-1), gate, ndelta, None


class SoftResidualRegimes(nn.Module):
    """Context-routed soft mixture of residual transition maps."""
    def __init__(self, hidden=64, regimes=4, correction_limit=1.0, gate_bias=-2.0, temperature=1.0):
        super().__init__()
        self.encoder = ContextEncoder(hidden)
        self.regimes = int(regimes)
        self.temperature = float(temperature)
        self.correction_limit = float(correction_limit)
        self.router = nn.Linear(hidden // 2, regimes)
        self.expert_maps = nn.Parameter(torch.zeros(regimes, 7, 6))
        self.gate = nn.Linear(hidden // 2, 6)
        nn.init.zeros_(self.router.weight); nn.init.zeros_(self.router.bias)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, sy, qx, static_support, ridge_query, force_uniform=False):
        feat, srn, scale = _features(sy, qx, static_support, ridge_query)
        h = self.encoder(feat)
        if force_uniform:
            weights = torch.full((h.shape[0], self.regimes), 1.0 / self.regimes, device=h.device, dtype=h.dtype)
        else:
            weights = torch.softmax(self.router(h) / self.temperature, dim=1)
        u = torch.cat([torch.ones(srn.shape[0], 1, device=srn.device), srn], dim=1)
        expert_raw = torch.einsum("bi,mio->bmo", u, self.expert_maps)
        raw = torch.einsum("bm,bmo->bo", weights, expert_raw)
        gate = torch.sigmoid(self.gate(h))
        ndelta = self.correction_limit * gate * torch.tanh(raw)
        pred = ridge_query.squeeze(-1) + scale * ndelta
        return pred.unsqueeze(-1), gate, ndelta, weights


def _group_robust_penalty(model_mse, anchor_mse, time_id, group_size):
    groups = torch.div(time_id, group_size, rounding_mode="floor")
    out = []
    for g in torch.unique(groups):
        idx = groups == g
        if idx.any():
            out.append(torch.relu(model_mse[idx].mean() - anchor_mse[idx].mean()))
    return torch.stack(out).mean() if out else model_mse.new_zeros(())


def _loss(pred, qy, anchor, gate, ndelta, time_id, args, weights=None):
    model_mse = ((pred - qy) ** 2).mean(dim=(1, 2))
    anchor_mse = ((anchor - qy) ** 2).mean(dim=(1, 2))
    query = model_mse.mean()
    safety = torch.relu(model_mse - anchor_mse).mean()
    robust = _group_robust_penalty(model_mse, anchor_mse, time_id, args.time_group_size)
    gate_reg = gate.mean()
    corr_reg = (ndelta ** 2).mean()
    balance = pred.new_zeros(())
    if weights is not None:
        usage = weights.mean(dim=0)
        balance = ((usage - 1.0 / weights.shape[1]) ** 2).mean()
    total = (query + args.safety_weight * safety + args.robust_weight * robust +
             args.gate_weight * gate_reg + args.correction_weight * corr_reg +
             args.balance_weight * balance)
    return total, [query, safety, robust, gate_reg, corr_reg, balance]


@torch.no_grad()
def _val_loss(model, experts, loader, labels_tensor, ridge_cluster, device, max_batches):
    model.eval(); losses = []
    for sx, sy, qx, qy, road, _ in _limited(loader, max_batches):
        sx, sy, qx, qy, road = sx.to(device), sy.to(device), qx.to(device), qy.to(device), road.to(device)
        task = labels_tensor[road]
        ss, _, _, rq = _ridge_query(experts, sx, sy, qx, task, ridge_cluster)
        pred = model(sy, qx, ss, rq)[0]
        losses.append(float(nn.functional.mse_loss(pred, qy).cpu()))
    return float(np.mean(losses)) if losses else float("inf")


@torch.no_grad()
def _evaluate(single, dynamic, experts, loader, labels_tensor, ridge_cluster, scale_data, device, max_batches):
    chunks = {k: [] for k in ["target", "static", "ridge", "single", "dynamic", "uniform"]}
    weights_all, gates_all, times_all, roads_all = [], [], [], []
    single.eval(); dynamic.eval()
    for sx, sy, qx, qy, road, time_id in _limited(loader, max_batches):
        sx, sy, qx, qy, road = sx.to(device), sy.to(device), qx.to(device), qy.to(device), road.to(device)
        task = labels_tensor[road]
        ss, sq, _, rq = _ridge_query(experts, sx, sy, qx, task, ridge_cluster)
        sp, _, _, _ = single(sy, qx, ss, rq)
        dp, dg, _, w = dynamic(sy, qx, ss, rq)
        up = dynamic(sy, qx, ss, rq, force_uniform=True)[0]
        vals = {"target": qy, "static": sq, "ridge": rq, "single": sp, "dynamic": dp, "uniform": up}
        for name, value in vals.items(): chunks[name].append((value * scale_data).cpu().numpy())
        weights_all.append(w.cpu().numpy()); gates_all.append(dg.cpu().numpy())
        times_all.append(time_id.numpy()); roads_all.append(road.cpu().numpy())

    y = np.concatenate(chunks["target"]).reshape(-1)
    result = {}
    static_mae = None; ridge_mae = None
    for name in ["static", "ridge", "single", "uniform", "dynamic"]:
        p = np.concatenate(chunks[name]).reshape(-1)
        met = _metrics(p, y)
        if name == "static": static_mae = met["MAE"]
        if name == "ridge": ridge_mae = met["MAE"]
        met["relative_mae_vs_static_pct"] = 100.0 * (met["MAE"] / static_mae - 1.0)
        met["relative_mae_vs_ridge_pct"] = None if ridge_mae is None else 100.0 * (met["MAE"] / ridge_mae - 1.0)
        result[name] = met

    w = np.concatenate(weights_all); g = np.concatenate(gates_all)
    t = np.concatenate(times_all); r = np.concatenate(roads_all); arg = w.argmax(axis=1)
    changes = transitions = roads_changed = 0
    for road in np.unique(r):
        idx = np.flatnonzero(r == road); idx = idx[np.argsort(t[idx])]
        seq = arg[idx]
        if len(seq) > 1:
            diff = seq[1:] != seq[:-1]
            changes += int(diff.sum()); transitions += int(diff.size); roads_changed += int(diff.any())
    ent = -(w * np.log(np.clip(w, 1e-12, 1.0))).sum(axis=1)
    result["routing"] = {
        "mean_entropy": float(ent.mean()),
        "mean_max_weight": float(w.max(axis=1).mean()),
        "mean_usage": w.mean(axis=0).tolist(),
        "mean_gate": float(g.mean()),
        "temporal_argmax_change_rate": float(changes / max(1, transitions)),
        "roads_with_temporal_change": float(roads_changed / max(1, len(np.unique(r)))),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beijing", choices=["beijing", "shanghai", "largest"])
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--train-max-batches", type=int, default=40)
    ap.add_argument("--val-max-batches", type=int, default=0)
    ap.add_argument("--test-max-batches", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--regimes", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--correction-limit", type=float, default=1.0)
    ap.add_argument("--gate-bias", type=float, default=-2.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--safety-weight", type=float, default=0.5)
    ap.add_argument("--robust-weight", type=float, default=1.0)
    ap.add_argument("--gate-weight", type=float, default=0.0)
    ap.add_argument("--correction-weight", type=float, default=1e-3)
    ap.add_argument("--balance-weight", type=float, default=1e-2)
    ap.add_argument("--time-group-size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed); device = _device(args.device)
    flow, ids, scale_data = _load_flow(args.dataset)
    labels = _cluster_features(args.dataset, "lstm", flow, ids, args.clusters, args.seed)
    fit, val, test = _episodic_windows(flow)
    fl_fit = _loader(fit, args.batch_size, device, False)
    tr = _loader(fit, args.batch_size, device, True)
    vl = _loader(val, args.batch_size, device, False)
    tl = _loader(test, args.batch_size, device, False)
    experts = _load_static_experts(Path(args.checkpoint_dir), args.clusters, device)
    labels_tensor = torch.from_numpy(labels).long().to(device)
    controls = _fit_controls(experts, fl_fit, labels_tensor, args.clusters, device, 0, args.ridge)
    ridge_cluster = controls["ridge_cluster"].to(device)

    single = SingleResidualMap(args.hidden, args.correction_limit, args.gate_bias).to(device)
    dynamic = SoftResidualRegimes(args.hidden, args.regimes, args.correction_limit, args.gate_bias, args.temperature).to(device)
    opt_s = torch.optim.AdamW(single.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_d = torch.optim.AdamW(dynamic.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_s = best_d = float("inf")
    state_s = copy.deepcopy(single.state_dict()); state_d = copy.deepcopy(dynamic.state_dict())
    history = []

    for epoch in range(args.epochs):
        single.train(); dynamic.train(); rows = []
        for sx, sy, qx, qy, road, time_id in _limited(tr, args.train_max_batches):
            sx, sy, qx, qy, road, time_id = sx.to(device), sy.to(device), qx.to(device), qy.to(device), road.to(device), time_id.to(device)
            task = labels_tensor[road]
            ss, _, _, rq = _ridge_query(experts, sx, sy, qx, task, ridge_cluster)

            sp, sg, snd, _ = single(sy, qx, ss, rq)
            sloss, srow = _loss(sp, qy, rq, sg, snd, time_id, args)
            opt_s.zero_grad(set_to_none=True); sloss.backward(); torch.nn.utils.clip_grad_norm_(single.parameters(), 5.0); opt_s.step()

            dp, dg, dnd, w = dynamic(sy, qx, ss, rq)
            dloss, drow = _loss(dp, qy, rq, dg, dnd, time_id, args, w)
            opt_d.zero_grad(set_to_none=True); dloss.backward(); torch.nn.utils.clip_grad_norm_(dynamic.parameters(), 5.0); opt_d.step()
            rows.append([float(sloss.detach()), float(dloss.detach()), float(srow[0].detach()), float(drow[0].detach()),
                         float(drow[1].detach()), float(drow[2].detach()), float(drow[5].detach()),
                         float((-(w * torch.log(w.clamp_min(1e-12))).sum(dim=1).mean()).detach())])

        sv = _val_loss(single, experts, vl, labels_tensor, ridge_cluster, device, args.val_max_batches)
        dv = _val_loss(dynamic, experts, vl, labels_tensor, ridge_cluster, device, args.val_max_batches)
        arr = np.asarray(rows)
        rec = {"epoch": epoch + 1, "single_loss": float(arr[:,0].mean()), "dynamic_loss": float(arr[:,1].mean()),
               "single_query": float(arr[:,2].mean()), "dynamic_query": float(arr[:,3].mean()),
               "dynamic_safety": float(arr[:,4].mean()), "dynamic_robust": float(arr[:,5].mean()),
               "dynamic_balance": float(arr[:,6].mean()), "dynamic_entropy": float(arr[:,7].mean()),
               "single_val_mse": sv, "dynamic_val_mse": dv}
        history.append(rec); print(json.dumps(rec))
        if sv < best_s: best_s = sv; state_s = copy.deepcopy(single.state_dict())
        if dv < best_d: best_d = dv; state_d = copy.deepcopy(dynamic.state_dict())

    single.load_state_dict(state_s); dynamic.load_state_dict(state_d)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save(state_s, out / "single_best.pt"); torch.save(state_d, out / "dynamic_best.pt")
    validation = _evaluate(single, dynamic, experts, vl, labels_tensor, ridge_cluster, scale_data, device, args.val_max_batches)
    future = _evaluate(single, dynamic, experts, tl, labels_tensor, ridge_cluster, scale_data, device, args.test_max_batches)
    result = {
        "experiment": "dynamic_soft_residual_regimes_v13",
        "protocol": "cluster-ridge anchor + bounded learned correction; single-map control vs dynamically routed soft residual regimes; best epochs selected on validation only",
        "config": vars(args), "best_single_val_mse": best_s, "best_dynamic_val_mse": best_d,
        "history": history, "validation": validation, "future": future,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("V13_RESULT"); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
