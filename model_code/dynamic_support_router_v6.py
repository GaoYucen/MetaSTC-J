"""V6 support-aware amortized dynamic task router.

V3 (query context only) did not generalize across the future temporal split, while
V4 (support error only) discovered real task changes but support/query task
identity was not always aligned. V6 combines both signals.

The task-adapter bank is the validation-selected V4 bank and is frozen. For each
training episode we compute:
  * support evidence: per-task errors on recently observed support targets;
  * current context: the latest 12 observed query-input values and statistics;
  * static prior: the original KMeans task identity.
A small router predicts the query-time task distribution. During training only,
per-task query errors define a soft oracle responsibility used to supervise the
router. At test time query targets are never used: routing depends only on past
support observations + current query context.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from optimized_runner import LSTMModel, _cluster_features, _device, _load_flow, _metrics, _set_seed
from dynamic_support_routing_v4 import (
    TaskAdapterBank,
    _all_samples,
    _episodic_windows,
    _load_static_experts,
    _static_prediction,
)


def _limited(loader, max_batches):
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        yield batch


def _make_loader(arrays, batch_size, device, shuffle):
    sx, sy, qx, qy, roads, times = _all_samples(arrays)
    ds = TensorDataset(
        torch.from_numpy(sx), torch.from_numpy(sy), torch.from_numpy(qx), torch.from_numpy(qy),
        torch.from_numpy(roads), torch.from_numpy(times),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=device.type == "cuda")


def _standardized_errors(errors: torch.Tensor):
    best = errors.min(dim=1, keepdim=True).values
    spread = errors.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
    return (errors - best) / spread


def _soft_oracle(query_errors: torch.Tensor, temperature: float):
    z = _standardized_errors(query_errors)
    return torch.softmax(-z / temperature, dim=1)


class SupportAwareRouter(nn.Module):
    def __init__(self, k: int, look_back: int = 12, hidden: int = 64, residual_scale: float = 1.0):
        super().__init__()
        self.k = k
        self.residual_scale = residual_scale
        # normalized support errors K + support posterior K + raw query 12
        # + mean/std/min/max/trend 5 + static prior K
        input_dim = k + k + look_back + 5 + k
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, k),
        )
        # Start at pure support-based routing. Learning only adds a residual.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def features(self, support_errors, query_x, static_task, support_temp, support_prior):
        z = _standardized_errors(support_errors)
        prior = torch.nn.functional.one_hot(static_task, self.k).to(query_x.dtype)
        base_logits = -z / support_temp + support_prior * prior
        support_p = torch.softmax(base_logits, dim=1)
        x = query_x.squeeze(-1)
        stats = torch.cat([
            x.mean(dim=1, keepdim=True),
            x.std(dim=1, keepdim=True, unbiased=False),
            x.min(dim=1, keepdim=True).values,
            x.max(dim=1, keepdim=True).values,
            x[:, -1:] - x[:, :1],
        ], dim=1)
        feat = torch.cat([z, support_p, x, stats, prior], dim=1)
        return feat, base_logits, support_p

    def forward(self, support_errors, query_x, static_task, support_temp, support_prior):
        feat, base_logits, support_p = self.features(
            support_errors, query_x, static_task, support_temp, support_prior
        )
        logits = base_logits + self.residual_scale * self.net(feat)
        return torch.softmax(logits, dim=1), support_p


def _evaluate(bank, router, experts, loader, labels, labels_tensor, scale, device, max_batches,
              support_temp, support_prior):
    bank.eval(); router.eval()
    names = ["target", "old_static", "support_soft", "support_hard", "router_soft", "router_hard", "oracle"]
    chunks = {name: [] for name in names}
    p_all, support_p_all, roads_all, times_all = [], [], [], []
    with torch.no_grad():
        for sx, sy, qx, qy, road, time_id in _limited(loader, max_batches):
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            road = road.to(device)
            static_task = labels_tensor[road]
            support_pred, _, _ = bank(sx)
            support_errors = ((support_pred - sy[:, None]) ** 2).mean(dim=(2, 3))
            query_pred, _, _ = bank(qx)
            query_errors = ((query_pred - qy[:, None]) ** 2).mean(dim=(2, 3))
            p, support_p = router(support_errors, qx, static_task, support_temp, support_prior)
            row = torch.arange(qx.shape[0], device=device)
            router_soft = (query_pred * p[:, :, None, None]).sum(dim=1)
            router_hard = query_pred[row, p.argmax(dim=1)]
            support_soft = (query_pred * support_p[:, :, None, None]).sum(dim=1)
            support_hard = query_pred[row, support_p.argmax(dim=1)]
            oracle = query_pred[row, query_errors.argmin(dim=1)]
            old_static = _static_prediction(experts, qx, static_task)
            vals = {"target": qy, "old_static": old_static, "support_soft": support_soft,
                    "support_hard": support_hard, "router_soft": router_soft,
                    "router_hard": router_hard, "oracle": oracle}
            for name, value in vals.items():
                chunks[name].append((value * scale).cpu().numpy())
            p_all.append(p.cpu().numpy()); support_p_all.append(support_p.cpu().numpy())
            roads_all.append(road.cpu().numpy()); times_all.append(time_id.numpy())

    target = np.concatenate(chunks["target"]).reshape(-1)
    pred = {name: np.concatenate(chunks[name]).reshape(-1) for name in names if name != "target"}
    p = np.concatenate(p_all); support_p = np.concatenate(support_p_all)
    roads = np.concatenate(roads_all); times = np.concatenate(times_all)
    static_task = labels[roads]; task = p.argmax(axis=1)
    entropy = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)
    support_entropy = -(support_p * np.log(np.clip(support_p, 1e-12, 1.0))).sum(axis=1)
    changes = transitions = roads_changed = 0
    for road in np.unique(roads):
        idx = np.flatnonzero(roads == road); idx = idx[np.argsort(times[idx])]
        seq = task[idx]
        if len(seq) > 1:
            d = seq[1:] != seq[:-1]
            changes += int(d.sum()); transitions += int(d.size); roads_changed += int(d.any())
    return {
        "reproduced_static_hard": _metrics(pred["old_static"], target),
        "v4_support_soft": _metrics(pred["support_soft"], target),
        "v4_support_hard": _metrics(pred["support_hard"], target),
        "v6_router_soft": _metrics(pred["router_soft"], target),
        "v6_router_hard": _metrics(pred["router_hard"], target),
        "oracle_best_task_diagnostic": _metrics(pred["oracle"], target),
        "routing": {
            "mean_entropy": float(entropy.mean()),
            "mean_max_weight": float(p.max(axis=1).mean()),
            "mean_support_entropy": float(support_entropy.mean()),
            "router_vs_support_argmax_disagreement": float((p.argmax(axis=1) != support_p.argmax(axis=1)).mean()),
            "router_vs_static_argmax_disagreement": float((task != static_task).mean()),
            "temporal_task_change_rate": float(changes / max(transitions, 1)),
            "roads_with_temporal_task_change": float(roads_changed / max(len(np.unique(roads)), 1)),
            "task_usage": p.mean(axis=0).tolist(),
        },
        "num_window_road_samples": int(len(p)),
    }


def _val_mse(bank, router, loader, labels_tensor, device, max_batches, support_temp, support_prior):
    bank.eval(); router.eval(); losses = []
    with torch.no_grad():
        for sx, sy, qx, qy, road, _ in _limited(loader, max_batches):
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            road = road.to(device); static_task = labels_tensor[road]
            sp, _, _ = bank(sx); qp, _, _ = bank(qx)
            se = ((sp - sy[:, None]) ** 2).mean(dim=(2, 3))
            p, _ = router(se, qx, static_task, support_temp, support_prior)
            mixed = (qp * p[:, :, None, None]).sum(dim=1)
            losses.append(float(nn.functional.mse_loss(mixed, qy).cpu()))
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser(description="Support-aware dynamic router V6")
    ap.add_argument("--dataset", default="beijing", choices=["beijing", "shanghai", "largest"])
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--checkpoint-dir", default="param/4090_tuned/lstm/beijing")
    ap.add_argument("--bank-checkpoint", default="param/journal/dynamic_support_v4/beijing_lstm_substantial/model_best.pt")
    ap.add_argument("--output-dir", default="param/journal/dynamic_support_router_v6/beijing_lstm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--gate-hidden", type=int, default=64)
    ap.add_argument("--support-temperature", type=float, default=1.0)
    ap.add_argument("--support-prior", type=float, default=0.25)
    ap.add_argument("--oracle-temperature", type=float, default=0.5)
    ap.add_argument("--ce-weight", type=float, default=0.02)
    ap.add_argument("--mixture-weight", type=float, default=1.0)
    ap.add_argument("--anchor-weight", type=float, default=0.002)
    ap.add_argument("--train-max-batches", type=int, default=50)
    ap.add_argument("--val-max-batches", type=int, default=0)
    ap.add_argument("--test-max-batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed); device = _device(args.device); k = args.clusters
    flow, ids, scale = _load_flow(args.dataset)
    labels = _cluster_features(args.dataset, "lstm", flow, ids, k, args.seed)
    fit, val, test = _episodic_windows(flow)
    train_loader = _make_loader(fit, args.batch_size, device, True)
    val_loader = _make_loader(val, args.batch_size, device, False)
    test_loader = _make_loader(test, args.batch_size, device, False)

    ckpt = Path(args.checkpoint_dir)
    base = LSTMModel(12, 6).to(device)
    base.load_state_dict(torch.load(ckpt / "global_best.pt", map_location=device, weights_only=True))
    bank = TaskAdapterBank(base, k, rank=6).to(device)
    bank.load_state_dict(torch.load(args.bank_checkpoint, map_location=device, weights_only=True))
    bank.eval()
    for p in bank.parameters(): p.requires_grad_(False)
    experts = _load_static_experts(ckpt, k, device)
    labels_tensor = torch.from_numpy(labels).long().to(device)

    router = SupportAwareRouter(k, hidden=args.gate_hidden).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=1e-5)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    best_val = float("inf"); best_state = copy.deepcopy(router.state_dict()); history = []

    for epoch in range(args.epochs):
        router.train(); rows = []
        for sx, sy, qx, qy, road, _ in _limited(train_loader, args.train_max_batches):
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            road = road.to(device); static_task = labels_tensor[road]
            with torch.no_grad():
                sp, _, _ = bank(sx); qp, _, _ = bank(qx)
                se = ((sp - sy[:, None]) ** 2).mean(dim=(2, 3))
                qe = ((qp - qy[:, None]) ** 2).mean(dim=(2, 3))
                oracle_q = _soft_oracle(qe, args.oracle_temperature)
            p, support_p = router(se, qx, static_task, args.support_temperature, args.support_prior)
            mixed = (qp * p[:, :, None, None]).sum(dim=1)
            mixture_loss = nn.functional.mse_loss(mixed, qy)
            ce = -(oracle_q * torch.log(p.clamp_min(1e-8))).sum(dim=1).mean()
            anchor = (p * (torch.log(p.clamp_min(1e-8)) - torch.log(support_p.clamp_min(1e-8)))).sum(dim=1).mean()
            loss = args.mixture_weight * mixture_loss + args.ce_weight * ce + args.anchor_weight * anchor
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(router.parameters(), 5.0); optimizer.step()
            rows.append((float(loss.detach().cpu()), float(mixture_loss.detach().cpu()), float(ce.detach().cpu()), float(anchor.detach().cpu()), float(p.max(dim=1).values.mean().detach().cpu())))
        val = _val_mse(bank, router, val_loader, labels_tensor, device, args.val_max_batches, args.support_temperature, args.support_prior)
        a = np.asarray(rows)
        record = {"epoch": epoch + 1, "loss": float(a[:,0].mean()), "mixture_loss": float(a[:,1].mean()),
                  "oracle_ce": float(a[:,2].mean()), "anchor_kl": float(a[:,3].mean()),
                  "mean_max_weight": float(a[:,4].mean()), "val_mse": val}
        history.append(record); print(json.dumps(record))
        if val < best_val:
            best_val = val; best_state = copy.deepcopy(router.state_dict()); torch.save(best_state, out / "router_best.pt")

    router.load_state_dict(best_state)
    validation = _evaluate(bank, router, experts, val_loader, labels, labels_tensor, scale, device, args.val_max_batches, args.support_temperature, args.support_prior)
    test_result = _evaluate(bank, router, experts, test_loader, labels, labels_tensor, scale, device, args.test_max_batches, args.support_temperature, args.support_prior)
    result = {"experiment":"dynamic_support_router_v6", "best_val_normalized_mse":best_val,
              "history":history, "validation":validation, "test":test_result}
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
