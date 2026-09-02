"""V10 continuous dynamic residual adaptation around the static MetaSTC expert.

V9 shows that the hindsight-best discrete adapter task changes too quickly to be
predicted reliably from causal history.  V10 therefore keeps the original
static spatial task/expert as a stable anchor and represents dynamic
heterogeneity as a *continuous deviation* from that anchor.

For each road and forecast origin, a small shared network consumes:
  - the latest 12 observed values (query context),
  - the static expert's 6-step forecast,
  - the static expert's residual on the immediately preceding observed
    support target (6 values), and
  - an embedding of the road's original static cluster.
It outputs a gated 6-step residual correction.  The static experts are frozen;
only this lightweight residual adapter is trained.  The final layer starts at
zero, so optimization begins exactly from the reproduced static baseline.
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

from optimized_runner import LSTMModel, _cluster_features, _device, _load_flow, _metrics, _set_seed
from dynamic_support_routing_v4 import _all_samples, _episodic_windows, _limited, _load_static_experts, _static_prediction


class ContinuousResidualAdapter(nn.Module):
    def __init__(self, num_clusters: int, emb_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.cluster_emb = nn.Embedding(num_clusters, emb_dim)
        # qx(12) + static forecast(6) + observed support residual(6) +
        # simple local statistics(6) + cluster embedding.
        in_dim = 12 + 6 + 6 + 6 + emb_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(hidden, 6)
        self.gate_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)

    def forward(self, qx, static_q, support_residual, static_task):
        q = qx.squeeze(-1)
        sq = static_q.squeeze(-1)
        sr = support_residual.squeeze(-1)
        # Causal, scale-free local dynamics: recent mean/std, last value,
        # short/long trend, and support residual RMS.
        stats = torch.stack([
            q.mean(dim=1),
            q.std(dim=1, unbiased=False),
            q[:, -1],
            q[:, -1] - q[:, -2],
            q[:, -1] - q[:, 0],
            torch.sqrt((sr * sr).mean(dim=1) + 1e-8),
        ], dim=1)
        feat = torch.cat([q, sq, sr, stats, self.cluster_emb(static_task)], dim=1)
        h = self.trunk(feat)
        delta = self.delta_head(h).unsqueeze(-1)
        gate = torch.sigmoid(self.gate_head(h)).unsqueeze(-1)
        pred = static_q + gate * delta
        return pred, delta, gate


def _make_loader(arrays, batch_size, shuffle):
    sx, sy, qx, qy, road, time_id = _all_samples(arrays)
    ds = TensorDataset(
        torch.from_numpy(sx), torch.from_numpy(sy), torch.from_numpy(qx), torch.from_numpy(qy),
        torch.from_numpy(road), torch.from_numpy(time_id),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def _validation_loss(model, experts, loader, labels_tensor, device, max_batches):
    model.eval(); losses = []
    with torch.no_grad():
        for sx, sy, qx, qy, road, _ in _limited(loader, max_batches):
            sx, sy, qx, qy, road = sx.to(device), sy.to(device), qx.to(device), qy.to(device), road.to(device)
            static_task = labels_tensor[road]
            static_s = _static_prediction(experts, sx, static_task)
            static_q = _static_prediction(experts, qx, static_task)
            pred, _, _ = model(qx, static_q, sy - static_s, static_task)
            losses.append(float(nn.functional.mse_loss(pred, qy).cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def _evaluate(model, experts, loader, labels_tensor, scale, device, max_batches):
    model.eval()
    chunks = {"target": [], "static": [], "dynamic": []}
    gate_values, correction_values = [], []
    with torch.no_grad():
        for sx, sy, qx, qy, road, _ in _limited(loader, max_batches):
            sx, sy, qx, qy, road = sx.to(device), sy.to(device), qx.to(device), qy.to(device), road.to(device)
            static_task = labels_tensor[road]
            static_s = _static_prediction(experts, sx, static_task)
            static_q = _static_prediction(experts, qx, static_task)
            pred, delta, gate = model(qx, static_q, sy - static_s, static_task)
            for name, value in (("target", qy), ("static", static_q), ("dynamic", pred)):
                chunks[name].append((value * scale).cpu().numpy())
            gate_values.append(gate.squeeze(-1).squeeze(-1).cpu().numpy())
            correction_values.append((gate * delta).abs().mean(dim=(1, 2)).cpu().numpy() * scale)
    target = np.concatenate(chunks["target"]).reshape(-1)
    static = np.concatenate(chunks["static"]).reshape(-1)
    dynamic = np.concatenate(chunks["dynamic"]).reshape(-1)
    gates = np.concatenate(gate_values)
    corrections = np.concatenate(correction_values)
    result = {
        "reproduced_static_hard": _metrics(static, target),
        "continuous_dynamic_residual": _metrics(dynamic, target),
        "routing": {
            "mean_gate": float(gates.mean()),
            "std_gate": float(gates.std()),
            "mean_abs_correction_raw_scale": float(corrections.mean()),
            "p90_abs_correction_raw_scale": float(np.quantile(corrections, 0.9)),
        },
        "num_samples": int(gates.size),
    }
    s = result["reproduced_static_hard"]["MAE"]
    d = result["continuous_dynamic_residual"]["MAE"]
    result["dynamic_vs_static_delta_mae"] = d - s
    result["dynamic_vs_static_relative_mae_pct"] = 100.0 * (d / s - 1.0)
    return result


def main():
    ap = argparse.ArgumentParser(description="Continuous dynamic residual adaptation V10")
    ap.add_argument("--dataset", default="beijing", choices=["beijing", "shanghai", "largest"])
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--checkpoint-dir", default="param/4090_tuned/lstm/beijing")
    ap.add_argument("--output-dir", default="param/journal/dynamic_continuous_residual_v10/beijing_lstm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--train-max-batches", type=int, default=20)
    ap.add_argument("--val-max-batches", type=int, default=0)
    ap.add_argument("--test-max-batches", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--emb-dim", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--correction-weight", type=float, default=1e-3)
    ap.add_argument("--gate-weight", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed)
    device = _device(args.device)
    flow, ids, scale = _load_flow(args.dataset)
    labels = _cluster_features(args.dataset, "lstm", flow, ids, args.clusters, args.seed)
    fit_arrays, val_arrays, test_arrays = _episodic_windows(flow)
    train_loader = _make_loader(fit_arrays, args.batch_size, True)
    val_loader = _make_loader(val_arrays, args.batch_size, False)
    test_loader = _make_loader(test_arrays, args.batch_size, False)
    labels_tensor = torch.from_numpy(labels).long().to(device)

    experts = _load_static_experts(Path(args.checkpoint_dir), args.clusters, device)
    model = ContinuousResidualAdapter(args.clusters, args.emb_dim, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val = float("inf"); best_state = copy.deepcopy(model.state_dict()); history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = mse_total = reg_total = gate_total = 0.0; batches = 0
        for sx, sy, qx, qy, road, _ in _limited(train_loader, args.train_max_batches):
            sx, sy, qx, qy, road = sx.to(device), sy.to(device), qx.to(device), qy.to(device), road.to(device)
            static_task = labels_tensor[road]
            with torch.no_grad():
                static_s = _static_prediction(experts, sx, static_task)
                static_q = _static_prediction(experts, qx, static_task)
            pred, delta, gate = model(qx, static_q, sy - static_s, static_task)
            mse = nn.functional.mse_loss(pred, qy)
            correction_reg = (gate * delta).pow(2).mean()
            gate_reg = gate.mean()
            loss = mse + args.correction_weight * correction_reg + args.gate_weight * gate_reg
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += float(loss.detach()); mse_total += float(mse.detach())
            reg_total += float(correction_reg.detach()); gate_total += float(gate_reg.detach()); batches += 1
        val_mse = _validation_loss(model, experts, val_loader, labels_tensor, device, args.val_max_batches)
        row = {"epoch": epoch, "loss": total/max(batches,1), "train_mse": mse_total/max(batches,1),
               "correction_reg": reg_total/max(batches,1), "mean_gate_train": gate_total/max(batches,1), "val_mse": val_mse}
        history.append(row); print(json.dumps(row), flush=True)
        if val_mse < best_val:
            best_val = val_mse; best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out / "model_best.pt")
    result = {
        "experiment": "dynamic_continuous_residual_v10",
        "best_val_normalized_mse": best_val,
        "history": history,
        "validation": _evaluate(model, experts, val_loader, labels_tensor, scale, device, args.val_max_batches),
        "test": _evaluate(model, experts, test_loader, labels_tensor, scale, device, args.test_max_batches),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
