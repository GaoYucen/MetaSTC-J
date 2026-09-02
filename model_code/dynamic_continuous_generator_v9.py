"""V9 continuous latent task discovery with task-conditioned parameter generation.

V4--V8 show that dynamic heterogeneity exists, but routing among a small set of
fixed KMeans tasks does not generalize reliably to the future temporal segment.
V9 removes discrete task identity from the predictor.  A continuous task encoder
uses only information available at forecast time (recent support residuals and
current query history) to produce a latent embedding.  A lightweight hypernetwork
maps that embedding to mixture coefficients over trainable low-rank adapter bases
for the frozen global LSTM prediction head.

For a forecast origin t:
  support input  : t-18 .. t-7
  support target : t-6  .. t-1   (already observed)
  query input    : t-12 .. t-1
  query target   : t    .. t+5   (training/evaluation target only)

The task encoder never receives query targets.  The old KMeans cluster experts
are loaded only for the reproduced MetaSTC baseline and optional basis
initialization; cluster labels are not inputs to the V9 task encoder.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from optimized_runner import LSTMModel, _cluster_features, _device, _load_flow, _metrics, _set_seed
from dynamic_support_routing_v4 import (
    _all_samples,
    _episodic_windows,
    _limited,
    _load_static_experts,
    _static_prediction,
)


class ContinuousTaskGenerator(nn.Module):
    """Frozen global LSTM + continuously generated low-rank head adapter."""

    def __init__(self, base: LSTMModel, num_bases: int = 8, rank: int = 4,
                 latent_dim: int = 16, hidden_dim: int = 48):
        super().__init__()
        self.num_bases = num_bases
        self.feature_dim = base.linear.in_features
        self.output_dim = base.linear.out_features
        self.rank = min(rank, self.feature_dim, self.output_dim)
        self.base_lstm = base.lstm
        self.base_linear = base.linear
        for p in self.base_lstm.parameters():
            p.requires_grad_(False)
        for p in self.base_linear.parameters():
            p.requires_grad_(False)

        # Encoder input: query history (12), observed support target (6),
        # and global-model support residual (6).  All are causal at forecast time.
        encoder_in = 12 + 6 + 6
        self.task_encoder = nn.Sequential(
            nn.Linear(encoder_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )
        self.coeff_head = nn.Linear(latent_dim, num_bases)
        # A learned context-independent adapter provides a same-capacity fixed
        # parameter-generation ablation inside the same trained model.
        self.global_coeff_logits = nn.Parameter(torch.zeros(num_bases))

        self.basis_a = nn.Parameter(torch.empty(num_bases, self.feature_dim, self.rank))
        self.basis_b = nn.Parameter(torch.zeros(num_bases, self.rank, self.output_dim))
        self.basis_bias = nn.Parameter(torch.zeros(num_bases, self.output_dim))
        nn.init.normal_(self.basis_a, std=0.02)
        # Start near the frozen global predictor.  This makes early optimization safe.
        nn.init.normal_(self.basis_b, std=1e-3)

    @torch.no_grad()
    def warmstart_bases_from_static_heads(self, cluster_states: List[dict]):
        """Use old cluster heads only as a low-rank basis initialization."""
        base_w = self.base_linear.weight.detach().cpu()
        base_b = self.base_linear.bias.detach().cpu()
        for task, state in enumerate(cluster_states[: self.num_bases]):
            delta = (state["linear.weight"].detach().cpu() - base_w).t().float()
            u, s, vh = torch.linalg.svd(delta, full_matrices=False)
            r = min(self.rank, s.numel())
            root = torch.sqrt(torch.clamp(s[:r], min=0.0))
            self.basis_a[task].zero_(); self.basis_b[task].zero_()
            self.basis_a[task, :, :r].copy_((u[:, :r] * root.unsqueeze(0)).to(self.basis_a.device))
            self.basis_b[task, :r, :].copy_((root.unsqueeze(1) * vh[:r, :]).to(self.basis_b.device))
            self.basis_bias[task].copy_((state["linear.bias"].detach().cpu() - base_b).to(self.basis_bias.device))

    def _features(self, x: torch.Tensor):
        self.base_lstm.flatten_parameters()
        sequence, _ = self.base_lstm(x)
        flat = sequence.reshape(sequence.shape[0], -1)
        base = self.base_linear(flat)
        return flat, base

    def infer_coefficients(self, sx: torch.Tensor, sy: torch.Tensor, qx: torch.Tensor,
                           temperature: float = 1.0):
        with torch.no_grad():
            _, support_base = self._features(sx)
        residual = sy.squeeze(-1) - support_base
        encoder_input = torch.cat([qx.squeeze(-1), sy.squeeze(-1), residual], dim=1)
        z = self.task_encoder(encoder_input)
        logits = self.coeff_head(z) / max(float(temperature), 1e-4)
        coeff = torch.softmax(logits, dim=1)
        return z, coeff

    def _apply_coeff(self, flat: torch.Tensor, base: torch.Tensor, coeff: torch.Tensor):
        # Per-basis low-rank residuals, followed by a continuous coefficient mixture.
        low = torch.einsum("bf,mfr->bmr", flat, self.basis_a)
        basis_delta = torch.einsum("bmr,mro->bmo", low, self.basis_b) + self.basis_bias.unsqueeze(0)
        delta = (basis_delta * coeff[:, :, None]).sum(dim=1)
        return (base + delta).unsqueeze(-1), basis_delta

    def forward(self, sx: torch.Tensor, sy: torch.Tensor, qx: torch.Tensor,
                temperature: float = 1.0):
        z, coeff = self.infer_coefficients(sx, sy, qx, temperature)
        qflat, qbase = self._features(qx)
        dynamic, basis_delta = self._apply_coeff(qflat, qbase, coeff)
        global_coeff = torch.softmax(self.global_coeff_logits, dim=0).expand(qx.shape[0], -1)
        fixed, _ = self._apply_coeff(qflat, qbase, global_coeff)
        return dynamic, fixed, qbase.unsqueeze(-1), z, coeff, basis_delta

    def support_prediction(self, sx: torch.Tensor, sy: torch.Tensor, qx: torch.Tensor,
                           temperature: float = 1.0):
        _, coeff = self.infer_coefficients(sx, sy, qx, temperature)
        sflat, sbase = self._features(sx)
        pred, _ = self._apply_coeff(sflat, sbase, coeff)
        return pred


def _make_loader(arrays, batch_size: int, device: torch.device, shuffle: bool):
    sx, sy, qx, qy, roads, times = _all_samples(arrays)
    ds = TensorDataset(
        torch.from_numpy(sx), torch.from_numpy(sy), torch.from_numpy(qx), torch.from_numpy(qy),
        torch.from_numpy(roads), torch.from_numpy(times),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=device.type == "cuda")


def _validation_loss(model, loader: Iterable, device, max_batches: int, temperature: float):
    model.eval(); values = []
    with torch.no_grad():
        for sx, sy, qx, qy, _, _ in _limited(loader, max_batches):
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            dynamic, _, _, _, _, _ = model(sx, sy, qx, temperature)
            values.append(float(nn.functional.mse_loss(dynamic, qy).cpu()))
    return float(np.mean(values)) if values else float("inf")


def _evaluate(model, experts, loader, labels, labels_tensor, scale, device,
              max_batches: int, temperature: float):
    model.eval()
    names = ["target", "global", "old_static", "fixed_generated", "dynamic_generated", "oracle_basis"]
    chunks = {name: [] for name in names}
    coeff_all, roads_all, times_all, latent_all = [], [], [], []
    with torch.no_grad():
        for sx, sy, qx, qy, road, time_id in _limited(loader, max_batches):
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            road = road.to(device)
            dynamic, fixed, base, z, coeff, basis_delta = model(sx, sy, qx, temperature)
            old_static = _static_prediction(experts, qx, labels_tensor[road])

            # Diagnostic capacity upper bound: best single learned basis for each query.
            base2 = base.squeeze(-1)
            basis_pred = base2[:, None, :] + basis_delta
            basis_pred = basis_pred.unsqueeze(-1)
            basis_err = ((basis_pred - qy[:, None]) ** 2).mean(dim=(2, 3))
            row_idx = torch.arange(qx.shape[0], device=device)
            oracle_basis = basis_pred[row_idx, basis_err.argmin(dim=1)]

            values = {
                "target": qy, "global": base, "old_static": old_static,
                "fixed_generated": fixed, "dynamic_generated": dynamic,
                "oracle_basis": oracle_basis,
            }
            for name, value in values.items():
                chunks[name].append((value * scale).cpu().numpy())
            coeff_all.append(coeff.cpu().numpy())
            latent_all.append(z.cpu().numpy())
            roads_all.append(road.cpu().numpy())
            times_all.append(time_id.numpy())

    target = np.concatenate(chunks["target"]).reshape(-1)
    preds = {n: np.concatenate(chunks[n]).reshape(-1) for n in names if n != "target"}
    coeff = np.concatenate(coeff_all); latent = np.concatenate(latent_all)
    roads = np.concatenate(roads_all); times = np.concatenate(times_all)
    argmax = coeff.argmax(axis=1)
    changes = transitions = roads_changed = 0
    coeff_step_l1 = []
    for road in np.unique(roads):
        idx = np.flatnonzero(roads == road); idx = idx[np.argsort(times[idx])]
        if idx.size > 1:
            seq = argmax[idx]
            diff = seq[1:] != seq[:-1]
            changes += int(diff.sum()); transitions += int(diff.size); roads_changed += int(diff.any())
            coeff_step_l1.append(np.abs(coeff[idx][1:] - coeff[idx][:-1]).sum(axis=1))
    entropy = -(coeff * np.log(np.clip(coeff, 1e-12, 1.0))).sum(axis=1)
    step_l1 = float(np.concatenate(coeff_step_l1).mean()) if coeff_step_l1 else 0.0
    result = {
        "global_base": _metrics(preds["global"], target),
        "reproduced_static_hard": _metrics(preds["old_static"], target),
        "fixed_generated_adapter": _metrics(preds["fixed_generated"], target),
        "continuous_dynamic_generator": _metrics(preds["dynamic_generated"], target),
        "oracle_single_basis_diagnostic": _metrics(preds["oracle_basis"], target),
        "latent": {
            "mean_entropy": float(entropy.mean()),
            "mean_max_coefficient": float(coeff.max(axis=1).mean()),
            "coefficient_std_mean": float(coeff.std(axis=0).mean()),
            "latent_std_mean": float(latent.std(axis=0).mean()),
            "temporal_argmax_change_rate": float(changes / max(transitions, 1)),
            "roads_with_temporal_argmax_change": float(roads_changed / max(len(np.unique(roads)), 1)),
            "mean_temporal_coefficient_l1_change": step_l1,
            "mean_basis_usage": coeff.mean(axis=0).tolist(),
        },
        "num_window_road_samples": int(coeff.shape[0]),
    }
    d = result["continuous_dynamic_generator"]["MAE"]
    s = result["reproduced_static_hard"]["MAE"]
    f = result["fixed_generated_adapter"]["MAE"]
    result["dynamic_vs_static_relative_mae_pct"] = 100.0 * (d / s - 1.0)
    result["dynamic_vs_fixed_relative_mae_pct"] = 100.0 * (d / f - 1.0)
    return result


def main():
    ap = argparse.ArgumentParser(description="Continuous latent task parameter generator V9")
    ap.add_argument("--dataset", choices=["beijing", "shanghai", "largest"], default="beijing")
    ap.add_argument("--clusters", type=int, default=5, help="only for reproduced baseline / optional basis init")
    ap.add_argument("--checkpoint-dir", default="param/4090_tuned/lstm/beijing")
    ap.add_argument("--output-dir", default="param/journal/dynamic_continuous_v9/beijing_lstm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--num-bases", type=int, default=8)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--hidden-dim", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--support-loss-weight", type=float, default=0.2)
    ap.add_argument("--fixed-loss-weight", type=float, default=0.2)
    ap.add_argument("--entropy-weight", type=float, default=0.0)
    ap.add_argument("--no-basis-warmstart", action="store_true")
    ap.add_argument("--train-max-batches", type=int, default=50)
    ap.add_argument("--val-max-batches", type=int, default=0)
    ap.add_argument("--test-max-batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed); device = _device(args.device)
    flow, ids, scale = _load_flow(args.dataset)
    labels = _cluster_features(args.dataset, "lstm", flow, ids, args.clusters, args.seed)
    fit, val, test = _episodic_windows(flow)
    train_loader = _make_loader(fit, args.batch_size, device, True)
    val_loader = _make_loader(val, args.batch_size, device, False)
    test_loader = _make_loader(test, args.batch_size, device, False)

    ckpt = Path(args.checkpoint_dir)
    base = LSTMModel(12, 6).to(device)
    base.load_state_dict(torch.load(ckpt / "global_best.pt", map_location=device, weights_only=True))
    base.eval(); base.lstm.flatten_parameters()
    model = ContinuousTaskGenerator(
        base, num_bases=args.num_bases, rank=args.rank,
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
    ).to(device)
    states = [torch.load(ckpt / f"cluster_{task}.pt", map_location="cpu", weights_only=True)
              for task in range(args.clusters)]
    if not args.no_basis_warmstart:
        model.warmstart_bases_from_static_heads(states)
    experts = _load_static_experts(ckpt, args.clusters, device)
    labels_tensor = torch.from_numpy(labels).long().to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    best_val = float("inf"); best_state = copy.deepcopy(model.state_dict()); history = []
    for epoch in range(args.epochs):
        model.train(); rows = []
        for sx, sy, qx, qy, _, _ in _limited(train_loader, args.train_max_batches):
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            dynamic, fixed, _, _, coeff, _ = model(sx, sy, qx, args.temperature)
            support_pred = model.support_prediction(sx, sy, qx, args.temperature)
            query_loss = nn.functional.mse_loss(dynamic, qy)
            fixed_loss = nn.functional.mse_loss(fixed, qy)
            support_loss = nn.functional.mse_loss(support_pred, sy)
            entropy = -(coeff * torch.log(coeff.clamp_min(1e-8))).sum(dim=1).mean()
            loss = query_loss + args.fixed_loss_weight * fixed_loss + args.support_loss_weight * support_loss
            if args.entropy_weight != 0.0:
                loss = loss + args.entropy_weight * entropy
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0); optimizer.step()
            rows.append([
                float(loss.detach().cpu()), float(query_loss.detach().cpu()),
                float(fixed_loss.detach().cpu()), float(support_loss.detach().cpu()),
                float(entropy.detach().cpu()), float(coeff.max(dim=1).values.mean().detach().cpu()),
            ])
        val_loss = _validation_loss(model, val_loader, device, args.val_max_batches, args.temperature)
        a = np.asarray(rows)
        row = {
            "epoch": epoch + 1, "loss": float(a[:,0].mean()), "query_loss": float(a[:,1].mean()),
            "fixed_loss": float(a[:,2].mean()), "support_loss": float(a[:,3].mean()),
            "coeff_entropy": float(a[:,4].mean()), "coeff_mean_max": float(a[:,5].mean()),
            "val_mse": val_loss,
        }
        history.append(row); print(json.dumps(row))
        if val_loss < best_val:
            best_val = val_loss; best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, out / "model_best.pt")

    model.load_state_dict(best_state)
    validation = _evaluate(model, experts, val_loader, labels, labels_tensor, scale, device,
                           args.val_max_batches, args.temperature)
    heldout = _evaluate(model, experts, test_loader, labels, labels_tensor, scale, device,
                        args.test_max_batches, args.temperature)
    result = {
        "experiment": "dynamic_continuous_generator_v9",
        "protocol": "continuous causal task embedding -> low-rank parameter generation; best epoch chosen on validation only",
        "dataset": args.dataset,
        "best_val_normalized_mse": best_val,
        "history": history,
        "validation": validation,
        "heldout_test": heldout,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL_RESULT")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
