"""V3 amortized-EM dynamic latent task discovery.

V1/V2 showed that a gate trained only through final mixture loss tends to remain
near the original static KMeans assignment. V3 gives task discovery an explicit
latent-variable learning signal while preserving test-time label independence.

Training iteration:
  E-like step: each task adapter predicts the current sample; prediction losses
               define a soft responsibility q(task | x, y).
  M-like step: update task adapters with q-weighted prediction loss, and train an
               amortized gate p(task | x) to match q using only current context.

At test time q is unavailable and unused; routing is entirely p(task | x).
The shared global LSTM is frozen. Only lightweight low-rank task adapters and the
small routing network are learned jointly.
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

from optimized_runner import (
    LSTMModel,
    _cluster_features,
    _device,
    _load_flow,
    _metrics,
    _set_seed,
    _windows,
)


def _all_samples(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    roads = x.shape[2]
    windows = x.shape[0]
    sx = x.transpose(2, 0, 1).reshape(-1, x.shape[1], 1).astype(np.float32)
    sy = y.transpose(2, 0, 1).reshape(-1, y.shape[1], 1).astype(np.float32)
    road_ids = np.repeat(np.arange(roads, dtype=np.int64), windows)
    time_ids = np.tile(np.arange(windows, dtype=np.int64), roads)
    return sx, sy, road_ids, time_ids


def _limited(loader: Iterable, max_batches: int):
    for index, batch in enumerate(loader):
        if max_batches > 0 and index >= max_batches:
            break
        yield batch


class TaskAdapterBank(nn.Module):
    def __init__(self, base: LSTMModel, num_tasks: int, rank: int = 6) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.feature_dim = base.linear.in_features
        self.output_dim = base.linear.out_features
        self.hidden_dim = base.lstm.hidden_size
        self.rank = min(rank, self.feature_dim, self.output_dim)
        self.base_lstm = base.lstm
        self.base_linear = base.linear
        for parameter in self.base_lstm.parameters():
            parameter.requires_grad_(False)
        for parameter in self.base_linear.parameters():
            parameter.requires_grad_(False)

        self.adapter_a = nn.Parameter(torch.empty(num_tasks, self.feature_dim, self.rank))
        self.adapter_b = nn.Parameter(torch.zeros(num_tasks, self.rank, self.output_dim))
        self.adapter_bias = nn.Parameter(torch.zeros(num_tasks, self.output_dim))
        nn.init.normal_(self.adapter_a, std=0.02)

    @torch.no_grad()
    def warmstart_from_static_heads(self, cluster_states: List[Dict[str, torch.Tensor]]) -> None:
        base_w = self.base_linear.weight.detach().cpu()
        base_b = self.base_linear.bias.detach().cpu()
        for task, state in enumerate(cluster_states):
            delta = (state["linear.weight"].detach().cpu() - base_w).t().float()
            u, s, vh = torch.linalg.svd(delta, full_matrices=False)
            r = min(self.rank, s.numel())
            root = torch.sqrt(torch.clamp(s[:r], min=0.0))
            self.adapter_a[task].zero_()
            self.adapter_b[task].zero_()
            self.adapter_a[task, :, :r].copy_((u[:, :r] * root.unsqueeze(0)).to(self.adapter_a.device))
            self.adapter_b[task, :r, :].copy_((root.unsqueeze(1) * vh[:r, :]).to(self.adapter_b.device))
            self.adapter_bias[task].copy_((state["linear.bias"].detach().cpu() - base_b).to(self.adapter_bias.device))

    def forward(self, x: torch.Tensor):
        self.base_lstm.flatten_parameters()
        sequence, _ = self.base_lstm(x)
        flat = sequence.reshape(sequence.shape[0], -1)
        base = self.base_linear(flat)
        compressed = torch.einsum("bf,kfr->bkr", flat, self.adapter_a)
        delta = torch.einsum("bkr,kro->bko", compressed, self.adapter_b)
        delta = delta + self.adapter_bias.unsqueeze(0)
        task_predictions = base.unsqueeze(1) + delta
        return task_predictions.unsqueeze(-1), base.unsqueeze(-1), sequence[:, -1, :], delta


class AmortizedTaskGate(nn.Module):
    def __init__(self, look_back: int, hidden_dim: int, num_tasks: int, gate_hidden: int = 64) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        input_dim = look_back + 5 + hidden_dim + num_tasks
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, gate_hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(gate_hidden, num_tasks)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x: torch.Tensor,
        last_hidden: torch.Tensor,
        static_task: torch.Tensor,
        prior_strength: float,
    ) -> torch.Tensor:
        context = x.squeeze(-1)
        mean = context.mean(dim=1, keepdim=True)
        std = context.std(dim=1, keepdim=True, unbiased=False)
        minimum = context.min(dim=1, keepdim=True).values
        maximum = context.max(dim=1, keepdim=True).values
        trend = context[:, -1:] - context[:, :1]
        prior = torch.nn.functional.one_hot(static_task, self.num_tasks).to(context.dtype)
        features = torch.cat(
            [context, mean, std, minimum, maximum, trend, last_hidden, prior], dim=1
        )
        logits = self.head(self.encoder(features)) + prior_strength * prior
        return torch.softmax(logits, dim=1)


def _responsibilities(
    errors: torch.Tensor,
    static_task: torch.Tensor,
    num_tasks: int,
    temperature: float,
    prior_strength: float,
) -> torch.Tensor:
    # Per-sample standardization makes the temperature independent of raw flow scale.
    best = errors.min(dim=1, keepdim=True).values
    spread = errors.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
    logits = -(errors - best) / (spread * temperature)
    if prior_strength != 0.0:
        prior = torch.nn.functional.one_hot(static_task, num_tasks).to(errors.dtype)
        logits = logits + prior_strength * prior
    return torch.softmax(logits, dim=1)


def _load_static_experts(checkpoint_dir: Path, k: int, device: torch.device) -> List[LSTMModel]:
    experts = []
    for task in range(k):
        model = LSTMModel(12, 6).to(device)
        model.load_state_dict(torch.load(
            checkpoint_dir / f"cluster_{task}.pt", map_location=device, weights_only=True
        ))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.lstm.flatten_parameters()
        experts.append(model)
    return experts


def _static_expert_prediction(experts, data, static_task):
    result = torch.empty((data.shape[0], 6, 1), device=data.device, dtype=data.dtype)
    with torch.no_grad():
        for task, expert in enumerate(experts):
            idx = torch.nonzero(static_task == task, as_tuple=False).squeeze(1)
            if idx.numel() > 0:
                result[idx] = expert(data[idx])
    return result


def _validation_loss(bank, gate, loader, labels_tensor, device, max_batches, gate_prior_strength):
    bank.eval(); gate.eval()
    losses = []
    with torch.no_grad():
        for data, target, road, _ in _limited(loader, max_batches):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            road = road.to(device, non_blocking=True)
            static_task = labels_tensor[road]
            task_pred, _, last_hidden, _ = bank(data)
            weights = gate(data, last_hidden, static_task, gate_prior_strength)
            mixed = (task_pred * weights[:, :, None, None]).sum(dim=1)
            losses.append(float(nn.functional.mse_loss(mixed, target).cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def _evaluate(
    bank,
    gate,
    static_experts,
    loader,
    labels,
    labels_tensor,
    scale,
    device,
    max_batches,
    gate_prior_strength,
):
    bank.eval(); gate.eval()
    chunks = {name: [] for name in ["target", "global", "old_static", "adapter_static", "dynamic_soft", "dynamic_hard", "oracle"]}
    weights_all, roads_all, times_all, delta_norm_all = [], [], [], []
    with torch.no_grad():
        for data, target, road, time_id in _limited(loader, max_batches):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            road = road.to(device, non_blocking=True)
            static_task = labels_tensor[road]
            task_pred, base, last_hidden, delta = bank(data)
            weights = gate(data, last_hidden, static_task, gate_prior_strength)
            dynamic_soft = (task_pred * weights[:, :, None, None]).sum(dim=1)
            dynamic_task = weights.argmax(dim=1)
            row = torch.arange(data.shape[0], device=device)
            dynamic_hard = task_pred[row, dynamic_task]
            adapter_static = task_pred[row, static_task]
            old_static = _static_expert_prediction(static_experts, data, static_task)
            errors = ((task_pred - target[:, None]) ** 2).mean(dim=(2, 3))
            oracle_task = errors.argmin(dim=1)
            oracle = task_pred[row, oracle_task]

            values = {
                "target": target,
                "global": base,
                "old_static": old_static,
                "adapter_static": adapter_static,
                "dynamic_soft": dynamic_soft,
                "dynamic_hard": dynamic_hard,
                "oracle": oracle,
            }
            for name, value in values.items():
                chunks[name].append((value * scale).cpu().numpy())
            weights_all.append(weights.cpu().numpy())
            roads_all.append(road.cpu().numpy())
            times_all.append(time_id.numpy())
            delta_norm_all.append(delta.norm(dim=2).mean(dim=1).cpu().numpy())

    target = np.concatenate(chunks["target"]).reshape(-1)
    predictions = {name: np.concatenate(chunks[name]).reshape(-1) for name in chunks if name != "target"}
    weights = np.concatenate(weights_all)
    roads = np.concatenate(roads_all)
    times = np.concatenate(times_all)
    delta_norm = np.concatenate(delta_norm_all)
    static_task = labels[roads]
    dynamic_task = weights.argmax(axis=1)

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

    result = {
        "global_base": _metrics(predictions["global"], target),
        "reproduced_static_hard": _metrics(predictions["old_static"], target),
        "adapter_static": _metrics(predictions["adapter_static"], target),
        "dynamic_soft": _metrics(predictions["dynamic_soft"], target),
        "dynamic_hard": _metrics(predictions["dynamic_hard"], target),
        "oracle_best_task_diagnostic": _metrics(predictions["oracle"], target),
        "routing": {
            "mean_entropy": float(entropy.mean()),
            "mean_max_weight": float(weights.max(axis=1).mean()),
            "dynamic_vs_static_argmax_disagreement": float((dynamic_task != static_task).mean()),
            "temporal_task_change_rate": float(changes / max(transitions, 1)),
            "roads_with_temporal_task_change": float(roads_with_change / max(len(np.unique(roads)), 1)),
            "task_usage": weights.mean(axis=0).tolist(),
            "mean_adapter_delta_norm_normalized": float(delta_norm.mean()),
        },
        "num_window_road_samples": int(weights.shape[0]),
    }
    result["dynamic_vs_reproduced_static_delta_mae"] = result["dynamic_soft"]["MAE"] - result["reproduced_static_hard"]["MAE"]
    result["dynamic_vs_reproduced_static_relative_mae_pct"] = 100.0 * (result["dynamic_soft"]["MAE"] / result["reproduced_static_hard"]["MAE"] - 1.0)
    result["dynamic_vs_adapter_static_delta_mae"] = result["dynamic_soft"]["MAE"] - result["adapter_static"]["MAE"]
    return result


def main():
    ap = argparse.ArgumentParser(description="Amortized-EM dynamic latent task discovery")
    ap.add_argument("--dataset", choices=["beijing", "shanghai", "largest"], default="beijing")
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--checkpoint-dir", default="param/4090_tuned/lstm/beijing")
    ap.add_argument("--output-dir", default="param/journal/dynamic_amortized_em_v3/beijing_lstm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--adapter-lr", type=float, default=0.002)
    ap.add_argument("--gate-lr", type=float, default=0.001)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--gate-hidden", type=int, default=64)
    ap.add_argument("--responsibility-temperature", type=float, default=1.0)
    ap.add_argument("--posterior-prior-strength", type=float, default=0.25)
    ap.add_argument("--gate-prior-strength", type=float, default=1.0)
    ap.add_argument("--gate-loss-weight", type=float, default=0.005)
    ap.add_argument("--mixture-loss-weight", type=float, default=0.5)
    ap.add_argument("--balance-weight", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--no-head-warmstart", action="store_true")
    ap.add_argument("--train-max-batches", type=int, default=50)
    ap.add_argument("--val-max-batches", type=int, default=0)
    ap.add_argument("--test-max-batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed)
    device = _device(args.device)
    k = args.clusters
    flow, ids, scale = _load_flow(args.dataset)
    labels = _cluster_features(args.dataset, "lstm", flow, ids, k, args.seed)
    fit_x, fit_y, val_x, val_y, test_x, test_y = _windows(flow, 12, 6)

    def loader(x, y, shuffle):
        sx, sy, roads, times = _all_samples(x, y)
        dataset = TensorDataset(torch.from_numpy(sx), torch.from_numpy(sy), torch.from_numpy(roads), torch.from_numpy(times))
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=0, pin_memory=device.type == "cuda")

    train_loader = loader(fit_x, fit_y, True)
    val_loader = loader(val_x, val_y, False)
    test_loader = loader(test_x, test_y, False)

    ckpt = Path(args.checkpoint_dir)
    base = LSTMModel(12, 6).to(device)
    base.load_state_dict(torch.load(ckpt / "global_best.pt", map_location=device, weights_only=True))
    base.eval(); base.lstm.flatten_parameters()
    cluster_states = [torch.load(ckpt / f"cluster_{task}.pt", map_location="cpu", weights_only=True) for task in range(k)]

    bank = TaskAdapterBank(base, k, rank=args.rank).to(device)
    if not args.no_head_warmstart:
        bank.warmstart_from_static_heads(cluster_states)
    gate = AmortizedTaskGate(12, bank.hidden_dim, k, gate_hidden=args.gate_hidden).to(device)
    labels_tensor = torch.from_numpy(labels).long().to(device)
    static_experts = _load_static_experts(ckpt, k, device)

    optimizer = torch.optim.AdamW([
        {"params": [bank.adapter_a, bank.adapter_b, bank.adapter_bias], "lr": args.adapter_lr},
        {"params": gate.parameters(), "lr": args.gate_lr},
    ], weight_decay=args.weight_decay)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    best_val = float("inf")
    best_bank = copy.deepcopy(bank.state_dict())
    best_gate = copy.deepcopy(gate.state_dict())
    history = []

    for epoch in range(args.epochs):
        bank.train(); gate.train()
        rows = []
        for data, target, road, _ in _limited(train_loader, args.train_max_batches):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            road = road.to(device, non_blocking=True)
            static_task = labels_tensor[road]
            task_pred, _, last_hidden, _ = bank(data)
            errors = ((task_pred - target[:, None]) ** 2).mean(dim=(2, 3))
            q = _responsibilities(errors.detach(), static_task, k, args.responsibility_temperature, args.posterior_prior_strength)
            p = gate(data, last_hidden, static_task, args.gate_prior_strength)
            mixed = (task_pred * p[:, :, None, None]).sum(dim=1)

            adapter_loss = (q * errors).sum(dim=1).mean()
            mixture_loss = nn.functional.mse_loss(mixed, target)
            gate_ce = -(q * torch.log(p.clamp_min(1e-8))).sum(dim=1).mean()
            usage = q.mean(dim=0).clamp_min(1e-8)
            balance = (usage * torch.log(usage * k)).sum()
            loss = adapter_loss + args.mixture_loss_weight * mixture_loss + args.gate_loss_weight * gate_ce + args.balance_weight * balance

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(bank.parameters()) + list(gate.parameters()), 5.0)
            optimizer.step()
            q_entropy = -(q * torch.log(q.clamp_min(1e-8))).sum(dim=1).mean()
            p_entropy = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1).mean()
            rows.append((float(loss.detach().cpu()), float(adapter_loss.detach().cpu()), float(mixture_loss.detach().cpu()), float(gate_ce.detach().cpu()), float(q_entropy.cpu()), float(p_entropy.detach().cpu()), float(q.max(dim=1).values.mean().cpu())))

        val_loss = _validation_loss(bank, gate, val_loader, labels_tensor, device, args.val_max_batches, args.gate_prior_strength)
        arr = np.asarray(rows)
        record = {
            "epoch": epoch + 1,
            "total_loss": float(arr[:, 0].mean()),
            "adapter_loss": float(arr[:, 1].mean()),
            "mixture_loss": float(arr[:, 2].mean()),
            "gate_ce": float(arr[:, 3].mean()),
            "q_entropy": float(arr[:, 4].mean()),
            "p_entropy": float(arr[:, 5].mean()),
            "q_mean_max": float(arr[:, 6].mean()),
            "val_mse": val_loss,
        }
        history.append(record)
        print(json.dumps(record))
        if val_loss < best_val:
            best_val = val_loss
            best_bank = copy.deepcopy(bank.state_dict())
            best_gate = copy.deepcopy(gate.state_dict())
            torch.save({"bank": best_bank, "gate": best_gate}, out / "model_best.pt")

    bank.load_state_dict(best_bank); gate.load_state_dict(best_gate)
    validation = _evaluate(bank, gate, static_experts, val_loader, labels, labels_tensor, scale, device, args.val_max_batches, args.gate_prior_strength)
    test = _evaluate(bank, gate, static_experts, test_loader, labels, labels_tensor, scale, device, args.test_max_batches, args.gate_prior_strength)
    result = {
        "experiment": "dynamic_amortized_em_v3",
        "dataset": args.dataset,
        "clusters": k,
        "best_val_normalized_mse": best_val,
        "history": history,
        "validation": validation,
        "test": test,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
