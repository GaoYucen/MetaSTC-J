"""V2 joint dynamic soft task discovery + task-conditioned adaptation.

This experiment addresses the main limitation exposed by V1: frozen experts that
were adapted under static KMeans assignments are not interchangeable after the
fact. V2 instead starts from the reproduced global LSTM, keeps its shared
backbone frozen, and jointly learns:

  1) K lightweight low-rank residual task adapters; and
  2) a context-dependent soft routing gate.

The static KMeans road label is used only as a *prior* for routing. The learned
gate can move probability mass to other latent tasks based on the current
12-step traffic context. Because the adapters and gate are optimized together,
task discovery and task specialization can co-evolve.

For a clean first test, this file currently targets the reproduced LSTM family.
The original static-hard cluster experts are loaded only for evaluation, so the
journal model can be compared directly with the reproduced conference baseline.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
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
    num_roads = x.shape[2]
    num_windows = x.shape[0]
    sx = x.transpose(2, 0, 1).reshape(-1, x.shape[1], 1).astype(np.float32)
    sy = y.transpose(2, 0, 1).reshape(-1, y.shape[1], 1).astype(np.float32)
    road_ids = np.repeat(np.arange(num_roads, dtype=np.int64), num_windows)
    time_ids = np.tile(np.arange(num_windows, dtype=np.int64), num_roads)
    return sx, sy, road_ids, time_ids


def _limited(loader: Iterable, max_batches: int):
    for index, batch in enumerate(loader):
        if max_batches > 0 and index >= max_batches:
            break
        yield batch


class DynamicTaskAdapterModel(nn.Module):
    """Frozen global LSTM + K jointly learned low-rank residual heads + soft gate."""

    def __init__(
        self,
        base: LSTMModel,
        num_tasks: int,
        look_back: int,
        look_forward: int,
        rank: int = 6,
        gate_hidden: int = 64,
        init_alpha: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0.0 < init_alpha < 1.0:
            raise ValueError("init_alpha must be in (0,1)")
        self.num_tasks = num_tasks
        self.look_back = look_back
        self.look_forward = look_forward
        self.feature_dim = base.linear.in_features
        self.hidden_dim = base.lstm.hidden_size
        self.rank = min(rank, self.feature_dim, look_forward)

        self.base_lstm = base.lstm
        self.base_linear = base.linear
        for parameter in self.base_lstm.parameters():
            parameter.requires_grad_(False)
        for parameter in self.base_linear.parameters():
            parameter.requires_grad_(False)

        # Linear low-rank residual: delta_k(h) = h A_k B_k + b_k.
        # Keeping it linear allows exact low-rank SVD warm-start from static heads.
        self.adapter_a = nn.Parameter(torch.empty(num_tasks, self.feature_dim, self.rank))
        self.adapter_b = nn.Parameter(torch.zeros(num_tasks, self.rank, look_forward))
        self.adapter_bias = nn.Parameter(torch.zeros(num_tasks, look_forward))
        nn.init.normal_(self.adapter_a, mean=0.0, std=0.02)

        # Current raw window + five simple statistics + last shared hidden state
        # + the original static task prior.
        gate_input = look_back + 5 + self.hidden_dim + num_tasks
        self.gate_encoder = nn.Sequential(
            nn.Linear(gate_input, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, gate_hidden),
            nn.ReLU(),
        )
        self.task_head = nn.Linear(gate_hidden, num_tasks)
        self.alpha_head = nn.Linear(gate_hidden, 1)
        nn.init.zeros_(self.task_head.weight)
        nn.init.zeros_(self.task_head.bias)
        nn.init.zeros_(self.alpha_head.weight)
        self.alpha_head.bias.data.fill_(math.log(init_alpha / (1.0 - init_alpha)))

    @torch.no_grad()
    def warmstart_from_static_heads(self, cluster_states: List[Dict[str, torch.Tensor]]) -> None:
        """Initialize each low-rank adapter from the reproduced static expert head.

        Only the final linear-head difference is used. The shared recurrent
        backbone stays equal to the reproduced global initialization, so V2 is
        not merely mixing the old full static experts.
        """
        base_w = self.base_linear.weight.detach().cpu()  # [O,F]
        base_b = self.base_linear.bias.detach().cpu()
        for task, state in enumerate(cluster_states):
            delta = (state["linear.weight"].detach().cpu() - base_w).t().float()  # [F,O]
            u, s, vh = torch.linalg.svd(delta, full_matrices=False)
            r = min(self.rank, s.numel())
            root = torch.sqrt(torch.clamp(s[:r], min=0.0))
            a = u[:, :r] * root.unsqueeze(0)
            b = root.unsqueeze(1) * vh[:r, :]
            self.adapter_a[task].zero_()
            self.adapter_b[task].zero_()
            self.adapter_a[task, :, :r].copy_(a.to(self.adapter_a.device, self.adapter_a.dtype))
            self.adapter_b[task, :r, :].copy_(b.to(self.adapter_b.device, self.adapter_b.dtype))
            self.adapter_bias[task].copy_((state["linear.bias"].detach().cpu() - base_b).to(
                self.adapter_bias.device, self.adapter_bias.dtype
            ))

    def _gate(self, x: torch.Tensor, last_hidden: torch.Tensor, static_task: torch.Tensor):
        context = x.squeeze(-1)
        mean = context.mean(dim=1, keepdim=True)
        std = context.std(dim=1, keepdim=True, unbiased=False)
        minimum = context.min(dim=1, keepdim=True).values
        maximum = context.max(dim=1, keepdim=True).values
        trend = context[:, -1:] - context[:, :1]
        prior = torch.nn.functional.one_hot(static_task, self.num_tasks).to(context.dtype)
        gate_features = torch.cat(
            [context, mean, std, minimum, maximum, trend, last_hidden, prior], dim=1
        )
        encoded = self.gate_encoder(gate_features)
        residual = torch.softmax(self.task_head(encoded), dim=1)
        alpha = torch.sigmoid(self.alpha_head(encoded))
        weights = (1.0 - alpha) * prior + alpha * residual
        return weights, alpha.squeeze(1)

    def forward(self, x: torch.Tensor, static_task: torch.Tensor):
        self.base_lstm.flatten_parameters()
        sequence, _ = self.base_lstm(x)
        flat = sequence.reshape(sequence.shape[0], -1)
        base = self.base_linear(flat)  # [B,O]

        compressed = torch.einsum("bf,kfr->bkr", flat, self.adapter_a)
        delta = torch.einsum("bkr,kro->bko", compressed, self.adapter_b)
        delta = delta + self.adapter_bias.unsqueeze(0)  # [B,K,O]

        weights, alpha = self._gate(x, sequence[:, -1, :], static_task)
        mixed_delta = (delta * weights.unsqueeze(-1)).sum(dim=1)
        dynamic = base + mixed_delta

        row = torch.arange(x.shape[0], device=x.device)
        static_adapter = base + delta[row, static_task]
        return (
            dynamic.unsqueeze(-1),
            base.unsqueeze(-1),
            static_adapter.unsqueeze(-1),
            weights,
            alpha,
            delta,
        )


def _load_static_experts(checkpoint_dir: Path, k: int, device: torch.device) -> List[LSTMModel]:
    experts: List[LSTMModel] = []
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


def _static_expert_prediction(
    experts: List[LSTMModel], data: torch.Tensor, static_task: torch.Tensor
) -> torch.Tensor:
    result = torch.empty((data.shape[0], 6, 1), device=data.device, dtype=data.dtype)
    with torch.no_grad():
        for task, expert in enumerate(experts):
            idx = torch.nonzero(static_task == task, as_tuple=False).squeeze(1)
            if idx.numel() > 0:
                result[idx] = expert(data[idx])
    return result


def _validation_loss(
    model: DynamicTaskAdapterModel,
    loader: DataLoader,
    labels_tensor: torch.Tensor,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for data, target, road, _ in _limited(loader, max_batches):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            road = road.to(device, non_blocking=True)
            static_task = labels_tensor[road]
            pred, *_ = model(data, static_task)
            losses.append(float(nn.functional.mse_loss(pred, target).cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def _evaluate(
    model: DynamicTaskAdapterModel,
    static_experts: List[LSTMModel],
    loader: DataLoader,
    labels: np.ndarray,
    labels_tensor: torch.Tensor,
    scale: float,
    device: torch.device,
    max_batches: int,
) -> Dict:
    model.eval()
    chunks = {"target": [], "global": [], "old_static": [], "joint_static": [], "dynamic": []}
    weights_all, alpha_all, roads_all, times_all, delta_norm_all = [], [], [], [], []
    with torch.no_grad():
        for data, target, road, time_id in _limited(loader, max_batches):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            road = road.to(device, non_blocking=True)
            static_task = labels_tensor[road]
            dynamic, base, joint_static, weights, alpha, delta = model(data, static_task)
            old_static = _static_expert_prediction(static_experts, data, static_task)

            chunks["target"].append((target * scale).cpu().numpy())
            chunks["global"].append((base * scale).cpu().numpy())
            chunks["old_static"].append((old_static * scale).cpu().numpy())
            chunks["joint_static"].append((joint_static * scale).cpu().numpy())
            chunks["dynamic"].append((dynamic * scale).cpu().numpy())
            weights_all.append(weights.cpu().numpy())
            alpha_all.append(alpha.cpu().numpy())
            roads_all.append(road.cpu().numpy())
            times_all.append(time_id.numpy())
            delta_norm_all.append(delta.norm(dim=2).mean(dim=1).cpu().numpy())

    target = np.concatenate(chunks["target"]).reshape(-1)
    global_pred = np.concatenate(chunks["global"]).reshape(-1)
    old_static = np.concatenate(chunks["old_static"]).reshape(-1)
    joint_static = np.concatenate(chunks["joint_static"]).reshape(-1)
    dynamic = np.concatenate(chunks["dynamic"]).reshape(-1)
    weights = np.concatenate(weights_all, axis=0)
    alpha = np.concatenate(alpha_all)
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
        "global_base": _metrics(global_pred, target),
        "reproduced_static_hard": _metrics(old_static, target),
        "joint_static_adapters": _metrics(joint_static, target),
        "joint_dynamic_soft": _metrics(dynamic, target),
        "routing": {
            "mean_alpha": float(alpha.mean()),
            "median_alpha": float(np.median(alpha)),
            "p90_alpha": float(np.quantile(alpha, 0.9)),
            "mean_entropy": float(entropy.mean()),
            "mean_max_weight": float(weights.max(axis=1).mean()),
            "dynamic_vs_static_argmax_disagreement": float((dynamic_task != static_task).mean()),
            "temporal_task_change_rate": float(changes / max(transitions, 1)),
            "roads_with_temporal_task_change": float(
                roads_with_change / max(len(np.unique(roads)), 1)
            ),
            "task_usage": weights.mean(axis=0).tolist(),
            "mean_adapter_delta_norm_normalized": float(delta_norm.mean()),
        },
        "num_window_road_samples": int(weights.shape[0]),
    }
    result["dynamic_vs_reproduced_static_delta_mae"] = (
        result["joint_dynamic_soft"]["MAE"] - result["reproduced_static_hard"]["MAE"]
    )
    result["dynamic_vs_reproduced_static_relative_mae_pct"] = 100.0 * (
        result["joint_dynamic_soft"]["MAE"] / result["reproduced_static_hard"]["MAE"] - 1.0
    )
    result["dynamic_vs_joint_static_delta_mae"] = (
        result["joint_dynamic_soft"]["MAE"] - result["joint_static_adapters"]["MAE"]
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Joint dynamic task discovery + low-rank task adapters V2")
    ap.add_argument("--dataset", choices=["beijing", "shanghai", "largest"], default="beijing")
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--checkpoint-dir", default="param/4090_tuned/lstm/beijing")
    ap.add_argument("--output-dir", default="param/journal/dynamic_joint_v2/beijing_lstm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--gate-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--gate-hidden", type=int, default=64)
    ap.add_argument("--init-alpha", type=float, default=0.05)
    ap.add_argument("--no-head-warmstart", action="store_true")
    ap.add_argument("--train-max-batches", type=int, default=50)
    ap.add_argument("--val-max-batches", type=int, default=0)
    ap.add_argument("--test-max-batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed)
    device = _device(args.device)
    look_back, look_forward, k = 12, 6, args.clusters
    flow, ids, scale = _load_flow(args.dataset)
    labels = _cluster_features(args.dataset, "lstm", flow, ids, k, args.seed)
    fit_x, fit_y, val_x, val_y, test_x, test_y = _windows(flow, look_back, look_forward)

    def loader(x, y, shuffle):
        sx, sy, roads, times = _all_samples(x, y)
        dataset = TensorDataset(
            torch.from_numpy(sx), torch.from_numpy(sy),
            torch.from_numpy(roads), torch.from_numpy(times),
        )
        return DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
            pin_memory=device.type == "cuda",
        )

    train_loader = loader(fit_x, fit_y, True)
    val_loader = loader(val_x, val_y, False)
    test_loader = loader(test_x, test_y, False)

    ckpt_dir = Path(args.checkpoint_dir)
    base = LSTMModel(look_back, look_forward).to(device)
    base.load_state_dict(torch.load(
        ckpt_dir / "global_best.pt", map_location=device, weights_only=True
    ))
    base.eval()
    base.lstm.flatten_parameters()

    cluster_states = [
        torch.load(ckpt_dir / f"cluster_{task}.pt", map_location="cpu", weights_only=True)
        for task in range(k)
    ]
    model = DynamicTaskAdapterModel(
        base=base,
        num_tasks=k,
        look_back=look_back,
        look_forward=look_forward,
        rank=args.rank,
        gate_hidden=args.gate_hidden,
        init_alpha=args.init_alpha,
    ).to(device)
    if not args.no_head_warmstart:
        model.warmstart_from_static_heads(cluster_states)

    static_experts = _load_static_experts(ckpt_dir, k, device)
    labels_tensor = torch.from_numpy(labels).long().to(device)

    adapter_parameters = [model.adapter_a, model.adapter_b, model.adapter_bias]
    gate_parameters = list(model.gate_encoder.parameters()) + list(model.task_head.parameters()) + list(model.alpha_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": args.lr},
            {"params": gate_parameters, "lr": args.gate_lr},
        ],
        weight_decay=args.weight_decay,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["head_warmstart"] = not args.no_head_warmstart
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses, alphas = [], []
        for data, target, road, _ in _limited(train_loader, args.train_max_batches):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            road = road.to(device, non_blocking=True)
            static_task = labels_tensor[road]
            pred, _, _, _, alpha, _ = model(data, static_task)
            loss = nn.functional.mse_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                adapter_parameters + gate_parameters, max_norm=5.0
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            alphas.append(float(alpha.detach().mean().cpu()))

        val_loss = _validation_loss(
            model, val_loader, labels_tensor, device, args.val_max_batches
        )
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
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, out / "model_best.pt")

    model.load_state_dict(best_state)
    validation = _evaluate(
        model, static_experts, val_loader, labels, labels_tensor, scale, device, args.val_max_batches
    )
    test = _evaluate(
        model, static_experts, test_loader, labels, labels_tensor, scale, device, args.test_max_batches
    )
    result = {
        "experiment": "joint_dynamic_task_adapter_v2",
        "dataset": args.dataset,
        "clusters": k,
        "checkpoint_dir": str(ckpt_dir),
        "best_val_normalized_mse": best_val,
        "history": history,
        "validation": validation,
        "test": test,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
