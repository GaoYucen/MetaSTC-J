"""V9 diagnostic: is the useful latent task causally predictable?

This is a diagnostic, not a new model.  For each episodic forecast origin it
computes the hindsight-best adapter task and asks whether that task can be
recovered from information that is legally available at prediction time:

* original static KMeans task;
* best task on the recent observed support pair;
* the query-best task from six origins ago (now fully observed);
* EMA / rolling averages of per-task query errors that have become observable.

The same task-conditioned adapter bank is used for all routing diagnostics.
This separates "large oracle headroom" from "actionable, temporally predictable
headroom" before adding more routing machinery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from optimized_runner import LSTMModel, _cluster_features, _device, _load_flow, _metrics, _set_seed
from dynamic_support_routing_v4 import TaskAdapterBank, _episodic_windows, _load_static_experts, _static_prediction
from dynamic_stream_tta_v7 import _episode_tensors


def _append(chunks, name, tensor, scale):
    chunks[name].append((tensor * scale).detach().cpu().numpy())


def _select(task_predictions, task_ids):
    row = torch.arange(task_predictions.shape[0], device=task_predictions.device)
    return task_predictions[row, task_ids]


def _evaluate_split(bank, experts, arrays, labels_tensor, scale, device, delay=6, ema_betas=(0.5, 0.8, 0.95), rolling=(3, 6)):
    bank.eval()
    chunks = {"target": [], "static_expert": [], "static_adapter": [], "support_best": [], "lag_oracle": [], "oracle": []}
    for beta in ema_betas:
        chunks[f"ema_{beta:g}"] = []
    for window in rolling:
        chunks[f"rolling_{window}"] = []

    agreements = {"static_adapter": [], "support_best": [], "lag_oracle": []}
    for beta in ema_betas:
        agreements[f"ema_{beta:g}"] = []
    for window in rolling:
        agreements[f"rolling_{window}"] = []

    query_error_history = []
    oracle_task_history = []
    ema_state = {beta: None for beta in ema_betas}
    observed_error_history = []
    oracle_transition_num = 0
    oracle_transition_den = 0

    episodes = arrays[0].shape[0]
    for i in range(episodes):
        sx, sy, qx, qy = _episode_tensors(arrays, i, device)
        with torch.no_grad():
            sp, _, _ = bank(sx)
            qp, _, _ = bank(qx)
            support_error = ((sp - sy[:, None]) ** 2).mean(dim=(2, 3))
            query_error = ((qp - qy[:, None]) ** 2).mean(dim=(2, 3))
            oracle_task = query_error.argmin(dim=1)
            support_task = support_error.argmin(dim=1)

            # Only feedback from episode i-delay is fully observed now.
            if i >= delay:
                newly_observed = query_error_history[i - delay]
                observed_error_history.append(newly_observed)
                for beta in ema_betas:
                    if ema_state[beta] is None:
                        ema_state[beta] = newly_observed.clone()
                    else:
                        ema_state[beta].mul_(beta).add_(newly_observed, alpha=1.0 - beta)

            static_task = labels_tensor
            lag_task = oracle_task_history[i - delay] if i >= delay else static_task
            strategy_tasks = {
                "static_adapter": static_task,
                "support_best": support_task,
                "lag_oracle": lag_task,
            }
            for beta in ema_betas:
                strategy_tasks[f"ema_{beta:g}"] = ema_state[beta].argmin(dim=1) if ema_state[beta] is not None else static_task
            for window in rolling:
                if observed_error_history:
                    recent = torch.stack(observed_error_history[-window:], dim=0).mean(dim=0)
                    strategy_tasks[f"rolling_{window}"] = recent.argmin(dim=1)
                else:
                    strategy_tasks[f"rolling_{window}"] = static_task

            _append(chunks, "target", qy, scale)
            _append(chunks, "static_expert", _static_prediction(experts, qx, static_task), scale)
            _append(chunks, "oracle", _select(qp, oracle_task), scale)
            for name, task in strategy_tasks.items():
                _append(chunks, name, _select(qp, task), scale)
                agreements[name].append((task == oracle_task).float().cpu().numpy())

            if oracle_task_history:
                oracle_transition_num += int((oracle_task != oracle_task_history[-1]).sum().item())
                oracle_transition_den += int(oracle_task.numel())

        query_error_history.append(query_error.detach())
        oracle_task_history.append(oracle_task.detach())

    target = np.concatenate(chunks["target"]).reshape(-1)
    metrics = {name: _metrics(np.concatenate(values).reshape(-1), target)
               for name, values in chunks.items() if name != "target"}
    agreement = {name: float(np.concatenate(values).mean()) for name, values in agreements.items()}

    oracle_tasks = torch.stack(oracle_task_history, dim=0).cpu().numpy()
    lag6_persistence = float((oracle_tasks[delay:] == oracle_tasks[:-delay]).mean()) if episodes > delay else float("nan")
    result = {
        "episodes": int(episodes),
        "roads": int(labels_tensor.numel()),
        "delay_origins": int(delay),
        "metrics": metrics,
        "oracle_task_agreement": agreement,
        "oracle_task_one_step_change_rate": float(oracle_transition_num / max(oracle_transition_den, 1)),
        "oracle_task_lag6_persistence": lag6_persistence,
    }
    static_mae = metrics["static_expert"]["MAE"]
    oracle_mae = metrics["oracle"]["MAE"]
    result["oracle_headroom_vs_static_pct"] = float((static_mae - oracle_mae) / static_mae * 100.0)
    result["causal_routes_vs_static_pct"] = {
        name: float((static_mae - m["MAE"]) / static_mae * 100.0)
        for name, m in metrics.items() if name not in {"static_expert", "oracle"}
    }
    return result


def main():
    ap = argparse.ArgumentParser(description="Causal latent-task predictability diagnostic")
    ap.add_argument("--dataset", default="beijing", choices=["beijing", "shanghai", "largest"])
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--checkpoint-dir", default="param/4090_tuned/lstm/beijing")
    ap.add_argument("--bank-checkpoint", default="param/journal/dynamic_support_v4/beijing_lstm_substantial/model_best.pt")
    ap.add_argument("--output-dir", default="param/journal/dynamic_task_predictability_v9/beijing_lstm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--delay", type=int, default=6)
    ap.add_argument("--ema-betas", type=float, nargs="+", default=[0.5, 0.8, 0.95])
    ap.add_argument("--rolling", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed)
    device = _device(args.device)
    flow, ids, scale = _load_flow(args.dataset)
    labels = _cluster_features(args.dataset, "lstm", flow, ids, args.clusters, args.seed)
    _, val_arrays, test_arrays = _episodic_windows(flow)
    labels_tensor = torch.from_numpy(labels).long().to(device)

    ckpt = Path(args.checkpoint_dir)
    base = LSTMModel(12, 6).to(device)
    base.load_state_dict(torch.load(ckpt / "global_best.pt", map_location=device, weights_only=True))
    bank = TaskAdapterBank(base, args.clusters, rank=6).to(device)
    bank.load_state_dict(torch.load(args.bank_checkpoint, map_location=device, weights_only=True))
    bank.eval()
    experts = _load_static_experts(ckpt, args.clusters, device)

    result = {
        "experiment": "dynamic_task_predictability_v9",
        "purpose": "diagnose whether oracle latent-task headroom is causally predictable",
        "validation": _evaluate_split(bank, experts, val_arrays, labels_tensor, scale, device, args.delay, tuple(args.ema_betas), tuple(args.rolling)),
        "heldout_test": _evaluate_split(bank, experts, test_arrays, labels_tensor, scale, device, args.delay, tuple(args.ema_betas), tuple(args.rolling)),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
