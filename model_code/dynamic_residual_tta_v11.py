"""V11 causal streaming adaptation for the continuous residual model.

The original static MetaSTC experts remain frozen.  A V10 residual adapter is
initialized from the offline training checkpoint and adapted only after the full
6-step target of an earlier forecast has become observable.  Hyperparameters
(parameter subset and learning rate) are selected on the validation stream;
the adapter is then reset to the identical offline checkpoint before the held-
out test stream.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from optimized_runner import _cluster_features, _device, _load_flow, _metrics, _set_seed
from dynamic_support_routing_v4 import _episodic_windows, _load_static_experts, _static_prediction
from dynamic_stream_tta_v7 import _episode_tensors
from dynamic_continuous_residual_v10 import ContinuousResidualAdapter


def _forward_episode(model, experts, arrays, index, labels_tensor, device):
    sx, sy, qx, qy = _episode_tensors(arrays, index, device)
    with torch.no_grad():
        static_s = _static_prediction(experts, sx, labels_tensor)
        static_q = _static_prediction(experts, qx, labels_tensor)
    pred, delta, gate = model(qx, static_q, sy - static_s, labels_tensor)
    return pred, static_q, qy, delta, gate


def _trainable(model, mode):
    for p in model.parameters():
        p.requires_grad_(False)
    if mode == "heads":
        modules = [model.delta_head, model.gate_head]
    elif mode == "all":
        modules = [model]
    else:
        raise ValueError(mode)
    params = []
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(True); params.append(p)
    return params


def _adapt_once(model, experts, optimizer, arrays, index, labels_tensor, device,
                offline_state, anchor_weight, correction_weight):
    model.train()
    pred, _, qy, delta, gate = _forward_episode(model, experts, arrays, index, labels_tensor, device)
    mse = nn.functional.mse_loss(pred, qy)
    correction_reg = (gate * delta).pow(2).mean()
    anchor = torch.zeros((), device=device)
    for name, p in model.named_parameters():
        if p.requires_grad:
            anchor = anchor + (p - offline_state[name]).pow(2).mean()
    loss = mse + correction_weight * correction_reg + anchor_weight * anchor
    optimizer.zero_grad(set_to_none=True); loss.backward()
    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
    optimizer.step()
    return float(loss.detach()), float(mse.detach())


def _stream(model, experts, arrays, labels_tensor, scale, device, offline_state,
            mode, lr, delay, adapt_steps, anchor_weight, correction_weight):
    model.load_state_dict({k: v.detach().clone() for k, v in offline_state.items()})
    params = _trainable(model, mode)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0) if lr > 0 else None
    dynamic_chunks, static_chunks, target_chunks = [], [], []
    gates, corrections = [], []
    updates = 0
    for i in range(arrays[0].shape[0]):
        if optimizer is not None and i >= delay:
            for _ in range(adapt_steps):
                _adapt_once(model, experts, optimizer, arrays, i - delay, labels_tensor, device,
                            offline_state, anchor_weight, correction_weight)
                updates += 1
        model.eval()
        with torch.no_grad():
            pred, static_q, qy, delta, gate = _forward_episode(model, experts, arrays, i, labels_tensor, device)
        dynamic_chunks.append((pred * scale).cpu().numpy())
        static_chunks.append((static_q * scale).cpu().numpy())
        target_chunks.append((qy * scale).cpu().numpy())
        gates.append(gate.squeeze(-1).squeeze(-1).cpu().numpy())
        corrections.append((gate * delta).abs().mean(dim=(1, 2)).cpu().numpy() * scale)
    target = np.stack(target_chunks).reshape(-1)
    dynamic = np.stack(dynamic_chunks).reshape(-1)
    static = np.stack(static_chunks).reshape(-1)
    gate_arr = np.stack(gates)
    correction_arr = np.stack(corrections)
    return {
        "MAE": _metrics(dynamic, target)["MAE"],
        "RMSE": _metrics(dynamic, target)["RMSE"],
        "MSE": _metrics(dynamic, target)["MSE"],
        "MAPE": _metrics(dynamic, target)["MAPE"],
        "R2": _metrics(dynamic, target)["R2"],
        "static_metrics": _metrics(static, target),
        "updates": updates,
        "mean_gate": float(gate_arr.mean()),
        "std_gate": float(gate_arr.std()),
        "mean_abs_correction_raw_scale": float(correction_arr.mean()),
    }


def main():
    ap = argparse.ArgumentParser(description="Causal streaming TTA for V10 residual adapter")
    ap.add_argument("--dataset", default="beijing", choices=["beijing", "shanghai", "largest"])
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--checkpoint-dir", default="param/4090_tuned/lstm/beijing")
    ap.add_argument("--residual-checkpoint", default="param/journal/dynamic_continuous_residual_v10/beijing_lstm_smoke/model_best.pt")
    ap.add_argument("--output-dir", default="param/journal/dynamic_residual_tta_v11/beijing_lstm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--emb-dim", type=int, default=8)
    ap.add_argument("--delay", type=int, default=6)
    ap.add_argument("--adapt-steps", type=int, default=1)
    ap.add_argument("--modes", nargs="+", default=["heads", "all"])
    ap.add_argument("--lr-grid", type=float, nargs="+", default=[0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3])
    ap.add_argument("--anchor-weight", type=float, default=1e-3)
    ap.add_argument("--correction-weight", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed)
    device = _device(args.device)
    flow, ids, scale = _load_flow(args.dataset)
    labels = _cluster_features(args.dataset, "lstm", flow, ids, args.clusters, args.seed)
    _, val_arrays, test_arrays = _episodic_windows(flow)
    labels_tensor = torch.from_numpy(labels).long().to(device)
    experts = _load_static_experts(Path(args.checkpoint_dir), args.clusters, device)

    model = ContinuousResidualAdapter(args.clusters, args.emb_dim, args.hidden).to(device)
    model.load_state_dict(torch.load(args.residual_checkpoint, map_location=device, weights_only=True))
    offline_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    validation_grid = []
    for mode in args.modes:
        for lr in args.lr_grid:
            result = _stream(model, experts, val_arrays, labels_tensor, scale, device, offline_state,
                             mode, lr, args.delay, args.adapt_steps, args.anchor_weight, args.correction_weight)
            validation_grid.append({"mode": mode, "lr": lr, **result})
            print("VAL", json.dumps({"mode": mode, "lr": lr, "MAE": result["MAE"],
                                      "static_MAE": result["static_metrics"]["MAE"],
                                      "updates": result["updates"], "mean_gate": result["mean_gate"]}), flush=True)
    validation_grid.sort(key=lambda x: (x["MAE"], x["RMSE"]))
    selected = {"mode": validation_grid[0]["mode"], "lr": validation_grid[0]["lr"]}
    print("SELECTED_BY_VALIDATION", json.dumps(selected), flush=True)

    selected_val = validation_grid[0]
    test_result = _stream(model, experts, test_arrays, labels_tensor, scale, device, offline_state,
                          selected["mode"], selected["lr"], args.delay, args.adapt_steps,
                          args.anchor_weight, args.correction_weight)
    result = {
        "experiment": "dynamic_residual_tta_v11",
        "protocol": "validation-selected delayed causal residual adaptation; reset before held-out test",
        "delay_origins": args.delay,
        "selected": selected,
        "validation_grid": validation_grid,
        "validation_selected": selected_val,
        "heldout_test": test_result,
    }
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL_RESULT")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
