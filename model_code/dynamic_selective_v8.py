"""V8 selective causal dynamic routing with a static safety anchor.

V7 shows that causal router TTA corrects a substantial part of the temporal
shift, but forcing the dynamic branch on every road/time can still underperform
the reproduced static-hard MetaSTC baseline on a future held-out segment.

V8 treats the static model as a safe anchor and the TTA-adapted dynamic model as
an online expert. Before forecast origin i, only feedback whose entire 6-step
query horizon is already observed is allowed to update a delayed EMA of the
realized dynamic-vs-static loss difference. The dynamic forecast is used only
when this causal performance estimate says it has been better by a validation-
selected margin. No held-out test labels are used for hyperparameter selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from optimized_runner import LSTMModel, _cluster_features, _device, _load_flow, _metrics, _set_seed
from dynamic_support_routing_v4 import TaskAdapterBank, _episodic_windows, _load_static_experts, _static_prediction
from dynamic_support_router_v6 import SupportAwareRouter
from dynamic_stream_tta_v7 import _adapt_once, _forward_episode


def _collect_stream(bank, router, experts, arrays, labels_tensor, device,
                    support_temp, support_prior, delay, tta_lr, adapt_steps,
                    oracle_temp, ce_weight, anchor_weight, offline_state):
    """Collect predictions from a causally adapted dynamic router and static anchor."""
    router.load_state_dict({k: v.detach().clone() for k, v in offline_state.items()})
    router.eval()
    optimizer = torch.optim.AdamW(router.parameters(), lr=tta_lr, weight_decay=0.0) if tta_lr > 0 else None
    dyn_all, static_all, target_all, weight_all = [], [], [], []
    adapt_updates = 0
    count = arrays[0].shape[0]
    for i in range(count):
        if optimizer is not None and i >= delay:
            for _ in range(adapt_steps):
                _adapt_once(
                    bank, router, optimizer, arrays, i - delay, labels_tensor, device,
                    support_temp, support_prior, oracle_temp, ce_weight, anchor_weight,
                    offline_state,
                )
                adapt_updates += 1
        router.eval()
        with torch.no_grad():
            _, _, qx, qy, _, _, p, _, mixed = _forward_episode(
                bank, router, arrays, i, labels_tensor, device, support_temp, support_prior
            )
            static = _static_prediction(experts, qx, labels_tensor)
        dyn_all.append(mixed.cpu().numpy())
        static_all.append(static.cpu().numpy())
        target_all.append(qy.cpu().numpy())
        weight_all.append(p.cpu().numpy())
    return (
        np.stack(dyn_all, axis=0),
        np.stack(static_all, axis=0),
        np.stack(target_all, axis=0),
        np.stack(weight_all, axis=0),
        adapt_updates,
    )


def _evaluate_raw(pred, target, scale):
    return _metrics((pred * scale).reshape(-1), (target * scale).reshape(-1))


def _selective_replay(dynamic, static, target, scale, delay, beta, margin, min_updates, mode):
    """Replay a delayed causal selector over already-produced streaming forecasts."""
    episodes, roads = dynamic.shape[:2]
    if mode == "global":
        ema = 0.0
    elif mode == "per_road":
        ema = np.zeros(roads, dtype=np.float32)
    else:
        raise ValueError(mode)

    selected = []
    masks = []
    feedback_count = 0
    for i in range(episodes):
        # Episode i-delay has become fully observed before origin i.
        if i >= delay:
            j = i - delay
            dyn_err = ((dynamic[j] - target[j]) ** 2).mean(axis=(1, 2))
            sta_err = ((static[j] - target[j]) ** 2).mean(axis=(1, 2))
            diff = dyn_err - sta_err
            if mode == "global":
                value = float(diff.mean())
                ema = value if feedback_count == 0 else beta * ema + (1.0 - beta) * value
            else:
                ema = diff if feedback_count == 0 else beta * ema + (1.0 - beta) * diff
            feedback_count += 1

        if feedback_count < min_updates:
            use_dynamic = np.zeros(roads, dtype=bool)
        elif mode == "global":
            use_dynamic = np.full(roads, ema < -margin, dtype=bool)
        else:
            use_dynamic = ema < -margin

        chosen = np.where(use_dynamic[:, None, None], dynamic[i], static[i])
        selected.append(chosen)
        masks.append(use_dynamic)

    selected = np.stack(selected, axis=0)
    masks = np.stack(masks, axis=0)
    result = _evaluate_raw(selected, target, scale)
    result["dynamic_usage_rate"] = float(masks.mean())
    result["feedback_updates"] = feedback_count
    if episodes > 1:
        result["decision_change_rate"] = float((masks[1:] != masks[:-1]).mean())
        result["roads_ever_dynamic"] = float(masks.any(axis=0).mean())
    else:
        result["decision_change_rate"] = 0.0
        result["roads_ever_dynamic"] = float(masks.any(axis=0).mean())
    return result


def main():
    ap = argparse.ArgumentParser(description="Selective causal dynamic routing V8")
    ap.add_argument("--dataset", default="beijing", choices=["beijing", "shanghai", "largest"])
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--checkpoint-dir", default="param/4090_tuned/lstm/beijing")
    ap.add_argument("--bank-checkpoint", default="param/journal/dynamic_support_v4/beijing_lstm_substantial/model_best.pt")
    ap.add_argument("--router-checkpoint", default="param/journal/dynamic_support_router_v6/beijing_lstm_smoke/router_best.pt")
    ap.add_argument("--output-dir", default="param/journal/dynamic_selective_v8/beijing_lstm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--delay", type=int, default=6)
    ap.add_argument("--adapt-steps", type=int, default=1)
    ap.add_argument("--tta-lr", type=float, default=1e-3)
    ap.add_argument("--support-temperature", type=float, default=1.0)
    ap.add_argument("--support-prior", type=float, default=0.25)
    ap.add_argument("--oracle-temperature", type=float, default=0.5)
    ap.add_argument("--ce-weight", type=float, default=0.02)
    ap.add_argument("--anchor-weight", type=float, default=0.0001)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.5, 0.8, 0.9, 0.95])
    ap.add_argument("--margins", type=float, nargs="+", default=[0.0, 1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3])
    ap.add_argument("--min-updates", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--modes", nargs="+", default=["per_road", "global"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed)
    device = _device(args.device)
    k = args.clusters
    flow, ids, scale = _load_flow(args.dataset)
    labels = _cluster_features(args.dataset, "lstm", flow, ids, k, args.seed)
    _, val_arrays, test_arrays = _episodic_windows(flow)
    labels_tensor = torch.from_numpy(labels).long().to(device)

    ckpt = Path(args.checkpoint_dir)
    base = LSTMModel(12, 6).to(device)
    base.load_state_dict(torch.load(ckpt / "global_best.pt", map_location=device, weights_only=True))
    bank = TaskAdapterBank(base, k, rank=6).to(device)
    bank.load_state_dict(torch.load(args.bank_checkpoint, map_location=device, weights_only=True))
    bank.eval()
    for p in bank.parameters():
        p.requires_grad_(False)

    router = SupportAwareRouter(k).to(device)
    router.load_state_dict(torch.load(args.router_checkpoint, map_location=device, weights_only=True))
    offline_state = {name: p.detach().clone() for name, p in router.state_dict().items()}
    experts = _load_static_experts(ckpt, k, device)

    print("collecting validation causal TTA stream")
    vd, vs, vy, vw, val_adapt_updates = _collect_stream(
        bank, router, experts, val_arrays, labels_tensor, device,
        args.support_temperature, args.support_prior, args.delay, args.tta_lr, args.adapt_steps,
        args.oracle_temperature, args.ce_weight, args.anchor_weight, offline_state,
    )
    val_static = _evaluate_raw(vs, vy, scale)
    val_dynamic = _evaluate_raw(vd, vy, scale)
    print("VALIDATION_STATIC", json.dumps(val_static))
    print("VALIDATION_TTA_DYNAMIC", json.dumps(val_dynamic))

    grid = []
    for mode in args.modes:
        for beta in args.betas:
            for margin in args.margins:
                for min_updates in args.min_updates:
                    m = _selective_replay(vd, vs, vy, scale, args.delay, beta, margin, min_updates, mode)
                    grid.append({
                        "mode": mode, "beta": beta, "margin": margin, "min_updates": min_updates,
                        "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"],
                        "dynamic_usage_rate": m["dynamic_usage_rate"],
                    })
    grid.sort(key=lambda x: (x["MAE"], x["RMSE"]))
    selected_cfg = {k: grid[0][k] for k in ["mode", "beta", "margin", "min_updates"]}
    print("VALIDATION_TOP10")
    for row in grid[:10]:
        print(json.dumps(row))
    print("SELECTED_BY_VALIDATION", json.dumps(selected_cfg))
    selected_val = _selective_replay(vd, vs, vy, scale, args.delay, **selected_cfg)

    print("collecting held-out test causal TTA stream")
    td, ts, ty, tw, test_adapt_updates = _collect_stream(
        bank, router, experts, test_arrays, labels_tensor, device,
        args.support_temperature, args.support_prior, args.delay, args.tta_lr, args.adapt_steps,
        args.oracle_temperature, args.ce_weight, args.anchor_weight, offline_state,
    )
    test_static = _evaluate_raw(ts, ty, scale)
    test_dynamic = _evaluate_raw(td, ty, scale)
    selected_test = _selective_replay(td, ts, ty, scale, args.delay, **selected_cfg)

    result = {
        "experiment": "dynamic_selective_v8",
        "protocol": "static safety anchor + delayed causal dynamic expert selection; selector chosen on validation only",
        "tta_lr_from_v7_validation": args.tta_lr,
        "delay_origins": args.delay,
        "selected_config": selected_cfg,
        "validation": {
            "static": val_static,
            "tta_dynamic": val_dynamic,
            "selective": selected_val,
            "tta_updates": val_adapt_updates,
        },
        "heldout_test": {
            "static": test_static,
            "tta_dynamic": test_dynamic,
            "selective": selected_test,
            "tta_updates": test_adapt_updates,
        },
        "validation_top10": grid[:10],
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL_RESULT")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
