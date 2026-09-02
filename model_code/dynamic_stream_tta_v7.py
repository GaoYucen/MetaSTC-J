"""V7 causal streaming test-time adaptation for dynamic task inference.

The temporal split reveals a shift in the mapping from support/context evidence
to the best latent task. V7 adapts only the small V6 router online; the shared
LSTM and task adapter bank remain frozen.

Causality is explicit. An episode predicts six future steps. Its query target is
not used until six forecast origins later, when all six values have become past
observations. Before predicting episode i, the router may therefore update from
episode i-6, never from the current/future target.

The adaptation learning rate is selected on the validation stream. The selected
configuration is then reset to the offline router and evaluated once on the
held-out test stream.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from optimized_runner import LSTMModel, _cluster_features, _device, _load_flow, _metrics, _set_seed
from dynamic_support_routing_v4 import TaskAdapterBank, _episodic_windows, _load_static_experts, _static_prediction
from dynamic_support_router_v6 import SupportAwareRouter, _soft_oracle


def _episode_tensors(arrays, index, device):
    sx, sy, qx, qy = arrays
    # input arrays are [window,time,road]; models consume [road,time,1]
    return tuple(
        torch.from_numpy(a[index].T[:, :, None].astype(np.float32)).to(device)
        for a in (sx, sy, qx, qy)
    )


def _forward_episode(bank, router, arrays, index, labels_tensor, device, support_temp, support_prior):
    sx, sy, qx, qy = _episode_tensors(arrays, index, device)
    static_task = labels_tensor
    with torch.no_grad():
        sp, _, _ = bank(sx)
        qp, _, _ = bank(qx)
        se = ((sp - sy[:, None]) ** 2).mean(dim=(2, 3))
    p, support_p = router(se, qx, static_task, support_temp, support_prior)
    mixed = (qp * p[:, :, None, None]).sum(dim=1)
    return sx, sy, qx, qy, se, qp, p, support_p, mixed


def _adapt_once(bank, router, optimizer, arrays, index, labels_tensor, device,
                support_temp, support_prior, oracle_temp, ce_weight, anchor_weight, offline_state):
    router.train()
    sx, sy, qx, qy = _episode_tensors(arrays, index, device)
    with torch.no_grad():
        sp, _, _ = bank(sx)
        qp, _, _ = bank(qx)
        se = ((sp - sy[:, None]) ** 2).mean(dim=(2, 3))
        qe = ((qp - qy[:, None]) ** 2).mean(dim=(2, 3))
        oracle_q = _soft_oracle(qe, oracle_temp)
    p, _ = router(se, qx, labels_tensor, support_temp, support_prior)
    mixed = (qp * p[:, :, None, None]).sum(dim=1)
    mixture_loss = nn.functional.mse_loss(mixed, qy)
    ce = -(oracle_q * torch.log(p.clamp_min(1e-8))).sum(dim=1).mean()
    anchor = torch.zeros((), device=device)
    if anchor_weight > 0:
        for name, param in router.named_parameters():
            anchor = anchor + (param - offline_state[name]).pow(2).mean()
    loss = mixture_loss + ce_weight * ce + anchor_weight * anchor
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(router.parameters(), 5.0)
    optimizer.step()
    return float(loss.detach().cpu()), float(mixture_loss.detach().cpu()), float(ce.detach().cpu())


def _stream_mae(bank, router, arrays, labels_tensor, scale, device, support_temp, support_prior,
                delay, lr, adapt_steps, oracle_temp, ce_weight, anchor_weight, offline_state,
                collect_details=False, experts=None):
    router.load_state_dict({k: v.detach().clone() for k, v in offline_state.items()})
    router.eval()
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=0.0) if lr > 0 else None
    predictions, targets = [], []
    support_predictions, static_predictions = [], []
    weights_all = []
    adaptation_log = []
    count = arrays[0].shape[0]
    for i in range(count):
        # Episode i-delay has fully entered the observed past by the current origin.
        if optimizer is not None and i >= delay:
            for _ in range(adapt_steps):
                adaptation_log.append(_adapt_once(
                    bank, router, optimizer, arrays, i - delay, labels_tensor, device,
                    support_temp, support_prior, oracle_temp, ce_weight, anchor_weight,
                    offline_state,
                ))
        router.eval()
        with torch.no_grad():
            _, _, qx, qy, se, qp, p, support_p, mixed = _forward_episode(
                bank, router, arrays, i, labels_tensor, device, support_temp, support_prior
            )
            predictions.append((mixed * scale).cpu().numpy())
            targets.append((qy * scale).cpu().numpy())
            weights_all.append(p.cpu().numpy())
            if collect_details:
                support_mixed = (qp * support_p[:, :, None, None]).sum(dim=1)
                support_predictions.append((support_mixed * scale).cpu().numpy())
                if experts is not None:
                    static_predictions.append((_static_prediction(experts, qx, labels_tensor) * scale).cpu().numpy())

    pred = np.concatenate(predictions).reshape(-1)
    target = np.concatenate(targets).reshape(-1)
    result = _metrics(pred, target)
    result["adaptation_updates"] = len(adaptation_log)
    if adaptation_log:
        a = np.asarray(adaptation_log)
        result["mean_adapt_total_loss"] = float(a[:,0].mean())
        result["mean_adapt_mixture_loss"] = float(a[:,1].mean())
        result["mean_adapt_oracle_ce"] = float(a[:,2].mean())
    p = np.concatenate(weights_all)
    result["routing_mean_entropy"] = float((-(p * np.log(np.clip(p,1e-12,1))).sum(axis=1)).mean())
    result["routing_mean_max_weight"] = float(p.max(axis=1).mean())
    result["routing_vs_static_disagreement"] = float((p.argmax(axis=1) != np.tile(labels_tensor.cpu().numpy(), count)).mean())
    if collect_details:
        support_pred = np.concatenate(support_predictions).reshape(-1)
        result["offline_support_router_metrics"] = _metrics(support_pred, target)
        if static_predictions:
            static_pred = np.concatenate(static_predictions).reshape(-1)
            result["reproduced_static_hard_metrics"] = _metrics(static_pred, target)
    return result


def main():
    ap = argparse.ArgumentParser(description="Causal streaming router TTA V7")
    ap.add_argument("--dataset", default="beijing", choices=["beijing", "shanghai", "largest"])
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--checkpoint-dir", default="param/4090_tuned/lstm/beijing")
    ap.add_argument("--bank-checkpoint", default="param/journal/dynamic_support_v4/beijing_lstm_substantial/model_best.pt")
    ap.add_argument("--router-checkpoint", default="param/journal/dynamic_support_router_v6/beijing_lstm_smoke/router_best.pt")
    ap.add_argument("--output-dir", default="param/journal/dynamic_stream_tta_v7/beijing_lstm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--delay", type=int, default=6)
    ap.add_argument("--adapt-steps", type=int, default=1)
    ap.add_argument("--support-temperature", type=float, default=1.0)
    ap.add_argument("--support-prior", type=float, default=0.25)
    ap.add_argument("--oracle-temperature", type=float, default=0.5)
    ap.add_argument("--ce-weight", type=float, default=0.02)
    ap.add_argument("--anchor-weight", type=float, default=0.0001)
    ap.add_argument("--lr-grid", type=float, nargs="+", default=[0.0,1e-5,3e-5,1e-4,3e-4,1e-3])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed); device = _device(args.device); k = args.clusters
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
    for p in bank.parameters(): p.requires_grad_(False)

    router = SupportAwareRouter(k).to(device)
    router.load_state_dict(torch.load(args.router_checkpoint, map_location=device, weights_only=True))
    offline_state = {name: p.detach().clone() for name, p in router.state_dict().items()}

    val_rows = []
    for lr in args.lr_grid:
        metrics = _stream_mae(
            bank, router, val_arrays, labels_tensor, scale, device,
            args.support_temperature, args.support_prior, args.delay, lr, args.adapt_steps,
            args.oracle_temperature, args.ce_weight, args.anchor_weight, offline_state,
            collect_details=False,
        )
        row = {"lr": lr, "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "R2": metrics["R2"],
               "updates": metrics["adaptation_updates"]}
        val_rows.append(row); print("VAL_CANDIDATE", json.dumps(row))
    val_rows.sort(key=lambda x: (x["MAE"], x["RMSE"]))
    selected_lr = val_rows[0]["lr"]
    print("SELECTED_BY_VALIDATION", selected_lr)

    experts = _load_static_experts(ckpt, k, device)
    selected_val = _stream_mae(
        bank, router, val_arrays, labels_tensor, scale, device,
        args.support_temperature, args.support_prior, args.delay, selected_lr, args.adapt_steps,
        args.oracle_temperature, args.ce_weight, args.anchor_weight, offline_state,
        collect_details=True, experts=experts,
    )
    # Reset inside _stream_mae; held-out test receives no validation-adapted parameters.
    selected_test = _stream_mae(
        bank, router, test_arrays, labels_tensor, scale, device,
        args.support_temperature, args.support_prior, args.delay, selected_lr, args.adapt_steps,
        args.oracle_temperature, args.ce_weight, args.anchor_weight, offline_state,
        collect_details=True, experts=experts,
    )
    result = {
        "experiment":"dynamic_stream_tta_v7",
        "protocol":"causal delayed prequential TTA; LR selected on validation, router reset before test",
        "delay_origins":args.delay,
        "selected_lr":selected_lr,
        "validation_grid":val_rows,
        "validation":selected_val,
        "heldout_test":selected_test,
    }
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print("FINAL_RESULT")
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
