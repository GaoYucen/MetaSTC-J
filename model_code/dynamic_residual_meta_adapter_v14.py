"""V14: paper-protocol Dynamic Residual Meta-Adapter for MetaSTC-J.

Frozen cluster-specific MetaSTC experts provide the anchor forecast. A small
context-conditioned residual adapter learns only a bounded correction and is
initialized to exactly recover the static anchor. No future labels are used at
inference time: inputs are the observed 12-step history, frozen MetaSTC 6-step
prediction, and static task id.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from optimized_runner import LSTMModel, _load_flow, _metrics, _set_seed, _windows


def _device(name: str) -> torch.device:
    if name == "cuda":
        name = "cuda:0"
    device = torch.device(name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    return device


def _load_labels(checkpoint_dir: Path, n_roads: int, k: int) -> np.ndarray:
    path = checkpoint_dir / "cluster_labels.txt"
    labels = np.asarray(np.loadtxt(path, dtype=np.int64)).reshape(-1)
    if labels.shape[0] != n_roads:
        raise ValueError(f"label count {labels.shape[0]} != roads {n_roads}")
    if labels.min() < 0 or labels.max() >= k:
        raise ValueError("invalid cluster labels")
    return labels


def _cluster_tensors(x: np.ndarray, y: np.ndarray, labels: np.ndarray, k: int) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    out = {}
    for c in range(k):
        roads = np.flatnonzero(labels == c)
        if roads.size == 0:
            raise ValueError(f"empty cluster {c}")
        cx = x[:, :, roads].transpose(2, 0, 1).reshape(-1, 12, 1)
        cy = y[:, :, roads].transpose(2, 0, 1).reshape(-1, 6, 1)
        out[c] = (torch.from_numpy(cx), torch.from_numpy(cy))
    return out


def _loaders(data, batch_size: int, device: torch.device, shuffle: bool):
    return {
        c: DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=device.type == "cuda", drop_last=False)
        for c, (x, y) in data.items()
    }


def _load_experts(checkpoint_dir: Path, k: int, device: torch.device):
    experts = []
    for c in range(k):
        model = LSTMModel(12, 6).to(device)
        model.load_state_dict(torch.load(checkpoint_dir / f"cluster_{c}.pt", map_location=device, weights_only=True))
        model.eval()
        model.lstm.flatten_parameters()
        for p in model.parameters():
            p.requires_grad_(False)
        experts.append(model)
    return experts


class DynamicResidualMetaAdapter(nn.Module):
    """Static task prior + dynamic state-conditioned residual modulation."""

    def __init__(self, clusters: int, cluster_dim: int = 8, latent_dim: int = 32,
                 hidden_dim: int = 64, correction_limit: float = 0.8, gate_bias: float = -1.5):
        super().__init__()
        self.correction_limit = correction_limit
        self.cluster_embedding = nn.Embedding(clusters, cluster_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(26, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim), nn.GELU(),
        )
        self.shared_residual = nn.Sequential(nn.Linear(18, hidden_dim), nn.GELU())
        cond_dim = latent_dim + cluster_dim
        self.modulator = nn.Linear(cond_dim, 2 * hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, 6)
        self.gate_head = nn.Linear(cond_dim, 1)
        nn.init.zeros_(self.modulator.weight); nn.init.zeros_(self.modulator.bias)
        nn.init.zeros_(self.delta_head.weight); nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight); nn.init.constant_(self.gate_head.bias, gate_bias)

    @staticmethod
    def _context(history: torch.Tensor, base: torch.Tensor):
        mean = history.mean(dim=1, keepdim=True)
        centered = history - mean
        std = torch.sqrt(torch.mean(centered * centered, dim=1, keepdim=True) + 1e-6)
        local_scale = std + 0.05 * mean.abs() + 1e-3
        hist_n = centered / local_scale
        base_n = (base - mean) / local_scale
        diff = history[:, 1:] - history[:, :-1]
        diff_mean_raw = diff.mean(dim=1, keepdim=True)
        diff_mean = diff_mean_raw / local_scale
        diff_std = torch.sqrt(torch.mean((diff - diff_mean_raw) ** 2, dim=1, keepdim=True) + 1e-6) / local_scale
        stats = torch.cat([
            hist_n[:, -1:],
            (history[:, -1:] - history[:, :1]) / local_scale,
            diff_mean,
            diff_std,
            (base[:, :1] - history[:, -1:]) / local_scale,
            (base[:, -1:] - history[:, -1:]) / local_scale,
            mean,
            local_scale,
        ], dim=1)
        state = torch.cat([hist_n, base_n, stats], dim=1)
        shared = torch.cat([hist_n, base_n], dim=1)
        return state, shared, local_scale

    def forward(self, x: torch.Tensor, base: torch.Tensor, task: torch.Tensor):
        history = x.squeeze(-1); base2 = base.squeeze(-1)
        state, shared_input, local_scale = self._context(history, base2)
        z = self.state_encoder(state)
        cond = torch.cat([z, self.cluster_embedding(task)], dim=1)
        shared = self.shared_residual(shared_input)
        gamma, beta = self.modulator(cond).chunk(2, dim=1)
        hidden = torch.nn.functional.gelu(shared * (1.0 + 0.2 * torch.tanh(gamma)) + 0.2 * beta)
        delta_norm = self.correction_limit * torch.tanh(self.delta_head(hidden))
        gate = torch.sigmoid(self.gate_head(cond))
        correction = local_scale * gate * delta_norm
        return (base2 + correction).unsqueeze(-1), gate, correction


def _eval(adapter, experts, loaders, device: torch.device, scale: float, max_batches: int):
    adapter.eval(); static_chunks=[]; dynamic_chunks=[]; target_chunks=[]
    gate_sum=0.0; corr_sum=0.0; count=0
    with torch.inference_mode():
        for c, loader in loaders.items():
            expert = experts[c]
            for bi, (x, y) in enumerate(loader):
                if max_batches and bi >= max_batches: break
                x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
                base=expert(x); task=torch.full((x.shape[0],), c, device=device, dtype=torch.long)
                pred, gate, correction=adapter(x, base, task)
                static_chunks.append((base*scale).cpu().numpy()); dynamic_chunks.append((pred*scale).cpu().numpy()); target_chunks.append((y*scale).cpu().numpy())
                gate_sum += float(gate.sum().cpu()); corr_sum += float(correction.abs().sum().cpu()); count += int(gate.numel())
    target=np.concatenate(target_chunks).reshape(-1); static=np.concatenate(static_chunks).reshape(-1); dynamic=np.concatenate(dynamic_chunks).reshape(-1)
    sm=_metrics(static,target); dm=_metrics(dynamic,target)
    return {"static":sm,"dynamic":dm,
            "relative_mae_vs_static_pct":100.0*(dm["MAE"]/sm["MAE"]-1.0),
            "relative_mse_vs_static_pct":100.0*(dm["MSE"]/sm["MSE"]-1.0),
            "mean_gate":gate_sum/max(count,1),
            "mean_abs_correction_normalized":corr_sum/max(count*6,1)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=["beijing","shanghai","largest"],default="beijing")
    ap.add_argument("--checkpoint-dir",default="param/4090_tuned/lstm/beijing")
    ap.add_argument("--output-dir",default="param/journal/dynamic_residual_meta_adapter_v14/beijing_lstm")
    ap.add_argument("--device",default="cuda:0"); ap.add_argument("--clusters",type=int,default=5)
    ap.add_argument("--batch-size",type=int,default=16384); ap.add_argument("--epochs",type=int,default=10)
    ap.add_argument("--max-batches",type=int,default=0); ap.add_argument("--eval-max-batches",type=int,default=0)
    ap.add_argument("--lr",type=float,default=1e-3); ap.add_argument("--weight-decay",type=float,default=1e-4)
    ap.add_argument("--mae-weight",type=float,default=0.02); ap.add_argument("--safety-weight",type=float,default=0.02)
    ap.add_argument("--correction-weight",type=float,default=1e-4); ap.add_argument("--correction-limit",type=float,default=0.8)
    ap.add_argument("--gate-bias",type=float,default=-1.5); ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args(); _set_seed(args.seed); device=_device(args.device)
    checkpoint_dir=Path(args.checkpoint_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    flow,_,scale=_load_flow(args.dataset); labels=_load_labels(checkpoint_dir,flow.shape[0],args.clusters)
    fit_x,fit_y,val_x,val_y,test_x,test_y=_windows(flow,12,6)
    train_loaders=_loaders(_cluster_tensors(fit_x,fit_y,labels,args.clusters),args.batch_size,device,True)
    val_loaders=_loaders(_cluster_tensors(val_x,val_y,labels,args.clusters),args.batch_size,device,False)
    test_loaders=_loaders(_cluster_tensors(test_x,test_y,labels,args.clusters),args.batch_size,device,False)
    experts=_load_experts(checkpoint_dir,args.clusters,device)
    adapter=DynamicResidualMetaAdapter(args.clusters,correction_limit=args.correction_limit,gate_bias=args.gate_bias).to(device)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    initial=_eval(adapter,experts,test_loaders,device,scale,args.eval_max_batches)
    print("INITIAL_TEST"); print(json.dumps(initial,indent=2))
    best_score=float("inf"); best_state=copy.deepcopy(adapter.state_dict()); history=[]
    for epoch in range(args.epochs):
        adapter.train(); loss_sum=0.0; seen=0
        for c,loader in train_loaders.items():
            expert=experts[c]
            for bi,(x,y) in enumerate(loader):
                if args.max_batches and bi>=args.max_batches: break
                x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
                with torch.no_grad(): base=expert(x)
                task=torch.full((x.shape[0],),c,device=device,dtype=torch.long)
                pred,gate,correction=adapter(x,base,task)
                mse=torch.mean((pred-y)**2); mae=torch.mean(torch.abs(pred-y))
                dyn_sample=torch.mean((pred-y)**2,dim=(1,2)); static_sample=torch.mean((base-y)**2,dim=(1,2))
                safety=torch.relu(dyn_sample-static_sample).mean(); corr_reg=torch.mean(correction**2)
                loss=mse+args.mae_weight*mae+args.safety_weight*safety+args.correction_weight*corr_reg
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(),5.0); optimizer.step()
                loss_sum += float(loss.detach().cpu())*x.shape[0]; seen += x.shape[0]
        vm=_eval(adapter,experts,val_loaders,device,scale,args.eval_max_batches)
        score=vm["dynamic"]["MSE"]+0.10*vm["dynamic"]["MAE"]
        row={"epoch":epoch+1,"train_loss":loss_sum/max(seen,1),"val_static_mae":vm["static"]["MAE"],"val_dynamic_mae":vm["dynamic"]["MAE"],"val_static_mse":vm["static"]["MSE"],"val_dynamic_mse":vm["dynamic"]["MSE"],"val_mae_pct":vm["relative_mae_vs_static_pct"],"val_mse_pct":vm["relative_mse_vs_static_pct"],"mean_gate":vm["mean_gate"],"mean_abs_correction_normalized":vm["mean_abs_correction_normalized"]}
        history.append(row); print("EPOCH",json.dumps(row))
        if score<best_score: best_score=score; best_state=copy.deepcopy(adapter.state_dict()); torch.save(best_state,out/"adapter_best.pt")
    adapter.load_state_dict(best_state)
    validation=_eval(adapter,experts,val_loaders,device,scale,args.eval_max_batches); test=_eval(adapter,experts,test_loaders,device,scale,args.eval_max_batches)
    paper_targets={
        "beijing":{"MAE":3.534,"MSE":27.433},
        "shanghai":{"MAE":4.524,"MSE":42.380},
        "largest":{"MAE":4.644,"MSE":45.520},
    }
    result={"experiment":"dynamic_residual_meta_adapter_v14","dataset":args.dataset,"protocol":"paper-style 80/20 temporal split, L=12, P=6; frozen 4090-tuned MetaSTC cluster experts","config":vars(args),"paper_target_l12":paper_targets[args.dataset],"reproduced_static_reference":{"MAE":initial["static"]["MAE"],"MSE":initial["static"]["MSE"]},"initial_test":initial,"validation":validation,"test":test,"history":history}
    (out/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print("V14_RESULT"); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
