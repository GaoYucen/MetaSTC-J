"""V11 static-anchored, shift-invariant dynamic residual correction.

The reproduced MetaSTC static expert is always the prediction anchor.  A small
causal network observes only locally normalized recent dynamics and the static
expert's already-observed support residual, then emits a bounded correction to
the static future forecast.  The dynamic component therefore cannot replace the
static expert wholesale as V9 did.

Training additionally penalizes temporal groups in which the dynamic correction
is worse than the static anchor, encouraging robust corrections across regimes.
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

from optimized_runner import _cluster_features, _device, _load_flow, _metrics, _set_seed
from dynamic_support_routing_v4 import _all_samples, _episodic_windows, _limited, _load_static_experts, _static_prediction


class StaticAnchoredResidual(nn.Module):
    def __init__(self, hidden=64, correction_limit=2.0, gate_bias=-2.0):
        super().__init__()
        # q shape 12 + observed support y 6 + static support residual 6 +
        # normalized static future forecast 6 + five shift-invariant summaries.
        d = 12 + 6 + 6 + 6 + 5
        self.correction_limit = float(correction_limit)
        self.encoder = nn.Sequential(
            nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
        )
        self.delta = nn.Linear(hidden // 2, 6)
        self.gate = nn.Linear(hidden // 2, 6)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, gate_bias)

    def make_features(self, sx, sy, qx, static_support, static_query):
        q = qx.squeeze(-1); s = sy.squeeze(-1)
        ss = static_support.squeeze(-1); sq = static_query.squeeze(-1)
        mean = q.mean(dim=1, keepdim=True)
        std = q.std(dim=1, unbiased=False, keepdim=True)
        # Include a small level-dependent floor only in the *scale*, not as input.
        scale = (std + 0.05 * mean.abs() + 1e-3).clamp_min(1e-3)
        qn = (q - mean) / scale
        sn = (s - mean) / scale
        support_resid = (s - ss) / scale
        static_q = (sq - mean) / scale
        trend = (q[:, -1:] - q[:, :1]) / scale
        shift = (q[:, 6:].mean(dim=1, keepdim=True) - q[:, :6].mean(dim=1, keepdim=True)) / scale
        diffstd = torch.diff(q, dim=1).std(dim=1, unbiased=False, keepdim=True) / scale
        rmean = support_resid.mean(dim=1, keepdim=True)
        rstd = support_resid.std(dim=1, unbiased=False, keepdim=True)
        feat = torch.cat([qn, sn, support_resid, static_q, trend, shift, diffstd, rmean, rstd], dim=1)
        return torch.nan_to_num(feat), scale

    def forward(self, sx, sy, qx, static_support, static_query):
        feat, scale = self.make_features(sx, sy, qx, static_support, static_query)
        h = self.encoder(feat)
        gate = torch.sigmoid(self.gate(h))
        normalized_delta = self.correction_limit * gate * torch.tanh(self.delta(h))
        correction = scale * normalized_delta
        pred = static_query.squeeze(-1) + correction
        return pred.unsqueeze(-1), gate, normalized_delta, feat


def _loader(arrays, batch_size, device, shuffle):
    sx, sy, qx, qy, roads, times = _all_samples(arrays)
    ds = TensorDataset(torch.from_numpy(sx), torch.from_numpy(sy), torch.from_numpy(qx), torch.from_numpy(qy),
                       torch.from_numpy(roads), torch.from_numpy(times))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=device.type == 'cuda')


def _group_robust_penalty(dynamic_sample_mse, static_sample_mse, time_id, group_size):
    groups = torch.div(time_id, group_size, rounding_mode='floor')
    penalties = []
    for g in torch.unique(groups):
        idx = groups == g
        if idx.any():
            delta = dynamic_sample_mse[idx].mean() - static_sample_mse[idx].mean()
            penalties.append(torch.relu(delta))
    if not penalties:
        return dynamic_sample_mse.new_zeros(())
    return torch.stack(penalties).mean()


def _val_loss(model, experts, loader, labels_tensor, device, max_batches):
    model.eval(); losses = []
    with torch.no_grad():
        for sx, sy, qx, qy, road, _ in _limited(loader, max_batches):
            sx, sy, qx, qy, road = sx.to(device), sy.to(device), qx.to(device), qy.to(device), road.to(device)
            task = labels_tensor[road]
            ss = _static_prediction(experts, sx, task); sq = _static_prediction(experts, qx, task)
            pred, _, _, _ = model(sx, sy, qx, ss, sq)
            losses.append(float(nn.functional.mse_loss(pred, qy).cpu()))
    return float(np.mean(losses)) if losses else float('inf')


def _evaluate(model, experts, loader, labels_tensor, scale_data, device, max_batches):
    model.eval(); target=[]; static=[]; dynamic=[]; gates=[]; ndeltas=[]; times=[]; roads=[]
    with torch.no_grad():
        for sx, sy, qx, qy, road, time_id in _limited(loader, max_batches):
            sx, sy, qx, qy, road = sx.to(device), sy.to(device), qx.to(device), qy.to(device), road.to(device)
            task = labels_tensor[road]
            ss = _static_prediction(experts, sx, task); sq = _static_prediction(experts, qx, task)
            pred, gate, ndelta, _ = model(sx, sy, qx, ss, sq)
            target.append((qy*scale_data).cpu().numpy()); static.append((sq*scale_data).cpu().numpy()); dynamic.append((pred*scale_data).cpu().numpy())
            gates.append(gate.cpu().numpy()); ndeltas.append(ndelta.cpu().numpy()); times.append(time_id.numpy()); roads.append(road.cpu().numpy())
    y=np.concatenate(target).reshape(-1); s=np.concatenate(static).reshape(-1); d=np.concatenate(dynamic).reshape(-1)
    g=np.concatenate(gates); nd=np.concatenate(ndeltas); t=np.concatenate(times); r=np.concatenate(roads)
    # Time-origin diagnostics.
    rows=[]
    for ti in np.unique(t):
        idx=t==ti
        # six horizons per sample
        ys=np.concatenate(target, axis=0)[idx].reshape(-1); ss=np.concatenate(static,axis=0)[idx].reshape(-1); dd=np.concatenate(dynamic,axis=0)[idx].reshape(-1)
        rows.append({'time_id':int(ti),'static_MAE':float(np.abs(ss-ys).mean()),'dynamic_MAE':float(np.abs(dd-ys).mean()),'mean_gate':float(g[idx].mean())})
    result={
      'reproduced_static_hard':_metrics(s,y),
      'static_anchored_dynamic':_metrics(d,y),
      'mean_gate':float(g.mean()), 'p90_gate':float(np.quantile(g,.9)),
      'mean_abs_normalized_correction':float(np.abs(nd).mean()),
      'dynamic_vs_static_relative_mae_pct':100.0*(_metrics(d,y)['MAE']/_metrics(s,y)['MAE']-1.0),
      'per_time':rows,
    }
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='beijing',choices=['beijing','shanghai','largest'])
    ap.add_argument('--clusters',type=int,default=5)
    ap.add_argument('--checkpoint-dir',default='param/4090_tuned/lstm/beijing')
    ap.add_argument('--output-dir',default='param/journal/dynamic_static_anchor_v11/beijing_lstm')
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--batch-size',type=int,default=8192)
    ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--train-max-batches',type=int,default=50)
    ap.add_argument('--val-max-batches',type=int,default=0); ap.add_argument('--test-max-batches',type=int,default=0)
    ap.add_argument('--hidden',type=int,default=64); ap.add_argument('--correction-limit',type=float,default=2.0)
    ap.add_argument('--gate-bias',type=float,default=-2.0); ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--weight-decay',type=float,default=1e-4)
    ap.add_argument('--safety-weight',type=float,default=0.5); ap.add_argument('--robust-weight',type=float,default=1.0)
    ap.add_argument('--gate-weight',type=float,default=0.005); ap.add_argument('--correction-weight',type=float,default=0.005)
    ap.add_argument('--time-group-size',type=int,default=20); ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    _set_seed(args.seed); device=_device(args.device)
    flow,ids,scale_data=_load_flow(args.dataset)
    labels=_cluster_features(args.dataset,'lstm',flow,ids,args.clusters,args.seed)
    fit,val,test=_episodic_windows(flow)
    tr=_loader(fit,args.batch_size,device,True); vl=_loader(val,args.batch_size,device,False); tl=_loader(test,args.batch_size,device,False)
    ckpt=Path(args.checkpoint_dir); experts=_load_static_experts(ckpt,args.clusters,device)
    labels_tensor=torch.from_numpy(labels).long().to(device)
    model=StaticAnchoredResidual(args.hidden,args.correction_limit,args.gate_bias).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    best=float('inf'); best_state=copy.deepcopy(model.state_dict()); history=[]
    for epoch in range(args.epochs):
        model.train(); rows=[]
        for sx,sy,qx,qy,road,time_id in _limited(tr,args.train_max_batches):
            sx,sy,qx,qy,road,time_id=sx.to(device),sy.to(device),qx.to(device),qy.to(device),road.to(device),time_id.to(device)
            task=labels_tensor[road]
            with torch.no_grad():
                ss=_static_prediction(experts,sx,task); sq=_static_prediction(experts,qx,task)
            pred,gate,ndelta,_=model(sx,sy,qx,ss,sq)
            dyn_sample=((pred-qy)**2).mean(dim=(1,2)); sta_sample=((sq-qy)**2).mean(dim=(1,2))
            query=dyn_sample.mean()
            safety=torch.relu(dyn_sample-sta_sample).mean()
            robust=_group_robust_penalty(dyn_sample,sta_sample,time_id,args.time_group_size)
            gate_reg=gate.mean(); corr_reg=(ndelta**2).mean()
            loss=query+args.safety_weight*safety+args.robust_weight*robust+args.gate_weight*gate_reg+args.correction_weight*corr_reg
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            rows.append([float(loss.detach()),float(query.detach()),float(safety.detach()),float(robust.detach()),float(gate_reg.detach()),float(corr_reg.detach())])
        val_loss=_val_loss(model,experts,vl,labels_tensor,device,args.val_max_batches)
        a=np.asarray(rows); row={'epoch':epoch+1,'loss':float(a[:,0].mean()),'query':float(a[:,1].mean()),'safety':float(a[:,2].mean()),'robust':float(a[:,3].mean()),'gate':float(a[:,4].mean()),'corr':float(a[:,5].mean()),'val_mse':val_loss}
        history.append(row); print(json.dumps(row))
        if val_loss<best:
            best=val_loss; best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); torch.save(best_state,out/'model_best.pt')
    validation=_evaluate(model,experts,vl,labels_tensor,scale_data,device,args.val_max_batches)
    future=_evaluate(model,experts,tl,labels_tensor,scale_data,device,args.test_max_batches)
    result={'experiment':'dynamic_static_anchor_v11','protocol':'static MetaSTC anchor + shift-invariant bounded dynamic residual + temporal-group safety training; best epoch validation only','config':vars(args),'best_val_mse':best,'history':history,'validation':validation,'exploratory_future_segment':future}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('FINAL_RESULT'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
