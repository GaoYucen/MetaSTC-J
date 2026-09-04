"""Mechanism-validation harness for MetaSTC-J V14."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from dynamic_residual_meta_adapter_v14 import DynamicResidualMetaAdapter, _device, _load_labels, _cluster_tensors, _loaders, _load_experts
from optimized_runner import _load_flow, _metrics, _set_seed, _windows

def _count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

class ConstantGateAdapter(DynamicResidualMetaAdapter):
    def __init__(self,clusters,correction_limit=.8,gate_bias=-1.5):
        super().__init__(clusters,correction_limit=correction_limit,gate_bias=gate_bias)
        self.global_gate_logit=nn.Parameter(torch.tensor(float(gate_bias)))
        for p in self.gate_head.parameters(): p.requires_grad_(False)
    def forward(self,x,base,task):
        history=x.squeeze(-1); base2=base.squeeze(-1)
        state,shared_input,local_scale=self._context(history,base2)
        z=self.state_encoder(state); cond=torch.cat([z,self.cluster_embedding(task)],1)
        shared=self.shared_residual(shared_input); gamma,beta=self.modulator(cond).chunk(2,1)
        hidden=torch.nn.functional.gelu(shared*(1+.2*torch.tanh(gamma))+.2*beta)
        delta=self.correction_limit*torch.tanh(self.delta_head(hidden)); gate=torch.sigmoid(self.global_gate_logit).expand(x.shape[0],1)
        corr=local_scale*gate*delta; return (base2+corr).unsqueeze(-1),gate,corr

class SharedNoTaskAdapter(DynamicResidualMetaAdapter):
    """V14-capacity control with no task/cluster identity in conditioning."""
    def forward(self,x,base,task):
        history=x.squeeze(-1); base2=base.squeeze(-1)
        state,shared_input,local_scale=self._context(history,base2)
        z=self.state_encoder(state)
        # Same embedding matrix/parameter count as V14, but every sample receives
        # the same pooled embedding, so cluster identity cannot affect the output.
        shared_emb=self.cluster_embedding.weight.mean(dim=0,keepdim=True).expand(x.shape[0],-1)
        cond=torch.cat([z,shared_emb],1)
        shared=self.shared_residual(shared_input); gamma,beta=self.modulator(cond).chunk(2,1)
        hidden=torch.nn.functional.gelu(shared*(1+.2*torch.tanh(gamma))+.2*beta)
        delta=self.correction_limit*torch.tanh(self.delta_head(hidden)); gate=torch.sigmoid(self.gate_head(cond))
        corr=local_scale*gate*delta; return (base2+corr).unsqueeze(-1),gate,corr

class SharedContextAdapter(nn.Module):
    def __init__(self,clusters,correction_limit=.8,gate_bias=-1.5):
        super().__init__(); self.correction_limit=correction_limit
        self.net=nn.Sequential(nn.Linear(26,96),nn.GELU(),nn.LayerNorm(96),nn.Linear(96,64),nn.GELU())
        self.delta_head=nn.Linear(64,6); self.gate_head=nn.Linear(26,1)
        nn.init.zeros_(self.delta_head.weight); nn.init.zeros_(self.delta_head.bias); nn.init.zeros_(self.gate_head.weight); nn.init.constant_(self.gate_head.bias,gate_bias)
    def forward(self,x,base,task):
        history=x.squeeze(-1); base2=base.squeeze(-1); state,_,local_scale=DynamicResidualMetaAdapter._context(history,base2)
        delta=self.correction_limit*torch.tanh(self.delta_head(self.net(state))); gate=torch.sigmoid(self.gate_head(state)); corr=local_scale*gate*delta
        return (base2+corr).unsqueeze(-1),gate,corr

class LinearClusterAdapter(nn.Module):
    def __init__(self,clusters,correction_limit=.8,gate_bias=-1.5):
        super().__init__(); self.correction_limit=correction_limit
        self.weight=nn.Parameter(torch.zeros(clusters,26,6)); self.bias=nn.Parameter(torch.zeros(clusters,6)); self.gate_logit=nn.Parameter(torch.full((clusters,),float(gate_bias)))
    def forward(self,x,base,task):
        history=x.squeeze(-1); base2=base.squeeze(-1); state,_,local_scale=DynamicResidualMetaAdapter._context(history,base2)
        raw=torch.bmm(state.unsqueeze(1),self.weight[task]).squeeze(1)+self.bias[task]
        delta=self.correction_limit*torch.tanh(raw); gate=torch.sigmoid(self.gate_logit[task]).unsqueeze(1); corr=local_scale*gate*delta
        return (base2+corr).unsqueeze(-1),gate,corr

def build(v,k,lim,bias):
    if v=='v14': return DynamicResidualMetaAdapter(k,correction_limit=lim,gate_bias=bias)
    if v=='constant_gate': return ConstantGateAdapter(k,lim,bias)
    if v=='shared_no_task': return SharedNoTaskAdapter(k,correction_limit=lim,gate_bias=bias)
    if v=='shared': return SharedContextAdapter(k,lim,bias)
    if v=='linear_cluster': return LinearClusterAdapter(k,lim,bias)
    raise ValueError(v)

def evaluate(adapter,experts,loaders,device,scale,max_batches=0):
    adapter.eval(); ss=[]; dd=[]; yy=[]; gg=[]
    with torch.inference_mode():
        for c,loader in loaders.items():
            for bi,(x,y) in enumerate(loader):
                if max_batches and bi>=max_batches: break
                x=x.to(device); y=y.to(device); base=experts[c](x); task=torch.full((x.shape[0],),c,device=device,dtype=torch.long); pred,gate,_=adapter(x,base,task)
                ss.append((base*scale).cpu().numpy()); dd.append((pred*scale).cpu().numpy()); yy.append((y*scale).cpu().numpy()); gg.append(gate.cpu().numpy().reshape(-1))
    s=np.concatenate(ss).reshape(-1); d=np.concatenate(dd).reshape(-1); y=np.concatenate(yy).reshape(-1); g=np.concatenate(gg)
    sm=_metrics(s,y); dm=_metrics(d,y)
    return {'static':sm,'dynamic':dm,'relative_mae_vs_static_pct':100*(dm['MAE']/sm['MAE']-1),'relative_mse_vs_static_pct':100*(dm['MSE']/sm['MSE']-1),'mean_gate':float(g.mean()),'std_gate':float(g.std())}

def _rank(x):
    x=np.asarray(x,float); o=np.argsort(x,kind='mergesort'); r=np.empty(len(x),float); r[o]=np.arange(len(x)); vals=x[o]; i=0
    while i<len(x):
        j=i+1
        while j<len(x) and vals[j]==vals[i]: j+=1
        if j-i>1: r[o[i:j]]=(i+j-1)/2
        i=j
    return r

def corr(a,b,rank=False):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if rank: a=_rank(a); b=_rank(b)
    return None if len(a)<2 or np.std(a)<1e-12 or np.std(b)<1e-12 else float(np.corrcoef(a,b)[0,1])

def diagnostics(adapter,experts,labels,fit_x,test_x,test_y,device,scale,k):
    train_mean=fit_x.mean((0,1)); train_std=fit_x.std((0,1))+1e-4; rows=[]; adapter.eval()
    with torch.inference_mode():
        for t in range(test_x.shape[0]):
            ss=[]; dd=[]; yy=[]; gg=[]
            for c in range(k):
                roads=np.flatnonzero(labels==c); x=torch.from_numpy(test_x[t][:,roads].T[:,:,None]).float().to(device); y=torch.from_numpy(test_y[t][:,roads].T[:,:,None]).float().to(device)
                base=experts[c](x); task=torch.full((x.shape[0],),c,device=device,dtype=torch.long); pred,gate,_=adapter(x,base,task)
                ss.append((base*scale).cpu().numpy().reshape(-1)); dd.append((pred*scale).cpu().numpy().reshape(-1)); yy.append((y*scale).cpu().numpy().reshape(-1)); gg.append(gate.cpu().numpy().reshape(-1))
            s=np.concatenate(ss); d=np.concatenate(dd); y=np.concatenate(yy); g=np.concatenate(gg); hist=test_x[t]; hm=hist.mean(0); hs=hist.std(0)+1e-4; trend=hist[-1]-hist[0]
            level=float(np.mean(np.abs((hm-train_mean)/train_std))); vol=float(np.mean(np.abs(np.log(hs/train_std)))); tr=float(np.mean(np.abs(trend/train_std))); shift=level+.5*vol+.25*tr
            smae=float(np.mean(np.abs(s-y))); dmae=float(np.mean(np.abs(d-y)))
            rows.append({'window':t,'static_mae':smae,'dynamic_mae':dmae,'gain_mae':smae-dmae,'relative_gain_pct':100*(1-dmae/smae),'mean_gate':float(g.mean()),'level_shift':level,'volatility_shift':vol,'trend_shift':tr,'shift_proxy':shift})
    gate=[r['mean_gate'] for r in rows]; err=[r['static_mae'] for r in rows]; gain=[r['gain_mae'] for r in rows]; shift=[r['shift_proxy'] for r in rows]
    return {'per_window':rows,'correlations':{'pearson_gate_static_error':corr(gate,err),'spearman_gate_static_error':corr(gate,err,True),'pearson_gate_gain':corr(gate,gain),'spearman_gate_gain':corr(gate,gain,True),'pearson_shift_gate':corr(shift,gate),'spearman_shift_gate':corr(shift,gate,True),'pearson_shift_gain':corr(shift,gain),'spearman_shift_gain':corr(shift,gain,True)}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['beijing','shanghai','largest'],required=True); ap.add_argument('--checkpoint-dir',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--variant',choices=['v14','constant_gate','shared_no_task','shared','linear_cluster'],required=True); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--clusters',type=int,default=5); ap.add_argument('--batch-size',type=int,default=16384); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--mae-weight',type=float,default=.02); ap.add_argument('--safety-weight',type=float,default=.02); ap.add_argument('--correction-weight',type=float,default=1e-4); ap.add_argument('--correction-limit',type=float,default=.8); ap.add_argument('--gate-bias',type=float,default=-1.5); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--max-batches',type=int,default=0); ap.add_argument('--eval-max-batches',type=int,default=0); ap.add_argument('--skip-diagnostics',action='store_true')
    a=ap.parse_args(); _set_seed(a.seed); device=_device(a.device); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); ck=Path(a.checkpoint_dir); flow,_,scale=_load_flow(a.dataset); labels=_load_labels(ck,flow.shape[0],a.clusters); fit_x,fit_y,val_x,val_y,test_x,test_y=_windows(flow,12,6)
    tr=_loaders(_cluster_tensors(fit_x,fit_y,labels,a.clusters),a.batch_size,device,True); va=_loaders(_cluster_tensors(val_x,val_y,labels,a.clusters),a.batch_size,device,False); te=_loaders(_cluster_tensors(test_x,test_y,labels,a.clusters),a.batch_size,device,False); experts=_load_experts(ck,a.clusters,device); adapter=build(a.variant,a.clusters,a.correction_limit,a.gate_bias).to(device); opt=torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad],lr=a.lr,weight_decay=a.weight_decay); initial=evaluate(adapter,experts,te,device,scale,a.eval_max_batches); best=1e99; best_state=copy.deepcopy(adapter.state_dict()); hist=[]
    for ep in range(a.epochs):
        adapter.train(); total=0.; seen=0
        for c,loader in tr.items():
            for bi,(x,y) in enumerate(loader):
                if a.max_batches and bi>=a.max_batches: break
                x=x.to(device); y=y.to(device)
                with torch.no_grad(): base=experts[c](x)
                task=torch.full((x.shape[0],),c,device=device,dtype=torch.long); pred,gate,cr=adapter(x,base,task); mse=torch.mean((pred-y)**2); mae=torch.mean(torch.abs(pred-y)); dyn=torch.mean((pred-y)**2,(1,2)); sta=torch.mean((base-y)**2,(1,2)); loss=mse+a.mae_weight*mae+a.safety_weight*torch.relu(dyn-sta).mean()+a.correction_weight*torch.mean(cr**2); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(),5); opt.step(); total+=float(loss.detach().cpu())*x.shape[0]; seen+=x.shape[0]
        vm=evaluate(adapter,experts,va,device,scale,a.eval_max_batches); score=vm['dynamic']['MSE']+.1*vm['dynamic']['MAE']; hist.append({'epoch':ep+1,'loss':total/max(seen,1),'mae_pct':vm['relative_mae_vs_static_pct'],'mse_pct':vm['relative_mse_vs_static_pct'],'gate':vm['mean_gate']}); print('EPOCH',a.variant,a.dataset,a.seed,json.dumps(hist[-1]))
        if score<best: best=score; best_state=copy.deepcopy(adapter.state_dict()); torch.save(best_state,out/'adapter_best.pt')
    adapter.load_state_dict(best_state); validation=evaluate(adapter,experts,va,device,scale,a.eval_max_batches); test=evaluate(adapter,experts,te,device,scale,a.eval_max_batches); diag={'per_window':[],'correlations':{}} if a.skip_diagnostics else diagnostics(adapter,experts,labels,fit_x,test_x,test_y,device,scale,a.clusters); result={'experiment':'dynamic_residual_mechanism_v14','variant':a.variant,'dataset':a.dataset,'config':vars(a),'trainable_params':_count_params(adapter),'validation':validation,'test':test,'diagnostics':diag,'history':hist}; (out/'metrics.json').write_text(json.dumps(result,indent=2)); print('RESULT',json.dumps({'variant':a.variant,'dataset':a.dataset,'seed':a.seed,'params':result['trainable_params'],'static_mae':test['static']['MAE'],'test_mae':test['dynamic']['MAE'],'rel_mae_pct':test['relative_mae_vs_static_pct'],'gate':test['mean_gate'],'corr':diag['correlations']},indent=2))
if __name__=='__main__': main()
