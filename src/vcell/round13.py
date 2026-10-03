"""Audited target/background interactions on frozen Round12 expanded data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .background_training import fingerprint
from .data import load_prepared
from .fixed_training import response_basis
from .knowledge import load_knowledge
from .round11 import freeze, verify_files
from .round12 import (encode_background, projection, target_features, design, fit_ridge,
                      score, save_complete)
from .selection import row_weights, mse
from .specialization import reference_responses
from .train import source_fingerprint
from .utils import seed_all, atomic_torch_save, write_json, file_sha256

ARMS = {
    'panel_additive_ridge': ('panel','ridge',False,False,0.),
    'diffusion_additive_ridge': ('diffusion','ridge',False,False,0.),
    'diffusion_interaction_ridge': ('diffusion','ridge',True,False,0.),
    'undiffused_interaction_ridge': ('undiffused_random','ridge',True,False,0.),
    'diffusion_additive': ('diffusion','factor',False,False,0.),
    'diffusion_factor': ('diffusion','factor',True,False,0.),
    'diffusion_factor_id': ('diffusion','factor',True,True,0.),
    'diffusion_factor_id_dropout': ('diffusion','factor',True,True,.5),
}
for _j in (1,2,3):
    ARMS[f'shuffled_diffusion_{_j}_interaction_ridge']=(f'shuffled_diffusion_{_j}','ridge',True,False,0.)
    ARMS[f'shuffled_diffusion_{_j}_factor_id_dropout']=(f'shuffled_diffusion_{_j}','factor',True,True,.5)


def retrieval(meta,truth,prediction,method):
    records=[]
    for context,ix in meta.groupby('context',sort=True).indices.items():
        local=meta.iloc[ix].reset_index(drop=True);groups=local.groupby('perturbation',sort=True).indices
        names=list(groups);y=np.stack([truth[ix[j]].mean(0) for j in groups.values()]).astype(np.float64)
        p=np.stack([prediction[ix[j]].mean(0) for j in groups.values()]).astype(np.float64)
        y-=y.mean(1,keepdims=True);p-=p.mean(1,keepdims=True)
        y/=np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12)
        p/=np.maximum(np.linalg.norm(p,axis=1,keepdims=True),1e-12)
        sim=p@y.T;n=len(names)
        for j,target in enumerate(names):
            own=float(sim[j,j]);higher=int((sim[j]>own+1e-10).sum())
            tied=int(np.isclose(sim[j],own,rtol=0,atol=1e-10).sum())
            other=float(np.delete(sim[j],j).mean()) if n>1 else 0.
            records.append({'context':context,'target':target,'method':method,'candidates':n,
                'own_pearson':own,'other_pearson':other,'specificity_gap':own-other,
                'midrank':1+higher+(tied-1)/2,'rank_fraction':(1+higher+(tied-1)/2)/n,
                'top1_credit':float(np.clip((1-higher)/tied,0,1)),
                'top5_credit':float(np.clip((min(5,n)-higher)/tied,0,1))})
    return records


def episode_mask(n_targets,seed,step,dropout):
    """One mask per whole target per optimizer step, shared across all its rows."""
    if not 0<=dropout<=1:raise ValueError('Invalid target mask probability')
    mask=np.random.default_rng(seed+37000+step).random(n_targets)>=dropout
    mask[0]=False # unknown ID is always disabled
    return mask


class FactorizedResponse(nn.Module):
    def __init__(self,target_dim,background_dim,n_ids,rank,interaction=True,use_id=False,width=32):
        super().__init__()
        self.target=nn.Linear(target_dim,width,bias=False)
        self.context=nn.Linear(background_dim,width)
        self.target_out=nn.Linear(width,rank,bias=False)
        self.context_out=nn.Linear(width,rank,bias=False)
        self.cross_out=nn.Linear(width,rank,bias=False)
        self.bias=nn.Parameter(torch.zeros(rank))
        # All arms start at zero response and share trunk/output initialization.
        for layer in (self.target_out,self.context_out,self.cross_out):nn.init.zeros_(layer.weight)
        self.ids=nn.Embedding(n_ids,width,padding_idx=0);nn.init.zeros_(self.ids.weight)
        self.interaction=interaction;self.use_id=use_id
        if not use_id:self.ids.weight.requires_grad_(False)
        if not interaction:self.cross_out.weight.requires_grad_(False)

    def forward(self,target,background,ids,mask=None):
        h=self.target(target)
        if self.use_id:
            residual=self.ids(ids)
            if mask is not None:residual=residual*mask[ids,None]
            h=h+residual
        hp=torch.tanh(h);hc=torch.tanh(self.context(background))
        prediction=self.target_out(hp)+self.context_out(hc)+self.bias
        if self.interaction:prediction=prediction+self.cross_out(hp*hc)
        return prediction


def identity_indices(data,fit):
    seen=np.unique(data['pert_idx'][fit]);lookup=np.zeros(len(data['perturbations']),np.int64)
    lookup[seen]=np.arange(1,len(seen)+1)
    return lookup[data['pert_idx']],seen


def fit_factor(target,background,ids,coeff,fit,cal,cfg,arm,dest,stamp):
    _,_,interaction,use_id,dropout=ARMS[arm]
    seed_all(cfg['seed']);device=cfg['device']
    model=FactorizedResponse(target.shape[1],background.shape[1],int(ids.max())+1,coeff.shape[1],interaction,use_id).to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0005,weight_decay=.01)
    tx=torch.tensor(target,device=device);bx=torch.tensor(background,device=device)
    ix=torch.tensor(ids,device=device);y=torch.tensor(coeff,device=device)
    weights=np.asarray(cfg['fit_weights']);cw=torch.tensor(cfg['cal_weights'],device=device)
    first=0;history=[];best=float('inf');best_state=None;best_step=0
    if (dest/'last.pt').exists():
        saved=torch.load(dest/'last.pt',map_location='cpu',weights_only=True)
        if saved['fingerprint']!=stamp:raise ValueError('Factor checkpoint mismatch')
        model.load_state_dict(saved['state']);optimizer.load_state_dict(saved['optimizer'])
        first=saved['step'];history=saved['history'];best=saved['best'];best_state=saved['best_state'];best_step=saved['best_step']
    for step in range(first,cfg['max_steps']):
        rows=np.random.default_rng(cfg['seed']+9000+step).choice(fit,cfg['batch_size'],p=weights)
        mask=torch.tensor(episode_mask(int(ids.max())+1,cfg['seed'],step,dropout),device=device)
        optimizer.zero_grad(set_to_none=True)
        loss=(model(tx[rows],bx[rows],ix[rows],mask)-y[rows]).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite factor loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        done=step+1
        if done==1 or done%100==0:print(f'ROUND13 {arm} step={done}/{cfg["max_steps"]} loss={loss.item():.6f}',flush=True)
        if done%cfg['save_every']==0 or done==cfg['max_steps']:
            with torch.no_grad():validation=float((cw*(model(tx[cal],bx[cal],ix[cal])-y[cal]).square().mean(1)).sum())
            if validation<best:
                best=validation;best_step=done;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            history.append({'step':done,'train_loss':float(loss.item()),'calibration_coefficient_mse':validation})
            atomic_torch_save({'fingerprint':stamp,'state':model.state_dict(),'optimizer':optimizer.state_dict(),
                'step':done,'best':best,'best_step':best_step,'best_state':best_state,'history':history},dest/'last.pt')
            write_json(dest/'progress.json',{'step':done,'max_steps':cfg['max_steps'],'best_step':best_step})
            pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():pred=np.concatenate([model(tx[i:i+512],bx[i:i+512],ix[i:i+512]).cpu().numpy() for i in range(0,len(tx),512)])
    atomic_torch_save({'state':best_state,'target_dim':target.shape[1],'background_dim':background.shape[1],
        'n_ids':int(ids.max())+1,'rank':coeff.shape[1],'interaction':interaction,'use_id':use_id,'arm':arm,
        'best_step':best_step,'fingerprint':stamp},dest/'model.pt')
    write_json(dest/'selection.json',{'best_step':best_step,'selection':'minimum source calibration coefficient MSE; outer never used',
        'dropout':dropout,'mask_unit':'whole target per optimizer step','pseudo_unseen_is_true_holdout':False,
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
    return pred


def representation(name,data,assets,diffusion,fit,seed):
    if name=='panel':return target_features(data,assets,fit,'string',seed,32)
    raw=diffusion['undiffused_random'] if name=='undiffused_random' else diffusion['diffusion']
    available=diffusion['available'];perm=None
    if name.startswith('shuffled_diffusion_'):
        perm=np.random.default_rng(13800+int(name.rsplit('_',1)[1])).permutation(len(raw))
        raw=raw[perm];available=available[perm]
    values,transform=projection(raw,np.unique(data['pert_idx'][fit]),32)
    if perm is not None:transform['permutation']=perm
    return np.c_[values,available.astype(np.float32)],transform,available


def worker(cfg,resume=False):
    torch.set_num_threads(2);dest=Path(cfg['output_dir']);dest.mkdir(parents=True,exist_ok=True)
    parent=Path(cfg['parent_worker']);verify_files(parent)
    pm=json.loads((parent/'manifest.json').read_text());pc=pm['config']
    if pc['variant']!='expanded' or pc['protocol']!=cfg['protocol'] or pc['seed']!=cfg['seed']:raise ValueError('Historical worker identity mismatch')
    data=load_prepared(pc['data_dir'])
    if data['audit']['fingerprint']!=pm['data']:raise ValueError('Historical prepared data changed')
    if file_sha256(pc['knowledge'])!=pm['knowledge'] or file_sha256(pc['encoder'])!=pm['encoder']:raise ValueError('Historical dependencies changed')
    identity={'config':cfg,'source':source_fingerprint(),'parent_manifest':file_sha256(parent/'manifest.json'),
              'parent_complete':file_sha256(parent/'COMPLETE.json'),'diffusion':file_sha256(cfg['diffusion'])}
    stamp=fingerprint(identity)
    if (dest/'manifest.json').exists() and not resume:raise ValueError('Use --resume or a new name')
    freeze(dest/'manifest.json',{'fingerprint':stamp,**identity})
    if (dest/'COMPLETE.json').exists():verify_files(dest);return dest
    assets=load_knowledge(pc['knowledge'],data)
    with np.load(cfg['diffusion'],allow_pickle=False) as f:diffusion={k:f[k] for k in f.files}
    if not np.array_equal(diffusion['targets'],data['perturbations']):raise ValueError('Diffusion target order mismatch')
    fit,cal,outer=[np.asarray(pc['partition'][k],int) for k in ('fit','calibration','outer')]
    with np.load(parent/'background_transform.npz',allow_pickle=False) as f:
        basis=f['basis'];bt={k:f[k] for k in ('mean','axes','scale')};response_scale=float(f['response_scale'])
    latent,norm=encode_background(data,{**pc,'device':cfg['device']})
    if not np.isclose(response_scale,norm['scale']):raise ValueError('Response normalization changed')
    bg=((latent-bt['mean'])@bt['axes'].T/bt['scale']).astype(np.float32)
    weights=row_weights(data['meta'].iloc[fit]);cw=row_weights(data['meta'].iloc[cal])
    ids,seen=identity_indices(data,fit)
    # No response transformation of Jurkat or outer rows is needed for fitting.
    coeff=np.zeros((len(data['meta']),len(basis)),np.float32)
    coeff[np.r_[fit,cal]]=data['delta'][np.r_[fit,cal]]@basis.T/response_scale
    meta=data['meta'].iloc[outer].reset_index(drop=True);truth=data['delta'][outer]
    records=[];ranks=[]
    ref,_=reference_responses(data,fit)
    baseline={'zero':np.zeros_like(truth),'mean_transfer':ref[data['pert_idx'][outer]]}
    for name in ('string_ridge','id_ridge','none_ridge'):
        with np.load(parent/name/'predictions.npz',allow_pickle=False) as f:
            if not np.array_equal(f['row_ids'],meta.row_id.to_numpy(dtype='U')) or not np.array_equal(f['genes'],data['genes']):raise ValueError('Baseline row/panel mismatch')
            baseline['r12_'+name]=f['prediction']
    for name,pred in baseline.items():
        records+=score(meta,truth,pred,name);ranks+=retrieval(meta,truth,pred,name)
    oracle=[]
    for rank in (32,64):
        b,audit=response_basis(data,fit,rank)
        if rank==32 and not np.allclose((basis@b.T)@b,basis,rtol=2e-4,atol=2e-4):raise ValueError('Historical basis not fit-only')
        rows=score(meta,truth,truth@b.T@b,f'oracle_rank{rank}')
        oracle+=rows;write_json(dest/f'basis_rank{rank}.json',audit)
    pd.DataFrame(oracle).to_csv(dest/'oracle_sweep.csv',index=False)
    ood=[]
    for label,ix in [('fit',fit),('calibration',cal),('outer',outer)]:
        for context,indices in data['meta'].iloc[ix].reset_index(drop=True).groupby('context').indices.items():
            norms=np.linalg.norm(bg[ix[indices]],axis=1)
            ood.append({'partition':label,'context':context,'rows':len(indices),'mean_latent_norm':float(norms.mean()),
                        'p95_latent_norm':float(np.quantile(norms,.95)),'max_latent_norm':float(norms.max())})
    pd.DataFrame(ood).to_csv(dest/'background_shift.csv',index=False)
    for arm in cfg['arms']:
        spec=ARMS[arm];folder=dest/arm;folder.mkdir(exist_ok=True)
        if (folder/'COMPLETE.json').exists():
            verify_files(folder)
            if json.loads((folder/'COMPLETE.json').read_text())['fingerprint']!=stamp:raise ValueError('Head fingerprint changed')
            records+=pd.read_csv(folder/'per_target.csv').to_dict('records')
            ranks+=pd.read_csv(folder/'retrieval.csv').to_dict('records');continue
        print(f'ROUND13 HEAD {cfg["protocol"]} seed={cfg["seed"]} {arm}',flush=True)
        tf,transform,available=representation(spec[0],data,assets,diffusion,fit,cfg['seed'])
        target=tf[data['pert_idx']]
        if spec[1]=='ridge':
            x=design(bg,target,interactions=spec[2]);mean=weights@x[fit]
            scale=np.maximum(np.sqrt(weights@np.square(x[fit]-mean)),.1);x=((x-mean)/scale).astype(np.float32)
            best=None;trials=[]
            for alpha in pc['ridge_alphas']:
                coef,bias=fit_ridge(x[fit],coeff[fit],weights,alpha)
                error=mse((x[cal]@coef+bias)@basis*response_scale,data['delta'][cal],cw)
                trials.append({'alpha':alpha,'calibration_mse':error})
                if best is None or error<best[0]:best=(error,coef,bias,alpha)
            _,coef,bias,alpha=best;pred=(x[outer]@coef+bias)@basis*response_scale
            np.savez_compressed(folder/'model.npz',coef=coef,intercept=bias,xmean=mean,xscale=scale)
            write_json(folder/'selection.json',{'chosen_alpha':alpha,'trials':trials,'selection':'source calibration only'})
        else:
            fc={**cfg,'fit_weights':weights.tolist(),'cal_weights':cw.tolist()}
            pred=fit_factor(target,bg,ids,coeff,fit,cal,fc,arm,folder,stamp)[outer]@basis*response_scale
        np.savez_compressed(folder/'transforms.npz',**transform,basis=basis,response_scale=response_scale,
                            background_mean=bt['mean'],background_axes=bt['axes'],background_scale=bt['scale'],fit_target_indices=seen)
        rows=score(meta,truth,pred,arm,available[data['pert_idx'][outer]]);rr=retrieval(meta,truth,pred,arm)
        pd.DataFrame(rows).to_csv(folder/'per_target.csv',index=False);pd.DataFrame(rr).to_csv(folder/'retrieval.csv',index=False)
        np.savez_compressed(folder/'predictions.npz',prediction=pred,row_ids=meta.row_id.to_numpy(dtype='U'),genes=data['genes'])
        write_json(folder/'coverage.json',{'fit_targets':len(seen),'outer_targets':int(meta.perturbation.nunique()),
            'outer_covered_targets':int(meta.loc[available[data['pert_idx'][outer]],'perturbation'].nunique()),
            'outer_seen_targets':int(meta.loc[ids[outer]>0,'perturbation'].nunique()),'missing_targets_retained':True})
        save_complete(folder,{'fingerprint':stamp});records+=rows;ranks+=rr
    pd.DataFrame(records).to_csv(dest/'per_target.csv',index=False);pd.DataFrame(ranks).to_csv(dest/'retrieval.csv',index=False)
    save_complete(dest,{'fingerprint':stamp,'test_evaluated':False})
    print(f'ROUND13 COMPLETE WORKER: {dest}',flush=True)
    return dest


def report(specs,root):
    root=Path(root);all_frames={name:[] for name in ('per_target','retrieval','oracle_sweep','background_shift')}
    for cfg in specs.values():
        folder=Path(cfg['output_dir']);verify_files(folder)
        for name,frames in all_frames.items():
            f=pd.read_csv(folder/(name+'.csv'));f['protocol']=cfg['protocol'];f['seed']=cfg['seed'];frames.append(f)
    frames={k:pd.concat(v,ignore_index=True) for k,v in all_frames.items()}
    for name,f in frames.items():f.to_csv(root/(name+'.csv'),index=False)
    f=frames['per_target'];r=frames['retrieval'];keys=['protocol','method','seed']
    scores=f.groupby(keys+['context']).agg(mse=('mse','mean'),pearson=('pearson','mean'),targets=('target','size')).reset_index()
    retrieval_scores=r.groupby(keys+['context']).agg(specificity_gap=('specificity_gap','mean'),top1=('top1_credit','mean'),
        top5=('top5_credit','mean'),median_rank=('midrank','median'),mean_rank_fraction=('rank_fraction','mean')).reset_index()
    scores=scores.merge(retrieval_scores,on=keys+['context'],validate='one_to_one');scores.to_csv(root/'context_scores.csv',index=False)
    per_seed=scores.groupby(keys).agg(mse=('mse','mean'),specificity_gap=('specificity_gap','mean'),top1=('top1','mean'),top5=('top5','mean')).reset_index()
    per_seed.to_csv(root/'comparison.csv',index=False)
    summary=per_seed.groupby(['protocol','method']).agg(mse_mean=('mse','mean'),seed_sd=('mse','std'),
            specificity_gap=('specificity_gap','mean'),top1=('top1','mean'),top5=('top5','mean'),seeds=('seed','nunique')).reset_index()
    summary.to_csv(root/'comparison_mean.csv',index=False)
    contrasts=[('zero',m) for m in sorted(f.method.unique()) if m!='zero']
    contrasts += [('r12_string_ridge','diffusion_interaction_ridge'),('panel_additive_ridge','r12_string_ridge'),
                  ('diffusion_additive_ridge','diffusion_interaction_ridge'),('diffusion_additive','diffusion_factor'),
                  ('diffusion_interaction_ridge','diffusion_factor'),('diffusion_factor','diffusion_factor_id'),
                  ('diffusion_factor_id','diffusion_factor_id_dropout'),('undiffused_interaction_ridge','diffusion_interaction_ridge')]
    for j in (1,2,3):
        contrasts += [(f'shuffled_diffusion_{j}_interaction_ridge','diffusion_interaction_ridge'),
                      (f'shuffled_diffusion_{j}_factor_id_dropout','diffusion_factor_id_dropout')]
    averaged=f.groupby(['protocol','method','context','target']).mse.mean();paired=[]
    for protocol in f.protocol.unique():
        for left,right in contrasts:
            try:a=averaged.loc[(protocol,left)];b=averaged.loc[(protocol,right)]
            except KeyError:continue
            aligned=pd.concat([a.rename('a'),b.rename('b')],axis=1)
            if aligned.isna().any().any():raise ValueError('Unpaired comparison rows')
            gain=(aligned.a-aligned.b).unstack('context');rng=np.random.default_rng(1701);draws=[];v=gain.to_numpy()
            for _ in range(2000):
                sample=v[rng.integers(len(v),size=len(v))]
                if np.isfinite(sample).any(0).all():draws.append(float(np.nanmean(sample,axis=0).mean()))
            lo,hi=np.quantile(draws,[.025,.975]);target=gain.mean(1);largest=target.idxmax()
            paired.append({'protocol':protocol,'baseline':left,'candidate':right,'gain_mse':float(gain.mean().mean()),
                'ci_low':lo,'ci_high':hi,'target_wins':int((target>0).sum()),'targets':len(target),'largest_gain_target':largest,
                'gain_without_largest_target':float(gain.drop(index=largest).mean().mean()) if len(gain)>1 else np.nan})
    pairs=pd.DataFrame(paired);pairs.to_csv(root/'paired_comparisons.csv',index=False)
    write_json(root/'evaluation_scope.json',{'test_evaluated':False,'jurkat_scored':False,'automatic_winner':False,
        'status':'previously inspected development cohorts; independent replication still required',
        'rank64':'oracle diagnostic only, not chosen automatically using outer labels',
        'target_dropout':'ID branch masked per target per step; STRING branch still sees that target response; not true leave-target-out training',
        'bootstrap':'2000 target resamples after averaging optimizer seeds; joint across contexts; no multiplicity correction',
        'success':'double MSE below zero plus positive specificity gap is an exploratory milestone, not proof of solving generalization',
        'no_posthoc_shrinkage':True,'background':'unchanged audited Round12 B2 encoders and projections'})
    (root/'report.html').write_text('<html><meta charset="utf-8"><title>Round13</title><h1>Round13</h1><p>Development only; no automatic winner. Full-network diffusion, low-rank interactions and ID-residual controls. Rank64 is diagnostic only.</p>'+summary.to_html(index=False)+'<h2>Paired exploratory comparisons</h2>'+pairs.to_html(index=False),encoding='utf-8')
