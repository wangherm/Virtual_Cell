"""Supported Rep1 -> sealed Rep2 task; response transforms exclude query targets."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..model_agent.artifacts import arrays,seal,verify,read
from ..model_agent.candidate_code import BASE_HEAD,build_head
from ..model_agent.contracts import digest
from ..round11 import freeze
from ..round20.signal import ALPHAS
from ..utils import seed_all,write_json,file_sha256,atomic_torch_save
from .signal import pairs,KEYS,POLICY,top
from .transfer import namespace


def repeat_plan(meta,rows,delta,baseline):
    units=pairs(meta,rows,delta);specs=[]
    for context in sorted(set(u['context'] for u in units)):
        for cond in sorted(set(tuple(u[k] for k in KEYS[2:]) for u in units if u['context']==context)):
            block=[u for u in units if u['context']==context and tuple(u[k] for k in KEYS[2:])==cond]
            roles=[int(digest(['round21_repeat_target',u['perturbation']])[:8],16)%3 for u in block]
            for fold in range(3):
                train=[u for u,r in zip(block,roles) if r!=fold];query=[u for u,r in zip(block,roles) if r==fold]
                if not query:continue
                specs.append({'context':context,'conditions':cond,'fold':fold,'train':train,'query':query})
    return specs


def pack(spec,baseline,dest):
    """Only Rep1 support is exported for query targets. Rep2 truth stays controller-private."""
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    def b(unit,rep):return baseline[unit[rep+'_rows']].mean(0)
    train=spec['train'];query=spec['query'];g=len(query[0]['rep1'])
    np.savez_compressed(dest/'data.npz',train_rep1=np.array([u['rep1'] for u in train]).reshape(-1,g),
        train_rep2=np.array([u['rep2'] for u in train]).reshape(-1,g),query_rep1=np.array([u['rep1'] for u in query]),
        train_control_difference=np.array([b(u,'rep2')-b(u,'rep1') for u in train]).reshape(-1,g),
        query_control_difference=np.array([b(u,'rep2')-b(u,'rep1') for u in query]),
        train_targets=np.array([u['perturbation'] for u in train],dtype='U'),query_targets=np.array([u['perturbation'] for u in query],dtype='U'))
    write_json(dest/'lineage.json',{'task':'supported technical repeat prediction, NOT unseen background or unsupported target prediction',
        'context':spec['context'],'conditions':spec['conditions'],'fold':spec['fold'],
        'train_rep1_rows':[r for u in train for r in u['rep1_rows'].tolist()],
        'train_rep2_rows':[r for u in train for r in u['rep2_rows'].tolist()],
        'query_support_rep1_rows':[r for u in query for r in u['rep1_rows'].tolist()],
        'sealed_query_rep2_rows':[r for u in query for r in u['rep2_rows'].tolist()],
        'query_selection':'identifier-only whole-target hash, all complete eligible pairs retained',
        'strong_mask':'query Rep1 support plus noise from OTHER training-target pairs; no query Rep2 filtering',
        'normalization':'original matched pseudobulk, independent Rep2 controls must be checked in raw audit','test_evaluated':False})
    seal(dest);return dest


def worker(cfg,resume=False):
    torch.set_num_threads(2);source=Path(cfg['cache']);verify(source);dest=Path(cfg['output_dir']);dest.mkdir(parents=True,exist_ok=True)
    identity={'config':cfg,'cache':file_sha256(source/'COMPLETE.json'),'code':namespace()};stamp=digest(identity)
    freeze(dest/'manifest.json',{'fingerprint':stamp,**identity})
    if (dest/'COMPLETE.json').exists():verify(dest);return dest
    d=arrays(source/'data.npz');a,b,q=d['train_rep1'],d['train_rep2'],d['query_rep1'];n=len(a);g=q.shape[1]
    if set(d['train_targets'])&set(d['query_targets']):raise ValueError('Repeat target fold overlap')
    preds={'zero':np.zeros_like(q),'rep1_copy':q.copy()};mask=np.zeros_like(q,dtype=bool)
    lineage={'train_targets':d['train_targets'].tolist(),'query_targets':d['query_targets'].tolist(),'train_pairs':n,
        'query_rep2_loaded':False,'response_transforms_fit':'other-target training pairs only','test_evaluated':False}
    if n<3:
        lineage.update(status='BLOCKED_TOO_FEW_TRAIN_PAIRS',steps=0)
    else:
        alpha=min(ALPHAS,key=lambda x:(float(np.square(x*a-b).mean()),x));mu1=a.mean(0);mu2=b.mean(0)
        centered_alpha=min(ALPHAS,key=lambda x:(float(np.square(mu2+x*(a-mu1)-b).mean()),x))
        preds.update(background_mean=np.tile(mu2,(len(q),1)),calibrated_rep1=alpha*q,demeaned_rep1=mu2+centered_alpha*(q-mu1))
        wrong=[a[int(digest(['repeat_wrong',cfg['seed'],str(t)])[:8],16)%n] for t in d['query_targets']];preds['wrong_rep1']=alpha*np.asarray(wrong)
        sigma=np.sqrt(np.median(np.square(a-b)/2,axis=0));threshold=np.maximum(POLICY['absolute_gene_floor'],POLICY['gene_sigma_multiplier']*sigma)
        for i in range(len(q)):
            mask[i,top(q[i],100)]=True;mask[i] &= np.abs(q[i])>=threshold
        # All response and control projections are fitted without query Rep2 or query targets.
        _,_,vh=np.linalg.svd(np.r_[a,b],full_matrices=False);basis=vh[:min(16,len(vh))].astype(np.float32)
        scale=max(float(np.sqrt(np.square(b).mean())),1e-4)
        cd=d['train_control_difference'];mean=cd.mean(0);sd=np.maximum(cd.std(0),.1)
        _,_,axes=np.linalg.svd((cd-mean)/sd,full_matrices=False);axes=axes[:min(16,len(axes))]
        x=a@basis.T/scale;qx=q@basis.T/scale
        z=((cd-mean)/sd)@axes.T;qz=((d['query_control_difference']-mean)/sd)@axes.T
        design=np.c_[x,z,np.ones(n)];qdesign=np.c_[qx,qz,np.ones(len(q))]
        penalty=np.eye(design.shape[1]);penalty[-1,-1]=0
        coef=np.linalg.solve(design.T@design+penalty,design.T@(b-alpha*a)/scale@basis.T)
        preds['ridge_correction']=alpha*q+(qdesign@coef)@basis*scale
        seed_all(cfg['seed']);model=build_head(BASE_HEAD,x.shape[1],z.shape[1],len(basis)).to(cfg['device'])
        tx,bx,bt=[torch.tensor(v,device=cfg['device']) for v in (x,z,basis)];truth=torch.tensor(b,device=cfg['device']);support=torch.tensor(a,device=cfg['device'])
        bound=.2*float(np.linalg.norm(b,axis=1).mean())/np.sqrt(len(basis));opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01);first=0
        if resume and (dest/'last.pt').exists():
            saved=torch.load(dest/'last.pt',map_location='cpu',weights_only=True)
            if saved['fingerprint']!=stamp:raise ValueError('Repeat checkpoint identity changed')
            model.load_state_dict(saved['state']);opt.load_state_dict(saved['optimizer']);first=saved['step']
        for step in range(first,cfg['max_steps']):
            seed_all(cfg['seed']+41000+step);ix=np.random.default_rng(cfg['seed']+41000+step).choice(n,min(cfg['batch_size'],n),replace=True)
            model.train();opt.zero_grad(set_to_none=True);p=alpha*support[ix]+torch.tanh(model(tx[ix],bx[ix]))@bt*bound;loss=(p-truth[ix]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite repeat head objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();done=step+1
            if done==1 or done%100==0:print(f'ROUND21 REPEAT step={done}/{cfg["max_steps"]} loss={float(loss.detach()):.6f}',flush=True)
            if done%cfg['save_every']==0 or done==cfg['max_steps']:
                atomic_torch_save({'fingerprint':stamp,'step':done,'state':model.state_dict(),'optimizer':opt.state_dict()},dest/'last.pt')
                write_json(dest/'progress.json',{'step':done,'max_steps':cfg['max_steps']})
        model.eval()
        with torch.no_grad():preds['mlp_correction']=alpha*q+(torch.tanh(model(torch.tensor(qx,device=cfg['device']),torch.tensor(qz,device=cfg['device'])))@bt*bound).cpu().numpy()
        atomic_torch_save({'state':model.state_dict(),'basis':torch.tensor(basis),'control_axes':torch.tensor(axes),'control_mean':torch.tensor(mean),
            'control_scale':torch.tensor(sd),'alpha':alpha,'bound':bound,'response_scale':scale,'target_dim':x.shape[1],'background_dim':z.shape[1]},dest/'model.pt')
        lineage.update(status='TRAINED',steps=cfg['max_steps'],alpha=float(alpha),centered_alpha=float(centered_alpha))
    np.savez_compressed(dest/'prediction.npz',**preds,strong_mask=mask,targets=d['query_targets'])
    write_json(dest/'lineage.json',lineage);seal(dest);return dest
