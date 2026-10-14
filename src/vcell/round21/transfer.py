"""Small correction heads receive source labels and control-only query features."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ..model_agent.artifacts import arrays,read,seal,verify
from ..model_agent.contracts import digest
from ..model_agent.nested import lines
from ..model_agent.candidate_code import BASE_HEAD,build_head
from ..round11 import freeze
from ..round20.signal import exposure_weights
from ..utils import file_sha256,write_json,seed_all,atomic_torch_save
from .signal import POLICY,KEYS,references


def cache(parent,metadata,held,dest,seed=17):
    import pandas as pd
    parent=Path(parent);verify(parent);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    identity={'parent_sha256':file_sha256(parent/'COMPLETE.json'),'metadata_sha256':file_sha256(metadata),
        'held_targets':held,'policy':POLICY,'seed':seed,'code':namespace()}
    freeze(dest/'manifest.json',identity)
    if (dest/'COMPLETE.json').exists():verify(dest);return dest
    meta=pd.read_csv(metadata,keep_default_na=False);d=arrays(parent/'features.npz');s=arrays(parent/'source_labels.npz')
    if not np.array_equal(d['row_ids'],meta.row_id.to_numpy(dtype='U')):raise ValueError('Cache row alignment changed')
    fit,cal,query=[np.asarray(v,int) for v in (s['fit'],s['calibration'],d['query'])]
    if set(lines(meta.iloc[query]))&set(lines(meta.iloc[np.r_[fit,cal]])):raise ValueError('Held cell line entered source labels')
    ref,donor,av,common,strong,table,audit=references(meta,fit,s['fit_delta'],d['control'],held,seed=seed,response_scale=float(d['response_scale']))
    mean=ref['mean'];coeff=mean@d['basis'].T
    reliable_keys={tuple(r[k] for k in KEYS) for _,r in table.loc[table.reliable].iterrows()}
    train=np.array([i for i in fit if av[i] and tuple(meta.iloc[i][k] for k in KEYS) in reliable_keys],int)
    np.savez_compressed(dest/'features.npz',target=np.c_[d['target'],coeff].astype(np.float32),
        background=np.c_[d['control'],(d['control']-donor)*av[:,None],av].astype(np.float32),
        basis=d['basis'],response_scale=d['response_scale'],query=query,row_ids=d['row_ids'],genes=d['genes'],
        reference=mean,nearest=ref['nearest'],wrong=ref['wrong'],shuffled=ref['shuffled'],background_mean=ref['background'],
        available=av,common_support=common,strong_mask=strong,
        ridge=arrays(parent/'teacher_query_predictions.npz')['prediction'][:,0]/float(d['response_scale']))
    np.savez_compressed(dest/'source_labels.npz',**s,train=train,
        train_weights=exposure_weights(meta.iloc[train]) if len(train) else np.empty(0))
    table.to_csv(dest/'replicate_qc_by_target.csv',index=False)
    from .signal import signature_table
    _,mismatch,_=signature_table(meta,fit,s['fit_delta']*float(d['response_scale']));mismatch.to_csv(dest/'signal_vs_mismatch.csv',index=False)
    write_json(dest/'lineage.json',{'fit_rows':fit.tolist(),'calibration_rows':cal.tolist(),'query_rows':query.tolist(),
        'correction_train_rows':train.tolist(),'references':audit,'reliability_label_scope':'source fit only, both technical repeats allowed',
        'held_targets':held,'normalization':'parent source-fit response scale and basis; inherited covariate encoder',
        'fallback':'zero without same-target support; background_mean remains target-agnostic',
        'parent_cache':str(parent),'parent_sha256':identity['parent_sha256'],'test_evaluated':False})
    seal(dest);return dest


def namespace():
    root=Path(__file__).resolve().parents[1]
    paths=[*sorted(Path(__file__).parent.glob('*.py')),root/'model_agent/candidate_code.py',root/'model_agent/contracts.py',
        root/'model_agent/nested.py',root/'round20/signal.py']
    return digest({p.relative_to(root).as_posix():file_sha256(p) for p in paths})


def worker(cfg,resume=False):
    torch.set_num_threads(2);source=Path(cfg['cache']);verify(source);dest=Path(cfg['output_dir']);dest.mkdir(parents=True,exist_ok=True)
    identity={'config':cfg,'cache_sha256':file_sha256(source/'COMPLETE.json'),'code':namespace()};stamp=digest(identity)
    freeze(dest/'manifest.json',{'fingerprint':stamp,**identity})
    if (dest/'COMPLETE.json').exists():verify(dest);return dest
    d=arrays(source/'features.npz');s=arrays(source/'source_labels.npz');train=s['train'].astype(int);fit=s['fit'].astype(int);cal=s['calibration'].astype(int);query=d['query'].astype(int)
    if not set(train)<=set(fit) or set(query)&set(np.r_[fit,cal]):raise ValueError('Correction label scope overlap')
    seed_all(cfg['seed']);device=cfg['device'];rank=min(cfg['rank'],len(d['basis']));basis=d['basis'][:rank]
    model=build_head(cfg['code'],d['target'].shape[1],d['background'].shape[1],rank).to(device)
    x,b,bt=[torch.tensor(v,device=device) for v in (d['target'],d['background'],basis)]
    y=torch.zeros((len(x),len(d['genes'])),device=device);y[fit]=torch.tensor(s['fit_delta'],device=device);y[cal]=torch.tensor(s['calibration_delta'],device=device)
    norm=float(np.linalg.norm(s['fit_delta'],axis=1).mean());bound=.2*norm/np.sqrt(rank)
    av=torch.tensor(d['available'],device=device);ref=torch.tensor(d['reference'],device=device)
    def predict(ix):return (cfg['alpha']*ref[ix]+torch.tanh(model(x[ix],b[ix]))@bt*bound)*av[ix,None]
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    first=0;history=[];steps=cfg['max_steps'] if len(train)>=4 else 0
    if resume and (dest/'last.pt').exists():
        checkpoint=torch.load(dest/'last.pt',map_location='cpu',weights_only=True)
        if checkpoint['fingerprint']!=stamp:raise ValueError('Resume fingerprint changed')
        model.load_state_dict(checkpoint['state']);opt.load_state_dict(checkpoint['optimizer']);first=checkpoint['step'];history=checkpoint['history']
    for step in range(first,steps):
        seed_all(cfg['seed']+31000+step);ix=np.random.default_rng(cfg['seed']+31000+step).choice(train,cfg['batch_size'],p=s['train_weights'])
        model.train();opt.zero_grad(set_to_none=True);p=predict(ix)
        loss=(p-y[ix]).square().mean() if cfg['loss']=='mse' else nn.functional.huber_loss(p,y[ix])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite correction objective')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();done=step+1
        if done==1 or done%100==0:print(f'ROUND21 HEAD step={done}/{steps} loss={loss.item():.6f}',flush=True)
        if done%cfg['save_every']==0 or done==steps:
            model.eval()
            with torch.no_grad():value=float((predict(cal)-y[cal]).square().mean())
            history.append({'step':done,'fit_loss':float(loss.detach()),'calibration_mse_normalized':value,'calibration_selects_checkpoint':False})
            atomic_torch_save({'fingerprint':stamp,'step':done,'state':model.state_dict(),'optimizer':opt.state_dict(),'history':history},dest/'last.pt')
            write_json(dest/'progress.json',{'step':done,'max_steps':steps})
    model.eval()
    with torch.no_grad():p=predict(query).cpu().numpy() if steps else cfg['alpha']*d['reference'][query]
    np.savez_compressed(dest/'prediction.npz',prediction=p.astype(np.float32)*float(d['response_scale']),row_ids=d['row_ids'][query],genes=d['genes'],
        reference_available=d['available'][query],common_support=d['common_support'][query])
    atomic_torch_save({'config':cfg,'state':model.state_dict(),'basis':torch.tensor(basis),'bound':bound,'response_scale':float(d['response_scale'])},dest/'model.pt')
    write_json(dest/'training_status.json',{'status':'TRAINED' if steps else 'BLOCKED_NO_RELIABLE_TRAIN_REFERENCE',
        'steps':steps,'train_rows':len(train),'fallback':'calibrated reference only when untrained','checkpoint':'fixed final step','test_evaluated':False})
    write_json(dest/'history.json',history);seal(dest);return dest
