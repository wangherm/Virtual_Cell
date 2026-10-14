"""Fit-only experimental reference library and bounded numerical correction."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ..model_agent.artifacts import arrays,read,seal,verify
from ..model_agent.candidate_code import build_head,BASE_HEAD
from ..model_agent.contracts import digest
from ..model_agent.nested import lines
from ..round11 import freeze
from ..utils import seed_all,atomic_torch_save,write_json,file_sha256
from .signal import CONDITIONS,exposure_weights


def references(meta,fit,fit_delta,control,held_targets,mode='matched',seed=17):
    """Query features use only listed fit responses; never query truth/cal labels."""
    fit=np.asarray(fit,int);held=set(held_targets)
    if set(meta.iloc[fit].perturbation.astype(str)) & held:raise ValueError('Globally unknown target entered reference fit library')
    if len(fit_delta)!=len(fit):raise ValueError('Reference fit labels misaligned')
    keys=[*CONDITIONS,'batch']
    if any(k not in meta for k in keys):raise ValueError('Reference conditions missing')
    line=lines(meta)
    values=meta[keys].fillna('').astype(str).to_numpy()
    targets=meta.perturbation.astype(str).to_numpy();known=sorted(set(targets[fit]))
    replacement=dict(zip(known,np.roll(known,1)))
    if mode=='shuffled':
        replacement=dict(zip(known,np.random.default_rng(seed).permutation(known)))
    ref=np.zeros((len(meta),fit_delta.shape[1]),np.float32);donor=np.zeros_like(control);available=np.zeros(len(meta),bool);audit=[]
    for row in range(len(meta)):
        target=targets[row]
        # Wrong/shuffled controls also cannot expose a held target's response.
        if target in held or target not in known or mode=='none':continue
        if any(v in ('','unknown','not_encoded_source_specific') for v in values[row]):continue
        requested=replacement[target] if mode in ('wrong','shuffled') else target
        keep=(line[fit]!=line[row])&(targets[fit]==requested)&np.all(values[fit]==values[row],axis=1)
        pos=np.flatnonzero(keep)
        if not len(pos):continue
        w=exposure_weights(meta.iloc[fit[pos]])
        ref[row]=w@fit_delta[pos];donor[row]=w@control[fit[pos]];available[row]=True
        audit.append({'row_id':str(meta.iloc[row].row_id),'query_target':target,'reference_target':requested,
            'donor_rows':fit[pos].tolist(),'donor_cell_lines':sorted(set(line[fit[pos]])),
            'query_cell_line':line[row],'conditions':dict(zip(keys,values[row]))})
    return ref,donor,available,audit


def build_inputs(meta,features,labels,held_targets,mode):
    ref,donor,available,audit=references(meta,labels['fit'],labels['fit_delta'],features['control'],held_targets,mode)
    basis=features['basis'];coeff=ref@basis.T
    # Strict paired ablation: correction-only does not receive reference content.
    target=np.c_[features['target'],coeff if mode!='correction_only' else np.zeros_like(coeff)]
    background=np.c_[features['control'],(features['control']-donor)*available[:,None] if mode!='correction_only' else np.zeros_like(donor),available]
    return target.astype(np.float32),background.astype(np.float32),coeff,available,audit


def worker(cfg,resume=False):
    torch.set_num_threads(2);cache=Path(cfg['cache']);verify(cache)
    dest=Path(cfg['output_dir']);dest.mkdir(parents=True,exist_ok=True)
    identity={'config':cfg,'cache':file_sha256(cache/'COMPLETE.json'),'code':file_sha256(__file__),
        'metadata':file_sha256(cfg['metadata'])}
    stamp=digest(identity);freeze(dest/'manifest.json',{'fingerprint':stamp,**identity})
    if (dest/'COMPLETE.json').exists():verify(dest);return dest
    import pandas as pd
    meta=pd.read_csv(cfg['metadata'],keep_default_na=False);d=arrays(cache/'features.npz');labels=arrays(cache/'source_labels.npz')
    if not np.array_equal(d['row_ids'],meta.row_id.to_numpy(dtype='U')):raise ValueError('Reference metadata row order changed')
    target,bg,ref,available,audit=build_inputs(meta,d,labels,cfg['held_targets'],cfg['mode'])
    fit,cal,query=[np.asarray(x,int) for x in (labels['fit'],labels['calibration'],d['query'])]
    if set(query)&set(np.r_[fit,cal]):raise ValueError('Reference worker query label leakage')
    seed_all(cfg['seed']);device=cfg['device'];rank=min(cfg.get('rank',32),len(d['basis']));basis=d['basis'][:rank];ref=ref[:,:rank]
    code=cfg.get('code',BASE_HEAD)
    model=build_head(code,target.shape[1],bg.shape[1],rank).to(device)
    tx,bx,bt=[torch.tensor(x,device=device) for x in (target,bg,basis)]
    rc=torch.tensor(ref,device=device);av=torch.tensor(available,device=device)
    y=torch.zeros((len(meta),len(d['genes'])),device=device);y[fit]=torch.tensor(labels['fit_delta'],device=device);y[cal]=torch.tensor(labels['calibration_delta'],device=device)
    w=exposure_weights(meta.iloc[fit]);cw=exposure_weights(meta.iloc[cal])
    # Bound L2 correction by radius * source-fit typical response norm.
    bound=float(cfg['radius'])*(w@np.linalg.norm(labels['fit_delta'],axis=1))/np.sqrt(rank)
    def prediction(rows):
        corr=torch.tanh(model(tx[rows],bx[rows]))*bound
        if cfg['mode']=='correction_only':return corr@bt
        if cfg['mode'] in ('reference_only','none'):corr=corr*0
        return ((float(cfg['alpha'])*rc[rows]+corr)*av[rows,None])@bt
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    first=0;history=[]
    if resume and (dest/'last.pt').exists():
        s=torch.load(dest/'last.pt',map_location='cpu',weights_only=True)
        if s['fingerprint']!=stamp:raise ValueError('Reference resume identity changed')
        model.load_state_dict(s['state']);opt.load_state_dict(s['optimizer']);first=s['step'];history=s['history']
    steps=cfg['max_steps'] if cfg['mode'] not in ('reference_only','none') else 0
    for step in range(first,steps):
        seed_all(cfg['seed']+21000+step);ix=np.random.default_rng(cfg['seed']+21000+step).choice(fit,cfg['batch_size'],p=w)
        model.train();opt.zero_grad(set_to_none=True);pred=prediction(ix)
        loss=(pred-y[ix]).square().mean() if cfg.get('loss','mse')=='mse' else torch.nn.functional.huber_loss(pred,y[ix])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite reference objective')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        done=step+1
        if done==1 or done%100==0:print(f'ROUND20 REFERENCE step={done}/{steps} loss={loss.item():.6f}',flush=True)
        if done%cfg['save_every']==0 or done==steps:
            model.eval()
            with torch.no_grad():
                history.append({'step':done,'fit_mse':float(w@((prediction(fit)-y[fit]).square().mean(1)).cpu().numpy()),
                    'calibration_mse':float(cw@((prediction(cal)-y[cal]).square().mean(1)).cpu().numpy())})
            atomic_torch_save({'fingerprint':stamp,'step':done,'state':model.state_dict(),'optimizer':opt.state_dict(),'history':history},dest/'last.pt')
            write_json(dest/'progress.json',{'step':done,'max_steps':steps})
    # Fixed final step; source OOF selects designs. Historical-only calibration
    # cannot select an IFNG reference correction checkpoint meaningfully.
    model.eval()
    with torch.no_grad():p=prediction(query).cpu().numpy()*float(d['response_scale'])
    np.savez_compressed(dest/'prediction.npz',prediction=p.astype(np.float32),row_ids=d['row_ids'][query],genes=d['genes'],reference_available=available[query])
    atomic_torch_save({'state':model.state_dict(),'code':code,'target_dim':target.shape[1],'background_dim':bg.shape[1],
        'bound':bound,'basis':torch.tensor(basis),'response_scale':float(d['response_scale']),'config':cfg},dest/'model.pt')
    write_json(dest/'lineage.json',{'references':audit,'fit_rows':fit.tolist(),'calibration_rows':cal.tolist(),
        'query_rows':query.tolist(),'held_targets':cfg['held_targets'],'no_reference_fallback':'zero except explicit correction_only control',
        'checkpoint_selection':'fixed final step, not historical-only calibration','fit_exposure':'50:50 historical/IFNG if both available',
        'parameter_count':sum(p.numel() for p in model.parameters()),'correction_L2_bound_normalized':float(bound*np.sqrt(rank)),
        'test_evaluated':False})
    write_json(dest/'history.json',history);seal(dest);return dest
