"""Executable Round15 P0/P1/P2 screen; future stages remain explicitly gated."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .background_training import fingerprint
from .data import load_prepared
from .fixed_training import response_basis
from .knowledge import load_knowledge
from .round11 import freeze, verify_files
from .round12 import encode_background, projection, score
from .round13 import identity_indices, retrieval
from .round14 import target_representation, response_diagnostics
from .round15_heads import ARMS, REUSE, VARIANTS, BLOCKED, ExperimentalHead, contrastive_loss
from .round15_references import reference_bank, ridge_reference
from .selection import row_weights
from .specialization import reference_responses
from .train import source_fingerprint
from .utils import atomic_torch_save, file_sha256, seed_all, write_json


def historical_source():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        if path.name.startswith('round15'): continue
        h.update(path.name.encode()); h.update(path.read_bytes())
    return h.hexdigest()


def seal(path, marker='COMPLETE.json', **extra):
    excluded = {'COMPLETE.json','SCREEN_COMPLETE.json','CORE_COMPLETE.json','INPUTS_COMPLETE.json','last.pt'}
    files = {p.relative_to(path).as_posix():file_sha256(p) for p in sorted(Path(path).rglob('*'))
             if p.is_file() and p.name not in excluded and not p.name.endswith('.tar.gz')}
    write_json(Path(path)/marker, {'files':files,**extra})


def verify_marker(path, marker):
    manifest = json.loads((Path(path)/marker).read_text())
    for name, digest in manifest['files'].items():
        if file_sha256(Path(path)/name)!=digest: raise ValueError('Artifact checksum mismatch: '+name)
    return manifest


def prepare_inputs(cfg, dest):
    """Recompute only once per fold/seed; all target heads share audited inputs."""
    parent14 = Path(cfg['parent_worker']); verify_files(parent14)
    manifest14 = json.loads((parent14/'manifest.json').read_text()); c14 = manifest14['config']
    parent13 = Path(c14['parent_worker']); verify_files(parent13)
    c13 = json.loads((parent13/'manifest.json').read_text())['config']
    parent12 = Path(c13['parent_worker']); verify_files(parent12)
    manifest12 = json.loads((parent12/'manifest.json').read_text()); pc = manifest12['config']
    for c in (c14,c13,pc):
        if c['protocol']!=cfg['protocol'] or c['seed']!=cfg['seed']: raise ValueError('Historical fold/seed mismatch')
    data = load_prepared(pc['data_dir'])
    if data['audit']['fingerprint']!=manifest12['data']: raise ValueError('Historical data changed')
    if file_sha256(pc['knowledge'])!=manifest12['knowledge'] or file_sha256(pc['encoder'])!=manifest12['encoder']:
        raise ValueError('Historical dependency changed')
    identity = {'config':cfg,'source':source_fingerprint(),'parent14':file_sha256(parent14/'COMPLETE.json'),
                'parent12':file_sha256(parent12/'COMPLETE.json'),'data':data['audit']['fingerprint']}
    stamp = fingerprint(identity); freeze(dest/'manifest.json',{'fingerprint':stamp,**identity})
    fit,cal,outer = [np.asarray(pc['partition'][k],int) for k in ('fit','calibration','outer')]
    inputs = dest/'inputs'; inputs.mkdir(exist_ok=True)
    if (inputs/'INPUTS_COMPLETE.json').exists():
        verified = verify_marker(inputs,'INPUTS_COMPLETE.json')
        if verified['fingerprint']!=stamp: raise ValueError('Input cache identity mismatch')
    else:
        with np.load(parent12/'background_transform.npz',allow_pickle=False) as f:
            bt = {k:f[k] for k in f.files}
        latent,norm = encode_background(data,{**pc,'device':cfg['device']})
        if not np.isclose(bt['response_scale'],norm['scale']): raise ValueError('Normalization changed')
        bg = ((latent-bt['mean'])@bt['axes'].T/bt['scale']).astype(np.float32)
        np.savez_compressed(inputs/'background.npz',background=bg,**bt)
        seal(inputs,'INPUTS_COMPLETE.json',fingerprint=stamp)
    with np.load(inputs/'background.npz',allow_pickle=False) as f: bt = {k:f[k] for k in f.files}
    assets = load_knowledge(pc['knowledge'],data); ids,seen = identity_indices(data,fit)
    reusable = manifest14['source']==historical_source() and all(c14[k]==cfg[k] for k in ('max_steps','batch_size','save_every','device'))
    return dict(data=data,assets=assets,bg=bt['background'],bt=bt,fit=fit,cal=cal,outer=outer,ids=ids,
                seen=seen,stamp=stamp,parent14=parent14,parent12=parent12,c14=c14,reusable=reusable)


@torch.no_grad()
def infer(model, t, b, ids, reference, device):
    model.eval(); out=[]
    for start in range(0,len(t),512):
        ix=slice(start,start+512)
        out.append(model(torch.as_tensor(t[ix],device=device),torch.as_tensor(b[ix],device=device),
                         torch.as_tensor(ids[ix],device=device),torch.as_tensor(reference[ix],device=device)).cpu().numpy())
    result=np.concatenate(out).astype(np.float32)
    if not np.isfinite(result).all(): raise FloatingPointError('Nonfinite model output')
    return result


def train_head(t,b,ids,reference,offset,data,basis,scale,fit,cal,cfg,arm,variant,dest,stamp):
    seed_all(cfg['seed']); device=cfg['device']; spec=ARMS[arm]
    model=ExperimentalHead(t.shape[1],b.shape[1],int(ids.max())+1,len(basis),spec,cfg['seed']).to(device)
    lr={'lr_1e4':1e-4,'lr_1e3':1e-3}.get(variant,5e-4)
    wd={'wd_1e3':1e-3,'wd_1e1':1e-1}.get(variant,.01)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=wd)
    # Only fit/calibration labels are materialized as training tensors.
    coef=np.zeros((len(t),len(basis)),np.float32)
    coef[np.r_[fit,cal]]=(data['delta'][np.r_[fit,cal]]-offset[np.r_[fit,cal]])@basis.T/scale
    tx,bx,ix,rx,y=[torch.as_tensor(v,device=device) for v in (t,b,ids,reference,coef)]
    fw=row_weights(data['meta'].iloc[fit]); cw=row_weights(data['meta'].iloc[cal])
    cw_tensor=torch.as_tensor(cw,dtype=torch.float32,device=device)
    fields=[c for c in ('context','dataset','batch','stimulus','dose','time','modality') if c in data['meta']]
    groups=pd.factorize(pd.MultiIndex.from_frame(data['meta'][fields].astype(str)))[0]
    gx=torch.as_tensor(groups,device=device); px=torch.as_tensor(data['pert_idx'],device=device)
    min_norm=max(float(np.quantile(np.linalg.norm(coef[fit],axis=1),.1)),1e-8)
    gene_weight=1/np.maximum(np.std(data['delta'][fit],axis=0),.05)
    gene_weight=np.clip(gene_weight/np.mean(gene_weight),.5,2).astype(np.float32)
    gw=torch.as_tensor(gene_weight,device=device); basis_t=torch.as_tensor(basis,device=device)
    full_train=torch.as_tensor((data['delta'][fit]-offset[fit])/scale,device=device)
    fit_lookup=np.full(len(t),-1,int);fit_lookup[fit]=np.arange(len(fit))
    first=0;best=float('inf');best_step=0;best_state=None;history=[];contrastive_rows=0;elapsed=0.
    if (dest/'last.pt').exists():
        saved=torch.load(dest/'last.pt',map_location='cpu',weights_only=True)
        if saved['fingerprint']!=stamp:raise ValueError('Checkpoint fingerprint mismatch')
        model.load_state_dict(saved['state']);optimizer.load_state_dict(saved['optimizer'])
        first,best,best_step=saved['step'],saved['best'],saved['best_step']
        best_state,history=saved['best_state'],saved['history'];contrastive_rows=saved['contrastive_rows'];elapsed=saved['elapsed']
    if device=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    started=time.perf_counter()
    for step in range(first,cfg['max_steps']):
        rows=np.random.default_rng(cfg['seed']+9000+step).choice(fit,cfg['batch_size'],p=fw)
        optimizer.zero_grad(set_to_none=True);pred=model(tx[rows],bx[rows],ix[rows],rx[rows]);truth=y[rows]
        loss=(pred-truth).square().mean()
        if variant=='huber':loss=F.huber_loss(pred,truth,delta=1.)
        elif variant=='mse_cosine':
            valid=truth.norm(dim=1)>min_norm
            if valid.any():loss=loss+.05*(1-F.cosine_similarity(pred[valid],truth[valid],dim=1,eps=1e-8)).mean()
        elif variant=='mse_target_contrastive':
            aux,count=contrastive_loss(pred,truth,gx[rows],px[rows],min_norm);loss=loss+.05*aux;contrastive_rows+=count
        elif variant=='gene_scale_robust':
            # Equal weights recover coefficient-MSE gradient scale for an orthonormal basis.
            loss=(((pred@basis_t-full_train[fit_lookup[rows]])**2)*gw).mean()*data['delta'].shape[1]/len(basis)
        elif variant not in ('mse_reference','lr_1e4','lr_1e3','wd_1e3','wd_1e1'):raise ValueError('Unsupported training variant')
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        done=step+1
        if done==1 or done%100==0:print(f'ROUND15 {arm}/{variant} step={done}/{cfg["max_steps"]} loss={loss.item():.6f}',flush=True)
        if done%cfg['save_every']==0 or done==cfg['max_steps']:
            prediction=infer(model,t[cal],b[cal],ids[cal],reference[cal],device)@basis*scale+offset[cal]
            raw_validation=float(cw@np.square(prediction-data['delta'][cal]).mean(1))
            # Same operation order and selection rule as Round14. For each fixed
            # orthonormal basis/offset this is affine-equivalent to raw-gene MSE.
            with torch.no_grad():
                validation=float((cw_tensor*(model(tx[cal],bx[cal],ix[cal],rx[cal])-y[cal]).square().mean(1)).sum())
            if not np.isfinite(validation) or not np.isfinite(raw_validation):raise FloatingPointError('Nonfinite calibration')
            if validation<best:
                best,best_step=validation,done;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            history.append({'step':done,'train_loss':float(loss.item()),'calibration_mse':raw_validation,'calibration_coefficient_mse':validation})
            atomic_torch_save({'fingerprint':stamp,'state':model.state_dict(),'optimizer':optimizer.state_dict(),
                'step':done,'best':best,'best_step':best_step,'best_state':best_state,'history':history,
                'contrastive_rows':contrastive_rows,'elapsed':elapsed+time.perf_counter()-started},dest/'last.pt')
            pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
            write_json(dest/'progress.json',{'step':done,'max_steps':cfg['max_steps'],'best_step':best_step})
    if device=='cuda':torch.cuda.synchronize()
    model.load_state_dict(best_state)
    atomic_torch_save({'state':best_state,'spec':spec,'target_dim':t.shape[1],'background_dim':b.shape[1],
        'n_ids':int(ids.max())+1,'rank':len(basis),'seed':cfg['seed'],'fingerprint':stamp},dest/'model.pt')
    write_json(dest/'selection.json',{'best_step':best_step,'calibration_coefficient_mse':best,'selection':'source calibration coefficient MSE; fixed orthonormal basis, affine-equivalent to gene MSE',
        'variant':variant,'learning_rate':lr,'weight_decay':wd,'minimum_direction_norm':min_norm,
        'contrastive_eligible_rows':contrastive_rows,'condition_fields':fields})
    write_json(dest/'resources.json',{'training_wall_seconds':elapsed+time.perf_counter()-started,
        'optimizer_steps':cfg['max_steps'],'examples_seen':cfg['max_steps']*cfg['batch_size'],
        'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'note':'wall time includes checkpoint I/O; concurrent worker times are not additive GPU hours'})
    np.savez_compressed(dest/'loss_parameters.npz',gene_weights=gene_weight)
    return model


def reconstruct(path,row_ids=None,genes=None):
    with np.load(path,allow_pickle=False) as f:
        if row_ids is not None and not np.array_equal(f['row_ids'],row_ids):raise ValueError('Prediction row alignment mismatch')
        if genes is not None and not np.array_equal(f['genes'],genes):raise ValueError('Prediction gene alignment mismatch')
        result=f['prediction'] if 'prediction' in f else f['coefficients']@f['basis']*f['scale']+f['offset']
        if result.shape!=(len(f['row_ids']),len(f['genes'])):raise ValueError('Prediction shape mismatch')
        if not np.isfinite(result).all():raise ValueError('Nonfinite saved predictions')
        return result.astype(np.float32)


def worker(cfg,resume=False):
    torch.set_num_threads(2);dest=Path(cfg['output_dir']);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'manifest.json').exists() and not resume:raise ValueError('Use --resume or a new name')
    d=prepare_inputs(cfg,dest)
    if (dest/'SCREEN_COMPLETE.json').exists():verify_marker(dest,'SCREEN_COMPLETE.json');return dest
    data=d['data'];fit,cal,outer=d['fit'],d['cal'],d['outer'];meta=data['meta'].iloc[outer].reset_index(drop=True)
    truth=data['delta'][outer];scale=float(d['bt']['response_scale']);records=[];ranks=[];diagnostics=[];states=[]
    threshold=max(float(np.quantile(np.abs(data['delta'][fit]),.9)),1e-6)
    bank_cache={};feature_cache={};basis_cache={32:d['bt']['basis']}
    coverage=(d['assets']['relations'][:,2,:].sum(1)>0)[data['pert_idx'][outer]]
    schedule=[('P0' if a=='id_background_fixed' else 'P1',a,'mse_reference',a) for a in cfg['arms']]
    schedule += [('P2',cfg['training_parent'],v,'training_'+v) for v in cfg['variants']]
    # The repaired control runs before any dependent interpretation.
    schedule.sort(key=lambda x: {'P0':0,'P1':1,'P2':2}[x[0]])
    def append_metrics(name,pred,available=None):
        if available is None:available=coverage
        rows=score(meta,truth,pred,name,available);rr=retrieval(meta,truth,pred,name)
        dd=response_diagnostics(meta,truth,pred,name,threshold)
        records.extend(rows);ranks.extend(rr);diagnostics.extend(dd)
        return rows,rr,dd
    reference,_=reference_responses(data,fit)
    append_metrics('zero',np.zeros_like(truth));append_metrics('mean_transfer',reference[data['pert_idx'][outer]])
    for name in ('string_ridge','id_ridge','none_ridge'):
        pred=reconstruct(d['parent12']/name/'predictions.npz',meta.row_id.to_numpy(dtype='U'),data['genes'])
        append_metrics('r12_'+name,pred)
    for stage,arm,variant,name in schedule:
        folder=dest/name;folder.mkdir(exist_ok=True)
        state={'stage':stage,'arm':arm,'variant':variant,'method':name,'status':'RUNNING'}
        reason=BLOCKED.get(arm) or BLOCKED.get(variant)
        if reason:
            state.update(status='BLOCKED',reason=reason);write_json(folder/'BLOCKED.json',state);states.append(state);continue
        if (folder/'COMPLETE.json').exists():
            done=verify_marker(folder,'COMPLETE.json')
            if done['fingerprint']!=d['stamp']:raise ValueError('Completed head identity mismatch')
            for file,target in [('per_target.csv',records),('retrieval.csv',ranks),('response_diagnostics.csv',diagnostics)]:
                target.extend(pd.read_csv(folder/file).to_dict('records'))
            states.append(done['state']);continue
        reused=None
        if stage=='P2' and variant=='mse_reference' and (dest/arm/'COMPLETE.json').exists():reused=dest/arm
        elif stage!='P2' and arm in REUSE and d['reusable'] and (d['parent14']/REUSE[arm]/'COMPLETE.json').exists():reused=d['parent14']/REUSE[arm]
        if reused:
            verify_files(reused)
            prediction_path=reused/'predictions.npz'
            if not prediction_path.exists() and (reused/'coefficients.npz').exists():
                prediction_path=reused/'coefficients.npz'
            elif not prediction_path.exists():
                ref=json.loads((reused/'REUSED.json').read_text());prediction_path=Path(ref['prediction_path'])
                if file_sha256(prediction_path)!=ref['prediction_sha256']:raise ValueError('Reused reference changed')
            pred=reconstruct(prediction_path,meta.row_id.to_numpy(dtype='U'),data['genes'])
            write_json(folder/'REUSED.json',{'parent':str(reused),'parent_complete_sha256':file_sha256(reused/'COMPLETE.json'),
                'prediction_path':str(prediction_path),'prediction_sha256':file_sha256(prediction_path),'historical_source':historical_source()})
            state['status']='REUSED';selection=json.loads((reused/'selection.json').read_text())
            write_json(folder/'selection.json',selection)
            rows,rr,dd=append_metrics(name,pred)
        else:
            spec=ARMS[arm];kind=spec['representation'];feature_kind='local' if kind=='coverage' else kind
            if feature_kind not in feature_cache:feature_cache[feature_kind]=target_representation(data,d['assets'],fit,feature_kind)
            tf,transform,available=feature_cache[feature_kind];tf=tf.copy()
            if kind=='coverage':tf[:,:-1]=0
            t=tf[data['pert_idx']];b=d['bg'].copy()
            if spec['background']=='zero':b[:]=0
            elif spec['background']=='control_pca':
                _,unique=np.unique(data['baseline'][fit],axis=0,return_index=True)
                b,btransform=projection(data['baseline'],fit[unique],d['bg'].shape[1])
                np.savez_compressed(folder/'control_pca.npz',**btransform)
            if spec['rank'] not in basis_cache:
                basis_cache[spec['rank']],audit=response_basis(data,fit,spec['rank']);write_json(dest/f'basis_rank{spec["rank"]}.json',audit)
            basis=basis_cache[spec['rank']];offset=np.zeros_like(data['delta']);present=np.zeros(len(t),bool)
            if spec['reference']:
                mode=spec['reference']
                if mode not in bank_cache:
                    if mode=='ridge':
                        base,audit=ridge_reference(data,d['assets'],d['bg'],fit,np.r_[cal,outer]);found=np.ones(len(t),bool)
                    else:base,found,audit=reference_bank(data,fit,d['assets']['relations'][:,2,:],mode)
                    bank_cache[mode]=(base,found)
                    write_json(dest/f'{mode}_donors.json',audit)
                offset,present=bank_cache[mode]
            ref_input=np.c_[offset@basis.T/scale,present.astype(np.float32)].astype(np.float32)
            print(f'ROUND15 HEAD {cfg["protocol"]} seed={cfg["seed"]} {name}',flush=True)
            model=train_head(t,b,d['ids'],ref_input,offset,data,basis,scale,fit,cal,cfg,arm,variant,folder,d['stamp'])
            coefficients=infer(model,t[outer],b[outer],d['ids'][outer],ref_input[outer],cfg['device'])
            # Save one explicit representation from which exactly the scored float32 matrix is rebuilt.
            np.savez_compressed(folder/'coefficients.npz',coefficients=coefficients,basis=basis.astype(np.float32),
                scale=np.float32(scale),offset=offset[outer].astype(np.float32),row_ids=meta.row_id.to_numpy(dtype='U'),genes=data['genes'])
            pred=reconstruct(folder/'coefficients.npz',meta.row_id.to_numpy(dtype='U'),data['genes'])
            np.savez_compressed(folder/'transforms.npz',**transform,fit_target_indices=d['seen'])
            rows,rr,dd=append_metrics(name,pred,available[data['pert_idx'][outer]])
            state['status']='COMPLETED';del model
            if cfg['device']=='cuda':torch.cuda.empty_cache()
        for file,values in [('per_target',rows),('retrieval',rr),('response_diagnostics',dd)]:pd.DataFrame(values).to_csv(folder/(file+'.csv'),index=False)
        seal(folder,fingerprint=d['stamp'],state=state);states.append(state)
        write_json(dest/'head_status.json',{'heads':states,'current':name})
    for file,values in [('per_target',records),('retrieval',ranks),('response_diagnostics',diagnostics)]:pd.DataFrame(values).to_csv(dest/(file+'.csv'),index=False)
    write_json(dest/'head_status.json',{'heads':states})
    seal(dest,'SCREEN_COMPLETE.json',fingerprint=d['stamp'],screen_finished=True,
         blocked=sum(s['status']=='BLOCKED' for s in states),test_evaluated=False)
    return dest


def paired_report(frame,value,pairs,smaller=True):
    """Paired target bootstrap, respecting equal-context macro weights."""
    averaged=frame.groupby(['protocol','method','context','target'])[value].mean();out=[]
    for protocol in frame.protocol.unique():
        for left,right in pairs:
            aligned=pd.concat([averaged.loc[(protocol,left)].rename('a'),averaged.loc[(protocol,right)].rename('b')],axis=1)
            if aligned.isna().any().any():raise ValueError('Unpaired comparison')
            gain=((aligned.a-aligned.b) if smaller else (aligned.b-aligned.a)).unstack('context')
            v=gain.to_numpy();rng=np.random.default_rng(1701);draws=[]
            for start in range(0,2000,200):
                sample=v[rng.integers(len(v),size=(min(200,2000-start),len(v)))];count=np.isfinite(sample).sum(1)
                valid=(count>0).all(1)
                means=np.nansum(sample[valid],axis=1)/count[valid]
                draws.extend(means.mean(1).tolist())
            if not draws:raise ValueError('No valid paired bootstrap draws')
            low,high=np.quantile(draws,[.025,.975])
            # Largest contribution to the actual equal-context/within-context-target mean.
            contribution=gain.div(gain.count(),axis=1).sum(1)/gain.shape[1]
            largest=contribution.idxmax();without=gain.drop(index=largest)
            remaining=float(without.mean().mean()) if len(without) and without.count().gt(0).all() else np.nan
            out.append(dict(protocol=protocol,baseline=left,candidate=right,metric=value,
                gain=float(gain.mean().mean()),ci_low=low,ci_high=high,targets=len(gain),
                target_wins=int((gain.mean(1)>0).sum()),largest_gain_target=largest,
                largest_macro_contribution=float(contribution.loc[largest]),gain_without_largest_target=remaining,
                valid_bootstrap_draws=len(draws)))
    return pd.DataFrame(out)


def report(specs,root):
    root=Path(root);frames={k:[] for k in ('per_target','retrieval','response_diagnostics')};statuses=[]
    for cfg in specs.values():
        dest=Path(cfg['output_dir']);verify_marker(dest,'SCREEN_COMPLETE.json')
        for name in frames:
            f=pd.read_csv(dest/(name+'.csv'));f['protocol']=cfg['protocol'];f['seed']=cfg['seed'];frames[name].append(f)
        for s in json.loads((dest/'head_status.json').read_text())['heads']:statuses.append({**s,'protocol':cfg['protocol'],'seed':cfg['seed']})
    frames={k:pd.concat(v,ignore_index=True) for k,v in frames.items()}
    for name,f in frames.items():f.to_csv(root/(name+'.csv'),index=False)
    f=frames['per_target'];r=frames['retrieval'];keys=['protocol','method','seed']
    f.groupby(keys+['context','annotation_available']).agg(mse=('mse','mean'),targets=('target','size')).reset_index().to_csv(root/'coverage_scores.csv',index=False)
    context=f.groupby(keys+['context']).mse.mean().rename('mse').reset_index()
    zero=context[context.method=='zero'][['protocol','seed','context','mse']].rename(columns={'mse':'zero_mse'})
    context=context.merge(zero,on=['protocol','seed','context'],validate='many_to_one')
    context['nmse']=context.mse/np.where(context.zero_mse>1e-10,context.zero_mse,np.nan)
    retrieval_scores=r.groupby(keys+['context'])[['specificity_gap','top1_credit','top5_credit']].mean().reset_index()
    context=context.merge(retrieval_scores,on=keys+['context'],validate='one_to_one');context.to_csv(root/'context_scores.csv',index=False)
    per_seed=context.groupby(keys)[['mse','nmse','specificity_gap','top1_credit','top5_credit']].mean().reset_index()
    per_seed.to_csv(root/'comparison.csv',index=False)
    summary=per_seed.groupby(['protocol','method']).agg(mse_mean=('mse','mean'),seed_sd=('mse','std'),nmse=('nmse','mean'),
        specificity_gap=('specificity_gap','mean'),top1=('top1_credit','mean'),top5=('top5_credit','mean')).reset_index()
    summary.to_csv(root/'comparison_mean.csv',index=False)
    methods=set(f.method);pairs=[('zero',m) for m in sorted(methods) if m!='zero']
    pairs += [('background_only','id_background_fixed'),('local_additive_noid','local_factor_noid'),
              ('local_additive_id','local_factor_id'),('r12_string_ridge','local_factor_noid')]
    pairs += [(f'shuffle{j}_factor_noid','local_factor_noid') for j in (1,2,3)]
    pairs += [('local_factor_noid',m) for m in methods if m not in ('local_factor_noid','zero') and not m.startswith('r12_')]
    pairs += [('training_mse_reference','training_'+v) for v in VARIANTS if v!='mse_reference']
    pairs=list(dict.fromkeys((a,b) for a,b in pairs if a in methods and b in methods))
    paired=paired_report(f,'mse',pairs);paired.to_csv(root/'paired_comparisons.csv',index=False)
    paired_report(r,'specificity_gap',pairs,smaller=False).to_csv(root/'paired_specificity.csv',index=False)
    pd.DataFrame(statuses).to_csv(root/'execution_status.csv',index=False)
    counts=pd.Series([s['status'] for s in statuses]).value_counts().to_dict()
    write_json(root/'screen_summary.json',{'counts':counts,'test_evaluated':False,'candidate_selection':'not performed; no independent confirmation claimed',
        'future_stages':'P3/P4/P5/P6/FS are not executed by this core-screen command'})
    scope={'raw_predictions_only':True,'test_evaluated':False,'Jurkat_scored':False,
        'P2_parent':'local_factor_noid predeclared from Round14 development results; not selected separately per loss',
        'bootstrap':'2000 target resamples after averaging optimizer seeds; exploratory, uncorrected multiplicity',
        'replication':'optimizer seeds on the same development split are not independent biological datasets',
        'reference_labels':'fit only; Ridge target-OOF; same-target reference excludes query context; neighbor/shared banks exclude query target',
        'background_OOF':'frozen outer-fit control-only encoder; no perturbation labels in encoder; inner OOF target PCA refitted',
        'storage':'new predictions reconstructed from float32 coefficients, basis, scale, offset and exact row/gene IDs',
        'unsupported':'blocked heads have BLOCKED.json and no COMPLETE.json; SCREEN/CORE completion applies only to the available screen'}
    write_json(root/'evaluation_scope.json',scope)
    (root/'report.html').write_text('<html><meta charset="utf-8"><h1>Round15 core screen</h1><p>Development only. Future stages remain gated.</p>'
        +pd.DataFrame(statuses).to_html(index=False)+summary.to_html(index=False)+'<h2>Exploratory paired comparisons</h2>'
        +paired.to_html(index=False),encoding='utf-8')
