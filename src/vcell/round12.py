"""Target representation benchmarks with fit-only bases and fixed evaluation rows."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import svd, solve
import torch
from torch import nn

from .background_training import control_encoder, fit_normalization, fingerprint
from .data import load_prepared
from .fixed_training import response_basis
from .knowledge import load_knowledge
from .round8 import validate_partition
from .round11 import freeze, verify_files
from .selection import row_weights, mse
from .specialization import reference_responses
from .train import source_fingerprint
from .utils import file_sha256, write_json, seed_all, atomic_torch_save

REPRESENTATIONS = ('none', 'id', 'random', 'text', 'string', 'shuffled_string_1',
                   'shuffled_string_2', 'shuffled_string_3')
PROTOCOLS = ('target', 'context', 'double')


def order_targets(values, salt):
    return sorted(set(map(str, values)), key=lambda x: hashlib.sha256((salt+':'+x).encode()).hexdigest())


def make_partition(old, data, protocol):
    """Identifier-only splits anchored on old rows; expansion cannot alter evaluation.

    Context: HepG2, fit-seen targets. Double: H1, targets excluded in every
    source context. Target: old source rows, globally held-out targets.
    Calibration targets are also excluded from fitting in all contexts.
    """
    if protocol not in PROTOCOLS: raise ValueError('Unknown protocol')
    m = old['meta']; current = data['meta']; train = m.split.eq('train')
    source = train & (m.context.ne('H1_hESC') if protocol == 'double' else True)
    outer_targets = set(m.loc[train & m.context.eq('H1_hESC'), 'perturbation']) if protocol == 'double' else set()
    candidates = order_targets(set(m.loc[source, 'perturbation']) - outer_targets, 'round12:817')
    if len(candidates) < 10: raise ValueError('Need >=10 source targets for explicit fit/calibration separation')
    n = max(2, len(candidates)//5)
    if protocol == 'target':
        outer_targets = set(candidates[:n]); calibration = set(candidates[n:2*n])
        outer_mask = train & m.perturbation.isin(outer_targets)
    else:
        calibration = set(candidates[:n])
        fit_targets = set(candidates) - calibration
        outer_mask = (m.context.eq('H1_hESC') & train if protocol == 'double' else
                      m.context.eq('HepG2') & m.perturbation.isin(fit_targets))
    fit_mask = current.split.eq('train') & ~current.perturbation.isin(calibration | outer_targets)
    if protocol == 'double': fit_mask &= current.context.ne('H1_hESC')
    lookup = dict(zip(current.row_id, range(len(current))))
    if not set(m.row_id) <= set(lookup): raise ValueError('Historical rows missing from expanded data')
    parts = {'fit': np.flatnonzero(fit_mask).tolist(),
             'calibration': [lookup[x] for x in m.loc[source & m.perturbation.isin(calibration), 'row_id']],
             'outer': [lookup[x] for x in m.loc[outer_mask, 'row_id']]}
    validate_partition(data, parts)
    fit_p = set(current.iloc[parts['fit']].perturbation)
    cal_p = set(current.iloc[parts['calibration']].perturbation)
    eval_p = set(current.iloc[parts['outer']].perturbation)
    if fit_p & cal_p: raise ValueError('Calibration target leakage')
    if protocol in ('target','double') and fit_p & eval_p: raise ValueError('Global target leakage')
    if protocol == 'context' and not eval_p <= fit_p: raise ValueError('Context benchmark must use fit-seen targets')
    fit_c = set(current.iloc[parts['fit']].context); eval_c = set(current.iloc[parts['outer']].context)
    if protocol == 'target' and not eval_c <= fit_c: raise ValueError('Target benchmark has unknown contexts')
    if protocol != 'target' and fit_c & eval_c: raise ValueError('Context leakage')
    return parts


def assert_extension(old, expanded):
    n = len(old['meta'])
    if not np.array_equal(old['genes'], expanded['genes']): raise ValueError('Panel changed')
    for key in ('baseline', 'delta'):
        if not np.array_equal(old[key], expanded[key][:n]): raise ValueError('Historical expression rows changed')
    for key in ('row_id','context','perturbation','split','batch'):
        if not np.array_equal(old['meta'][key], expanded['meta'][key].iloc[:n]): raise ValueError('Historical metadata changed')
    extra = expanded['meta'].iloc[n:]
    if extra.empty or not extra.context.isin(['K562','RPE1']).all() or not extra.split.eq('train').all():
        raise ValueError('Expansion must add only training K562/RPE1 rows')


def projection(x, fit, dim):
    """Fit-only, padded low-rank feature projection; constant inputs remain zero."""
    x = np.asarray(x, np.float64); fit = np.asarray(fit, int)
    mean = x[fit].mean(0); centered = x[fit] - mean
    _, s, vt = svd(centered, full_matrices=False, check_finite=True)
    count = min(dim, int((s > 1e-8).sum()))
    axes = np.zeros((dim, x.shape[1])); axes[:count] = vt[:count]
    values = (x-mean)@axes.T
    scale = np.maximum(values[fit].std(0), .1)
    return (values/scale).astype(np.float32), {'mean': mean, 'axes': axes, 'scale': scale}


def target_features(data, knowledge, fit, representation, seed, dim=32):
    n = len(data['perturbations']); seen = np.unique(data['pert_idx'][fit])
    if representation == 'none':
        return np.zeros((n,dim),np.float32), {'definition':'no target identity; background-only'}, np.ones(n,bool)
    if representation == 'id':
        values = np.zeros((n,len(seen)), np.float32); values[seen,np.arange(len(seen))] = 1
        return values, {'fit_ids':seen, 'definition':'one-hot fit targets; unseen all-zero; capacity not dimension-matched'}, np.isin(np.arange(n),seen)
    if representation == 'random':
        # Keyed by gene, so expanding the vocabulary never changes old controls.
        values = np.stack([np.random.default_rng(int.from_bytes(hashlib.sha256(f'{seed}:{g}'.encode()).digest()[:8],'little')).normal(size=dim)
                           for g in data['perturbations']]).astype(np.float32)
        return values, {'definition':'fixed label-free gene-keyed random vectors'}, np.ones(n,bool)
    if representation == 'text':
        raw = knowledge['semantic']; present = knowledge.get('text_available',np.linalg.norm(raw,axis=1)>0)
    else:
        raw = knowledge['relations'][:,2,:].copy()
        if representation.startswith('shuffled_string_'):
            if representation in knowledge:
                raw = knowledge[representation]
                perm = knowledge[representation+'_mapping']
            else:
                perm = np.random.default_rng(817+int(representation.rsplit('_',1)[1])).permutation(n)
                raw = raw[perm]
        elif representation != 'string': raise ValueError('Unknown representation')
        present = raw.sum(1)>0
    values, transform = projection(raw, seen, dim)
    # Missing annotation is explicit. It is not a fabricated all-negative effect.
    values = np.c_[values, present.astype(np.float32)]
    transform['definition'] = 'frozen gene text' if representation=='text' else 'STRING v12 top-16 functional neighbors on historical measured panel; not full-network embeddings'
    if representation.startswith('shuffled_'): transform['permutation']=perm
    return values, transform, present


def design(background, targets, interactions=True):
    if not interactions:
        return np.c_[background, targets].astype(np.float32)
    interaction = (background[:,:,None]*targets[:,None,:]).reshape(len(background),-1)
    return np.c_[background, targets, interaction].astype(np.float32)


def fit_ridge(x, y, weights, alpha):
    weights = np.asarray(weights,np.float64); weights /= weights.sum()
    xm = weights@x; ym = weights@y; xc = x-xm
    coef = solve(xc.T@(weights[:,None]*xc)+alpha*np.eye(x.shape[1]),
                 xc.T@(weights[:,None]*(y-ym)), assume_a='pos')
    return coef, ym-xm@coef


def correlations(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    a=a-a.mean(-1,keepdims=True);b=b-b.mean(-1,keepdims=True)
    return (a*b).sum(-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-12)


def score(meta, truth, prediction, method, present=None):
    records=[]
    for (context,target),ix in meta.groupby(['context','perturbation'],sort=True).indices.items():
        records.append({'context':context,'target':target,'method':method,'rows':len(ix),
                        'mse':float(np.square(prediction[ix]-truth[ix]).mean()),
                        'pearson':float(correlations(prediction[ix].mean(0),truth[ix].mean(0))),
                        'annotation_available':bool(present[ix].all()) if present is not None else True})
    return records


def specificity(meta, truth, prediction, method):
    records=[]
    for context, group in meta.groupby('context',sort=True):
        ix=meta.index.get_indexer(group.index); local=meta.iloc[ix].reset_index(drop=True)
        groups=local.groupby('perturbation',sort=True).indices
        names=list(groups); y=np.stack([truth[ix[j]].mean(0) for j in groups.values()])
        p=np.stack([prediction[ix[j]].mean(0) for j in groups.values()])
        # Avoid materializing targets x targets x genes (several GiB at full scale).
        p=p.astype(np.float64);y=y.astype(np.float64)
        p-=p.mean(1,keepdims=True);y-=y.mean(1,keepdims=True)
        p/=np.maximum(np.linalg.norm(p,axis=1,keepdims=True),1e-12)
        y/=np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12)
        sim=p@y.T; n=len(names)
        for j,target in enumerate(names):
            own=sim[j,j]; higher=int((sim[j]>own+1e-10).sum()); tied=int(np.isclose(sim[j],own,atol=1e-10,rtol=0).sum())
            records.append({'context':context,'target':target,'method':method,'candidates':n,
                            'own_pearson':float(own),'other_pearson':float(np.delete(sim[j],j).mean()) if n>1 else 0.,
                            'midrank':1+higher+(tied-1)/2,'top1_credit':1/tied if higher==0 else 0.})
    return records


def save_complete(dest, extra=None):
    files={p.relative_to(dest).as_posix():file_sha256(p) for p in sorted(dest.rglob('*'))
           if p.is_file() and p.name not in ('COMPLETE.json','last.pt') and not p.name.endswith('.tar.gz')}
    write_json(dest/'COMPLETE.json', {'files':files, **(extra or {})})


@torch.no_grad()
def encode_background(data,cfg):
    folder=Path(cfg['encoder']).parent;verify_files(folder)
    saved=torch.load(cfg['encoder'],map_location='cpu',weights_only=True)
    if saved['partition']!=cfg['partition'] or saved['genes']!=data['genes'].tolist() or saved['background']!='b2' or saved['family']!='qwen':
        raise ValueError('Background encoder partition/panel/family mismatch')
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['data']!=data['audit']['fingerprint'] or manifest['config']['seed']!=cfg['seed']:
        raise ValueError('Background data/seed mismatch')
    norm=fit_normalization(data,cfg['partition'])
    for k in norm: np.testing.assert_allclose(saved['normalization'][k].numpy(),norm[k],rtol=1e-6,atol=1e-8)
    encoder=control_encoder(len(data['genes']),'qwen',cfg['model']).to(cfg['device'])
    encoder.load_state_dict(saved['state']);encoder.eval()
    outputs=[]
    for start in range(0,len(data['meta']),256):
        x=torch.tensor((data['baseline'][start:start+256]-norm['mean'])/norm['std'],device=cfg['device'])
        outputs.append(encoder(x).reshape(len(x),cfg['model']['control_tokens'],-1).mean(1).cpu().numpy())
    return np.concatenate(outputs),norm


def mlp_fit(x,y,fit,weights,cfg,dest,stamp):
    seed_all(cfg['seed']); device=cfg['device']
    model=nn.Sequential(nn.Linear(x.shape[1],128),nn.GELU(),nn.Linear(128,y.shape[1])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    first=0
    if (dest/'last.pt').exists():
        saved=torch.load(dest/'last.pt',map_location='cpu',weights_only=True)
        if saved['fingerprint']!=stamp: raise ValueError('MLP checkpoint mismatch')
        model.load_state_dict(saved['state']);optimizer.load_state_dict(saved['optimizer']);first=saved['step']
    tx=torch.tensor(x,device=device);ty=torch.tensor(y,device=device)
    for step in range(first,cfg['max_steps']):
        indices=np.random.default_rng(cfg['seed']+step+9000).choice(fit,cfg['batch_size'],p=weights)
        optimizer.zero_grad(set_to_none=True);loss=(model(tx[indices])-ty[indices]).square().mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite MLP loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step==first or (step+1)%100==0: print(f'ROUND12 {dest.name} step={step+1}/{cfg["max_steps"]} loss={loss.item():.6f}',flush=True)
        if (step+1)%cfg['save_every']==0 or step+1==cfg['max_steps']:
            atomic_torch_save({'fingerprint':stamp,'state':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1},dest/'last.pt')
            write_json(dest/'progress.json',{'step':step+1,'max_steps':cfg['max_steps']})
    model.eval()
    with torch.no_grad(): prediction=np.concatenate([model(tx[i:i+512]).cpu().numpy() for i in range(0,len(tx),512)])
    atomic_torch_save({'state':{k:v.cpu() for k,v in model.state_dict().items()},'input_dim':x.shape[1],
                      'output_dim':y.shape[1],'fingerprint':stamp},dest/'model.pt')
    return prediction


def worker(cfg,resume=False):
    torch.set_num_threads(2)
    data=load_prepared(cfg['data_dir']); assets=load_knowledge(cfg['knowledge'],data)
    dest=Path(cfg['output_dir']);dest.mkdir(parents=True,exist_ok=True)
    identity={'config':cfg,'source':source_fingerprint(),'data':data['audit']['fingerprint'],
              'knowledge':file_sha256(cfg['knowledge']),'encoder':file_sha256(cfg['encoder'])}
    stamp=fingerprint(identity)
    if (dest/'manifest.json').exists() and not resume: raise ValueError('Use --resume or new run')
    freeze(dest/'manifest.json',{'fingerprint':stamp,**identity})
    if (dest/'COMPLETE.json').exists(): verify_files(dest);return dest
    parts=cfg['partition'];validate_partition(data,parts)
    fit,cal,outer=[np.asarray(parts[k],int) for k in ('fit','calibration','outer')]
    basis,basis_audit=response_basis(data,fit,cfg['rank']);write_json(dest/'basis_audit.json',basis_audit)
    latent,norm=encode_background(data,cfg)
    bg,bg_transform=projection(latent,fit,cfg['background_dim'])
    coeff=data['delta']@basis.T/norm['scale'];weights=row_weights(data['meta'].iloc[fit]);weights/=weights.sum()
    meta=data['meta'].iloc[outer].copy().reset_index(drop=True);truth=data['delta'][outer]
    ref,seen=reference_responses(data,fit)
    records=[];specific=[]
    for method,pred in [('zero',np.zeros_like(truth)),('mean_transfer',ref[data['pert_idx'][outer]]),
                        ('basis_oracle',truth@basis.T@basis)]:
        records+=score(meta,truth,pred,method)
        if method!='basis_oracle':specific+=specificity(meta,truth,pred,method)
    # Saved numerical transforms are sufficient for reproducing the small head.
    np.savez_compressed(dest/'background_transform.npz',**bg_transform,basis=basis,response_scale=norm['scale'])
    for rep in cfg['representations']:
        target,transform,present=target_features(data,assets,fit,rep,cfg['seed'],cfg['target_dim'])
        # The one-hot control is additive: its feature count grows with targets.
        # Avoid an O(targets * background_dim) high-capacity memorization head.
        x=design(bg,target[data['pert_idx']],interactions=rep!='id');xmean=weights@x[fit]
        xscale=np.maximum(np.sqrt(weights@np.square(x[fit]-xmean)),.1)
        x=((x-xmean)/xscale).astype(np.float32)
        for predictor in cfg['predictors']:
            name=rep+'_'+predictor;folder=dest/name;folder.mkdir(exist_ok=True)
            if (folder/'COMPLETE.json').exists():
                verify_files(folder)
                if json.loads((folder/'COMPLETE.json').read_text())['fingerprint']!=stamp: raise ValueError('Head stamp changed')
                records+=pd.read_csv(folder/'per_target.csv').to_dict('records')
                specific+=pd.read_csv(folder/'specificity.csv').to_dict('records');continue
            print(f'ROUND12 HEAD: {cfg["variant"]}/{cfg["protocol"]} seed={cfg["seed"]} {name}',flush=True)
            np.savez_compressed(folder/'feature_transform.npz',**transform,xmean=xmean,xscale=xscale)
            if predictor=='ridge':
                trials=[];models=[]
                for alpha in cfg['ridge_alphas']:
                    coef,intercept=fit_ridge(x[fit],coeff[fit],weights,alpha)
                    pred=(x[cal]@coef+intercept)@basis*norm['scale']
                    trials.append({'alpha':alpha,'calibration_mse':mse(pred,data['delta'][cal],row_weights(data['meta'].iloc[cal]))})
                    models.append((coef,intercept))
                best=int(np.argmin([t['calibration_mse'] for t in trials]));coef,intercept=models[best]
                prediction=(x[outer]@coef+intercept)@basis*norm['scale']
                np.savez_compressed(folder/'model.npz',coef=coef,intercept=intercept)
                write_json(folder/'selection.json',{'trials':trials,'chosen':trials[best],'selection':'source calibration only'})
            elif predictor=='mlp':
                prediction=mlp_fit(x,coeff,fit,weights,cfg,folder,stamp)[outer]@basis*norm['scale']
            else: raise ValueError('Unknown predictor')
            rows=score(meta,truth,prediction,name,present[data['pert_idx'][outer]])
            ranks=specificity(meta,truth,prediction,name)
            pd.DataFrame(rows).to_csv(folder/'per_target.csv',index=False)
            pd.DataFrame(ranks).to_csv(folder/'specificity.csv',index=False)
            np.savez_compressed(folder/'predictions.npz',prediction=prediction,row_ids=meta.row_id.to_numpy(dtype='U'),genes=data['genes'])
            write_json(folder/'coverage.json',{'fit_targets':int(len(np.unique(data['pert_idx'][fit]))),
                       'outer_targets':int(meta.perturbation.nunique()),'input_dim':x.shape[1],
                       'outer_covered_targets':int(meta.loc[present[data['pert_idx'][outer]],'perturbation'].nunique()),
                       'missing_annotations_retained':True,'encoder_frozen':True})
            save_complete(folder,{'fingerprint':stamp});records+=rows;specific+=ranks
    pd.DataFrame(records).to_csv(dest/'per_target.csv',index=False)
    pd.DataFrame(specific).to_csv(dest/'specificity.csv',index=False)
    save_complete(dest,{'fingerprint':stamp,'test_evaluated':False})
    print(f'ROUND12 WORKER COMPLETE: {dest}',flush=True)
    return dest


def half_diagnostic(folder,output):
    folder,output=Path(folder),Path(output);output.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(folder/'half_metadata.csv')
    with np.load(folder/'halves.npz',allow_pickle=False) as f: a,b=f['a'],f['b']
    if len(meta):
        pd.DataFrame(specificity(meta,a,b,'technical_half_b_to_a')).to_csv(output/'half_specificity.csv',index=False)
        pd.DataFrame(score(meta,a,b,'technical_split_half')).to_csv(output/'half_per_target.csv',index=False)
    write_json(output/'half_scope.json',{'rows':len(meta),'contexts':sorted(meta.context.unique().tolist()),
               'note':'K562/RPE1 only; independent cell halves with disjoint control halves. Technical repeatability, not biological replicates. No H1 raw cells available. Not used for fitting/selection.'})


def report(specs,root):
    root=Path(root);frames=[];ranks=[]
    for cfg in specs.values():
        folder=Path(cfg['output_dir'])
        verify_files(folder)
        for name,dest in [('per_target.csv',frames),('specificity.csv',ranks)]:
            f=pd.read_csv(folder/name)
            for k in ('variant','protocol','seed'):f[k]=cfg[k]
            dest.append(f)
    frame=pd.concat(frames,ignore_index=True);frame.to_csv(root/'per_target.csv',index=False)
    rank=pd.concat(ranks,ignore_index=True);rank.to_csv(root/'target_specificity.csv',index=False)
    keys=['variant','protocol','method','seed']
    # Equal context, equal target inside each context. Never pool by raw row count.
    scores=frame.groupby(keys+['context']).agg(mse=('mse','mean'),pearson=('pearson','mean'),targets=('target','size')).reset_index()
    scores.to_csv(root/'context_scores.csv',index=False)
    summary=scores.groupby(keys).agg(mse=('mse','mean'),pearson=('pearson','mean'),targets=('targets','sum')).reset_index()
    summary.to_csv(root/'comparison.csv',index=False)
    means=summary.groupby(['variant','protocol','method']).agg(mse_mean=('mse','mean'),seed_sd=('mse','std'),seeds=('seed','nunique')).reset_index()
    means.to_csv(root/'comparison_mean.csv',index=False)
    # Pair after averaging optimizer seeds; resample biological target identities
    # jointly across contexts. This is exploratory, with no multiplicity correction.
    paired=[]
    methods=sorted(set(frame.method)-{'basis_oracle'})
    contrasts=[]
    for variant in sorted(frame.variant.unique()):
        for method in methods:
            if method!='zero':contrasts.append((variant,'zero',variant,method,'vs_zero'))
        for predictor in ('ridge','mlp'):
            for rep in ('text','string'):
                for base in ('none','random','shuffled_string_1','shuffled_string_2','shuffled_string_3'):
                    if rep=='text' and base.startswith('shuffled'):continue
                    contrasts.append((variant,base+'_'+predictor,variant,rep+'_'+predictor,'representation'))
    for method in methods:contrasts.append(('original',method,'expanded',method,'expansion'))
    averaged=frame.groupby(['variant','protocol','method','context','target']).mse.mean()
    for protocol in sorted(frame.protocol.unique()):
        for va,ma,vb,mb,label in contrasts:
            try: a=averaged.loc[(va,protocol,ma)];b=averaged.loc[(vb,protocol,mb)]
            except KeyError:continue
            joined=pd.concat([a.rename('a'),b.rename('b')],axis=1)
            if joined.isna().any().any():raise ValueError('Unpaired evaluation targets')
            gain=(joined.a-joined.b).unstack('context'); rng=np.random.default_rng(1701)
            array=gain.to_numpy();draws=[]
            for _ in range(2000):
                sample=array[rng.integers(len(array),size=len(array))]
                if np.all(np.isfinite(sample).any(0)):draws.append(float(np.nanmean(sample,axis=0).mean()))
            ci=np.quantile(draws,[.025,.975])
            per_target=gain.mean(axis=1);drop=per_target.idxmax()
            removed=gain.drop(index=drop).mean().mean() if len(gain)>1 else np.nan
            paired.append({'protocol':protocol,'contrast':label,'baseline':va+'/'+ma,'candidate':vb+'/'+mb,
                           'gain_mse':float(gain.mean().mean()),'ci_low':ci[0],'ci_high':ci[1],
                           'targets':len(gain),'target_wins':int((per_target>0).sum()),
                           'largest_gain_target':drop,'gain_without_largest_target':removed})
    pd.DataFrame(paired).to_csv(root/'paired_comparisons.csv',index=False)
    write_json(root/'evaluation_scope.json',{'test_evaluated':False,'jurkat_scored':False,
        'outer_labels_previously_used':True,'claim':'development diagnostics; no new untouched test',
        'uncertainty':'2000 target bootstrap draws after averaging optimizer seeds; exploratory, not multiple-testing corrected',
        'basis_oracle':'projection ceiling diagnostic only; uses outer truth without fitting any parameters',
        'background':'new fold-specific B2 projector pretraining, then frozen; no historical supervised checkpoint reuse',
        'deferred':['Qwen end-to-end fusion','contrastive loss','confidence gate','ESM','new external dataset release']})
    html='<html><meta charset="utf-8"><title>Round12 target benchmark</title><body><h1>Round12</h1><p>Development only. Equal-context/equal-target MSE. Seed SD is not biological uncertainty. Basis oracle is a diagnostic, not a predictor.</p>'
    html+=means.to_html(index=False)+'<h2>Paired exploratory comparisons</h2>'+pd.DataFrame(paired).to_html(index=False)
    html+='<p>Inspect coverage, shuffled controls, split-half repeatability and basis oracle before choosing a representation. No automatic winner.</p></body></html>'
    (root/'report.html').write_text(html,encoding='utf-8')
