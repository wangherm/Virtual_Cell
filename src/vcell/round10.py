"""Background-granularity controls and additional context holdouts."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .background_training import fingerprint
from .round8 import partitions, validate_partition, report as base_report
from .selection import mse, row_weights
from .utils import file_sha256, write_json

PROTOCOLS=('context','target','context_rpe1','context_h1')


def arm_matrix():
    return [('mlp','b0','none'),('qwen','b1','none'),('qwen','b2','none'),('qwen','b3','none'),
            ('qwen','b2','aux_true_warmup'),('qwen','b2','aux_random_warmup')]


def arm_allowed(protocol,mode):
    return not mode.endswith('_warmup') or protocol=='context'


def warmup_weight(step,warmup,maximum):
    if step<0 or warmup<1 or maximum<0:raise ValueError('Invalid module warmup')
    return maximum*min(step/warmup,1.)


def make_partition(data,protocol):
    if protocol in ('context','target'):return partitions(data,protocol)
    held={'context_rpe1':'RPE1','context_h1':'H1_hESC'}.get(protocol)
    if held is None:raise ValueError('Unknown Round10 protocol')
    train=np.asarray(data['splits']['train'])
    labels=data['meta'].iloc[train].context.to_numpy()
    outer=train[labels==held];others=train[labels!=held]
    if not len(outer) or not len(others):raise ValueError(f'Missing holdout or fit contexts: {held}')
    local=copy.copy(data);local['splits']={**data['splits'],'train':others,'val':outer}
    parts=partitions(local,'context')
    validate_partition(data,parts)
    if held in set(data['meta'].iloc[parts['fit']+parts['calibration']].context):raise ValueError('Context leakage')
    return parts


def partition_audit(data,parts,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    seen=set(data['meta'].iloc[parts['fit']].perturbation)
    rows=[]
    for label,ix in parts.items():
        frame=data['meta'].iloc[ix]
        for context,g in frame.groupby('context'):
            rows.append({'partition':label,'context':context,'rows':len(g),'targets':g.perturbation.nunique(),
                'seen_in_fit_targets':len(set(g.perturbation)&seen),
                'unseen_in_fit_targets':len(set(g.perturbation)-seen),'batches':g.batch.nunique()})
    pd.DataFrame(rows).to_csv(folder/'coverage.csv',index=False)
    write_json(folder/'scope.json',{'test_evaluated':False,
        'fit_contexts':sorted(set(data['meta'].iloc[parts['fit']].context)),
        'outer_contexts':sorted(set(data['meta'].iloc[parts['outer']].context)),
        'note':'New development holdouts from previously used training contexts, not new independent datasets. '
               'Seen and unseen fit targets are scored separately; batch IDs may be pseudo-batches.'})


def prepare_aggregates(cache,dest):
    """One mean per source/donor/tissue/type/assay/sample, never across splits."""
    cache,dest=Path(cache),Path(dest)
    fields=['dataset','donor_group','tissue','cell_type','assay','mask_index']
    meta=pd.read_csv(cache/'train_metadata.csv',keep_default_na=False)
    if set(meta['split'])!={'train'} or not meta.dataset.str.startswith('ts_').all():
        raise ValueError('Aggregate preparation accepts Tabula train cells only')
    for field in ('study','sample_id','sample'):
        if field in meta:fields.append(field)
    actual={'source_complete':file_sha256(cache/'COMPLETE.json'),'fields':fields,
            'normalization':'mean(log1p(10000*counts/full_library))',
            'sampling':'Use the same original cell indices as B2, replacing each by its group mean.'}
    stamp=fingerprint(actual)
    if (dest/'COMPLETE.json').exists():
        if json.loads((dest/'COMPLETE.json').read_text())['fingerprint']!=stamp:raise ValueError('Aggregate plan changed')
        load_aggregates(dest,cache);return dest
    dest.mkdir(parents=True,exist_ok=True)
    x=np.load(cache/'train.npy',mmap_mode='r')
    if len(x)!=len(meta):raise ValueError('Background metadata/data mismatch')
    masks=np.load(cache/'masks.npy')
    group_indices=meta.groupby(fields,sort=True,dropna=False).indices
    values=np.lib.format.open_memmap(dest/'means.npy',mode='w+',dtype='float32',shape=(len(group_indices),x.shape[1]))
    mapping=np.full(len(x),-1,dtype=np.int64);records=[]
    for number,(key,ix) in enumerate(group_indices.items()):
        record=dict(zip(fields,key));record['group_id']=number;record['n_cells']=len(ix)
        mean=x[ix].astype(np.float64).mean(0)
        mask=masks[int(record['mask_index'])]
        if not np.isfinite(mean).all() or np.any(mean[~mask]!=0):raise ValueError('Invalid aggregate/missing-value mask')
        values[number]=mean;mapping[ix]=number;records.append(record)
    values.flush();del values
    if np.any(mapping<0):raise ValueError('Some training cells were not assigned')
    np.save(dest/'cell_to_group.npy',mapping)
    pd.DataFrame(records).to_csv(dest/'groups.csv',index=False)
    write_json(dest/'audit.json',{'fingerprint':stamp,**actual,'input_cells':len(meta),'groups':len(records),
        'singleton_groups':sum(r['n_cells']==1 for r in records),'donors':meta.donor_group.nunique(),
        'n_cells_quantiles':{str(q):float(v) for q,v in pd.Series([r['n_cells'] for r in records]).quantile([0,.5,.9,1]).items()},
        'independent_sample_count_is_not_group_count':True,'test_evaluated':False})
    names=['means.npy','cell_to_group.npy','groups.csv','audit.json']
    # Build a separate held-out grouped view for like-for-like reconstruction diagnostics.
    # Its values never enter means.npy or the optimizer's cell-index mapping.
    val_meta=pd.read_csv(cache/'val_metadata.csv',keep_default_na=False)
    for field in ('study','sample_id','sample'):
        if field in fields and field not in val_meta:val_meta[field]=''
    if set(val_meta['split'])!={'val'} or set(val_meta.donor_group)&set(meta.donor_group):
        raise ValueError('Aggregate validation donor leakage')
    vx=np.load(cache/'val.npy',mmap_mode='r');val_values=[];val_records=[]
    for number,(key,ix) in enumerate(val_meta.groupby(fields,sort=True,dropna=False).indices.items()):
        val_values.append(vx[ix].astype(np.float64).mean(0).astype(np.float32))
        val_records.append({**dict(zip(fields,key)),'group_id':number,'n_cells':len(ix)})
    np.save(dest/'val_means.npy',np.stack(val_values))
    pd.DataFrame(val_records).to_csv(dest/'val_groups.csv',index=False)
    names+=['val_means.npy','val_groups.csv']
    write_json(dest/'COMPLETE.json',{'fingerprint':stamp,'source_complete':actual['source_complete'],
                                   'files':{n:file_sha256(dest/n) for n in names}})
    print(f'ROUND10 AGGREGATES: {len(meta)} training cells -> {len(records)} groups',flush=True)
    return dest


def load_aggregates(folder,cache):
    folder,cache=Path(folder),Path(cache)
    done=json.loads((folder/'COMPLETE.json').read_text())
    if done['source_complete']!=file_sha256(cache/'COMPLETE.json'):raise ValueError('Aggregate source changed')
    for name,sha in done['files'].items():
        if file_sha256(folder/name)!=sha:raise ValueError('Aggregate checksum mismatch')
    values=np.load(folder/'means.npy',mmap_mode='r');mapping=np.load(folder/'cell_to_group.npy')
    if len(mapping)!=len(np.load(cache/'train.npy',mmap_mode='r')) or mapping.min()<0 or mapping.max()>=len(values):
        raise ValueError('Invalid aggregate membership')
    return values,mapping


@torch.no_grad()
def diagnose_grouped_background(model,norm,folder,cache,dest,device):
    folder,cache,dest=Path(folder),Path(cache),Path(dest)
    # Check the complete aggregate artifact, including its held-out diagnostic view.
    load_aggregates(folder,cache)
    values=np.load(folder/'val_means.npy',mmap_mode='r')
    meta=pd.read_csv(folder/'val_groups.csv',keep_default_na=False)
    masks=np.load(cache/'masks.npy');errors=[];reference=[]
    model.eval()
    for start in range(0,len(values),512):
        end=min(start+512,len(values));mask=masks[meta.mask_index.iloc[start:end].to_numpy()]
        x=torch.tensor(np.where(mask,(values[start:end]-norm['mean'])/norm['std'],0),device=device)
        corrupt=torch.tensor(np.random.default_rng(99000+start).random(x.shape)<.2,device=device)
        pred,_=model(x.masked_fill(corrupt,0));m=torch.tensor(mask,device=device)
        errors.extend((((pred-x).square()*m).sum(1)/m.sum(1)).cpu().tolist())
        reference.extend(((x.square()*m).sum(1)/m.sum(1)).cpu().tolist())
    meta['normalized_reconstruction_mse']=errors;meta['mean_control_mse']=reference
    meta.to_csv(dest/'grouped_background_diagnostics.csv',index=False)
    write_json(dest/'grouped_background_summary.json',{'partition':'Tabula val','groups':len(meta),
        'cells':int(meta.n_cells.sum()),'donors':meta.donor_group.nunique(),
        'group_equal_mse':float(np.mean(errors)),'cell_weighted_mse':float(np.average(errors,weights=meta.n_cells)),
        'test_evaluated':False,'note':'Same held-out group means for B1/B2/B3; groups are not independent donors.'})


PAIRS=[('qwen_b1_none','qwen_b2_none'),('qwen_b2_none','qwen_b3_none'),('qwen_b1_none','qwen_b3_none'),
       ('mlp_b0_none','qwen_b2_none'),('mlp_b0_none','qwen_b3_none'),
       ('qwen_b2_none','qwen_b2_aux_true_warmup'),('qwen_b2_none','qwen_b2_aux_random_warmup'),
       ('qwen_b2_aux_random_warmup','qwen_b2_aux_true_warmup')]


def primary_gain(units):
    return float(units.groupby('context').gain.mean().mean())


def paired_summary(units,n_boot=1000):
    """Average seeds first; bootstrap whole targets while retaining all contexts."""
    units=units.groupby(['context','target']).gain.mean().reset_index()
    targets=sorted(units.target.unique());lookup={t:i for i,t in enumerate(targets)}
    code=units.target.map(lookup).to_numpy();rng=np.random.default_rng(1701);bootstrap=[]
    groups=list(units.groupby('context').indices.values());v=units.gain.to_numpy()
    for _ in range(n_boot*10):
        counts=np.bincount(rng.integers(len(targets),size=len(targets)),minlength=len(targets))[code]
        if any(counts[ix].sum()==0 for ix in groups):continue
        bootstrap.append(np.mean([np.average(v[ix],weights=counts[ix]) for ix in groups]))
        if len(bootstrap)==n_boot:break
    # Contribution respects equal context / equal target weights of the primary metric.
    units['contribution']=units.gain/units.groupby('context').target.transform('size')/len(groups)
    largest=units.groupby('target').contribution.sum().idxmax()
    rest=units[units.target!=largest]
    remaining=primary_gain(rest) if rest.context.nunique()==len(groups) else None
    return {'primary_mse_improvement':primary_gain(units),'unit_win_fraction':float((v>0).mean()),
        'unit_median_improvement':float(np.median(v)),'units':len(units),'targets':len(targets),
        'largest_contribution_target':largest,'primary_gain_without_largest':remaining,
        'bootstrap_low':float(np.quantile(bootstrap,.025)) if bootstrap else None,
        'bootstrap_high':float(np.quantile(bootstrap,.975)) if bootstrap else None,'bootstrap_replicates':len(bootstrap),
        'interval_scope':'Exploratory target-cluster resampling, conditional on every context being represented; seeds averaged first.'}


def report(data,specs,root):
    root=Path(root);frame=base_report(data,specs,root)
    if frame.empty:return frame
    units=pd.read_csv(root/'per_target.csv');paired=[];seed_pairs=[];coverage=[]
    for (protocol,mode),group in units.groupby(['protocol','mode']):
        if mode not in ('outer','shrunk','calibrated'):continue
        for baseline,candidate in PAIRS:
            a=group[group.arm==baseline];b=group[group.arm==candidate]
            joined=a.merge(b,on=['context','target','seed'],suffixes=('_base','_candidate'))
            if joined.empty:continue
            joined['gain']=joined.mse_delta_base-joined.mse_delta_candidate
            paired.append({'protocol':protocol,'mode':mode,'baseline':baseline,'candidate':candidate,
                'paired_seeds':joined.seed.nunique(),**paired_summary(joined)})
            for seed,g in joined.groupby('seed'):
                seed_pairs.append({'protocol':protocol,'mode':mode,'baseline':baseline,'candidate':candidate,
                    'seed':seed,'primary_mse_improvement':primary_gain(g)})
    for cfg in specs.values():
        folder=Path(cfg['output_dir'])
        if not (folder/'COMPLETE.json').exists():continue
        rows=np.array(cfg['partition']['outer']);meta=data['meta'].iloc[rows].reset_index(drop=True)
        seen=set(data['meta'].iloc[cfg['partition']['fit']].perturbation)
        with np.load(folder/'predictions.npz') as f:
            for label,keep in [('seen_in_fit',meta.perturbation.isin(seen)),('unseen_in_fit',~meta.perturbation.isin(seen))]:
                ix=np.flatnonzero(keep)
                if not len(ix):continue
                for mode in ('outer','shrunk','calibrated','reference','zero'):
                    pred=f[mode][ix] if mode!='zero' else np.zeros_like(data['delta'][rows[ix]])
                    coverage.append({'protocol':cfg['protocol'],'arm':cfg['arm'],'seed':cfg['seed'],'mode':mode,
                        'target_coverage':label,'mse_delta':mse(pred,data['delta'][rows[ix]],row_weights(meta.iloc[ix])),
                        'rows':len(ix),'targets':meta.iloc[ix].perturbation.nunique(),'contexts':meta.iloc[ix].context.nunique()})
    pd.DataFrame(paired).to_csv(root/'paired_diagnostics.csv',index=False)
    pd.DataFrame(seed_pairs).to_csv(root/'paired_comparisons.csv',index=False)
    pd.DataFrame(coverage).to_csv(root/'target_coverage_scores.csv',index=False)
    units.groupby(['protocol','arm','seed','mode','context']).mse_delta.mean().reset_index().to_csv(root/'per_context.csv',index=False)
    write_json(root/'evaluation_scope.json',{'test_evaluated':False,'default_module':False,'teacher_kd':False,
        'primary':'Raw MSE with equal contexts, equal targets within context, and equal rows within target.',
        'background':'Same Tabula train cells and sample-index schedule for B2/B3; B3 substitutes group means.',
        'heldout_contexts':['HepG2','RPE1','H1_hESC'],'historical_holdouts_are_not_new_datasets':True,
        'limitations':['Previously inspected development data; no final significance or universal-winner claim.',
            'Fixed original Tabula donor split, not new donor cross-validation.',
            'Background aggregation preserves group exposure weights but reduces within-group variation.',
            'One module permutation and a predeclared 500-step warmup; no automatic loss-weight tuning.',
            'Existing Qwen pretraining overlap with public biology is unverified.']})
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Round10</title>'
        '<style>body{font:14px system-ui;margin:30px}td,th{padding:6px}table{border-collapse:collapse}</style>'
        '<h1>Round10: background granularity and context transfer</h1><p>Development comparisons; Jurkat remains sealed. '
        'Main arms use no module loss. Warmup is a separate diagnostic.</p>'
        +frame.groupby(['protocol','arm','mode']).mse_delta.agg(['mean','std','count']).reset_index().to_html(index=False),encoding='utf-8')
    return frame
