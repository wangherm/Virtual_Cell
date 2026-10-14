"""All thresholds and model choices derive from declared source rows only."""
import re
import numpy as np
import pandas as pd
from ..selection import row_weights

ALPHAS=(0.,.01,.03,.05,.1,.2,.4,.7,1.)
CONDITIONS=('dataset','stimulus','time','modality')


def exposure_weights(meta):
    """50:50 historical:IFNG exposure if both exist; equal context/target inside."""
    is_ifng=meta.get('stimulus',pd.Series('unknown',index=meta.index)).astype(str).eq('IFNG').to_numpy()
    w=np.zeros(len(meta))
    for ix in (np.flatnonzero(is_ifng),np.flatnonzero(~is_ifng)):
        if len(ix):w[ix]=row_weights(meta.iloc[ix])/(2 if is_ifng.any() and (~is_ifng).any() else 1)
    return w


def shuffle_target_prompts(texts,codes,seed):
    """Permute gene names consistently; leave cell and stimulus text untouched."""
    names={int(c):re.search(r' target (.*?); cell ',text).group(1) for c,text in zip(codes,texts)}
    keys=sorted(names);values=[names[k] for k in keys]
    if len(keys)>1:
        order=np.random.default_rng(seed).permutation(len(keys));values=np.asarray(values)[np.roll(order,1)].tolist()
    lookup=dict(zip(keys,values))
    return [re.sub(r'( target ).*?(; cell )',lambda m:m.group(1)+lookup[int(c)]+m.group(2),t, count=1) for c,t in zip(codes,texts)]


def replicate_qc(data,rows):
    """Technical Rep1/Rep2 consistency; differences estimate technical noise only."""
    meta=data['meta'].iloc[np.asarray(rows,int)].copy();meta['_row']=rows
    required=['context','perturbation',*CONDITIONS,'replicate']
    if any(k not in meta for k in required):return pd.DataFrame(),np.empty((0,len(data['genes']))),np.empty((0,len(data['genes'])))
    meta=meta.loc[meta.stimulus.eq('IFNG') & meta.replicate.isin(['Rep1','Rep2'])]
    records=[];effects=[];noise=[]
    for key,g in meta.groupby(required[:-1],sort=True):
        reps={r:data['delta'][s._row.to_numpy(int)].mean(0) for r,s in g.groupby('replicate')}
        if set(reps)!={'Rep1','Rep2'}:continue
        a,b=reps['Rep1'],reps['Rep2'];an,bn=np.linalg.norm(a),np.linalg.norm(b)
        corr=float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else 0.
        records.append(dict(zip(required[:-1],key),pearson=corr,cosine=float(a@b/max(an*bn,1e-12)),
            direction_agreement=float(((a*b)>0).mean()),replicate_mse=float(np.square(a-b).mean()),
            mean_effect_norm=float(np.linalg.norm((a+b)/2)),n_rows=len(g)))
        effects.append((a+b)/2);noise.append(np.square(a-b)/2)
    return pd.DataFrame(records),np.asarray(effects).reshape(-1,len(data['genes'])),np.asarray(noise).reshape(-1,len(data['genes']))


def fit_signal(data,fit):
    pairs,effects,noise=replicate_qc(data,fit)
    sigma=np.sqrt(np.median(noise,axis=0)) if len(noise) else np.zeros(len(data['genes']))
    threshold=np.maximum(.05,3*sigma)
    mean=np.mean(np.abs(effects),axis=0) if len(effects) else np.zeros(len(threshold))
    strong=mean>=threshold
    strong_targets=sorted(set(pairs.loc[(np.linalg.norm(effects[:,strong],axis=1)>.1) & (pairs.cosine.to_numpy()>0),'perturbation'])) if len(pairs) and strong.any() else []
    gate=bool(len(pairs)>=8 and pairs.context.nunique()>=2 and pairs.pearson.median()>=.1 and (pairs.cosine>0).mean()>=.6 and strong.any()) if len(pairs) else False
    report={'paired_units':len(pairs),'contexts':sorted(pairs.context.unique()) if len(pairs) else [],
        'median_pearson':float(pairs.pearson.median()) if len(pairs) else None,
        'positive_cosine_fraction':float((pairs.cosine>0).mean()) if len(pairs) else None,
        'strong_genes':int(strong.sum()),'strong_targets':strong_targets,'signal_gate':gate,
        'threshold':'source-fit technical sigma; max(0.05, 3*sigma); gene mean absolute replicate-average exceeds threshold',
        'gate_rule':'>=8 pairs, >=2 contexts, median Pearson >=0.1, >=60% positive cosine, >=1 strong gene',
        'scope':'Technical replicate noise; not biological replication or within-cell variance; exploratory predeclared gate',
        'fit_rows':list(map(int,fit)),'outer_used':False}
    genes=pd.DataFrame({'gene':data['genes'],'technical_sigma':sigma,'threshold':threshold,'mean_absolute_effect':mean,'strong':strong})
    return report,pairs,genes,strong


def macro(frame,column='mse'):
    return float(frame.groupby(['context','target'])[column].mean().groupby('context').mean().mean())


def summarize(frame):
    result={'mse':macro(frame),'zero_mse':macro(frame,'zero_mse'),
        'amplitude_ratio':macro(frame,'prediction_norm')/max(macro(frame,'truth_norm'),1e-12),
        'specificity_gap':macro(frame,'specificity_gap'),'strong_mse':macro(frame,'strong_mse'),
        'target_win_fraction':float((frame.groupby(['context','target']).mse.mean()<frame.groupby(['context','target']).zero_mse.mean()).mean()),
        'rows':len(frame),'distinct_targets':int(frame.target.nunique())}
    for group,part in frame.groupby('reference_available'):
        result['reference_available' if group else 'reference_unavailable']={'mse':macro(part),'rows':len(part),'targets':int(part.target.nunique())}
    if 'globally_unknown' in frame:
        for group,part in frame.groupby('globally_unknown'):
            result['globally_unknown' if group else 'known_target']={'mse':macro(part),'rows':len(part),'targets':int(part.target.nunique())}
    return {k:(float(v) if np.isfinite(v) else None) if isinstance(v,(float,np.floating)) else v for k,v in result.items()}


def promotion(candidate,baseline):
    # Undefined specificity/strong response cannot count as positive evidence.
    finite=all(x is not None and np.isfinite(x) for x in (candidate['mse'],baseline['mse'],candidate['strong_mse'],baseline['strong_mse'],candidate['specificity_gap'],baseline['specificity_gap']))
    return bool(finite and candidate['mse']<=.99*baseline['mse'] and candidate['strong_mse']<=baseline['strong_mse'] and candidate['specificity_gap']>=baseline['specificity_gap'] and candidate['amplitude_ratio']>=.05)
