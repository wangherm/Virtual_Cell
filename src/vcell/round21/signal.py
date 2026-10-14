"""Technical signatures are learned only inside a declared source label scope."""
import numpy as np
import pandas as pd
from ..model_agent.nested import lines
from ..model_agent.contracts import digest,protected
from ..round20.signal import CONDITIONS

POLICY={'minimum_pearson':.1,'minimum_specificity_gap':.05,'minimum_mismatch_percentile':.8,
        'top_genes':100,'gene_sigma_multiplier':2.,'absolute_gene_floor':.05}
KEYS=('context','perturbation',*CONDITIONS)


def cosine(a,b):return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
def pearson(a,b):return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else 0.
def top(a,n):return np.argsort(-np.abs(a),kind='stable')[:min(n,len(a))]


def pairs(meta,rows,delta):
    """delta has exactly len(rows) entries, not a globally addressable truth array."""
    rows=np.asarray(rows,int)
    if len(delta)!=len(rows) or len(set(rows))!=len(rows):raise ValueError('Source labels misaligned')
    m=meta.iloc[rows].copy();m['_position']=np.arange(len(rows));m['_global']=rows
    if any(k not in m for k in (*KEYS,'replicate')):raise ValueError('Explicit technical repeat metadata required')
    m=m.loc[m.stimulus.eq('IFNG') & m.replicate.isin(['Rep1','Rep2'])]
    records=[]
    for key,g in m.groupby(list(KEYS),sort=True):
        r={r:s for r,s in g.groupby('replicate')}
        if set(r)!={'Rep1','Rep2'}:continue
        v={name:delta[s._position.to_numpy(int)].mean(0) for name,s in r.items()}
        records.append({**dict(zip(KEYS,key)),'rep1':v['Rep1'],'rep2':v['Rep2'],
            'rep1_rows':r['Rep1']._global.to_numpy(int),'rep2_rows':r['Rep2']._global.to_numpy(int),
            'cell_line':str(lines(meta.iloc[[int(g._global.iloc[0])]])[0])})
    return records


def signature_table(meta,rows,delta,policy=POLICY):
    units=pairs(meta,rows,delta);reports=[];mismatch=[];masks={}
    for unit in units:
        a,b=unit['rep1'],unit['rep2'];own=pearson(a,b);own_cos=cosine(a,b)
        peers=[u for u in units if u['perturbation']!=unit['perturbation'] and all(u[k]==unit[k] for k in ('context',*CONDITIONS))]
        pc=[pearson(a,u['rep2']) for u in peers];cc=[cosine(a,u['rep2']) for u in peers]
        median=float(np.median(pc)) if pc else None
        percentile=float(np.mean(np.asarray(pc)<own)) if pc else None
        common1=np.mean([u['rep1'] for u in peers],axis=0) if peers else np.zeros_like(a)
        common2=np.mean([u['rep2'] for u in peers],axis=0) if peers else np.zeros_like(b)
        residual_own=pearson(a-common1,b-common2)
        mean=(a+b)/2;sigma=np.sqrt(np.square(a-b)/2);threshold=np.maximum(policy['absolute_gene_floor'],policy['gene_sigma_multiplier']*sigma)
        mask=(a*b>0)&(np.abs(mean)>=threshold);keep=np.zeros(len(a),bool);keep[top(mean,policy['top_genes'])]=True;mask &= keep
        reliable=bool(len(peers)>=3 and own>=policy['minimum_pearson'] and own-median>=policy['minimum_specificity_gap'] and percentile>=policy['minimum_mismatch_percentile'] and mask.any())
        row={k:unit[k] for k in KEYS};row.update(cell_line=unit['cell_line'],own_pearson=own,own_cosine=own_cos,
            mismatch_median=median,mismatch_percentile=percentile,specificity_gap=(own-median if pc else None),
            demeaned_own_pearson=residual_own,rep1_norm=float(np.linalg.norm(a)),rep2_norm=float(np.linalg.norm(b)),
            mean_norm=float(np.linalg.norm(mean)),technical_mse=float(np.square(a-b).mean()),strong_genes=int(mask.sum()),
            reliable=reliable,mismatch_targets=len(peers),rep1_rows=','.join(map(str,unit['rep1_rows'])),rep2_rows=','.join(map(str,unit['rep2_rows'])))
        row['pair_status']='COMPLETE'
        for n in (50,100):
            ix=np.intersect1d(top(a,n),top(b,n));row['top'+str(n)+'_intersection']=len(ix)
            row['top'+str(n)+'_sign_agreement']=float((a[ix]*b[ix]>0).mean()) if len(ix) else None
        reports.append(row);masks[tuple(unit[k] for k in KEYS)]=mask
        for peer,cor,c in zip(peers,pc,cc):mismatch.append({'context':unit['context'],'target':unit['perturbation'],
            'other_target':peer['perturbation'],'dataset':unit['dataset'],'stimulus':unit['stimulus'],'time':unit['time'],
            'modality':unit['modality'],'own_pearson':own,'wrong_pearson':cor,'wrong_cosine':c})
    # Retain one-repeat targets as explicit unavailable evidence, not silent drops.
    source=meta.iloc[np.asarray(rows,int)].copy();source['_position']=np.arange(len(rows));source['_global']=rows
    source=source.loc[source.stimulus.eq('IFNG') & source.replicate.isin(['Rep1','Rep2'])]
    complete={tuple(u[k] for k in KEYS) for u in units}
    for key,g in source.groupby(list(KEYS),sort=True):
        if key in complete:continue
        r={name:part for name,part in g.groupby('replicate')};missing=sorted({'Rep1','Rep2'}-set(r))
        row={**dict(zip(KEYS,key)),'cell_line':str(lines(g)[0]),'pair_status':'MISSING_'+','.join(missing),
            'reliable':False,'strong_genes':0,'mismatch_targets':0}
        for name in ('Rep1','Rep2'):
            label=name.lower();part=r.get(name)
            row[label+'_rows']=','.join(map(str,part._global)) if part is not None else ''
            row[label+'_norm']=float(np.linalg.norm(delta[part._position.to_numpy(int)].mean(0))) if part is not None else None
        reports.append(row)
    columns=[*KEYS,'cell_line','own_pearson','own_cosine','mismatch_median','mismatch_percentile','specificity_gap',
        'demeaned_own_pearson','rep1_norm','rep2_norm','mean_norm','technical_mse','strong_genes','reliable','mismatch_targets','rep1_rows','rep2_rows','pair_status']
    return pd.DataFrame(reports,columns=columns if not reports else None),pd.DataFrame(mismatch,columns=['context','target','other_target',*CONDITIONS,'own_pearson','wrong_pearson','wrong_cosine']),masks


def reliable_library(meta,fit,fit_delta,policy=POLICY):
    table,_,masks=signature_table(meta,fit,fit_delta,policy)
    keys={tuple(r[k] for k in KEYS) for _,r in table.loc[table.reliable].iterrows()}
    allowed=np.array([tuple(meta.iloc[i][k] for k in KEYS) in keys for i in fit])
    return np.asarray(fit,int)[allowed],fit_delta[allowed],table,masks


def references(meta,fit,fit_delta,control,held_targets,policy=POLICY,seed=17,response_scale=1.):
    """Matched, nearest and mismatches use the SAME reliable, allowed donor pool."""
    fit=np.asarray(fit,int)
    if set(meta.iloc[fit].perturbation)&set(held_targets):raise ValueError('Globally unknown target in source response library')
    if not np.isfinite(response_scale) or response_scale<=0:raise ValueError('Positive source response scale required')
    donors,values,table,masks=reliable_library(meta,fit,fit_delta*response_scale,policy);values=values/response_scale
    ll=lines(meta);target=meta.perturbation.astype(str).to_numpy();conditions=meta[[*CONDITIONS,'batch']].astype(str).to_numpy()
    n,g=len(meta),fit_delta.shape[1];result={k:np.zeros((n,g),np.float32) for k in ('mean','nearest','wrong','shuffled','background')}
    source_control=np.zeros_like(control);available=np.zeros(n,bool);common=np.zeros(n,bool);strong=np.zeros((n,g),bool);audit=[]
    fit_by_condition={};donors_by_condition={};background_cache={}
    for pos,row in enumerate(fit):fit_by_condition.setdefault(tuple(conditions[row]),[]).append(pos)
    for pos,row in enumerate(donors):donors_by_condition.setdefault(tuple(conditions[row]),[]).append(pos)
    for i in range(n):
        cond=conditions[i]
        if protected(ll[i]) or protected(str(meta.iloc[i].context)):continue
        if any(v in ('','unknown','not_encoded_source_specific') for v in cond):continue
        bgkey=(ll[i],*cond)
        if bgkey not in background_cache:
            candidates=np.array(fit_by_condition.get(tuple(cond),[]),int);legal=candidates[ll[fit[candidates]]!=ll[i]]
            # Average targets within donor cell lines before averaging cell lines.
            chunks=[]
            for line in sorted(set(ll[fit[legal]])):
                pos=legal[ll[fit[legal]]==line];chunks.append(np.mean([fit_delta[pos[target[fit[pos]]==t]].mean(0) for t in sorted(set(target[fit[pos]]))],axis=0))
            background_cache[bgkey]=np.mean(chunks,axis=0) if chunks else np.zeros(g,np.float32)
        result['background'][i]=background_cache[bgkey]
        if target[i] in held_targets:continue
        candidates=np.array(donors_by_condition.get(tuple(cond),[]),int);pool=candidates[ll[donors[candidates]]!=ll[i]]
        matched=pool[target[donors[pool]]==target[i]]
        if not len(matched):continue
        available[i]=True
        def average(pos):
            per=[values[pos[ll[donors[pos]]==line]].mean(0) for line in sorted(set(ll[donors[pos]]))]
            return np.mean(per,axis=0)
        result['mean'][i]=average(matched);source_control[i]=control[donors[matched]].mean(0)
        nearest=matched[np.argmin(np.square(control[donors[matched]]-control[i]).mean(1))];result['nearest'][i]=values[nearest]
        for d in donors[matched]:strong[i] |= masks[tuple(meta.iloc[d][k] for k in KEYS)]
        alternatives=sorted(set(target[donors[pool]])-{target[i]})
        wrong_rows=[];shuffled_rows=[]
        if alternatives:
            wrong=alternatives[0];shuffled=sorted(alternatives,key=lambda t:digest(['round21_mismatch',seed,str(meta.iloc[i].row_id),t]))[0]
            wp=pool[target[donors[pool]]==wrong];sp=pool[target[donors[pool]]==shuffled]
            wrong_rows=donors[wp].tolist();shuffled_rows=donors[sp].tolist()
            result['wrong'][i]=average(wp);result['shuffled'][i]=average(sp);common[i]=True
        audit.append({'row_id':str(meta.iloc[i].row_id),'query_cell_line':ll[i],'query_target':target[i],
            'matched_rows':donors[matched].tolist(),'wrong_rows':wrong_rows,'shuffled_rows':shuffled_rows,
            'conditions':dict(zip([*CONDITIONS,'batch'],cond)),'common_support':bool(common[i])})
    return result,source_control,available,common,strong,table,audit
