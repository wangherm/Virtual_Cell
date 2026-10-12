"""Paired-to-zero shrinkage and source-context diagnostics, not confidence claims."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from .artifacts import read,seal,verify
from .contracts import Action,FEATURES,digest
from ..selection import row_weights
from ..utils import write_json


def macro(frame,value='mse'):
    return float(frame.groupby(['context','target'])[value].mean().groupby(level='context').mean().mean())


def fit_relative(frame,alpha=1.):
    if 'zero' not in set(frame.expert_id):raise ValueError('Paired zero evidence required')
    zeros=frame.loc[frame.expert_id.eq('zero')].set_index('query_id').mse
    models={};contexts=int(frame.context.nunique());shrink=contexts/(contexts+5.)
    for name,g in frame.groupby('expert_id'):
        w=row_weights(g.rename(columns={'target':'perturbation','query_id':'row_id'}));x=g[list(FEATURES)].to_numpy(float)
        mean=w@x;std=np.sqrt(w@np.square(x-mean));scale=np.maximum(std,.1);z=np.c_[np.ones(len(x)),(x-mean)/scale]
        y=np.log(np.maximum(g.mse.to_numpy(),1e-12)/np.maximum(zeros.loc[g.query_id].to_numpy(),1e-12))
        penalty=np.eye(z.shape[1])*alpha;penalty[0,0]=0
        coef=np.linalg.solve(z.T@(w[:,None]*z)+penalty,z.T@(w*y));coef[1:]*=shrink
        residual=y-z@coef
        by_context=pd.DataFrame({'context':g.context.to_numpy(),'residual':residual}).groupby('context').residual.mean()
        models[name]={'mean':mean.tolist(),'scale':scale.tolist(),'ratio_coef':coef.tolist(),
            'minimum':x.min(0).tolist(),'maximum':x.max(0).tolist(),'active':(std>1e-6).tolist(),
            'residual_penalty':max(float(np.abs(by_context).max()),.01) if name!='zero' else 0.,
            'contexts':contexts,'shrink_factor':shrink,'source_mse':macro(g)}
    return models


def estimate(model,features):
    x=np.array([features[f] for f in FEATURES]);z=(x-np.array(model['mean']))/np.array(model['scale'])
    span=np.maximum(np.array(model['maximum'])-np.array(model['minimum']),.1)
    # A formerly constant binary/count covariate changing is also extrapolation.
    ood=bool(np.any((x<np.array(model['minimum'])-span*.5)|(x>np.array(model['maximum'])+span*.5)))
    ratio=float(np.exp(np.clip(np.r_[1.,np.clip(z,-2,2)]@model['ratio_coef'],-8,8)))
    return ratio,ood


def choose(models,features,policy,fixed,margin=.01):
    if policy=='zero':return 'zero'
    if policy=='fixed':return fixed
    values={n:estimate(m,features[n]) for n,m in models.items() if n in features}
    if policy=='risk_shrunk':return min(values,key=lambda n:(values[n][0],n))
    if policy!='conservative':raise ValueError('Unknown reliability policy')
    candidates={n:r*np.exp(models[n]['residual_penalty']) for n,(r,ood) in values.items() if n!='zero' and not ood}
    candidates={n:r for n,r in candidates.items() if r<1-margin}
    return min(candidates,key=lambda n:(candidates[n],n)) if candidates else 'zero'


def fit_reliability(frame,base_profile,output):
    """No outer inputs. Meta CV is diagnostic, not a fully independent third nesting."""
    dest=Path(output)
    if (dest/'COMPLETE.json').exists():verify(dest);return dest
    if frame.context.nunique()<3:raise ValueError('Round18 requires >=3 source OOF backgrounds')
    records=[];contexts=sorted(set(frame.context))
    for held in contexts:
        train=frame.loc[~frame.context.eq(held)];val=frame.loc[frame.context.eq(held)]
        models=fit_relative(train);fixed=min(models,key=lambda n:(models[n]['source_mse'],n))
        for q,group in val.groupby('query_id'):
            features={r.expert_id:{f:float(r[f]) for f in FEATURES} for _,r in group.iterrows()};scores=group.set_index('expert_id').mse
            for policy in ('zero','fixed','risk_shrunk','conservative'):
                name=choose(models,features,policy,fixed)
                records.append({'source_held_context':held,'query_id':q,'context':held,'target':group.iloc[0].target,
                    'policy':policy,'selected_expert':name,'mse':float(scores[name]),'zero_mse':float(scores['zero']),
                    'posthoc_regret':float(scores[name]-scores.min())})
    diag=pd.DataFrame(records);dest.mkdir(parents=True,exist_ok=True);diag.to_csv(dest/'private_source_meta_cv.csv',index=False)
    models=fit_relative(frame);base=read(Path(base_profile)/'profile.json');value=copy.deepcopy(base)
    zero_mse=models['zero']['source_mse'];cards=[]
    for name,m in models.items():
        coef=np.array(m['ratio_coef']);coef[0]+=np.log(max(zero_mse,1e-12))
        value['models'][name].update(mean=m['mean'],scale=m['scale'],coef=coef.tolist(),clip_z=2.,
            capability={k:m[k] for k in ('minimum','maximum','active','residual_penalty','contexts','shrink_factor','source_mse')})
        for (context,seen),g in frame.loc[frame.expert_id.eq(name)].groupby(['context','fit_target_seen']):
            cards.append({'expert_id':name,'source_context':context,'target_seen':bool(seen),'mse':macro(g),'queries':len(g),'targets':g.target.nunique()})
    value['risk']='Paired log-MSE ratio to zero; source-context shrinkage, clipped covariates; heuristic residual penalty, not a calibrated confidence interval'
    value['safe_routing']={'models':models,'fixed':base['fixed_expert'],'minimum_gain':.01,
        'meta_cv_scope':'Leave-source-context-out on an OOF error table. Underlying expert fits may share labels across these meta folds; untouched outer remains the transfer evaluation.'}
    write_json(dest/'profile.json',value)
    pd.DataFrame(cards).to_csv(dest/'capability_cards.csv',index=False)
    write_json(dest/'reliability_summary.json',{'source_contexts':contexts,'meta_cv_mse':{p:macro(g) for p,g in diag.groupby('policy')},
        'diagnostic_only':True,'outer_used':False,'source_covariates':FEATURES,'interval_claim':False,
        'meta_cv_limitation':value['safe_routing']['meta_cv_scope']})
    seal(dest);return dest


class ReliabilityPlanner:
    def __init__(self,profile,registry,q,policy):
        self.value=read(Path(profile)/'profile.json')['safe_routing'];self.policy=policy
        self.identity='source_reliability:'+policy+':'+digest(self.value)
        self.features={n:e.features(q) for n,e in registry.experts.items() if n in self.value['models'] and e.compatible(q)[0]}
    def decide(self,obs):
        available={n:f for n,f in self.features.items() if n not in obs['failed_experts']}
        if not available:return Action('abstain',reason='No available scoped expert')
        name=choose(self.value['models'],available,self.policy,self.value['fixed'],self.value['minimum_gain'])
        if name not in available:name='zero'
        return Action('finish' if name in obs['predictions'] else 'predict',name,'Source-only paired reliability; conservative fallback is a zero prediction, not abstention')
