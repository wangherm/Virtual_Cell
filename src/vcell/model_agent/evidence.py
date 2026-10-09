"""Source-context OOF risk fitting; private truth exists only in this offline role."""
from pathlib import Path

import numpy as np

from .artifacts import arrays, read, seal, verify
from .contracts import FEATURES, context_key, load_query, protected
from .experts import Expert
from ..utils import file_sha256, write_json


def fit_profiles(spec_path,output,alpha=.1):
    """Evaluate supplied already-refitted OOF bundles; do not invent OOF results.

    Expert training and calibration must exclude each query's entire background.
    Dedicated nested OOF refits are a prerequisite, not performed by this command.
    """
    if not np.isfinite(alpha) or alpha<=0:raise ValueError('Positive ridge regularization required')
    path=Path(spec_path).resolve();spec=read(path)
    if set(spec)!={'source_contexts','excluded_contexts','episodes'}:raise ValueError('Invalid evidence specification')
    source={context_key(c) for c in spec['source_contexts']};excluded={context_key(c) for c in spec['excluded_contexts']}
    if not source or source&excluded or any(protected(c) for c in source):raise ValueError('Invalid source-only evidence boundary')
    root=Path(output)
    if root.exists() and any(root.iterdir()):raise ValueError('Choose new profile directory')
    samples=[];lineage=[];recipes={};synthetic=False;panel=None
    for episode in spec['episodes']:
        if set(episode)!={'bundle','query','truth','fold_id'}:raise ValueError('Invalid OOF episode')
        expert=Expert(path.parent/episode['bundle']);q=load_query(path.parent/episode['query'])
        synthetic=synthetic or expert.card['synthetic']
        if q.source_partition!='train':raise ValueError('Source OOF requires training-source queries, not old validation or unlabelled queries')
        if not expert.card['synthetic'] and q.source_fingerprint is None:raise ValueError('Real source OOF query lineage required')
        if panel is None:panel=list(q.genes)
        if panel!=list(q.genes):raise ValueError('OOF output panel changed across queries')
        if context_key(q.context) not in source or context_key(q.context) in excluded:raise ValueError('Evidence query outside source-only boundary')
        p=expert.provenance
        if (context_key(q.context) in {context_key(c) for c in p['fit_contexts']+p['calibration_contexts']} or
                q.query_id in p['fit_query_ids']+p['calibration_query_ids']):
            raise ValueError('OOF query participated in expert training or calibration')
        if {context_key(c) for c in p['fit_contexts']+p['calibration_contexts']} & excluded:
            raise ValueError('Outer background participated in expert fitting')
        name=expert.card['expert_id'];recipe=expert.card['recipe_fingerprint']
        if name in recipes and recipes[name]!=recipe:raise ValueError('Expert recipe changed across OOF folds')
        recipes[name]=recipe
        truth_path=path.parent/episode['truth'];truth=arrays(truth_path)
        if set(truth)!={'genes','delta','query_fingerprint'} or not np.array_equal(truth['genes'],q.genes) or str(truth['query_fingerprint'])!=q.fingerprint:
            raise ValueError('OOF response alignment mismatch')
        y=truth['delta']
        if y.shape!=(len(q.genes),) or not np.isfinite(y).all():raise ValueError('Invalid OOF response')
        prediction=expert.predict(q);features=expert.features(q)
        if prediction.shape!=y.shape or not np.isfinite(prediction).all():raise ValueError('Invalid OOF prediction')
        samples.append({'expert_id':name,'query_id':q.query_id,'context':context_key(q.context),'target':q.target,
                        'query_fingerprint':q.fingerprint,'fold_id':episode['fold_id'],'mse':float(np.square(prediction-y).mean()),**features})
        lineage.append({'expert_id':name,'fold_id':episode['fold_id'],'bundle_complete':expert.marker_sha,
                        'query':q.fingerprint,'private_truth_sha256':file_sha256(truth_path)})
    if not samples:raise ValueError('No actual source OOF episodes supplied')
    # Each query must have the same complete model pool and consistent metadata.
    import pandas as pd
    frame=pd.DataFrame(samples)
    if frame[['expert_id','query_id']].duplicated().any():raise ValueError('Duplicate OOF unit; average repeated seeds before using distinct query IDs')
    sets=frame.groupby('query_id').expert_id.apply(lambda x:set(x))
    if any(s!=set(recipes) for s in sets):raise ValueError('OOF model pool is not matched across queries')
    if (frame.groupby('query_id').context.nunique()>1).any() or (frame.groupby('query_id').target.nunique()>1).any():
        raise ValueError('OOF query metadata changed across experts')
    if (frame.groupby('query_id').query_fingerprint.nunique()>1).any():raise ValueError('OOF query inputs changed across experts')
    from ..selection import row_weights
    models={};fixed_scores={}
    for name,group in frame.groupby('expert_id',sort=True):
        if group.context.nunique()<2:raise ValueError('Need >=2 independent source OOF backgrounds per expert')
        w=row_weights(group.rename(columns={'target':'perturbation','query_id':'row_id'}))
        x=group[list(FEATURES)].to_numpy(float);mean=w@x;scale=np.maximum(np.sqrt(w@np.square(x-mean)),.1)
        z=np.c_[np.ones(len(x)),(x-mean)/scale];y=np.log(np.maximum(group.mse.to_numpy(),1e-12))
        penalty=np.eye(z.shape[1])*alpha;penalty[0,0]=0
        coef=np.linalg.solve(z.T@(w[:,None]*z)+penalty,z.T@(w*y))
        models[name]={'mean':mean.tolist(),'scale':scale.tolist(),'coef':coef.tolist(),'recipe_fingerprint':recipes[name],
                      'units':len(group),'contexts':sorted(set(group.context))}
        fixed_scores[name]=float(w@group.mse.to_numpy())
    fixed=min(fixed_scores,key=lambda name:(fixed_scores[name],name))
    ceiling=frame.groupby('query_id').mse.min().mean()
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/'profile.json',{'schema_version':1,'features':FEATURES,'models':models,'fixed_expert':fixed,
        'fixed_source_mse':fixed_scores,'source_contexts':sorted(source),'excluded_contexts':sorted(excluded),
        'fit_alpha':alpha,'synthetic':synthetic,'genes':panel,
        'scope':'Supplied nested source OOF bundles; background-disjoint checks, not independent reproduction of refitting',
        'risk':'log-MSE Ridge; extrapolation risk is not a calibrated confidence interval',
        'no_outer_selection':True})
    # Oracle ceiling is an offline diagnostic, never included in planner profiles.
    frame.to_csv(root/'private_oof_scores.csv',index=False)
    write_json(root/'private_diagnostics.json',{'oracle_micro_mse':float(ceiling),'note':'Truth-based posthoc oracle; not a deployed selector'})
    write_json(root/'lineage.json',{'spec_sha256':file_sha256(path),'episodes':lineage})
    seal(root);return root


class Profiles:
    def __init__(self,path,registry):
        self.path=Path(path);verify(self.path);self.value=read(self.path/'profile.json')
        if tuple(self.value['features'])!=FEATURES:raise ValueError('Unknown risk feature schema')
        self.marker_sha=file_sha256(self.path/'COMPLETE.json')
        for name,model in self.value['models'].items():
            if name not in registry.experts or model['recipe_fingerprint']!=registry.experts[name].card['recipe_fingerprint']:
                raise ValueError('Risk profile/model recipe mismatch')
            if list(registry.experts[name].stats['genes'])!=self.value['genes']:raise ValueError('Risk profile/model panel mismatch')

    def risk(self,name,features):
        if name not in self.value['models']:return None
        model=self.value['models'][name]
        x=np.array([features[key] for key in FEATURES],float)
        z=np.r_[1.,(x-np.array(model['mean']))/np.array(model['scale'])]
        value=float(np.exp(np.clip(z@model['coef'],-25,15)))
        if not np.isfinite(value):raise ValueError('Invalid estimated risk')
        return value
