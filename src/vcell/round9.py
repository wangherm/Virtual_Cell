"""Round9 background/module attribution and development-only diagnostics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import solve

from .background_training import initialize_encoder
from .fixed_training import response_basis
from .round8 import KnowledgeResponse, report as basic_report
from .selection import mse, row_weights
from .utils import write_json, file_sha256


FAMILIES=('mlp','qwen')
BACKGROUNDS=('b0','b1','b2')
MODULES=('none','aux_true','aux_random','projection_true','projection_random')


def arm_matrix():
    # Full background x family x original-module factorial, plus projection at B2.
    return [(f,b,m) for f in FAMILIES for b in BACKGROUNDS for m in MODULES
            if not m.startswith('projection') or b=='b2']


def module_matrix(values,mode,seed=818):
    if mode not in MODULES:raise ValueError('Unknown module loss')
    matrix=np.asarray(values,dtype=np.float32).copy()
    if mode.endswith('random'):matrix=matrix[:,np.random.default_rng(seed).permutation(matrix.shape[1])]
    return matrix


class BackgroundResponse(KnowledgeResponse):
    def __init__(self,data,assets,cfg,basis,norm,load_encoder=True):
        # Existing architecture: semantics, graph and KD are disabled in this factorial.
        super().__init__(data,assets,cfg['model'],cfg['family']+'_base',basis)
        encoder=initialize_encoder(len(data['genes']),cfg['family'],cfg['model'],cfg['seed'])
        if cfg['background']!='b0' and load_encoder:
            path=Path(cfg['encoder']);done=json.loads((path.parent/'COMPLETE.json').read_text())
            if file_sha256(path)!=done['files']['encoder.pt']:raise ValueError('Background encoder checksum mismatch')
            saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved['genes']!=data['genes'].tolist() or saved['family']!=cfg['family'] or saved['background']!=cfg['background'] or saved['partition']!=cfg['partition']:
                raise ValueError('Background encoder panel/family/partition mismatch')
            for k in norm:
                if not np.allclose(saved['normalization'][k].numpy(),norm[k],rtol=1e-7,atol=1e-9):raise ValueError('Background normalization mismatch')
            encoder.load_state_dict(saved['state'])
        if cfg['family']=='qwen':self.qwen.input_projector=encoder
        else:self.control=encoder
        self.module_matrix=torch.tensor(module_matrix(assets['modules'],cfg['module_mode'],cfg['module_permutation_seed']))


def module_loss(pred,module,truth,matrix,mode):
    if mode=='none':return pred.sum()*0
    if mode.startswith('aux_'):return (module-truth@matrix.T).square().mean()
    if mode.startswith('projection_'):return ((pred-truth)@matrix.T).square().mean()
    raise ValueError('Unknown module loss')


def gradient_diagnostic(model,supervised,aux,weight):
    # Shared control encoder only: avoid claiming this describes every LoRA gradient.
    parameters=list((model.qwen.input_projector if hasattr(model,'qwen') else model.control).parameters())
    a=torch.autograd.grad(supervised,parameters,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(weight*aux,parameters,retain_graph=True,allow_unused=True)
    aa=bb=dot=0.
    for x,y in zip(a,b):
        if x is not None:aa+=float(x.detach().float().square().sum())
        if y is not None:bb+=float(y.detach().float().square().sum())
        if x is not None and y is not None:dot+=float((x.detach().float()*y.detach().float()).sum())
    return {'supervised_norm':aa**.5,'weighted_module_norm':bb**.5,
            'cosine':dot/(aa*bb)**.5 if aa*bb>1e-30 else None}


def diagnostics(data,assets,parts,dest,ranks=(16,32,64)):
    """Fixed-basis oracle and cheap held-out representation probes; no test rows."""
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    fit,cal,outer=[np.array(parts[k]) for k in ('fit','calibration','outer')]
    records=[]
    for rank in ranks:
        basis,audit=response_basis(data,fit,rank)
        for label,ix in [('fit',fit),('calibration',cal),('outer',outer)]:
            truth=data['delta'][ix];projected=truth@basis.T@basis
            records.append({'rank':rank,'actual_rank':len(basis),'partition':label,
                'oracle_projection_mse':mse(projected,truth,row_weights(data['meta'].iloc[ix])),
                'fit_energy_fraction':audit['energy_fraction'],
                'note':'Uses answers for diagnosis only; not a deployable predictor or score for model selection.'})
    pd.DataFrame(records).to_csv(dest/'basis_oracle.csv',index=False)
    # Same rank32 target and calibration-only alpha selection for each representation.
    basis,_=response_basis(data,fit,32);y=data['delta'][fit]@basis.T
    probes={'semantic':assets['semantic'],'string_neighbors':assets['relations'][:,2,:]}
    probes['shuffled_semantic']=probes['semantic'][np.random.default_rng(818).permutation(len(data['perturbations']))]
    probes['shuffled_string']=probes['string_neighbors'][np.random.default_rng(818).permutation(len(data['perturbations']))]
    probes['constant']=np.ones((len(data['perturbations']),1),dtype=np.float32)
    results=[]
    for name,features in probes.items():
        x=features[data['pert_idx'][fit]].astype(np.float64)
        mean=x.mean(0);std=np.maximum(x.std(0),.01);x=(x-mean)/std
        xc=(features[data['pert_idx'][cal]]-mean)/std;xo=(features[data['pert_idx'][outer]]-mean)/std
        best=None
        for alpha in (.1,1.,10.,100.):
            coef,intercept=fit_probe(x,y,row_weights(data['meta'].iloc[fit])*len(fit),alpha)
            cp=(xc@coef+intercept)@basis
            score=mse(cp,data['delta'][cal],row_weights(data['meta'].iloc[cal]))
            if best is None or score<best[0]:best=(score,alpha,coef,intercept)
        op=(xo@best[2]+best[3])@basis
        results.append({'representation':name,'alpha':best[1],'calibration_mse':best[0],
            'outer_mse':mse(op,data['delta'][outer],row_weights(data['meta'].iloc[outer])),
            'feature_dim':features.shape[1],'targets_without_features':int((np.abs(features).sum(1)==0).sum()),
            'note':'Adapted ridge probe, not a reproduction of SLIM; STRING features are existing measured-neighbor weights.'})
    pd.DataFrame(results).to_csv(dest/'representation_probes.csv',index=False)
    true=assets['modules'];random=module_matrix(true,'aux_random')
    write_json(dest/'module_audit.json',{'modules':len(true),'genes':true.shape[1],
        'gene_coverage':int((true!=0).any(0).sum()),'column_permutation_seed':818,
        'row_gram_preserved':bool(np.allclose(true@true.T,random@random.T,rtol=1e-5,atol=1e-7)),
        'module_sizes_preserved':bool(np.array_equal((true!=0).sum(1),(random!=0).sum(1))),
        'randomization_replicates':1,'test_evaluated':False})


def fit_probe(x,y,weights,alpha):
    # Collapse identical target vectors; preserves weighted least squares exactly.
    unique,inverse=np.unique(x,axis=0,return_inverse=True)
    w=np.bincount(inverse,weights=weights)
    target=np.zeros((len(unique),y.shape[1]),dtype=np.float64)
    np.add.at(target,inverse,y*weights[:,None]);target/=w[:,None]
    xm=np.average(unique,axis=0,weights=w);ym=np.average(target,axis=0,weights=w)
    a=(unique-xm)*np.sqrt(w[:,None]);b=(target-ym)*np.sqrt(w[:,None])
    if len(a)<=a.shape[1]:coef=a.T@solve(a@a.T+alpha*np.eye(len(a)),b,assume_a='pos')
    else:coef=solve(a.T@a+alpha*np.eye(a.shape[1]),a.T@b,assume_a='pos')
    return coef,ym-xm@coef


def planned_pairs(specs):
    lookup={(c['family'],c['background'],c['module_mode']):c['arm'] for c in specs.values()}
    pairs=[]
    def add(a,b):
        if a in lookup and b in lookup:pairs.append((lookup[a],lookup[b]))
    for family in FAMILIES:
        for mode in ('none','aux_true','aux_random'):
            add((family,'b0',mode),(family,'b1',mode));add((family,'b1',mode),(family,'b2',mode))
        for background in BACKGROUNDS:
            for mode in ('aux_true','aux_random','projection_true','projection_random'):
                add((family,background,'none'),(family,background,mode))
            add((family,background,'aux_random'),(family,background,'aux_true'))
        add((family,'b2','projection_random'),(family,'b2','projection_true'))
        add((family,'b2','aux_true'),(family,'b2','projection_true'))
    return list(dict.fromkeys(pairs))


def report(data,specs,root):
    root=Path(root);frame=basic_report(data,specs,root)
    if frame.empty:return frame
    units=pd.read_csv(root/'per_target.csv');details=[];sensitivity=[]
    pairs=planned_pairs(specs)
    for (protocol,mode),group in units.groupby(['protocol','mode']):
        if mode not in ('outer','shrunk','calibrated'):continue
        for baseline,candidate in pairs:
            a=group[group.arm==baseline];b=group[group.arm==candidate]
            joined=a.merge(b,on=['protocol','mode','seed','context','target'],suffixes=('_base','_candidate'))
            if joined.empty:continue
            joined['improvement']=joined.mse_delta_base-joined.mse_delta_candidate
            for seed,g in joined.groupby('seed'):
                details.append({'protocol':protocol,'mode':mode,'baseline':baseline,'candidate':candidate,
                    'seed':seed,'unit_win_fraction':float((g.improvement>0).mean()),
                    'unit_median_improvement':float(g.improvement.median()),'unit_mean_improvement':float(g.improvement.mean()),
                    'units':len(g),'targets':g.target.nunique()})
            # Average paired seeds first, cluster whole targets across every context.
            aggregate=joined.groupby(['context','target']).improvement.mean().reset_index()
            target=aggregate.groupby('target').improvement.mean();v=target.to_numpy()
            rng=np.random.default_rng(1701)
            bootstrap=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(1000)])
            top=target.idxmax()
            sensitivity.append({'protocol':protocol,'mode':mode,'baseline':baseline,'candidate':candidate,
                'target_macro_improvement':float(v.mean()),'target_cluster_ci_low':float(np.quantile(bootstrap,.025)),
                'target_cluster_ci_high':float(np.quantile(bootstrap,.975)),'paired_seeds':joined.seed.nunique(),
                'targets':len(v),'largest_target':top,
                'target_macro_improvement_without_largest':float(target.drop(top).mean()) if len(target)>1 else None,
                'note':'Exploratory target-cluster interval; seeds/rows are not independent biological samples.'})
    pd.DataFrame(details).to_csv(root/'paired_diagnostics.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(root/'target_sensitivity.csv',index=False)
    units.groupby(['protocol','arm','seed','mode','context']).mse_delta.mean().reset_index().to_csv(root/'per_context.csv',index=False)
    # Replace Round8-specific pair list/scope with this run's predeclared comparisons.
    comparisons=[]
    for (protocol,seed,mode),g in frame.groupby(['protocol','seed','mode']):
        scores=g.set_index('arm').mse_delta.to_dict()
        for baseline,candidate in pairs:
            if baseline in scores and candidate in scores:
                comparisons.append({'protocol':protocol,'seed':seed,'mode':mode,'baseline':baseline,'candidate':candidate,
                    'baseline_mse':scores[baseline],'candidate_mse':scores[candidate],
                    'improvement':scores[baseline]-scores[candidate]})
    pd.DataFrame(comparisons).to_csv(root/'paired_comparisons.csv',index=False)
    write_json(root/'evaluation_scope.json',{'test_evaluated':False,'teacher_kd':False,'semantic_graph_inputs':False,
        'primary':'Raw context/target macro MSE; shrunk and calibrated scores are separate.',
        'background_validation':'Held-out Tabula donors; report by tissue, donor, cell type and assay.',
        'external':'Unverified-overlap reconstruction diagnostics only; never fitted.',
        'limitations':['Development comparisons on previously examined contexts/targets.',
            'Matched backgrounds are pseudobulks; public references are single cells.',
            'Two validation donors, one for marrow; many cells are not independent donors.',
            'Single fixed random-module permutation; three optimizer seeds are not three module permutations.',
            'No causal reasoning or cross-species claim. No automatic final-model selection.']})
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Round9 comparison</title>'
        '<style>body{font:14px system-ui;margin:30px}td,th{padding:6px}table{border-collapse:collapse}</style>'
        '<h1>Round9 background and module comparison</h1><p>Development results; Jurkat test remains sealed. '
        'Raw model predictions and calibration results are reported separately.</p>'
        +frame.groupby(['protocol','arm','mode']).mse_delta.agg(['mean','std','count']).reset_index().to_html(index=False),encoding='utf-8')
    return frame
