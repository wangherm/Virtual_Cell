"""Allowlisted predictors; inference never loads prepared response/test labels."""
import copy
from pathlib import Path
import shutil

import numpy as np
import torch

from .artifacts import arrays, read, seal, verify
from .contracts import FEATURES, NORMALIZATION, context_key, digest, protected, safe_name
from ..utils import file_sha256, write_json


class Expert:
    def __init__(self,path):
        self.path=Path(path).resolve();verify(self.path);self.card=read(self.path/'expert.json')
        safe_name(self.card['expert_id'])
        if self.card['kind'] not in ('p4','reference','zero','demo_linear'):raise ValueError('Unimplemented expert kind')
        self.stats=arrays(self.path/'input.npz')
        n=len(self.stats['genes'])
        if n<1 or len(set(self.stats['genes']))!=n:raise ValueError('Invalid expert output panel')
        for key in ('control_mean','control_std'):
            if self.stats[key].shape!=(n,) or not np.isfinite(self.stats[key]).all():raise ValueError('Invalid expert control statistics')
        if (self.stats['control_std']<=0).any():raise ValueError('Invalid expert control scale')
        self.provenance=self.card['provenance']
        if any(protected(c) for c in self.provenance['fit_contexts']+self.provenance['calibration_contexts']):
            raise ValueError('Reserved source in expert fitting')
        self.marker_sha=file_sha256(self.path/'COMPLETE.json')

    def compatible(self,q):
        if tuple(self.stats['genes'])!=tuple(q.genes):return False,'gene panel/order mismatch'
        if q.normalization!=self.card['normalization']:return False,'normalization mismatch'
        if q.species not in self.card['species'] or q.modality not in self.card['modalities']:return False,'unsupported task'
        if self.card['kind']=='p4' and q.target not in self.stats['targets']:return False,'target annotation unavailable'
        if self.card['kind']=='reference' and not self.donors(q).any():return False,'no fit-only same-target cross-context reference'
        return True,'supported; empirical accuracy is assessed separately'

    def donors(self,q):
        if 'reference_targets' not in self.stats:return np.zeros(0,bool)
        return (self.stats['reference_targets']==q.target)&np.array([context_key(c)!=context_key(q.context) for c in self.stats['reference_contexts']])

    def features(self,q):
        distance=np.sqrt(np.mean(np.square((np.asarray(q.control)-self.stats['control_mean'])/self.stats['control_std'])))
        covered=0.
        if 'targets' in self.stats and q.target in self.stats['targets']:
            index=list(self.stats['targets']).index(q.target)
            covered=float((self.stats['relations'][index]>0).any())
        return dict(zip(FEATURES,(float(distance),covered,float(self.donors(q).sum()),
            float(q.target in self.provenance['fit_targets']),float(context_key(q.context) in {context_key(c) for c in self.provenance['fit_contexts']}))))

    def predict(self,q):
        ok,reason=self.compatible(q)
        if not ok:raise ValueError(reason)
        kind=self.card['kind']
        if kind=='zero':return np.zeros(len(q.genes),np.float32)
        if kind=='demo_linear':
            f=arrays(self.path/'model.npz');return np.asarray(q.control)@f['coef']+f['bias']
        if kind=='reference':return self.stats['reference_delta'][self.donors(q)].mean(0)
        return self._p4(q)

    @torch.no_grad()
    def _p4(self,q):
        from ..round15_heads import ExperimentalHead
        checkpoint=torch.load(self.path/'encoder.pt',map_location='cpu',weights_only=True)
        state=checkpoint['state']
        # Reconstruct the saved expression projector from its tensor dimensions;
        # no Hub config, language model or training data is loaded at inference.
        hidden=state['0.weight'].shape[0];output=state['2.weight'].shape[0]
        encoder=torch.nn.Sequential(torch.nn.Linear(len(q.genes),hidden),torch.nn.GELU(),torch.nn.Linear(hidden,output))
        encoder.load_state_dict(state);encoder.eval()
        norm=checkpoint['normalization'];x=torch.tensor(q.control,dtype=torch.float32)[None]
        z=encoder((x-norm['mean'])/norm['std']).reshape(1,output//hidden,hidden).mean(1).numpy()
        bg=arrays(self.path/'background_transform.npz')
        b=((z-bg['mean'])@bg['axes'].T/bg['scale']).astype(np.float32)
        tr=arrays(self.path/'transforms.npz');ix=list(self.stats['targets']).index(q.target)
        raw=self.stats['relations'][ix:ix+1]
        t=np.c_[((raw-tr['mean'])@tr['axes'].T/tr['scale']).astype(np.float32),(raw.sum(1)>0).astype(np.float32)]
        if self.card['method']=='string_ridge':
            from ..round12 import design
            f=arrays(self.path/'model.npz');design_values=(design(b,t,True)-tr['xmean'])/tr['xscale']
            coefficients=design_values@f['coef']+f['intercept']
        else:
            f=torch.load(self.path/'model.pt',map_location='cpu',weights_only=True)
            if f['spec']['use_id'] or f['spec']['reference']:raise ValueError('Only no-ID, no-reference P4 factor heads are supported')
            model=ExperimentalHead(f['target_dim'],f['background_dim'],f['n_ids'],f['rank'],f['spec'],f['seed'])
            model.load_state_dict(f['state']);model.eval()
            coefficients=model(torch.tensor(t),torch.tensor(b),torch.zeros(1,dtype=torch.long),torch.zeros((1,1))).numpy()
        basis=arrays(self.path/'response_basis.npz')
        return (np.asarray(coefficients,np.float32)@basis['basis']*basis['scale'])[0].astype(np.float32)


class Registry:
    def __init__(self,path):
        self.path=Path(path).resolve();self.config=read(self.path)
        if set(self.config)!={'schema_version','experts'} or self.config['schema_version']!=1:raise ValueError('Invalid registry schema')
        self.experts={};self.costs={}
        for record in self.config['experts']:
            if set(record)!={'expert_id','bundle','cost_units','cost_source'}:raise ValueError('Invalid registry entry')
            name=safe_name(record['expert_id']);cost=float(record['cost_units'])
            if name in self.experts or not np.isfinite(cost) or cost<=0:raise ValueError('Duplicate expert or invalid cost')
            if record['cost_source'] not in ('unmeasured','measured','synthetic'):raise ValueError('Unknown cost provenance')
            expert=Expert(self.path.parent/record['bundle'])
            if expert.card['expert_id']!=name:raise ValueError('Registry/expert identity mismatch')
            self.experts[name]=expert;self.costs[name]=cost
        if not self.experts:raise ValueError('Empty model pool')
        self.fingerprint=digest({'registry_sha':file_sha256(self.path),
            'bundles':{name:e.marker_sha for name,e in self.experts.items()}})

    def candidates(self,q):
        return {name:{'expert_id':name,'kind':e.card['kind'],'cost_units':self.costs[name],
                      'supported':e.compatible(q)[0],'reason':e.compatible(q)[1]}
                for name,e in self.experts.items()}


def export_p4(worker_path,output,methods=('string_ridge','factor_rank16','factor_rank32')):
    """Read completed fit artifacts; export only deployment inputs and fit references.

    Existing historical outer metrics are deliberately not copied into the model
    pool or used as performance profiles. This exporter performs no refitting.
    """
    from ..data import load_prepared
    from ..round11 import verify_files
    parent=Path(worker_path).resolve();verify_files(parent)
    manifest=read(parent/'manifest.json');cfg=manifest['config'];data=load_prepared(cfg['data_dir'])
    if data['audit']['fingerprint']!=manifest['data']:raise ValueError('Expert training lineage changed')
    if data['audit'].get('normalization')!=NORMALIZATION:raise ValueError('Unknown expert response units')
    if file_sha256(Path(cfg['data_dir'])/'metadata.csv')!=manifest['metadata']:raise ValueError('Expert metadata changed')
    if file_sha256(cfg['relations'])!=manifest['relations'] or file_sha256(cfg['encoder'])!=manifest['encoder']:
        raise ValueError('Expert external dependency changed')
    fit=np.asarray(cfg['partition']['fit'],int);cal=np.asarray(cfg['partition']['calibration'],int)
    meta=data['meta'].iloc[fit].reset_index(drop=True);calmeta=data['meta'].iloc[cal]
    groups=meta.groupby(['perturbation','context'],sort=True).indices
    references={
        'reference_targets':np.array([key[0] for key in groups],dtype='U'),
        'reference_contexts':np.array([key[1] for key in groups],dtype='U'),
        'reference_delta':np.stack([data['delta'][fit[ix]].mean(0) for ix in groups.values()])}
    provenance={'source':'P4 fit artifacts','data_fingerprint':manifest['data'],'parent_complete_sha':file_sha256(parent/'COMPLETE.json'),
        'fit_contexts':sorted(set(meta.context)),'calibration_contexts':sorted(set(calmeta.context)),
        'fit_targets':sorted(set(meta.perturbation)),'fit_query_ids':meta.row_id.tolist(),
        'calibration_query_ids':calmeta.row_id.tolist(),'protocol':cfg['protocol'],'seed':cfg['seed'],
        'source_oof_claim':'not automatic; query background must be excluded from fitting and calibration'}
    relations=arrays(cfg['relations']);lookup={str(p):i for i,p in enumerate(relations['perturbations'])}
    if not np.array_equal(relations['genes'],data['genes']):raise ValueError('Expert relation panel changed')
    indices=[lookup[str(p)] for p in data['perturbations']]
    inputs={'genes':data['genes'],'targets':data['perturbations'],'relations':relations['relations'][indices,2,:],
        'control_mean':data['baseline'][fit].mean(0),'control_std':np.maximum(data['baseline'][fit].std(0),.1),**references}
    root=Path(output)
    if root.exists() and any(root.iterdir()):raise ValueError('Choose new export directory')
    root.mkdir(parents=True,exist_ok=True);entries=[]
    for method in [*methods,'reference']:
        if method not in ('string_ridge','factor_rank16','factor_rank32','reference'):raise ValueError('Unsupported P4 expert')
        name=safe_name(cfg['variant']+'_'+method);dest=root/name;dest.mkdir()
        np.savez_compressed(dest/'input.npz',**inputs)
        card={'expert_id':name,'kind':'reference' if method=='reference' else 'p4','method':method,
              'species':['human'],'modalities':['CRISPRi'],'normalization':NORMALIZATION,
              'provenance':provenance,'recipe':{'method':method,'variant':cfg['variant'],'steps':cfg['max_steps'],
                  'basis_rank':16 if method=='factor_rank16' else 32,'max_batch':cfg['batch_size']},'synthetic':False}
        if method!='reference':
            folder=parent/method;verify_files(folder)
            for name_file in ('background_transform.npz','transforms.npz','model.npz' if method=='string_ridge' else 'model.pt'):
                shutil.copyfile(folder/name_file,dest/name_file)
            shutil.copyfile(cfg['encoder'],dest/'encoder.pt')
            numerical=arrays(folder/'coefficients.npz')
            if np.any(numerical['offset']):raise ValueError('This P4 exporter supports zero-offset heads only')
            np.savez_compressed(dest/'response_basis.npz',basis=numerical['basis'],scale=numerical['scale'])
        card['recipe_fingerprint']=digest(card['recipe']);write_json(dest/'expert.json',card);seal(dest)
        entries.append({'expert_id':card['expert_id'],'bundle':card['expert_id'],'cost_units':1.,'cost_source':'unmeasured'})
    write_json(root/'registry.json',{'schema_version':1,'experts':entries});seal(root)
    return root/'registry.json'
