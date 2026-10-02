"""Masked background pretraining for the existing MLP/Qwen control projectors.

Public atlases supply expression backgrounds only, never perturbation labels.
"""
import copy
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch import nn

from .fixed_training import step_rows
from .train import normalization, source_fingerprint
from .utils import atomic_torch_save, file_sha256, seed_all, write_json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def safe_child(root, name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Artifact path escapes its source directory')
    return path


def prepare_background(source, dest, genes):
    """Verify all shards, then cache selected cells as on-disk float32 arrays."""
    source, dest=Path(source),Path(dest)
    report=json.loads((source/'report.json').read_text())
    if report.get('stage')!='complete' or not (source/'COMPLETE.json').exists():
        raise ValueError('Background cleaning must complete first')
    if (source/'panel.txt').read_text().splitlines()!=list(genes):raise ValueError('Background panel/order mismatch')
    records=[]
    for key in report['datasets']:
        if key not in ('ts_blood','ts_bone_marrow','ts_lung','hlca_core','immune_global'):
            raise ValueError('Unreviewed background source')
        folder=source/key
        completion=json.loads((folder/'COMPLETE.json').read_text())
        if file_sha256(folder/'membership.csv.gz')!=completion['membership_sha256']:
            raise ValueError('Background membership checksum mismatch')
        for shard in completion['shards']:
            path=safe_child(folder,shard['file'])
            print(f'BACKGROUND VERIFY: {key}/{shard["file"]}',flush=True)
            if file_sha256(path)!=shard['sha256']:raise ValueError('Background shard checksum mismatch')
            split=shard['split']
            if split not in ('train','val','external_candidate') or (split!='external_candidate')!=key.startswith('ts_'):
                raise ValueError('External background cannot enter train or validation')
            records.append({'dataset':key,**shard,'path':str(path)})
        if sum(s['cells'] for s in completion['shards'])!=report['datasets'][key]['selected_cells']:
            raise ValueError('Background shard cell count mismatch')
    identity={'records':records,'panel':list(genes),'source_report':file_sha256(source/'report.json')}
    stamp=fingerprint(identity)
    if (dest/'COMPLETE.json').exists():
        done=json.loads((dest/'COMPLETE.json').read_text())
        if done['fingerprint']!=stamp:raise ValueError('Background source changed; choose a new run')
        for name,sha in done['files'].items():
            if file_sha256(dest/name)!=sha:raise ValueError('Background cache corrupted')
        return dest
    dest.mkdir(parents=True,exist_ok=True)
    masks=[];dataset_names=[];donors={};files=[];totals={}
    for split in ('train','val','external_candidate'):
        items=[s for s in records if s['split']==split];n=sum(s['cells'] for s in items)
        if n==0:raise ValueError(f'Empty public background partition: {split}')
        array=np.lib.format.open_memmap(dest/(split+'.npy'),mode='w+',dtype='float32',shape=(n,len(genes)))
        meta=[];offset=0
        for item in items:
            a=ad.read_h5ad(item['path'])
            if a.n_obs!=item['cells'] or a.var_names.tolist()!=list(genes):raise ValueError('Shard axes mismatch')
            if set(a.obs['split'].astype(str))!={split}:raise ValueError('Mixed shard split')
            if 'available' not in a.var:raise ValueError('Missing measured-gene mask')
            mask=a.var.available.to_numpy(dtype=bool)
            if not mask.any():raise ValueError('No measured panel genes')
            key=item['dataset']
            if key not in dataset_names:dataset_names.append(key);masks.append(mask)
            elif not np.array_equal(masks[dataset_names.index(key)],mask):raise ValueError('Mask changed across shards')
            x=a.X.toarray() if hasattr(a.X,'toarray') else np.asarray(a.X)
            if not np.isfinite(x).all() or (x<0).any():raise ValueError('Invalid log expression')
            array[offset:offset+len(x)]=x
            m=a.obs.copy();m['cell_id']=a.obs_names.astype(str);m['dataset']=key
            m['mask_index']=dataset_names.index(key);m['cache_row']=np.arange(offset,offset+len(x))
            if not m.donor_group.astype(str).str.startswith('tabula:').all() and split!='external_candidate':
                raise ValueError('Unexpected training donor namespace')
            meta.append(m.reset_index(drop=True));offset+=len(x)
        array.flush();del array
        m=pd.concat(meta,ignore_index=True)
        if m[['dataset','cell_id']].duplicated().any():raise ValueError('Duplicate background cells within split')
        donors[split]=set(m.donor_group.astype(str))
        m.to_csv(dest/(split+'_metadata.csv'),index=False)
        files.extend([split+'.npy',split+'_metadata.csv']);totals[split]=n
    if donors['train'] & donors['val']:raise ValueError('Background donor leakage')
    np.save(dest/'masks.npy',np.asarray(masks));files.append('masks.npy')
    write_json(dest/'audit.json',{'fingerprint':stamp,'source':identity,'cells':totals,'datasets':dataset_names,
        'train_donors':sorted(donors['train']),'val_donors':sorted(donors['val']),
        'external_overlap_verified':False,'test_evaluated':False})
    files.append('audit.json')
    write_json(dest/'COMPLETE.json',{'fingerprint':stamp,'files':{f:file_sha256(dest/f) for f in files}})
    return dest


def control_encoder(genes, family, spec):
    if family=='mlp':return nn.Sequential(nn.Linear(genes,256),nn.GELU(),nn.LayerNorm(256))
    if family!='qwen':raise ValueError('Unknown model family')
    hidden=int(json.loads((Path(spec['model_id'])/'config.json').read_text())['hidden_size'])
    return nn.Sequential(nn.Linear(genes,hidden),nn.GELU(),nn.Linear(hidden,int(spec['control_tokens'])*hidden))


def initialize_encoder(genes, family, spec, seed):
    # Isolate RNG consumption: every background condition starts from this state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+200)
        return control_encoder(genes,family,spec)


class BackgroundAutoencoder(nn.Module):
    def __init__(self,genes,family,spec,seed):
        super().__init__()
        self.encoder=initialize_encoder(genes,family,spec,seed)
        self.tokens=int(spec['control_tokens']) if family=='qwen' else 1
        h=int(json.loads((Path(spec['model_id'])/'config.json').read_text())['hidden_size']) if family=='qwen' else 256
        self.decoder=nn.Linear(h,genes)

    def forward(self,x):
        z=self.encoder(x).reshape(len(x),self.tokens,-1).mean(1)
        return self.decoder(z),z


def masked_loss(pred,target,available):
    return ((pred-target).square()*available).sum()/available.sum().clamp_min(1)


def fit_normalization(data,partition):
    from .round8 import validate_partition
    validate_partition(data,partition)
    local=copy.copy(data);local['splits']={**data['splits'],'train':np.array(partition['fit'])}
    return normalization(local)


def matched_controls(data,partition):
    # Baseline pseudobulks may be repeated for many perturbations; deduplicate.
    return np.unique(data['baseline'][partition['fit']],axis=0).astype(np.float32)


def pretrain(cfg,resume=False):
    from .data import load_prepared
    torch.set_num_threads(2)
    data=load_prepared(cfg['data_dir']);norm=fit_normalization(data,cfg['partition'])
    dest=Path(cfg['output_dir']);cache=Path(cfg['background_cache'])
    actual={'config':cfg,'source':source_fingerprint(),'data':data['audit']['fingerprint'],
            'background':file_sha256(cache/'COMPLETE.json'),
            'model_config':file_sha256(Path(cfg['model']['model_id'])/'config.json')}
    stamp=fingerprint(actual)
    if dest.exists() and any(dest.iterdir()):
        if not resume or not (dest/'manifest.json').exists() or json.loads((dest/'manifest.json').read_text())['fingerprint']!=stamp:
            raise ValueError('Resume identical background pretraining or choose a new run')
        if (dest/'COMPLETE.json').exists():
            done=json.loads((dest/'COMPLETE.json').read_text())
            for f,sha in done['files'].items():
                if file_sha256(dest/f)!=sha:raise ValueError('Background pretraining artifact changed')
            return dest
    dest.mkdir(parents=True,exist_ok=True);write_json(dest/'manifest.json',{'fingerprint':stamp,**actual})
    public=np.load(cache/'train.npy',mmap_mode='r');meta=pd.read_csv(cache/'train_metadata.csv')
    masks=np.load(cache/'masks.npy');mask_index=meta.mask_index.to_numpy()
    controls=matched_controls(data,cfg['partition'])
    seed_all(cfg['seed']);model=BackgroundAutoencoder(len(data['genes']),cfg['family'],cfg['model'],cfg['seed']).to(cfg['device'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.01)
    first=0;history=[];steps=int(cfg['pretrain_steps']);batch=int(cfg['pretrain_batch'])
    if cfg['background'] not in ('b1','b2') or batch<2 or steps<1:raise ValueError('Invalid background pretraining plan')
    if resume and (dest/'last.pt').exists():
        saved=torch.load(dest/'last.pt',map_location='cpu',weights_only=True)
        if saved['fingerprint']!=stamp:raise ValueError('Background checkpoint mismatch')
        model.load_state_dict(saved['state']);optimizer.load_state_dict(saved['optimizer'])
        first,history=saved['step'],saved['history']
    for step in range(first,steps):
        seed_all(cfg['seed']+30000+step)
        npublic=batch//2 if cfg['background']=='b2' else 0
        ci=step_rows(np.arange(len(controls)),cfg['seed'],step,batch-npublic)
        values=controls[ci];available=np.ones_like(values,dtype=bool)
        if npublic:
            pi=step_rows(np.arange(len(public)),cfg['seed'],step,npublic)
            values=np.concatenate([values,public[pi]]);available=np.concatenate([available,masks[mask_index[pi]]])
        x=torch.tensor(np.where(available,(values-norm['mean'])/norm['std'],0),device=cfg['device'])
        mask=torch.tensor(available,device=cfg['device'])
        damaged=x.masked_fill(torch.rand_like(x)<.2,0)
        model.train();optimizer.zero_grad(set_to_none=True)
        pred,_=model(damaged);loss=masked_loss(pred,x,mask)
        if not torch.isfinite(loss):raise FloatingPointError('Background pretraining nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        done=step+1
        if done==1 or done%100==0:print(f'BACKGROUND {cfg["family"]}/{cfg["background"]} step={done}/{steps} loss={loss.item():.5f}',flush=True)
        if done%cfg['save_every']==0 or done==steps:
            history.append({'step':done,'masked_reconstruction_mse':float(loss.item())})
            atomic_torch_save({'fingerprint':stamp,'state':model.state_dict(),'optimizer':optimizer.state_dict(),
                              'step':done,'history':history},dest/'last.pt')
            pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
            write_json(dest/'progress.json',{'step':done,'max_steps':steps,'test_evaluated':False})
    atomic_torch_save({'fingerprint':stamp,'state':{k:v.cpu().clone() for k,v in model.encoder.state_dict().items()},
        'normalization':{k:torch.tensor(v) for k,v in norm.items()},'genes':data['genes'].tolist(),
        'family':cfg['family'],'background':cfg['background'],'partition':cfg['partition']},dest/'encoder.pt')
    diagnose_background(model,norm,cache,dest,cfg)
    write_json(dest/'exposure.json',{'optimizer_steps':steps,'cells_seen':steps*batch,
        'unique_matched_control_pseudobulks':len(controls),'public_train_cells':len(public),
        'public_cells_seen':steps*(batch//2) if cfg['background']=='b2' else 0,
        'note':'B1 repeats matched control pseudobulks; B2 mixes these with public single cells 1:1. '
               'Added data scale and aggregation level differ. No validation or external cells fitted.',
        'test_evaluated':False})
    names=['encoder.pt','history.csv','background_diagnostics.csv','exposure.json','val_latent.json','external_candidate_latent.json']
    write_json(dest/'COMPLETE.json',{'fingerprint':stamp,'files':{n:file_sha256(dest/n) for n in names}})
    print(f'BACKGROUND PRETRAIN COMPLETE: {dest}',flush=True)
    return dest


@torch.no_grad()
def diagnose_background(model,norm,cache,dest,cfg):
    model.eval();records=[];masks=np.load(cache/'masks.npy')
    train_meta=pd.read_csv(cache/'train_metadata.csv');known=set(train_meta.cell_type)
    for split in ('val','external_candidate'):
        values=np.load(cache/(split+'.npy'),mmap_mode='r');meta=pd.read_csv(cache/(split+'_metadata.csv'))
        errors=[];baseline=[];latent=[]
        for start in range(0,len(values),512):
            end=min(start+512,len(values));mask=masks[meta.mask_index.iloc[start:end].to_numpy()]
            x=torch.tensor(np.where(mask,(values[start:end]-norm['mean'])/norm['std'],0),device=cfg['device'])
            # Fixed corruption shared by every pretraining condition and seed.
            corrupt=torch.tensor(np.random.default_rng(91000+start).random(x.shape)<.2,device=x.device)
            pred,z=model(x.masked_fill(corrupt,0));m=torch.tensor(mask,device=x.device)
            errors.extend((((pred-x).square()*m).sum(1)/m.sum(1)).cpu().tolist())
            baseline.extend(((x.square()*m).sum(1)/m.sum(1)).cpu().tolist())
            latent.append(z.cpu().numpy())
        meta['mse']=errors;meta['mean_control_mse']=baseline;meta['in_tabula_train_types']=meta.cell_type.isin(known)
        for fields in [[],['dataset'],['dataset','donor_id'],['dataset','tissue'],['dataset','cell_type'],['dataset','assay'],['in_tabula_train_types']]:
            groups=[('all',meta)] if not fields else meta.groupby(fields,observed=True)
            for key,group in groups:
                records.append({'split':split,'group_by':'+'.join(fields) or 'all','group':str(key),
                    'cells':len(group),'normalized_reconstruction_mse':float(group.mse.mean()),
                    'mean_control_mse':float(group.mean_control_mse.mean()),
                    'independent_test':False,'type_coverage_reference':'Tabula train pool; B1 does not fit this pool'})
        z=np.concatenate(latent)
        write_json(dest/(split+'_latent.json'),{'mean_latent_variance':float(z.var(0).mean()),
            'near_constant_dimension_fraction':float((z.var(0)<1e-8).mean()),'cells':len(z),
            'note':'Endpoint reconstruction/latent diagnostics only, not perturbation-response scores.'})
    pd.DataFrame(records).to_csv(dest/'background_diagnostics.csv',index=False)
