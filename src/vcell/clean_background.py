"""Streaming public-background preparation. No perturbation labels or model fitting."""
from collections import Counter
from contextlib import contextmanager
from itertools import combinations
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import tarfile

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from .background_inspect import column, digest, write_json


@contextmanager
def run_lock(path):
    """Prevent concurrent writers; the operating system releases locks on exit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Background preparation is already active in this output directory') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def check_counts(values):
    if not np.isfinite(values).all() or (values < 0).any() or (np.abs(values-np.rint(values)) > 1e-5).any():
        raise ValueError('raw/X failed full finite/nonnegative/integer count checks')


def load_panel(path, expected):
    path = Path(path)
    if path.is_dir(): path = path/'dataset.npz'
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as f:
            genes = f['genes'].astype(str).tolist()  # Do not load expression or sealed labels.
    else:
        genes = [x.strip() for x in path.read_text().splitlines() if x.strip()]
    if len(genes)!=expected or len(set(genes))!=len(genes):
        raise ValueError(f'Expected {expected} unique panel genes, found {len(genes)}; use the Round8 plus_h1 panel')
    return genes


def gene_mapping(f, panel):
    var = f['raw/var']
    key = var.attrs.get('_index','_index')
    if isinstance(key,bytes):key=key.decode()
    ids = column(var,key)
    if len(set(ids))!=len(ids): raise ValueError('Duplicate source gene IDs; resolve explicitly')
    symkey = next((k for k in ('feature_name','gene_symbols','gene_name','gene_symbol','symbol') if k in var),None)
    symbols = column(var,symkey) if symkey else ids
    exact = {g:i for i,g in enumerate(ids)}
    bysymbol = {}
    for i,s in enumerate(symbols):bysymbol.setdefault(s,[]).append(i)
    filtered = column(var,'feature_is_filtered') if 'feature_is_filtered' in var else np.full(len(ids),'unknown')
    records = []
    for g in panel:
        candidates = [exact[g]] if g in exact else bysymbol.get(g,[])
        status = 'exact_id' if g in exact else 'unique_symbol'
        if len(candidates)!=1:
            idx = -1; status = 'ambiguous_symbol' if candidates else 'missing'
        else:
            idx = candidates[0]
            if filtered[idx].lower()=='true':idx=-1;status='source_feature_filtered'
        records.append({'panel_gene':g,'source_index':idx,'source_gene_id':ids[idx] if idx>=0 else '',
                        'source_symbol':symbols[idx] if idx>=0 else '', 'status':status,'available':idx>=0})
    mapping = pd.DataFrame(records)
    matched = mapping[mapping.available]
    if matched.source_index.duplicated().any():
        raise ValueError('Multiple panel entries map to the same source gene; resolve panel aliases explicitly')
    return mapping, np.array([s.upper().startswith('MT-') for s in symbols])


def obs_table(f):
    obs=f['obs']; key=obs.attrs.get('_index','_index')
    if isinstance(key,bytes):key=key.decode()
    ids=column(obs,key)
    required=('donor_id','tissue','cell_type','assay','disease')
    if any(k not in obs for k in required):raise ValueError('Missing required donor/tissue/cell_type/assay/disease metadata')
    data={'cell_id':ids,'source_row':np.arange(len(ids))}
    for k in required:data[k]=column(obs,k)
    for k in ('study','sample_id','sample','is_primary_data','lung_condition','Manually_curated_celltype','cell_line','condition','perturbation'):
        data[k]=column(obs,k) if k in obs else np.full(len(ids),'')
    return pd.DataFrame(data)


def read_rows(x, start, end):
    encoding=x.attrs.get('encoding-type','') if isinstance(x,h5py.Group) else 'dense'
    if isinstance(encoding,bytes):encoding=encoding.decode()
    if encoding=='csr_matrix':
        ptr=x['indptr'][start:end+1];lo,hi=int(ptr[0]),int(ptr[-1])
        values=x['data'][lo:hi]
        check_counts(values)  # Check before combining duplicate sparse entries.
        mat=sparse.csr_matrix((values.astype(np.float64),x['indices'][lo:hi],ptr-ptr[0]),shape=(end-start,int(x.attrs['shape'][1])))
        mat.sum_duplicates();mat.eliminate_zeros()
        return mat
    if encoding=='dense':
        values=x[start:end,:];check_counts(values)
        return sparse.csr_matrix(values.astype(np.float64))
    raise ValueError('Preparation supports CSR or dense raw/X only; no implicit CSC densification')


def filter_reasons(obs, totals, detected, mito_pct, cfg):
    reasons=[[] for _ in range(len(obs))]
    def flag(mask,label):
        for i in np.flatnonzero(np.asarray(mask)):reasons[i].append(label)
    flag(~obs.assay.str.lower().str.startswith('10x'),'non_10x')
    flag(~obs.disease.str.lower().eq('normal'),'not_normal_annotation')
    flag(obs.donor_id.str.lower().isin(['','unknown','nan','none','<missing>']),'missing_donor')
    flag(obs.tissue.str.lower().isin(['','unknown','nan','none','<missing>']),'missing_tissue')
    flag(obs.cell_type.str.lower().isin(['','unknown','nan','none','<missing>']),'unknown_cell_type')
    text=obs[['cell_type','Manually_curated_celltype']].agg(' '.join,axis=1)
    flag(text.str.contains('doublet',case=False,regex=False),'annotated_doublet')
    flag(obs.lung_condition.str.contains('tumor|tumour',case=False,regex=True),'tumor_adjacent')
    flag(obs[['cell_line','condition','perturbation']].agg(' '.join,axis=1).str.contains(r'hep[ -]?g2|jurkat',case=False,regex=True),'heldout_background')
    flag(obs.cell_id.duplicated(keep='first'),'duplicate_obs_id_within_file')
    flag(totals<cfg['min_counts'],'low_counts')
    flag(detected<cfg['min_genes'],'low_detected_genes')
    flag(mito_pct>cfg['max_mito_pct'],'high_mito_fraction')
    return [';'.join(r) for r in reasons]


def qc_dataset(path, output, key, panel, cfg):
    output.mkdir(parents=True,exist_ok=True)
    with h5py.File(path,'r') as f:
        if 'raw/X' not in f:raise ValueError('raw/X missing: count-layer policy requires explicit revision')
        obs=obs_table(f);mapping,mito=gene_mapping(f,panel)
        if not mito.any():raise ValueError('No MT- symbols found; mitochondrial QC unavailable, do not silently pass')
        mapping.to_csv(output/'gene_mapping.csv',index=False)
        totals=np.zeros(len(obs));detected=np.zeros(len(obs),dtype=int);mitofrac=np.zeros(len(obs))
        x=f['raw/X']
        shape=tuple(x.attrs['shape']) if isinstance(x,h5py.Group) else x.shape
        if shape != (len(obs),len(mito)):raise ValueError('raw/X axes do not match obs and raw/var')
        for start in range(0,len(obs),cfg['chunk_rows']):
            end=min(start+cfg['chunk_rows'],len(obs));a=read_rows(x,start,end)
            totals[start:end]=np.asarray(a.sum(axis=1)).ravel()
            detected[start:end]=a.getnnz(axis=1)
            mitofrac[start:end]=100*np.asarray(a[:,mito].sum(axis=1)).ravel()/np.maximum(totals[start:end],1)
            if start%(cfg['chunk_rows']*20)==0 or end==len(obs):print(f'QC {key}: {end}/{len(obs)} cells',flush=True)
    obs['total_counts']=totals;obs['n_genes']=detected;obs['mito_pct']=mitofrac
    obs['exclusion_reasons']=filter_reasons(obs,totals,detected,mitofrac,cfg)
    obs['qc_pass']=obs.exclusion_reasons.eq('');obs['dataset']=key
    obs['donor_group']=('tabula:' if key.startswith('ts_') else key+':')+obs.donor_id
    obs.to_csv(output/'qc_cells.csv.gz',index=False)
    reasons=Counter(x for text in obs.exclusion_reasons for x in text.split(';') if x)
    counts={'input_cells':len(obs),'qc_pass':int(obs.qc_pass.sum()),'excluded_cells':int((~obs.qc_pass).sum()),
            'overlapping_exclusion_counts':dict(reasons),'panel_available':int(mapping.available.sum()),
            'panel_total':len(panel),'raw_count_check':'all stored entries scanned',
            'mitochondrial_features':int(mito.sum())}
    counts['qc_quantiles']={name:{str(q):float(value) for q,value in obs[name].quantile([0,.01,.1,.5,.9,.99,1]).items()}
                           for name in ('total_counts','n_genes','mito_pct')}
    for field in ('cell_type','assay','donor_id','tissue'):
        summary=obs.groupby(field,dropna=False).qc_pass.agg(input_cells='size',qc_pass='sum').reset_index()
        summary.to_csv(output/f'qc_by_{field}.csv',index=False)
    write_json(output/'qc_summary.json',counts)
    return obs,mapping,counts


def stable_score(seed,text):
    return hashlib.sha256(f'{seed}|{text}'.encode()).hexdigest()


def split_donors(frames,seed):
    available=pd.concat([f[f.qc_pass] for k,f in frames.items() if k.startswith('ts_')],ignore_index=True)
    donors=sorted(available.donor_group.unique())
    if len(donors)<3:raise ValueError('Need at least three eligible Tabula donors for a donor-held-out split')
    n=max(1,round(len(donors)*.2))
    # Metadata-only selection: prioritize train/val donor coverage in every tissue dataset.
    options=combinations(donors,n) if len(donors)<=20 else [tuple(sorted(donors,key=lambda d:stable_score(seed,d))[:n])]
    groups={k:set(f.loc[f.qc_pass,'donor_group']) for k,f in frames.items() if k.startswith('ts_')}
    def score(choice):
        val=set(choice)
        uncovered=sum(not bool(g&val) or not bool(g-val) for g in groups.values())
        return uncovered,stable_score(seed,'|'.join(choice))
    held=set(min(options,key=score))
    return {d:'val' if d in held else 'train' for d in donors}


def select_cells(obs,key,splits,cfg):
    obs=obs.copy()
    obs['split']=obs.donor_group.map(splits).fillna('excluded') if key.startswith('ts_') else 'external_candidate'
    obs.loc[~obs.qc_pass,'split']='excluded'
    eligible=obs[obs.qc_pass].copy()
    eligible['_score']=[stable_score(cfg['seed'],key+'|'+c) for c in eligible.cell_id]
    eligible=eligible.sort_values('_score')
    strata=['donor_group','tissue','cell_type','assay']
    eligible['_rank']=eligible.groupby(strata,sort=False).cumcount()
    eligible=eligible[eligible['_rank']<cfg['per_stratum']]
    # Round-robin strata ranks instead of sampling proportional to abundant cell types.
    eligible=eligible.sort_values(['_rank','_score']).head(cfg['max_cells_per_dataset'])
    obs['selected']=obs.index.isin(eligible.index)
    obs.loc[obs.qc_pass & ~obs.selected,'exclusion_reasons']='sampling_cap'
    obs.loc[~obs.selected,'split']='excluded'
    return obs


def export_shards(path,output,obs,mapping,cfg):
    selected=obs.index[obs.selected].to_numpy();selected.sort()
    matrices=[]; metas=[]; count=0; manifest=[]
    available=mapping.available.to_numpy(dtype=bool)
    source=mapping.loc[available,'source_index'].to_numpy(dtype=int)
    destinations=np.flatnonzero(available)
    with h5py.File(path,'r') as f:
        x=f['raw/X']
        n_source=int(x.attrs['shape'][1]) if isinstance(x,h5py.Group) else x.shape[1]
        projection=sparse.csr_matrix((np.ones(len(source)),(source,destinations)),shape=(n_source,len(mapping)))
        def flush():
            nonlocal matrices,metas,count
            if not matrices:return
            raw=sparse.vstack(matrices,format='csr').astype(np.float32)
            meta=pd.concat(metas,ignore_index=True)
            normalized=raw.multiply((10000/meta.total_counts.to_numpy())[:,None]).tocsr()
            normalized.data=np.log1p(normalized.data)
            for split in sorted(meta.split.unique()):
                keep=np.flatnonzero(meta.split.eq(split))
                var=mapping.set_index('panel_gene').copy()
                data=ad.AnnData(normalized[keep],obs=meta.iloc[keep].set_index('cell_id'),var=var)
                data.layers['counts']=raw[keep]
                data.uns['background_preparation']={
                    'normalization':'log1p(10000 * panel counts / full-source raw library size)',
                    'missing_policy':'unavailable columns contain placeholders; use var.available as a loss/input mask',
                    'external_overlap_verified':False,'perturbation_labels_used':False}
                if shutil.disk_usage(output).free < cfg['reserve_gib']*1024**3+raw.data.nbytes*8:
                    raise ValueError('Insufficient free data-disk reserve during shard export')
                filename=f'{split}_part_{len(manifest):04d}.h5ad';dest=output/filename;temp=output/(filename+'.tmp')
                data.write_h5ad(temp,compression='gzip');temp.replace(dest)
                manifest.append({'file':filename,'cells':len(keep),'sha256':digest(dest).hexdigest(),'split':split})
                print(f'EXPORT {output.name}: {filename}, cells={len(keep)}',flush=True)
            matrices=[];metas=[];count=0
        x=f['raw/X']
        for start in range(0,len(obs),cfg['chunk_rows']):
            end=min(start+cfg['chunk_rows'],len(obs))
            keep=selected[(selected>=start)&(selected<end)]
            if not len(keep):continue
            a=read_rows(x,start,end)[keep-start]@projection
            matrices.append(a);metas.append(obs.loc[keep].copy());count+=len(keep)
            if count>=cfg['shard_rows']:flush()
        flush()
    return manifest


def report(root,state):
    write_json(root/'report.json',state)
    rows=[]
    for key,d in state['datasets'].items():
        splits=d.get('split_counts',{})
        rows.append({'dataset':key,'input_cells':d['input_cells'],'qc_pass':d['qc_pass'],
                     'qc_pass_percent':round(100*d['qc_pass']/max(d['input_cells'],1),2),
                     'selected_cells':d.get('selected_cells',0),'train':splits.get('train',0),
                     'val':splits.get('val',0),'external_candidate':splits.get('external_candidate',0),
                     'panel_available':d['panel_available'],'panel_total':d['panel_total']})
    table=pd.DataFrame(rows)
    table.to_csv(root/'summary.csv',index=False)
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Background v1 preparation</title>'
        '<style>body{font:15px system-ui;max-width:1200px;margin:30px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
        'table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ddd}th{background:#eef4f8}</style>'
        '<h1>Human background preparation</h1><p>Stage: <strong>'+html.escape(state['stage'])+'</strong>. '
        'No model training or test evaluation has been performed.</p>'
        '<p>Tabula Sapiens supplies donor-separated train/validation candidates. HLCA and immune reference '
        'remain external candidates pending overlap review. Missing genes require the exported feature mask.</p>'
        +table.to_html(index=False,escape=True,border=0)+'<h2>Audit details</h2><pre>'
        +html.escape(json.dumps(state,indent=2,ensure_ascii=False))+'</pre>',encoding='utf-8')
    with tarfile.open(root/'background_clean_review.tar.gz','w:gz') as tar:
        for name in ('report.json','report.html','summary.csv','run_config.json','donor_split.json','panel.txt','COMPLETE.json'):
            if (root/name).exists():tar.add(root/name,arcname=name)
        for path in sorted(root.glob('*/*.json')):tar.add(path,arcname=path.relative_to(root))
        for path in sorted(root.glob('*/*.csv')):tar.add(path,arcname=path.relative_to(root))


def run(args):
    if args.min_counts<=0 or args.min_genes<=0 or not 0<=args.max_mito_pct<=100:
        raise ValueError('QC thresholds must be positive, with mitochondrial percentage in [0,100]')
    if any(getattr(args,k)<=0 for k in ('expected_genes','chunk_rows','shard_rows','per_stratum','max_cells_per_dataset')):
        raise ValueError('Panel size, chunk size, shard size and sampling caps must be positive')
    if not np.isfinite(args.reserve_gib) or args.reserve_gib<0:raise ValueError('Invalid disk reserve')
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    source_root=Path(args.input).resolve()
    source_report=json.loads((source_root/'inspect_report.json').read_text())
    if source_report.get('stage')!='inspect_complete':raise ValueError('Input public inspection must have completed')
    keys=[d['key'] for d in source_report['datasets']]
    if len(keys)!=len(set(keys)) or not any(k.startswith('ts_') for k in keys):
        raise ValueError('Need unique dataset keys and at least one Tabula Sapiens dataset')
    if set(keys)-{'ts_blood','ts_bone_marrow','ts_lung','hlca_core','immune_global'}:
        raise ValueError('Unknown source dataset; review its preparation policy explicitly')
    panel=load_panel(args.panel,args.expected_genes)
    cfg={k:v for k,v in vars(args).items() if k not in ('input','output','panel')}
    cfg.update(panel=panel,input=str(source_root),
               implementation_sha256=digest(__file__).hexdigest(),
               inspect_helper_sha256=digest(Path(__file__).with_name('background_inspect.py')).hexdigest(),
               source_report_sha256=digest(source_root/'inspect_report.json').hexdigest())
    p=root/'run_config.json'
    if p.exists() and json.loads(p.read_text())!=cfg:raise ValueError('Configuration changed; use a new output directory')
    (root/'COMPLETE.json').unlink(missing_ok=True)
    write_json(p,cfg);(root/'panel.txt').write_text('\n'.join(panel)+'\n')
    state={'stage':'preflight','datasets':{},'test_evaluated':False,'training_started':False,
           'limitations':['QC thresholds are fixed pilot defaults, not universal biological quality criteria.',
           'Primary-data False is retained; it is not a low-quality flag.',
           'Cross-atlas donor/source overlap remains unverified; external sets are candidates only.',
           'Normal tissue references do not replace matched cell-line controls.']}
    report(root,state)
    try:
        if shutil.disk_usage(root).free<args.reserve_gib*1024**3:raise ValueError('Insufficient free data-disk reserve')
        frames={};mappings={};paths={}
        for entry in source_report['datasets']:
            key=entry['key']
            if not re.fullmatch(r'[a-z0-9_]+',key):raise ValueError('Invalid dataset key')
            if entry['status']!='inspected_quarantine':raise ValueError('Dataset has not passed acquisition inspection')
            path=source_root/'h5ad'/(key+'.h5ad');paths[key]=path
            print(f'VERIFY SOURCE: {key}',flush=True)
            if digest(path).hexdigest()!=entry['download']['sha256']:raise ValueError(f'Source SHA256 mismatch: {key}')
            dest=root/key;dest.mkdir(exist_ok=True)
            state['stage']='full_qc';state['active_dataset']=key;report(root,state)
            cached=dest/'qc_complete.json'
            if cached.exists():
                checks=json.loads(cached.read_text())
                if any(digest(dest/f).hexdigest()!=h for f,h in checks.items()):raise ValueError('QC cache corrupted')
                obs=pd.read_csv(dest/'qc_cells.csv.gz',keep_default_na=False,float_precision='round_trip',
                                dtype={c:str for c in ('cell_id','donor_id','tissue','cell_type','assay','disease','sample_id','sample','study')})
                mapping=pd.read_csv(dest/'gene_mapping.csv',keep_default_na=False)
                summary=json.loads((dest/'qc_summary.json').read_text())
                print(f'REUSE QC: {key}',flush=True)
            else:
                obs,mapping,summary=qc_dataset(path,dest,key,panel,cfg)
                write_json(cached,{p.name:digest(p).hexdigest() for p in [dest/'qc_cells.csv.gz',dest/'gene_mapping.csv',dest/'qc_summary.json']})
            frames[key]=obs;mappings[key]=mapping;state['datasets'][key]=summary
            report(root,state)
        splits=split_donors(frames,args.seed);write_json(root/'donor_split.json',splits)
        training_donors=set();val_donors=set()
        for key,obs in frames.items():
            dest=root/key;state.update(stage='exporting',active_dataset=key);report(root,state)
            selected=select_cells(obs,key,splits,cfg)
            membership_fingerprint=hashlib.sha256(selected[['source_row','split','selected','exclusion_reasons']].to_csv(index=False).encode()).hexdigest()
            for split,target in [('train',training_donors),('val',val_donors)]:target.update(selected.loc[selected.split==split,'donor_group'])
            done=dest/'COMPLETE.json'
            if done.exists():
                completion=json.loads(done.read_text())
                if digest(dest/'membership.csv.gz').hexdigest()!=completion['membership_sha256']:
                    raise ValueError('Membership cache corrupted')
                if membership_fingerprint!=completion['membership_fingerprint']:
                    raise ValueError('Membership changed across resume')
                shards=completion['shards']
                if any(digest(dest/s['file']).hexdigest()!=s['sha256'] for s in shards):raise ValueError('Export shard checksum mismatch')
                print(f'REUSE EXPORT: {key}',flush=True)
            else:
                selected.to_csv(dest/'membership.csv.gz',index=False)
                shards=export_shards(paths[key],dest,selected,mappings[key],cfg)
                write_json(done,{'shards':shards,'membership_sha256':digest(dest/'membership.csv.gz').hexdigest(),
                    'membership_fingerprint':membership_fingerprint})
            state['datasets'][key].update(selected_cells=int(selected.selected.sum()),
                sampling_excluded=int((selected.exclusion_reasons=='sampling_cap').sum()),
                split_counts=selected.loc[selected.selected,'split'].value_counts().to_dict(),
                selected_donors=int(selected.loc[selected.selected,'donor_group'].nunique()),
                panel_complete=bool(mappings[key].available.all()),
                requires_feature_mask=not bool(mappings[key].available.all()), shards=len(shards))
            selected[selected.selected].groupby(['split','donor_id','tissue','cell_type','assay']).size().rename('cells').reset_index().to_csv(dest/'selected_coverage.csv',index=False)
            report(root,state)
        if training_donors & val_donors:raise ValueError('Donor leakage detected')
        if not training_donors or not val_donors:raise ValueError('Empty training or validation donor partition after sampling')
        state.update(stage='complete',train_donors=sorted(training_donors),val_donors=sorted(val_donors),donor_disjoint=True)
        write_json(root/'COMPLETE.json',{'stage':'complete','test_evaluated':False,'training_started':False})
    except Exception as exc:
        state.update(stage='failed',error=str(exc));raise
    finally:report(root,state)
    print(f'BACKGROUND CLEAN COMPLETE: {root / "background_clean_review.tar.gz"}',flush=True)
