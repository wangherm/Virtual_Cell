"""Trusted data broker. Download/inspect live here, never in generated workers."""
import hashlib
from pathlib import Path
import shutil
from urllib.parse import urlparse,urljoin
import numpy as np
import requests
from ..model_agent.artifacts import read
from ..model_agent.contracts import protected,cell_line_key
from ..utils import write_json,file_sha256

ALLOWED_HOSTS={'zenodo.org','www.ncbi.nlm.nih.gov','ftp.ncbi.nlm.nih.gov'}


def public_response(url,headers):
    for _ in range(5):
        parsed=urlparse(url)
        if parsed.scheme!='https' or parsed.hostname not in ALLOWED_HOSTS or parsed.port not in (None,443):raise ValueError('Dataset redirect outside approved public hosts')
        response=requests.get(url,headers=headers,stream=True,timeout=(15,120),allow_redirects=False)
        if not response.is_redirect:return response
        location=response.headers.get('Location','');response.close()
        if not location:raise ValueError('Public download redirect missing Location')
        url=urljoin(url,location)
    raise ValueError('Dataset redirect limit exceeded')


def fetch(entry,dest,max_bytes=2*1024**3,reserve_bytes=20*1024**3):
    """Bounded download; preserve interrupted/failed files, never delete old data."""
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    if urlparse(entry['url']).scheme!='https' or urlparse(entry['url']).hostname not in ALLOWED_HOSTS:raise ValueError('Dataset URL outside approved public hosts')
    filename=entry['filename']
    if Path(filename).name!=filename:raise ValueError('Dataset filename must be a basename')
    target=dest/filename
    def verified(path):
        h=hashlib.md5()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
        return h.hexdigest()==entry['md5']
    if target.exists():
        if not verified(target):raise ValueError('Existing dataset checksum mismatch; retained for inspection')
        return target
    if shutil.disk_usage(dest).free<reserve_bytes+max_bytes:raise ValueError('Insufficient reserved data-disk space')
    part=target.with_suffix(target.suffix+'.part');offset=part.stat().st_size if part.exists() else 0
    if part.exists() and verified(part):part.replace(target);return target
    if offset>max_bytes:raise ValueError('Partial file exceeds download cap')
    with public_response(entry['url'],{'Range':f'bytes={offset}-'} if offset else {}) as response:
        response.raise_for_status()
        if response.status_code==206:
            if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('Invalid resume range')
            mode='ab'
        else:offset=0;mode='wb'
        if int(response.headers.get('Content-Length',0))+offset>max_bytes:raise ValueError('Download size cap exceeded')
        with part.open(mode) as f:
            for chunk in response.iter_content(8*1024**2):
                if chunk:
                    offset+=len(chunk)
                    if offset>max_bytes:raise ValueError('Download size cap exceeded')
                    f.write(chunk)
                    print('DATA DOWNLOAD',entry['id'],offset,flush=True)
    if not verified(part):raise ValueError('Downloaded checksum mismatch; partial file retained')
    part.replace(target);return target


def inspect_h5ad(path):
    import anndata as ad
    a=ad.read_h5ad(path,backed='r')
    try:
        sample=a.X[:min(64,a.n_obs),:]
        values=sample.data if hasattr(sample,'toarray') else np.asarray(sample).ravel()
        count_like=bool(np.isfinite(values).all() and (values>=0).all() and np.allclose(values,np.round(values),rtol=0.,atol=1e-5))
        return {'cells':a.n_obs,'genes':a.n_vars,'obs_columns':list(a.obs.columns),'var_columns':list(a.var.columns),
            'var_column_samples':{c:a.var[c].astype(str).head(5).tolist() for c in a.var.columns},
            'layers':list(a.layers),'var_names_sample':a.var_names[:8].tolist(),'sample_X_count_like':count_like,
            'perturbation_candidates':{c:a.obs[c].astype(str).value_counts().head(12).to_dict() for c in a.obs.columns if c.lower() in ('perturbation','gene','gene_target','target','condition','guide_id','cell_type','cell_line','batch','replicate')},
            'training_ready':False,'reason':'Requires explicit count layer, target/control/stratum and gene-ID mapping; count-like sample is not sufficient evidence'}
    finally:a.file.close()


def inventory(work):
    work=Path(work);records=[]
    for p in sorted((work/'prepared').rglob('dataset.npz')):
        if p.is_symlink():continue
        metadata=p.parent/'metadata.csv';audit=p.parent/'data_audit.json'
        records.append({'id':str(p.parent.relative_to(work)),'kind':'prepared','path':str(p.parent),'bytes':p.stat().st_size,
            'dataset_sha256':file_sha256(p),'metadata_sha256':file_sha256(metadata) if metadata.exists() else None,
            'audit':read(audit) if audit.exists() else {},'role':'unassigned; broker needs source policy; no automatic test-to-train conversion'})
    return {'datasets':records,'scope':'All discovered prepared artifacts are catalogued. Sealed/test labels are not authorized merely by inventory.'}


def expanded_catalog(work,base,canonical_path):
    """Admit all compatible, nonsynthetic prepared sources with frozen provenance."""
    from ..data import load_prepared
    sources=list(base['sources']);report=[];wanted=base.get('output_panel',{}).get('genes')
    if wanted is None:raise ValueError('Freeze the common output panel before prepared-source expansion')
    for record in inventory(work)['datasets']:
        path=Path(record['path'])
        if path.resolve()==Path(canonical_path).resolve():continue
        try:
            d=load_prepared(path)
            if d['audit'].get('synthetic'):raise ValueError('Synthetic/demo data')
            if not set(wanted)<=set(d['genes']):raise ValueError('Frozen output-panel coverage incomplete')
            normalization=str(d['audit'].get('normalization',''))
            if 'log1p' not in normalization or '10000' not in normalization:raise ValueError('Original mean-log normalized response scale not verified')
            sources.append({'id':record['id'],'role':'source','prepared':str(path),'dataset_sha256':record['dataset_sha256'],
                'metadata_sha256':record['metadata_sha256'],'audit_sha256':file_sha256(path/'data_audit.json')})
            report.append({'id':record['id'],'status':'ADMITTED_SOURCE_CANDIDATE','rows':len(d['meta'])})
        except (ValueError,KeyError,OSError) as exc:report.append({'id':record['id'],'status':'BLOCKED','reason':str(exc)})
    return {**base,'sources':sources,'prepared_admission_report':report,'prepared_policy':'All compatible nonsynthetic preparations; same physical-unit identities retain the canonical artifact first. Fold exclusions still apply.'}


def legal_auxiliary(meta,outer_context,query_contexts,query_targets,confirmation_targets,held_targets,task,phase,target_aliases=None):
    """Whole-cell-line and target exclusions apply to every external source too."""
    contexts=meta.context.astype(str)
    def canonical(c):return cell_line_key(c).replace('-','').replace(' ','')
    forbidden={canonical(outer_context),*(canonical(c) for c in query_contexts)}
    cells=meta.cell_line.astype(str) if 'cell_line' in meta else contexts.map(cell_line_key)
    cells=cells.where(~cells.str.lower().isin(('unknown','unspecified','nan','none','')),contexts.map(cell_line_key))
    ok=~contexts.map(protected) & ~cells.map(lambda c:protected(c) or canonical(c) in forbidden)
    aliases=target_aliases or {}
    def gene(s):
        s=str(s);s=s.split('.')[0] if s.startswith('ENSG') else s
        return aliases.get(s,s)
    targets=meta.perturbation.astype(str).map(gene)
    ok &= ~targets.isin({gene(s) for s in held_targets})
    if phase=='search':ok &= ~targets.isin({gene(s) for s in confirmation_targets})
    if task=='repeat':ok &= ~targets.isin({gene(s) for s in query_targets})
    if 'split' in meta:ok &= meta.split.astype(str).str.lower().isin(('train','val'))
    return ok.to_numpy()


def load_source_bundle(entry,genes):
    """Audited numerical interchange contract, also usable by future adapters."""
    if entry.get('role')!='source':raise ValueError('Only source-role bundles can enter training')
    path=Path(entry['bundle'])
    if file_sha256(path)!=entry['sha256']:raise ValueError('Source bundle checksum changed')
    with np.load(path,allow_pickle=False) as f:d={k:f[k] for k in f.files}
    required={'genes','baseline','delta','context','cell_line','target','dataset','stimulus','modality','batch','time'}
    if not required<=d.keys() or not np.array_equal(d['genes'],genes):raise ValueError('Source bundle schema or frozen gene panel mismatch; no silent zero fill')
    n=len(d['target'])
    if d['delta'].shape!=(n,len(genes)) or d['baseline'].shape!=d['delta'].shape:raise ValueError('Source expression shape mismatch')
    if not np.isfinite(d['delta']).all() or not np.isfinite(d['baseline']).all():raise ValueError('Nonfinite source response')
    for k in required-{'genes','baseline','delta'}:
        if d[k].shape!=(n,):raise ValueError('Source metadata shape mismatch: '+k)
    if any(str(c).lower() in ('','unknown','nan','none','unspecified') for c in d['cell_line']):raise ValueError('External source needs explicit canonical cell-line identity')
    return d
