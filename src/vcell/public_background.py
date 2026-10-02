"""Anonymous, pinned public-atlas transfers and layer-aware inspection. No training."""
from collections import Counter
import html
import json
from pathlib import Path
import re
import shutil
import tarfile
import time
from urllib.parse import urlparse

import h5py
import requests

from .background_inspect import column, digest, inspect_h5ad, write_json

DEFAULT_DATASETS = ['ts_blood', 'ts_bone_marrow', 'ts_lung', 'hlca_core', 'immune_global']


def selection(manifest, keys):
    entries = {x['key']: x for x in manifest['datasets']}
    if not keys or len(set(keys)) != len(keys) or set(keys)-set(entries):
        raise ValueError('Choose unique dataset keys present in the manifest')
    selected = [entries[k] for k in keys]
    for x in selected:
        u = urlparse(x['url'])
        if (u.scheme != 'https' or u.netloc != 'datasets.cellxgene.cziscience.com'
                or u.path != '/' + x['dataset_version_id'] + '.h5ad' or u.query or u.fragment):
            raise ValueError('Only pinned public CELLxGENE H5AD version URLs are supported')
        if int(x['size_bytes']) <= 0:
            raise ValueError('Missing positive source file size')
    return selected


def identity(session, entry):
    with session.head(entry['url'], allow_redirects=True, timeout=(20,60),
                      headers={'Accept-Encoding': 'identity'}) as r:
        r.raise_for_status()
        size = int(r.headers.get('Content-Length', -1))
        if size != int(entry['size_bytes']):
            raise ValueError('Remote size differs from pinned manifest; do not silently update the release')
        etag = r.headers.get('ETag')
        if not etag or etag.startswith('W/'):
            raise ValueError('A strong HTTP ETag is required for safe resume')
        return {'url':entry['url'], 'size':size, 'etag':etag,
                'last_modified':r.headers.get('Last-Modified')}


def transfer(session, ident, dest, reserve=2*1024**3, chunk_size=16*1024**2):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix('.h5ad.part')
    sidecar = dest.with_suffix('.download.json')
    old = json.loads(sidecar.read_text()) if sidecar.exists() else None
    if old and old['identity'] != ident:
        raise ValueError('Remote identity changed; keep existing data and use a new output directory')
    if dest.exists():
        if not old or dest.stat().st_size != ident['size'] or digest(dest).hexdigest() != old.get('sha256'):
            raise ValueError('Existing file failed local size/SHA256 verification')
        return old
    if part.exists() and not old:
        raise ValueError('Unidentified partial file; will not append')
    offset = part.stat().st_size if part.exists() else 0
    if offset > ident['size']:
        raise ValueError('Partial file exceeds source size')
    if shutil.disk_usage(dest.parent).free < ident['size']-offset+reserve:
        raise ValueError('Insufficient data-disk space including reserve')
    write_json(sidecar, {'identity':ident, 'state':'downloading'})
    with open(part,'ab') as f:
        while offset < ident['size']:
            end = min(offset+chunk_size, ident['size'])-1
            headers = {'Range':f'bytes={offset}-{end}', 'If-Match':ident['etag'], 'Accept-Encoding':'identity'}
            for attempt in range(4):
                try:
                    with session.get(ident['url'], headers=headers, stream=True, timeout=(20,120)) as r:
                        r.raise_for_status()
                        expected = f'bytes {offset}-{end}/{ident["size"]}'
                        if r.status_code != 206 or r.headers.get('Content-Range') != expected:
                            raise ValueError('Server ignored or changed byte range; refusing to append a full file')
                        if r.headers.get('ETag') != ident['etag']:
                            raise ValueError('ETag changed during transfer')
                        written = 0
                        for block in r.iter_content(1024**2):
                            if written+len(block) > end-offset+1:
                                raise IOError('Range response exceeds requested length')
                            f.write(block); written += len(block)
                        if written != end-offset+1:
                            raise IOError('Incomplete range response')
                        f.flush()
                    break
                except Exception:
                    # Roll back a failed range before retrying, keeping earlier ranges.
                    f.flush(); f.truncate(offset); f.seek(offset)
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
            offset = end+1
            print(f'DOWNLOAD {dest.name}: {offset}/{ident["size"]} ({100*offset/ident["size"]:.1f}%)',flush=True)
    with open(part,'rb') as f:
        if f.read(8) != b'\x89HDF\r\n\x1a\n':
            raise ValueError('Downloaded file does not have an HDF5 header')
    record = {'identity':ident, 'state':'downloaded', 'sha256':digest(part).hexdigest(),
              'verification':'Pinned URL + strong ETag + exact ranges/size + local SHA256; no publisher SHA256 supplied'}
    write_json(sidecar,record)
    part.replace(dest)
    return record


def annotations(path):
    """Scan annotation columns in bounded chunks, not expression matrices."""
    result = {}
    pattern = r'donor|tissue|cell.?type|cell.?line|assay|disease|study|batch|sample|is_primary_data|organism|condition|perturb'
    with h5py.File(path,'r') as f:
        obs = f['obs']
        for key in obs:
            if not re.search(pattern,key,re.I):
                continue
            node = obs[key]
            if isinstance(node,h5py.Group) and 'codes' in node and 'categories' in node:
                cats = column(node,'categories')
                n = len(node['codes'])
                counts = Counter()
                for start in range(0,n,50000):
                    codes = Counter(node['codes'][start:start+50000].tolist())
                    counts.update({str(cats[k]) if k>=0 else '<missing>':v for k,v in codes.items()})
            elif isinstance(node,h5py.Dataset) and node.ndim == 1:
                n, counts = len(node), Counter()
                for start in range(0,n,50000):
                    values = node[start:start+50000]
                    counts.update(v.decode(errors='replace') if isinstance(v,bytes) else str(v) for v in values)
            else:
                result[key] = {'unsupported_encoding':True}
                continue
            # Annotation cardinality is not independently verified donor/study count.
            result[key] = {'n_cells_scanned':n,'n_distinct_labels':len(counts),
                           'top_values':counts.most_common(30)}
            if key=='assay':
                result[key]['cells_labelled_10x'] = sum(v for k,v in counts.items() if '10x' in k.lower())
                result[key]['cells_labelled_smartseq'] = sum(v for k,v in counts.items() if 'smart' in k.lower())
    return result


def inspect_atlas(path, panel=None, sample_rows=128):
    with h5py.File(path,'r') as f:
        matrices = ['X']
        if 'raw/X' in f:
            matrices.append('raw/X')
        matrices += ['layers/'+k for k in f.get('layers',{})]
    if len(matrices)>12:
        raise ValueError('More than 12 matrices; choose an explicit inspection policy first')
    inspected = [inspect_h5ad(path,panel,sample_rows,matrix=m) for m in matrices]
    obs = annotations(path)
    return {'matrices':inspected,'full_annotation_counts':obs,'training_admitted':False,
            'count_layer_status':'unverified; integer/nonnegative samples alone do not establish raw counts',
            'limitations':['Normal tissue atlases do not replace matched cell-line controls.',
             'Full objects may mix 10x and Smart-seq; no cells have been filtered yet.',
             'Cross-atlas donor/study/cell overlap is not resolved.',
             'Expression QC is sampled; annotation labels are counted but not externally verified.']}


def save_report(output, state):
    write_json(output/'inspect_report.json',state)
    table = '<table><tr><th>Dataset</th><th>Status</th><th>GiB</th></tr>' + ''.join(
        f'<tr><td>{html.escape(x["key"])}</td><td>{html.escape(x["status"])}</td><td>{x["size_bytes"]/1024**3:.2f}</td></tr>'
        for x in state['datasets']) + '</table>'
    page = ('<!doctype html><meta charset="utf-8"><title>Public background inspection</title>'
            '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
            'td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style>'
            '<h1>Public background inspection</h1><p>No training admission; no normalization or cell filtering performed.</p>'
            +table+'<pre>'+html.escape(json.dumps(state,indent=2,ensure_ascii=False))+'</pre>')
    (output/'inspect_report.html').write_text(page,encoding='utf-8')
    with tarfile.open(output/'public_background_review.tar.gz','w:gz') as tar:
        for filename in ('inspect_report.json','inspect_report.html','manifest.json','run_config.json'):
            if (output/filename).exists():
                tar.add(output/filename,arcname=filename)


def run(args):
    output = Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    selected = selection(manifest,args.datasets)
    config = dict(datasets=args.datasets, manifest_sha256=digest(args.manifest).hexdigest(),
                  panel_sha256=digest(args.panel).hexdigest() if args.panel else None,
                  max_total_gib=args.max_total_gib, sample_rows=args.sample_rows, catalog_only=args.catalog_only)
    config_path = output/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Configuration changed; use a different output directory')
    write_json(config_path,config); write_json(output/'manifest.json',manifest)
    state = {'stage':'starting','training_admitted':False,'test_evaluated':False,'datasets':[
        {**x,'status':'pending'} for x in selected]}
    save_report(output,state)
    panel = Path(args.panel).read_text().splitlines() if args.panel else None
    try:
        size = sum(x['size_bytes'] for x in selected)
        print(f'PUBLIC BACKGROUND PLAN: {len(selected)} files, {size/1024**3:.2f} GiB',flush=True)
        if size>args.max_total_gib*1024**3:
            raise ValueError('Selected files exceed total download cap; reduce --datasets or explicitly raise --max-total-gib')
        if args.catalog_only:
            state['stage']='catalog_only'; return
        state['stage']='acquiring_and_inspecting'
        # A single process owns the output; use separate output directories for concurrent runs.
        with requests.Session() as session:
            for item in state['datasets']:
                try:
                    item['status']='probing'; save_report(output,state)
                    ident = identity(session,item)
                    dest = output/'h5ad'/(item['key']+'.h5ad')
                    item['status']='downloading'; save_report(output,state)
                    item['download']=transfer(session,ident,dest)
                    item['status']='inspecting'; save_report(output,state)
                    item['inspection']=inspect_atlas(dest,panel,args.sample_rows)
                    item['status']='inspected_quarantine'
                    print(f'INSPECT COMPLETE: {item["key"]}',flush=True)
                except Exception as exc:
                    item.update(status='failed',error=str(exc))
                    print(f'FAILED {item["key"]}: {exc}',flush=True)
                save_report(output,state)
        state['stage']='partial_failure' if any(x['status']=='failed' for x in state['datasets']) else 'inspect_complete'
    except Exception as exc:
        state.update(stage='blocked',error=str(exc)); raise
    finally:
        save_report(output,state)
    if state['stage']!='inspect_complete':
        raise RuntimeError(f'Some files failed; rerun the same command. Report: {output / "inspect_report.html"}')
    print(f'PUBLIC BACKGROUND COMPLETE: {output / "public_background_review.tar.gz"}',flush=True)
