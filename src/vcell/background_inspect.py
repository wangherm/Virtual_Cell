"""Bounded scBaseCount acquisition for inspection, never training admission."""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import tarfile
import time

import h5py
import numpy as np
import pandas as pd

BUCKET = 'arc-institute-virtual-cell-atlas'
RELEASE = '2026-01-12'
FEATURES = ('Gene', 'GeneFull', 'GeneFull_ExonOverIntron', 'GeneFull_Ex50pAS')
ALIASES = {'K562': r'(?<![a-z0-9])k[ -]?562(?![a-z0-9])',
           'RPE1': r'(?<![a-z0-9])(?:htert[- ]*)?rpe[- ]?1(?![a-z0-9])',
           'H1': r'(?<![a-z0-9])(?:h1|wa[ -]?01)(?![a-z0-9])'}
HELDOUT = re.compile(r'hep[ -]?g2|jurkat', re.I)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h


def object_name(uri, release, feature):
    """Only a pinned release/feature/species; metadata cannot redirect downloads."""
    uri = str(uri)
    prefix = f'gs://{BUCKET}/scbasecount/{release}/h5ad/{feature}/Homo_sapiens/'
    if not uri.startswith(prefix) or not uri.endswith('.h5ad'):
        raise ValueError('Not a human h5ad in the selected release/count feature')
    if any(x in ('..', '.') for x in uri[len(prefix):].split('/')):
        raise ValueError('Invalid object path')
    return uri[len(f'gs://{BUCKET}/'):]


def candidates(frame, release, feature):
    required = {'srx_accession', 'file_path', 'cell_line', 'perturbation'}
    if not required <= set(frame):
        raise ValueError(f'Metadata schema missing {sorted(required-set(frame))}')
    rows = []
    seen = set()
    for _, original in frame.iterrows():
        row = {str(k): '' if pd.isna(v) else str(v) for k, v in original.items()}
        label = row['cell_line']
        matches = [k for k, pattern in ALIASES.items() if re.search(pattern, label, re.I)]
        heldout = bool(HELDOUT.search(' '.join(row.values())))
        if not matches and not heldout:
            continue
        row.update(background=','.join(matches), training_admitted=False,
                   control_status='unverified', decision='candidate', reason='metadata-only candidate')
        perturbation = row['perturbation'].strip().lower()
        if perturbation in ('untreated', 'unperturbed', 'control', 'no perturbation', 'none', 'no'):
            row['control_status'] = 'metadata_claims_baseline_not_verified'
        if heldout:
            row.update(decision='excluded', reason='held-out background mentioned in metadata')
        elif len(matches) != 1:
            row.update(decision='excluded', reason='multiple target backgrounds in sample')
        elif re.search(r'[,;/+]|\band\b|\bmix', label, re.I):
            row.update(decision='excluded', reason='possible mixed background; resolve first')
        try:
            object_name(row['file_path'], release, feature)
        except ValueError as exc:
            row.update(decision='excluded', reason=str(exc))
        if row['file_path'] in seen:
            row.update(decision='excluded', reason='duplicate file path in metadata')
        seen.add(row['file_path'])
        row['source_group'] = next((row[k] for k in ('study_accession', 'bioproject', 'czi_collection_id')
                                         if row.get(k) and row[k].lower() not in ('none', 'nan', 'unsure')), '')
        row['review_note'] = ('Verify cell-line identity/subline, cell-level controls and source overlap; '
                              'H1 aliases may be ambiguous. Missing study IDs do not establish independent studies.')
        rows.append(row)
    return pd.DataFrame(rows, columns=list(frame.columns) + [k for k in
        ('background', 'training_admitted', 'control_status', 'decision', 'reason', 'source_group', 'review_note')
        if k not in frame.columns])


class GCS:
    def __init__(self, project):
        if not project:
            raise ValueError('Pass --billing-project for the Marketplace-subscribed GCP project; see docs/BACKGROUND_INSPECT.md')
        from google.cloud import storage
        self.bucket = storage.Client(project=project).bucket(BUCKET, user_project=project)

    def stat(self, name):
        blob = self.bucket.blob(name)
        blob.reload(timeout=60)
        return {'object': name, 'generation': str(blob.generation), 'size': int(blob.size),
                'md5': blob.md5_hash, 'crc32c': blob.crc32c}

    def chunk(self, identity, start, end):
        blob = self.bucket.blob(identity['object'], generation=int(identity['generation']))
        return blob.download_as_bytes(start=start, end=end, timeout=120, checksum=None,
                                      if_generation_match=int(identity['generation']))


def download(remote, identity, dest, reserve_bytes=2 * 1024**3):
    """Generation-pinned range resume; full cloud checksum before promotion."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    stamp = dest.with_suffix(dest.suffix + '.download.json')
    old = json.loads(stamp.read_text()) if stamp.exists() else None
    if old and old['identity'] != identity:
        raise ValueError(f'Source identity changed; use a new output directory: {dest}')
    if dest.exists():
        if not old or dest.stat().st_size != identity['size'] or digest(dest).hexdigest() != old.get('sha256'):
            raise ValueError(f'Existing file failed validation: {dest}')
        print(f'REUSE VERIFIED: {dest.name}', flush=True)
        return
    if part.exists() and not old:
        raise ValueError(f'Unidentified partial download: {part}')
    offset = part.stat().st_size if part.exists() else 0
    if offset > identity['size']:
        raise ValueError('Partial file exceeds cloud object size')
    if shutil.disk_usage(dest.parent).free < identity['size'] - offset + reserve_bytes:
        raise ValueError('Insufficient disk space including reserve; nothing deleted')
    write_json(stamp, {'identity': identity, 'state': 'downloading'})
    with open(part, 'ab') as f:
        while offset < identity['size']:
            end = min(offset + 16 * 1024**2, identity['size']) - 1
            for attempt in range(4):
                try:
                    block = remote.chunk(identity, offset, end)
                    if len(block) != end - offset + 1:
                        raise IOError('Short range response')
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            f.write(block)
            f.flush()
            offset += len(block)
            print(f'DOWNLOAD {dest.name}: {offset}/{identity["size"]} bytes ({100*offset/max(1,identity["size"]):.1f}%)', flush=True)
    if identity.get('md5'):
        actual = base64.b64encode(digest(part, 'md5').digest()).decode()
        expected = identity['md5']
    elif identity.get('crc32c'):
        import google_crc32c
        checksum = google_crc32c.Checksum()
        with open(part, 'rb') as f:
            for block in iter(lambda: f.read(8 * 1024**2), b''):
                checksum.update(block)
        actual = base64.b64encode(checksum.digest()).decode()
        expected = identity['crc32c']
    else:
        raise ValueError('Cloud object has no checksum; refusing unverified promotion')
    if actual != expected:
        raise IOError(f'Cloud checksum mismatch; partial retained at {part}')
    write_json(stamp, {'identity': identity, 'state': 'verified', 'sha256': digest(part).hexdigest()})
    part.replace(dest)


def decode(values):
    return np.array([v.decode('utf-8', errors='replace') if isinstance(v, bytes) else str(v) for v in values])


def column(group, key, limit=None):
    node = group[key]
    if isinstance(node, h5py.Dataset):
        return decode(node[:limit])
    if 'codes' in node and 'categories' in node:
        codes, categories = node['codes'][:limit], decode(node['categories'][:])
        return np.array([categories[c] if c >= 0 else '<missing>' for c in codes])
    raise ValueError(f'Unsupported annotation encoding: {key}')


def inspect_h5ad(path, panel=None, sample_rows=256, matrix='X'):
    """Bounded HDF5 reads: no dense full-matrix conversion, no label transformations."""
    result = {'file': str(path), 'bytes': Path(path).stat().st_size,
              'training_admitted': False, 'warnings': [], 'qc_scope': f'sampled {matrix}, not full cell QC'}
    with h5py.File(path, 'r') as f:
        x = f[matrix]
        result['matrix_path'] = matrix
        if isinstance(x, h5py.Dataset):
            shape, encoding = x.shape, 'dense'
        else:
            shape = tuple(int(v) for v in x.attrs['shape'])
            encoding = x.attrs.get('encoding-type', '')
            if isinstance(encoding, bytes):
                encoding = encoding.decode()
        if len(shape) != 2 or not all(shape):
            raise ValueError('Empty or non-matrix X')
        result.update(n_cells=int(shape[0]), n_genes=int(shape[1]), x_encoding=encoding,
                      layers=list(f.get('layers', {})), obs_columns=list(f['obs']), var_columns=list(f['var']))
        var = f['raw/var'] if matrix == 'raw/X' else f['var']
        result['var_columns'] = list(var)
        index_key = var.attrs.get('_index', '_index')
        if isinstance(index_key, bytes):
            index_key = index_key.decode()
        ids = column(var, index_key)
        result['duplicate_var_ids'] = int(len(ids) - len(set(ids)))
        symbol_key = next((k for k in ('gene_symbols', 'gene_symbol', 'gene_name', 'feature_name', 'symbol') if k in var), None)
        symbols = column(var, symbol_key) if symbol_key else ids
        result['symbol_column'] = symbol_key or 'var_index (not verified as symbols)'
        result['duplicate_symbols'] = int(len(symbols) - len(set(symbols)))
        result['gene_examples'] = [{'id': a, 'symbol': b} for a,b in zip(ids[:8], symbols[:8])]
        if panel is not None:
            # Match exact IDs/symbols only; no unreviewed aliases or version stripping.
            missing = sorted(set(panel) - set(ids) - set(symbols))
            result['panel'] = {'n': len(set(panel)), 'matched': len(set(panel))-len(missing), 'missing': missing}
        else:
            result['warnings'].append('No gene panel supplied; output-panel coverage not assessed')
        annotations = {}
        for key in f['obs']:
            if re.search('cell.?type|cell.?line|condition|perturb|donor|batch|study|sample|treatment|control', key, re.I):
                try:
                    values = column(f['obs'], key, 5000)
                    annotations[key] = {'first_n_cells': len(values), 'top_values': Counter(values).most_common(15)}
                    if any(HELDOUT.search(v) for v in values):
                        result['warnings'].append(f'Held-out background found in sampled obs column {key}')
                except ValueError as exc:
                    annotations[key] = {'unsupported': str(exc)}
        result['sampled_annotations'] = annotations
        # At most two million dense-equivalent values in the sampled row set.
        if shape[1] > 200000:
            raise ValueError('Gene axis exceeds bounded inspection limit')
        row_limit = min(sample_rows, shape[0], max(1, 2_000_000 // shape[1]))
        indices = np.unique(np.linspace(0, shape[0]-1, row_limit, dtype=int))
        totals, detected, entries = [], [], []
        # Sparse QC includes all genes for sampled CSR rows; dense reads one row at a time.
        if encoding in ('csr_matrix', 'dense'):
            for i in indices:
                if encoding == 'dense':
                    row = np.asarray(x[int(i), :], dtype=np.float64)
                else:
                    start, end = x['indptr'][int(i):int(i)+2]
                    if end-start > 2_000_000:
                        raise ValueError('Sparse row exceeds bounded inspection limit')
                    row = np.asarray(x['data'][int(start):int(end)], dtype=np.float64)
                totals.append(float(row.sum()))
                detected.append(int(np.count_nonzero(row)))
                entries.append(row)
            values = np.concatenate(entries)
            result['sampled_cells'] = len(indices)
            result['cell_sum_quantiles'] = [float(v) if np.isfinite(v) else None for v in np.quantile(totals, [0,.5,1])]
            result['detected_gene_quantiles'] = np.quantile(detected, [0,.5,1]).tolist()
        elif encoding == 'csc_matrix':
            values = np.asarray(x['data'][:200000], dtype=np.float64)
            result['warnings'].append('CSC: checked first 200000 stored entries only; per-cell QC deferred')
        else:
            raise ValueError(f'Unsupported X encoding: {encoding}')
        finite = values[np.isfinite(values)]
        result['sampled_values'] = {'n': int(len(values)), 'nonfinite': int((~np.isfinite(values)).sum()),
            'negative': int((finite < 0).sum()), 'noninteger': int((np.abs(finite-np.rint(finite)) > 1e-5).sum())}
        result['warnings'].append('Baseline status, source duplication and per-cell intervention labels require review; this report does not certify counts or training eligibility')
    return result


def report(output, state, rows, inspections):
    write_json(output/'inspect_report.json', {**state, 'files': inspections})
    fields = ['background','srx_accession','control_status','decision','reason']
    table = rows[[k for k in fields if k in rows]].to_html(index=False, escape=True)
    content = ('<!doctype html><meta charset="utf-8"><title>Background inspection</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto}td,th{padding:6px;border:1px solid #ddd}'
        'table{border-collapse:collapse}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
        '<h1>scBaseCount background inspection</h1><p>Inspection only. No training admission or model fitting.</p>'
        '<h2>Run status and coverage</h2><pre>'+html.escape(json.dumps(state, indent=2, ensure_ascii=False))+'</pre>'
        '<h2>Candidate decisions</h2>'+table+'<h2>File inspections</h2>' + ''.join(
            '<pre>'+html.escape(json.dumps(x, indent=2, ensure_ascii=False))+'</pre>' for x in inspections))
    (output/'inspect_report.html').write_text(content, encoding='utf-8')
    with tarfile.open(output/'background_inspect_review.tar.gz', 'w:gz') as tar:
        for name in ('inspect_report.json','inspect_report.html','candidates.csv','download_manifest.json','run_config.json'):
            if (output/name).exists():
                tar.add(output/name, arcname=name)


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {k: v for k,v in vars(args).items() if k not in ('billing_project',)}
    cfg = output/'run_config.json'
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise ValueError('Configuration changed; use a new output directory')
    write_json(cfg, config)
    rows, inspections, manifest = pd.DataFrame(), [], []
    state = {'stage': 'starting', 'training_admitted': False, 'errors': [], 'test_evaluated': False}
    try:
        remote = None
        if args.metadata:
            metadata = Path(args.metadata)
        else:
            remote = GCS(args.billing_project)
            name = f'scbasecount/{args.release}/metadata/{args.feature}/Homo_sapiens/sample_metadata.parquet'
            identity = remote.stat(name)
            if identity['size'] > 256 * 1024**2:
                raise ValueError('Sample metadata exceeds 256 MiB cap')
            metadata = output/'sample_metadata.parquet'
            download(remote, identity, metadata)
        metadata_hash = digest(metadata).hexdigest()
        identity_path = output/'metadata_identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text())['sha256'] != metadata_hash:
            raise ValueError('Metadata contents changed; use a new output directory')
        write_json(identity_path, {'sha256': metadata_hash, 'path': str(metadata)})
        frame = pd.read_parquet(metadata) if metadata.suffix == '.parquet' else pd.read_csv(metadata, keep_default_na=False)
        rows = candidates(frame, args.release, args.feature)
        state.update(stage='metadata_inspected', metadata_sha256=metadata_hash, metadata_rows=len(frame),
                     candidate_rows=int((rows.decision=='candidate').sum()),
                     coverage={bg: int(((rows.background==bg)&(rows.decision=='candidate')).sum()) for bg in ALIASES},
                     limitations=['Candidate identity and baseline claims are unverified.',
                     'Independent-study coverage and overlap with existing data are not established.',
                     'Sampled QC cannot establish whole-file baseline status or exclude every held-out cell.'])
        rows.to_csv(output/'candidates.csv', index=False)
        report(output, state, rows, inspections)
        if args.metadata_only:
            print(f'METADATA INSPECT COMPLETE: {output / "inspect_report.html"}', flush=True)
            return
        panel = None
        if args.panel:
            panel = [s.strip() for s in Path(args.panel).read_text().splitlines() if s.strip()]
        total, counts, used_groups = 0, Counter(), set()
        eligible = rows[rows.decision=='candidate'].copy()
        eligible['_priority'] = eligible.control_status.ne('metadata_claims_baseline_not_verified').astype(int)
        eligible = eligible.sort_values(['_priority','background','srx_accession'])
        # Round-robin source groups within each background, keeping unknown groups explicit.
        ordering = []
        remaining = list(eligible.index)
        while remaining:
            for bg in ALIASES:
                same = [i for i in remaining if rows.at[i,'background']==bg]
                if not same:
                    continue
                i = next((i for i in same if rows.at[i,'source_group'] and (bg,rows.at[i,'source_group']) not in used_groups), same[0])
                used_groups.add((bg,rows.at[i,'source_group']))
                ordering.append(i); remaining.remove(i)
        for i in ordering:
            row = rows.loc[i]
            bg = row.background
            if counts[bg] >= args.per_background:
                rows.at[i,'reason'] = 'candidate beyond per-background download cap'
                continue
            name = object_name(row.file_path, args.release, args.feature)
            local_name = hashlib.sha256(name.encode()).hexdigest()[:12]+'_'+Path(name).name
            dest = output/'h5ad'/local_name
            try:
                if args.local_dir:
                    dest = Path(args.local_dir)/Path(name).name
                    if not dest.is_file():
                        rows.at[i,'reason'] = 'not available in local directory'
                        continue
                    size = dest.stat().st_size
                    identity = {'local_file': str(dest), 'size': size}
                else:
                    remote = remote or GCS(args.billing_project)
                    identity = remote.stat(name)
                    size = identity['size']
                if size > args.max_file_gib*1024**3 or total+size > args.max_total_gib*1024**3:
                    rows.at[i,'reason'] = 'candidate exceeds per-file or cumulative size cap'
                    continue
                # Failed inspections still count against transfer and file budgets.
                total += size; counts[bg] += 1
                print(f'INSPECT {bg} {row.srx_accession}: {size/1024**3:.3f} GiB', flush=True)
                if not args.local_dir:
                    download(remote, identity, dest)
                item = inspect_h5ad(dest, panel, args.sample_rows)
                item.update(background=bg, accession=row.srx_accession, source_identity=identity)
                inspections.append(item)
                manifest.append({'uri': row.file_path, 'local': str(dest), 'identity': identity, 'sha256': digest(dest).hexdigest()})
                rows.at[i,'decision'] = 'inspected_quarantine'
                rows.at[i,'reason'] = 'downloaded/inspected; not admitted to training'
            except Exception as exc:
                state['errors'].append({'accession': row.srx_accession, 'error': str(exc)})
                rows.at[i,'decision'] = 'failed'
                rows.at[i,'reason'] = str(exc)
                print(f'FAILED {row.srx_accession}: {exc}', flush=True)
            state.update(stage='inspecting', inspected_files=len(inspections), planned_bytes=total)
            rows.to_csv(output/'candidates.csv', index=False)
            write_json(output/'download_manifest.json', manifest)
            report(output, state, rows, inspections)
        state['stage'] = 'partial_failure' if state['errors'] else ('inspect_complete' if inspections else 'no_files_inspected')
    except Exception as exc:
        state.update(stage='blocked', errors=state['errors']+[{'error': str(exc)}])
        raise
    finally:
        rows.to_csv(output/'candidates.csv', index=False)
        write_json(output/'download_manifest.json', manifest)
        report(output, state, rows, inspections)
    if state['errors']:
        raise RuntimeError(f'Some files failed; see {output / "inspect_report.html"}. Rerun the same command to resume.')
    print(f'BACKGROUND INSPECT: {state["stage"]}: {output / "inspect_report.html"}', flush=True)
