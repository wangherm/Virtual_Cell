"""Append uncapped K562/RPE1 groups without changing any historical evaluation row."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import open_source, labels, count_chunks, save_prepared, load_prepared
from .round11 import freeze, verify_files
from .utils import file_sha256, write_json


def aggregate_halves(entry, genes, cfg):
    """Stream raw counts; split cells by stable ID, including controls separately.

    These are technical split-half diagnostics, not biological replicates.
    Available intervention/time strata must already be constant within batch.
    """
    a, names = open_source(entry, Path('/'))
    try:
        if not a.obs_names.is_unique:
            raise ValueError('Nonunique cell IDs: split-half assignment is ambiguous')
        for col in ('study', 'study_id', 'intervention_type', 'perturbation_type',
                    'time', 'timepoint', 'duration'):
            if col in a.obs:
                batch = entry.get('batch_key')
                variation = a.obs.groupby(batch, observed=True)[col].nunique(dropna=False) if batch else pd.Series([a.obs[col].nunique(dropna=False)])
                if (variation > 1).any():
                    raise ValueError(f'{entry["id"]}: {col} varies within batch; explicitly stratify before expansion')
        loc = {g: i for i, g in enumerate(names)}
        if not set(genes) <= set(loc):
            raise ValueError('Raw source does not cover the frozen panel')
        p, batches = labels(a, entry)
        sides = np.array([hashlib.sha256(('round12-half:' + str(x)).encode()).digest()[0] % 2
                          for x in a.obs_names], dtype=np.int8)
        groups = {}; rejected = set(); positive = 0
        for start, stop, x, keep in count_chunks(a, entry, np.array([loc[g] for g in genes]),
                                                cfg['chunk_size'], 10000):
            for j in np.flatnonzero(keep):
                target = p[start+j]
                if target != '__control__' and not re.fullmatch(r'[A-Za-z0-9_.-]+', target):
                    rejected.add(target); continue
                key = (str(batches[start+j]), str(target))
                if key not in groups:
                    groups[key] = [np.zeros((2, len(genes)), np.float64), np.zeros(2, np.int64)]
                side = sides[start+j]; groups[key][0][side] += x[j]; groups[key][1][side] += 1
                positive += 1
            if start == 0 or stop == a.n_obs or start // cfg['chunk_size'] % 25 == 0:
                print(f'ROUND12 RAW {entry["context"]} cells={stop}/{a.n_obs} groups={len(groups)}', flush=True)
        return groups, {'positive_library_cells': positive, 'rejected_nonsingle_labels': sorted(rejected),
                        'half_assignment': 'SHA256(round12-half:cell_id) first-byte parity',
                        'cell_qc': 'existing row filters, finite integer counts, positive full-library size; no new mitochondrial threshold'}
    finally:
        a.file.close()


def expand(work, old, dest):
    work, dest = Path(work), Path(dest)
    original = work/'prepared/real_min10_01/data_audit.json'
    info = json.loads(original.read_text())
    cfg = info['manifest']; paths = {v['id']: v['path'] for v in info['sources']}
    entries = []
    for item in cfg['datasets']:
        if item['context'] not in ('K562', 'RPE1'): continue
        entry = dict(item); path = Path(paths.get(entry['id'], entry['path']))
        if not path.is_file():
            matches = list((work/'raw').rglob(path.name))
            if len(matches) != 1: raise ValueError(f'Cannot uniquely locate raw source: {path.name}')
            path = matches[0]
        entry['path'] = str(path.resolve()); entries.append(entry)
    if sorted(e['context'] for e in entries) != ['K562', 'RPE1']:
        raise ValueError('Require one original K562 and one RPE1 source')
    dest.mkdir(parents=True, exist_ok=True)
    identity = {'original': old['audit']['fingerprint'], 'original_manifest': file_sha256(original),
                'builder': file_sha256(__file__), 'sources': []}
    for e in entries:
        print(f'ROUND12 VERIFY RAW: {e["path"]}', flush=True)
        identity['sources'].append({'entry': e, 'sha256': file_sha256(e['path'])})
    freeze(dest/'plan.json', identity)
    if (dest/'COMPLETE.json').exists():
        verify_files(dest); return load_prepared(dest)
    oldmeta = old['meta']; oldkeys = set(zip(oldmeta.context, oldmeta.perturbation))
    extra_meta, baselines, deltas, qc, audits = [], [], [], [], []
    half_a, half_b, half_meta = [], [], []
    for e in entries:
        cache = dest/('raw_' + e['context']); cache.mkdir(exist_ok=True)
        if (cache/'COMPLETE.json').exists():
            verify_files(cache)
            with np.load(cache/'groups.npz', allow_pickle=False) as f:
                groups = {(b, p): [s, n] for b, p, s, n in zip(f['batch'], f['target'], f['sums'], f['counts'])}
            audit = json.loads((cache/'audit.json').read_text())
        else:
            groups, audit = aggregate_halves(e, old['genes'], cfg)
            keys = sorted(groups)
            np.savez_compressed(cache/'groups.npz', batch=np.array([k[0] for k in keys], dtype='U'),
                                target=np.array([k[1] for k in keys], dtype='U'),
                                sums=np.array([groups[k][0] for k in keys]), counts=np.array([groups[k][1] for k in keys]))
            write_json(cache/'audit.json', audit)
            write_json(cache/'COMPLETE.json', {'files': {n: file_sha256(cache/n) for n in ('groups.npz', 'audit.json')}})
        audits.append({'context': e['context'], **audit})
        for (batch, target), (s, n) in sorted(groups.items()):
            if target == '__control__': continue
            control = groups.get((batch, '__control__'))
            nc = int(control[1].sum()) if control else 0
            eligible = int(n.sum()) >= cfg['min_cells'] and nc >= cfg['min_control_cells']
            old_group = (e['context'], target) in oldkeys
            qc.append({'context': e['context'], 'batch': batch, 'target': target, 'n_cells': int(n.sum()),
                       'n_controls': nc, 'eligible': eligible, 'historical_target': old_group,
                       'added': bool(eligible and not old_group)})
            if not eligible: continue
            base = control[0].sum(0)/nc; effect = s.sum(0)/n.sum() - base
            if np.min(n) >= 5 and np.min(control[1]) >= 15:
                half_a.append(s[0]/n[0] - control[0][0]/control[1][0])
                half_b.append(s[1]/n[1] - control[0][1]/control[1][1])
                half_meta.append({'context': e['context'], 'batch': batch, 'perturbation': target,
                                  'half_a_cells': int(n[0]), 'half_b_cells': int(n[1]),
                                  'half_a_controls': int(control[1][0]), 'half_b_controls': int(control[1][1])})
            if old_group: continue
            baselines.append(base); deltas.append(effect)
            extra_meta.append({'row_id': json.dumps([e['id'], batch, target], separators=(',', ':')),
                               'dataset': e['id'], 'context': e['context'], 'batch': batch,
                               'perturbation': target, 'split': 'train', 'n_cells': int(n.sum()), 'n_controls': nc})
    if not extra_meta: raise ValueError('No extra QC-eligible targets; cannot claim a data-expansion comparison')
    meta = pd.concat([oldmeta, pd.DataFrame(extra_meta)], ignore_index=True)
    vocab = sorted(set(meta.perturbation)); pi = np.array([vocab.index(p) for p in meta.perturbation])
    save_prepared(dest, np.concatenate([old['baseline'], baselines]), np.concatenate([old['delta'], deltas]),
                  pi, old['genes'], vocab, meta, {**{k: old['audit'][k] for k in ('normalization', 'target_sum')},
                  'historical_rows': len(oldmeta), 'historical_fingerprint': old['audit']['fingerprint'],
                  'panel_selection': 'unchanged historical panel; no new feature selection',
                  'scope': 'append new K562/RPE1 targets only; all historical rows unchanged', 'raw_qc': audits})
    pd.DataFrame(qc).to_csv(dest/'expansion_qc.csv', index=False)
    pd.DataFrame(half_meta, columns=['context','batch','perturbation','half_a_cells','half_b_cells','half_a_controls','half_b_controls']).to_csv(dest/'half_metadata.csv', index=False)
    np.savez_compressed(dest/'halves.npz', a=np.asarray(half_a, np.float32).reshape(-1,len(old['genes'])),
                        b=np.asarray(half_b, np.float32).reshape(-1,len(old['genes'])))
    names = ['plan.json','dataset.npz','metadata.csv','data_audit.json','expansion_qc.csv','half_metadata.csv','halves.npz']
    write_json(dest/'COMPLETE.json', {'files': {n: file_sha256(dest/n) for n in names}})
    return load_prepared(dest)
