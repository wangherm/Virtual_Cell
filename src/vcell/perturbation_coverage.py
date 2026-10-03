"""Metadata/count-only audit before the historical target cap; no new training rows."""
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

from .utils import file_sha256, write_json


def annotation_counts(obs, entry, min_cells=10, min_controls=30):
    """Retain original batch/condition annotations; count-only candidates are not QC-certified."""
    key = entry['perturbation_key']
    if key not in obs or obs[key].isna().any():
        raise ValueError('Missing perturbation annotation')
    frame = obs.copy()
    for col, values in entry.get('row_filter', {}).items():
        values = values if isinstance(values, list) else [values]
        frame = frame[frame[col].astype(str).isin(list(map(str, values)))]
    mapping = entry.get('perturbation_map', {})
    frame['target'] = frame[key].astype(str).map(lambda x: str(mapping.get(x, x)))
    batch_key = entry.get('batch_key')
    frame['audit_batch'] = frame[batch_key].astype(str) if batch_key else 'single_batch'
    strata = []
    for col in ('study','study_id','intervention_type','perturbation_type','time','timepoint','duration'):
        if col in frame:
            frame[col] = frame[col].fillna('unknown').astype(str); strata.append(col)
    grouped = frame.groupby(['target','audit_batch']+strata, dropna=False, observed=True).size().reset_index(name='n_cells')
    controls = set(map(str, entry['control_values']))
    keys = ['audit_batch']+strata
    ctrl = grouped[grouped.target.isin(controls)].groupby(keys, dropna=False).n_cells.sum().reset_index(name='n_controls')
    result = grouped[~grouped.target.isin(controls)].merge(ctrl,on=keys,how='left')
    result['n_controls'] = result.n_controls.fillna(0).astype(int)
    result['count_eligible'] = (result.n_cells>=min_cells)&(result.n_controls>=min_controls)
    result['control_matching'] = 'same declared batch and available study/intervention/time strata; no expression QC'
    result['batch_semantics'] = 'original declared column' if batch_key else 'no batch annotation; single pool'
    return result


def audit_available(work, data, output):
    work, output = Path(work), Path(output); output.mkdir(parents=True, exist_ok=True)
    records, sources = [], []
    original = work/'prepared/real_min10_01/data_audit.json'
    if original.exists():
        audit = json.loads(original.read_text()); manifest = audit.get('manifest', {})
        paths = {s['id']:s['path'] for s in audit.get('sources', [])}
        for entry in manifest.get('datasets', []):
            if entry['context'] not in ('K562','RPE1'):continue
            path = Path(paths.get(entry['id'], entry['path']))
            if not path.exists():
                matches = list((work/'raw').rglob(path.name)) if (work/'raw').exists() else []
                if len(matches)==1:path=matches[0]
            if not path.is_file():
                sources.append({'context':entry['context'],'status':'raw_missing','path':str(path)});continue
            # Deliberately read only obs: no counts, response magnitudes or Jurkat labels.
            with h5py.File(path,'r') as handle:obs=read_elem(handle['obs'])
            counts=annotation_counts(obs,entry,manifest.get('min_cells',10),manifest.get('min_control_cells',30))
            counts['context']=entry['context'];counts['dataset']=entry['id']
            if 'study' not in counts:counts['study']=counts['study_id'] if 'study_id' in counts else entry.get('study','unknown')
            counts['source_level']='raw_obs_before_target_cap'
            counts['source_path']=str(path);records.append(counts)
            sources.append({'context':entry['context'],'status':'raw_obs_audited','path':str(path),
                            'bytes':path.stat().st_size,'obs_columns':list(obs.columns)})
    else:sources.append({'status':'original_manifest_missing','path':str(original)})
    # H1 was streamed remotely. Its full pre-filter aggregate retains target/batch
    # counts even when the raw H5AD is not present locally. Do not call this raw obs.
    aggregate=work/'prepared/vcc2025_c_01_aggregation/groups.npz'
    complete=aggregate.parent/'COMPLETE.json'
    if aggregate.exists() and complete.exists():
        info=json.loads(complete.read_text())
        if file_sha256(aggregate)!=info['sha256']:raise ValueError('H1 annotation aggregate checksum mismatch')
        source=info['plan']['source'];control=source.get('control','non-targeting')
        with np.load(aggregate,allow_pickle=False) as f:
            counts=pd.DataFrame({'target':f['perturbations'],'audit_batch':f['batches'],'n_cells':f['counts']})
        controls=counts[counts.target==control].set_index('audit_batch').n_cells
        counts=counts[counts.target!=control].copy()
        counts['n_controls']=counts.audit_batch.map(controls).fillna(0).astype(int)
        counts['count_eligible']=(counts.n_cells>=10)&(counts.n_controls>=30)
        counts['context']='H1_hESC';counts['dataset']='vcc2025_training';counts['study']=source.get('study','unknown')
        counts['source_level']='pre_filter_aggregate_counts';counts['source_path']=str(aggregate)
        counts['control_matching']='same original aggregate batch; zero-library cells already removed'
        counts['batch_semantics']='source batch; biological independence unverified';records.append(counts)
        sources.append({'context':'H1_hESC','status':'pre_filter_counts_only','source':source,
                        'aggregate_sha256':info['sha256'],'limitation':'Original obs intervention/time annotations unavailable locally.'})
    else:sources.append({'context':'H1_hESC','status':'pre_filter_aggregate_missing','path':str(aggregate)})
    if records:
        frame=pd.concat(records,ignore_index=True)
        for col in ('study','intervention_type','timepoint'):
            if col not in frame:frame[col]='unknown'
            frame[col]=frame[col].fillna('unknown')
        current=set(zip(data['meta'].context,data['meta'].perturbation))
        frame['already_prepared']=[(c,t) in current for c,t in zip(frame.context,frame.target)]
        frame.to_csv(output/'original_annotation_counts.csv',index=False)
        eligible=frame[frame.count_eligible]
        matrix=eligible.groupby(['target','context','dataset','study'],dropna=False).n_cells.sum().unstack('context',fill_value=0)
        matrix.to_csv(output/'target_context_study.csv')
        cross=eligible.groupby(['target','context']).n_cells.sum().unstack('context',fill_value=0)
        cross['contexts']=(cross>0).sum(axis=1)
        cross[cross.contexts>=2].to_csv(output/'common_candidates.csv')
    write_json(output/'audit.json',{'sources':sources,'test_evaluated':False,
        'note':'Counts before old target cap/filter, not proof of high-quality extra interventions. '
               'No new expression data downloaded or imported; unavailable study/time metadata stays unknown. '
               'H1 aggregate counts cannot recover missing original intervention annotations.'})
