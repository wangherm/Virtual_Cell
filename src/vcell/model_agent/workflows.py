"""Human-invoked preparation and review tools; absent from the planner tool list."""
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

from .artifacts import read
from .contracts import NORMALIZATION, Query
from .experts import Registry
from ..utils import write_json


def combine(registry_paths,output):
    dest=Path(output).resolve()
    if dest.exists():raise ValueError('Registry output already exists')
    entries=[];ids=set()
    for path in registry_paths:
        registry=Registry(path)
        for entry in registry.config['experts']:
            if entry['expert_id'] in ids:raise ValueError('Duplicate expert ID across registries')
            ids.add(entry['expert_id'])
            entries.append({**entry,'bundle':str((registry.path.parent/entry['bundle']).resolve())})
    write_json(dest,{'schema_version':1,'experts':entries});Registry(dest)
    return dest


def query_from_prepared(data_path,row_id,output):
    root=Path(data_path);audit=read(root/'data_audit.json')
    if audit.get('normalization')!=NORMALIZATION:raise ValueError('Unknown control normalization')
    meta=pd.read_csv(root/'metadata.csv',dtype=str,keep_default_na=False)
    rows=meta.index[meta.row_id.eq(row_id)]
    if len(rows)!=1:raise ValueError('Query row not unique/present')
    i=int(rows[0]);row=meta.iloc[i]
    # Decode only baseline/genes. The response member is never read here.
    with np.load(root/'dataset.npz',allow_pickle=False) as data:
        genes=tuple(data['genes'].tolist());baseline=tuple(data['baseline'][i].tolist())
    q=Query(str(row_id),row.context,row.perturbation,row.get('stimulus','unspecified'),genes,baseline,
            source_partition=row.split,source_fingerprint=audit['fingerprint'])
    dest=Path(output)
    if dest.exists():raise ValueError('Query output already exists')
    write_json(dest,asdict(q));return dest


def package_review(root):
    root=Path(root);names={'expert.json','registry.json','profile.json','result.json','trace.json',
                         'stage.json','manifest.json','lineage.json','COMPLETE.json'}
    path=root/'model_agent_review_light.tar.gz';entries={}
    with tarfile.open(path,'w:gz') as tar:
        for p in sorted(root.rglob('*.json')):
            if p.name not in names:continue
            value=p.read_bytes();name=p.relative_to(root).as_posix();member=tarfile.TarInfo(name);member.size=len(value)
            tar.addfile(member,io.BytesIO(value));entries[name]={'sha256':hashlib.sha256(value).hexdigest(),'bytes':len(value)}
        value=json.dumps({'files':entries,'scope':'Architecture artifacts only; no response/control arrays or private score tables'},indent=2).encode()
        member=tarfile.TarInfo('ARCHIVE_MANIFEST.json');member.size=len(value);tar.addfile(member,io.BytesIO(value))
    return path
