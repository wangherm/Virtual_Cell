"""Content-addressed read-only expert bundles and immutable execution results."""
import json
from pathlib import Path
import numpy as np

from ..utils import file_sha256, write_json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def seal(root,**metadata):
    root=Path(root)
    files={p.relative_to(root).as_posix():file_sha256(p) for p in sorted(root.rglob('*'))
           if p.is_file() and p.name!='COMPLETE.json' and not p.name.endswith('.tmp')}
    write_json(root/'COMPLETE.json',{'files':files,**metadata})


def verify(root):
    root=Path(root).resolve();marker=read(root/'COMPLETE.json')
    if not marker.get('files'):raise ValueError('Checksummed completed artifact required')
    for name,sha in marker['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or file_sha256(p)!=sha:raise ValueError('Artifact checksum mismatch: '+name)
    return marker


def arrays(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def save_prediction(path,query,values):
    x=np.asarray(values,np.float32)
    if x.shape!=(len(query.genes),) or not np.isfinite(x).all():raise ValueError('Invalid expert prediction')
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream:np.savez_compressed(stream,delta=x,genes=np.array(query.genes,dtype='U'),query_fingerprint=np.array(query.fingerprint))
    tmp.replace(path)


def load_prediction(path,query):
    f=arrays(path)
    if str(f['query_fingerprint'])!=query.fingerprint or not np.array_equal(f['genes'],query.genes):
        raise ValueError('Prediction identity/order mismatch')
    if f['delta'].shape!=(len(query.genes),) or not np.isfinite(f['delta']).all():raise ValueError('Invalid cached prediction')
    return f['delta']
