"""Run a registered research method with fixed source blocks in a protected worker."""
from pathlib import Path
import sys
import numpy as np
from ..model_agent.artifacts import read
from ..model_agent.contracts import protected
from ..round22.evaluator import execute
from ..round24.modules import compose
from ..round24.runtime import public_data
from ..utils import file_sha256,write_json
from .runtime import fingerprint


def predict(root,work,module_sha,task,input_path,output_name):
    root=Path(root).resolve();work=Path(work).resolve()
    if Path(output_name).name!=output_name or output_name in ('','.','..'):raise ValueError('Simple output name required')
    complete=read(root/'COMPLETE.json');registry=root/'model_registry.json'
    if complete['files'].get('model_registry.json')!=file_sha256(registry):raise ValueError('Frozen model registry changed')
    cfg=read(root/'campaign_plan.json')
    if fingerprint(cfg['repo'])!=cfg['framework_sha256']:raise ValueError('Prediction runtime differs from the frozen campaign')
    entries=[r for r in read(registry)['models'] if r['task']==task and r['module_sha256']==module_sha]
    if not entries:raise ValueError('Method is not registered for this task')
    model=entries[0]
    if file_sha256(model['module'])!=module_sha or file_sha256(model['template_input'])!=model['template_sha256']:
        raise ValueError('Registered module/source template changed')
    with np.load(input_path,allow_pickle=False) as f:data={k:f[k] for k in f.files}
    with np.load(model['template_input'],allow_pickle=False) as f:template={k:f[k] for k in f.files}
    if set(data)!=set(template):raise ValueError('Use the exact public input schema; no query truth fields')
    query_keys={'reference_available','background_reference'}
    for key in template:
        if key.startswith('query_') or key in query_keys:continue
        if not np.array_equal(data[key],template[key]):raise ValueError('Frozen legal source block changed: '+key)
    for key in ('query_source_context','query_context'):
        if key in data and any(protected(str(c)) for c in data[key]):raise ValueError('Sealed context is unavailable')
    if task=='transfer':
        cell=lambda x:str(x).upper().split('_')[0]
        source={cell(x) for k in ('train_source_context','calibration_source_context','aux_context') for x in data.get(k,[])}
        if source & {cell(x) for x in data['query_source_context']}:raise ValueError('Transfer query cell line overlaps source labels')
    dest=work/'predictions'/output_name
    if dest.exists():raise ValueError('Use a fresh prediction output name')
    budget={'root':str(dest),'worker_root':str(dest),'seed':cfg['config']['seed'],'device':cfg['config']['device'],
        'steps':cfg['config']['steps'],'seconds':cfg['config']['worker_seconds'],'ram_bytes':12*1024**3,
        'evaluation_seconds_max':cfg['config']['worker_seconds']}
    result=execute(compose(Path(model['module']).read_text(encoding='utf-8')),public_data(data,'all',cfg['config']['new_source_datasets']),dest/'worker',budget,sys.executable)
    np.savez_compressed(dest/'prediction.npz',prediction=result,genes=data['genes'],row_ids=data['query_ids'])
    write_json(dest/'provenance.json',{'method':module_sha,'task':task,'campaign':str(root),'input_sha256':file_sha256(input_path),
        'evaluation':False,'support_provenance':'Caller supplies query support; not independently validated biological support.',
        'independent_generalization_claim':False})
    return dest/'prediction.npz'
