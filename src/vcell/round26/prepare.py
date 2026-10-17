"""Freeze matched protocols and replay source methods; never select on confirmation."""
from pathlib import Path
import numpy as np
from ..round25.prepare import verify_parent
from ..round22.evaluator import inputs
from ..round11 import freeze
from ..model_agent.artifacts import read
from ..model_agent.contracts import digest,protected
from ..evolution.self_correction import source_records as previous_records
from ..utils import file_sha256
from .runtime import assess,fingerprint


def pool_plan(pool,ledger_root,task,arm,cfg,folds,baseline,common,repo,python,hours):
    pool=Path(pool);pool.mkdir(parents=True,exist_ok=True)
    budget={'root':str(ledger_root),'worker_root':str((pool/'assessments').resolve()),'seed':cfg['seed'],
        'device':cfg['device'],'steps':cfg['steps'],'seconds':cfg['worker_seconds'],
        'ram_bytes':12*1024**3,'evaluation_seconds_max':hours*3600}
    freeze(pool/'run_plan.json',{'repo':str(repo),'python':str(python),'framework_sha256':fingerprint(repo),
        'task':task,'arm':arm,'config':cfg,'budget':budget,'test_evaluated':False})
    freeze(pool/'folds.json',folds);freeze(pool/'baseline_selection.json',baseline)
    freeze(pool/'common_sources.json',common)
    body=Path(baseline['search_record']['module']).read_text(encoding='utf-8')
    if file_sha256(baseline['search_record']['module'])!=baseline['module_sha256']:raise ValueError('Fixed comparator code changed')
    seed=pool/'seed.py'
    if seed.exists() and seed.read_text(encoding='utf-8')!=body:raise ValueError('Frozen seed changed')
    seed.write_text(body,encoding='utf-8',newline='\n')


def audit_protocol(parent,cfg):
    verify_parent(parent);protocol={};audit=[]
    old_cfg=read(parent/'campaign_plan.json')['config']
    for key in ('seed','steps','panel_genes','counterfactual_variants','new_source_datasets','behavior_quantum'):
        if cfg[key]!=old_cfg[key]:raise ValueError('Round25 numerical protocol changed: '+key)
    for task in cfg['tasks']:
        reference=read(parent/'baselines'/task/'folds.json')
        for arm in cfg['arms']:
            if read(parent/'arms'/arm/task/'folds.json')!=reference:raise ValueError('Historical arms have different folds')
        for phase,folds in reference.items():
            if phase not in ('search','confirmation') or not folds:raise ValueError('Invalid inherited phase')
            for fold in folds:
                d,g,_=inputs(fold)
                if len(d['genes'])!=cfg['panel_genes']:raise ValueError('Output gene order/panel changed')
                contexts=np.r_[d['train_source_context'],d['calibration_source_context'],d['query_source_context']]
                if any(protected(str(c)) for c in contexts):raise ValueError('Sealed context is unavailable')
                if task=='transfer':
                    cell=lambda c:str(c).upper().split('_')[0]
                    source={cell(c) for c in np.r_[d['train_source_context'],d['calibration_source_context'],d.get('aux_context',np.array([],dtype='U'))]}
                    if source & {cell(c) for c in d['query_source_context']}:raise ValueError('Transfer query cell line seen in fit/calibration')
                unknown=set(map(str,d['query_targets'][g['unknown']]))
                known=set(map(str,np.r_[d['train_targets'],d['calibration_targets'],d.get('aux_target',np.array([],dtype='U'))]))
                if unknown & known:raise ValueError('Globally unseen target response appears in source labels')
                audit.append({'task':task,'phase':phase,'name':fold['name'],'input_sha256':fold['input_sha256'],
                    'grade_sha256':fold['grade_sha256'],'gene_order_sha256':digest(d['genes'].tolist()),
                    'strong_mask_sha256':digest(g['strong_mask'].tolist())})
        protocol[task]=reference
    return protocol,audit


def historical_methods(parent,task):
    baseline=read(parent/'baselines'/task/'baseline_selection.json')
    rows=previous_records(parent/'arms/meta'/task)
    # Source-only, nonzero signal candidates are starts for all arms, not discoveries.
    candidates=[r for r in rows if r['status']=='valid' and r['module_sha256']!=baseline['module_sha256']
        and r['metrics']['public'].get('strong_mse_ratio') is not None
        and r['metrics']['public']['strong_mse_ratio']<1
        and (r['metrics']['public'].get('specificity_gap') or 0)>0
        and r['metrics']['public']['amplitude_ratio']>=.05
        and r['metrics']['public']['gain_over_fixed']>=-.02]
    selected=[]
    for key in ('strong_mse','mse'):
        if candidates:
            r=min(candidates,key=lambda r:r['metrics']['public'][key])
            if r['module_sha256'] not in {s['module_sha256'] for s in selected}:selected.append(r)
    return baseline,[baseline['search_record'],*selected]


def prepare(root,parent,cfg,repo,python,stage):
    marker=root/'protocol.json'
    if marker.exists():
        old=read(marker)
        if old['parent_sha256']!=file_sha256(parent/'COMPLETE.json'):raise ValueError('Completed parent changed')
        protocol=old['folds']
    else:
        stage(root,'audit unchanged Round25 source protocols and exact fixed comparators')
        protocol,audit=audit_protocol(parent,cfg)
        freeze(marker,{'folds':protocol,'checks':audit,'parent_sha256':file_sha256(parent/'COMPLETE.json'),
            'independent_confirmation':False,'test_evaluated':False,'HT29_evaluated':False})
    result={};replay=[]
    for task in cfg['tasks']:
        baseline,methods=historical_methods(parent,task)
        p=root/'baselines'/task
        pool_plan(p,root/'baseline_budget',task,'fixed',cfg,protocol[task],baseline,[],repo,python,cfg['baseline_hours'])
        ready=[]
        for old in methods:
            body=Path(old['module']).read_text(encoding='utf-8')
            if file_sha256(old['module'])!=old['module_sha256']:raise ValueError('Historical source code changed')
            stage(root,'replay common source method',task=task,module_sha256=old['module_sha256'])
            r=assess(p,body)
            if r['status']!='valid':raise RuntimeError('Common source replay failed: '+(r.get('error') or ''))
            for fold in protocol[task]['search']:
                with np.load(Path(old['directory'])/(fold['name']+'.npz'),allow_pickle=False) as a, np.load(Path(r['directory'])/(fold['name']+'.npz'),allow_pickle=False) as b:
                    if not np.array_equal(a['genes'],b['genes']) or not np.array_equal(a['row_ids'],b['row_ids']) or not np.allclose(a['prediction'],b['prediction'],rtol=1e-5,atol=1e-7):
                        raise ValueError('Historical SOURCE prediction did not reproduce: '+old['module_sha256'])
            replay.append({'task':task,'module_sha256':r['module_sha256'],'passed':True})
            path=Path(r['directory'])/'assessment.json';ready.append({'record':str(path),'sha256':file_sha256(path)})
        result[task]={'folds':protocol[task],'baseline':baseline,'common':ready}
        for arm in cfg['arms']:
            pool_plan(root/'arms'/arm/task,root/'arms'/arm,task,arm,cfg,protocol[task],baseline,ready,repo,python,cfg['arm_hours'])
    freeze(root/'baseline_replay.json',{'passed':True,'source_only':True,'checks':replay})
    return result
