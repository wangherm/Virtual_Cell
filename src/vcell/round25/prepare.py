"""Read-only Round24 source lineage and frozen Round25 baselines."""
import inspect
from pathlib import Path
import numpy as np
from ..model_agent.artifacts import read
from ..model_agent.contracts import protected
from ..round11 import freeze
from ..round22.evaluator import inputs, fixed_predictions
from ..round24.modules import simple_module, START, END
from ..utils import file_sha256, write_json
from .runtime import assess


def verify_parent(parent):
    """Verify full server artifacts, hashing identical hardlinked inputs once."""
    parent=Path(parent).resolve();marker=read(parent/'COMPLETE.json');seen={}
    if not marker.get('files'): raise ValueError('Full completed Round24 run required')
    for name, expected in marker['files'].items():
        p=(parent/name).resolve()
        if not p.is_relative_to(parent): raise ValueError('Parent payload outside run')
        stat=p.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if stat.st_ino==0: key=(*key,str(p))
        actual=seen.setdefault(key,None)
        if actual is None: actual=file_sha256(p);seen[key]=actual
        if actual!=expected: raise ValueError('Round24 payload changed: '+name)
    return marker


def inherited(parent, cfg):
    parent=Path(parent);verify_parent(parent)
    folds=read(parent/'folds.json')['tasks']
    for task in cfg['tasks']:
        if len(folds[task]['search'])<2: raise ValueError('At least two source folds per task required')
        for phase in ('search','confirmation'):
            if not folds[task][phase]: raise ValueError('Missing development '+phase+' folds')
            for fold in folds[task][phase]:
                d,_,_=inputs(fold)
                if len(d['genes'])!=cfg['panel_genes']: raise ValueError('Frozen output panel changed')
                for key in ('train_source_context','calibration_source_context','query_source_context'):
                    if any(protected(str(c)) for c in d[key]): raise ValueError('Reserved context in source fold')
                if task=='transfer':
                    # Full cell-line exclusion, not just condition-string inequality.
                    def cell(s):return str(s).upper().split('_')[0]
                    known={cell(c) for key in ('train_source_context','calibration_source_context') for c in d[key]}
                    if known & {cell(c) for c in d['query_source_context']}: raise ValueError('Transfer cell line overlaps training')
    return folds


def methods(parent):
    parent=Path(parent)
    result={'uniform_ensemble':(parent/'methods/uniform_ensemble.py').read_text(encoding='utf-8'),
            'simple_support':simple_module(), 'simple_shared_ridge':simple_module(ridge=True)}
    # This is a historical SOURCE-selected method, not a winner selected by confirmation.
    selected=read(parent/'search_frozen.json')['finalists']
    for item in selected:
        if item['identity']['phase']!='search': raise ValueError('Only source-selected seeds are eligible')
        path=Path(item['module'])
        if file_sha256(path)!=item['module_sha256']: raise ValueError('Inherited source method changed')
        result['round24_source_'+item['module_sha256'][:10]]=path.read_text(encoding='utf-8')
    body=inspect.getsource(fixed_predictions)
    for name in ('zero','background_mean','calibrated_support','demeaned_support',
                 'raw_control_STRING_ridge','raw_control_STRING_factor16','raw_control_STRING_factor32'):
        result[name]=START+'\n'+body+f'\ndef fit_predict(data,budget):\n    return fixed_predictions(data)[{name!r}]\n'+END+'\n'
    return result


def freeze_baselines(pool, definitions):
    pool=Path(pool)
    if (pool/'baseline_selection.json').exists(): return read(pool/'baseline_selection.json')
    task=read(pool/'run_plan.json')['task'];rows=[]
    for name, body in definitions.items():
        if task=='transfer' and name=='demeaned_support': continue
        print('ROUND25 BASELINE:',task,name,flush=True)
        r=assess(pool,body,baseline=True,controls=False)
        if r['status']!='valid': raise RuntimeError('Fixed baseline failed: '+name+' '+r.get('error',''))
        rows.append({'name':name,**r})
    winner=min(rows,key=lambda r:r['metrics']['public']['mse'])
    # Only the source winner is evaluated on development confirmation, before any agent runs.
    later=assess(pool,definitions[winner['name']],phase='confirmation',baseline=True,controls=False)
    if later['status']!='valid': raise RuntimeError('Fixed confirmation replay failed: '+later.get('error',''))
    value={'method':winner['name'],'module_sha256':winner['module_sha256'],
           'selection':'minimum SOURCE macro MSE; never confirmation',
           'search_record':winner,'confirmation_record':later}
    for phase,r in [('search',winner),('confirmation',later)]:
        value[phase]={f['name']:{'path':str(Path(r['directory'])/(f['name']+'.npz')),
            'sha256':file_sha256(Path(r['directory'])/(f['name']+'.npz'))} for f in read(pool/'folds.json')[phase]}
    write_json(pool/'fixed_scores.json',rows);freeze(pool/'baseline_selection.json',value)
    (pool/'seed.py').write_text(definitions[winner['name']],encoding='utf-8')
    return value


def replay_parent(root,parent):
    """Reproduce available historical source predictions, never use scores for selection."""
    old=read(Path(parent)/'diagnostics_frozen.json')
    old+=[r for r in read(Path(parent)/'search_frozen.json')['finalists']]
    checks=[]
    for task in ('repeat','transfer'):
        pool=Path(root)/'baselines'/task
        current=read(pool/'fixed_scores.json')+[read(pool/'baseline_selection.json')['confirmation_record']]
        for r in current:
            phase=r['identity']['phase']
            matches=[s for s in old if s['identity']['task']==task and s['identity']['phase']==phase
                     and s['module_sha256']==r['module_sha256'] and s['identity']['arm']=='all']
            if not matches:continue
            previous=Path(matches[0]['directory'])
            for fold in read(pool/'folds.json')[phase]:
                a=previous/(fold['name']+'.npz');b=Path(r['directory'])/(fold['name']+'.npz')
                with np.load(a,allow_pickle=False) as left,np.load(b,allow_pickle=False) as right:
                    passed=np.array_equal(left['genes'],right['genes']) and np.array_equal(left['row_ids'],right['row_ids']) and np.allclose(left['prediction'],right['prediction'],rtol=1e-5,atol=1e-7)
                checks.append({'task':task,'phase':phase,'fold':fold['name'],'module_sha256':r['module_sha256'],'passed':bool(passed)})
    write_json(Path(root)/'baseline_replay.json',{'checks':checks,'passed':bool(checks) and all(c['passed'] for c in checks),
        'scope':'Available identical historical methods/inputs; no modification or reselection using confirmation'})
    if not checks or not all(c['passed'] for c in checks):raise ValueError('Round24 numerical replay failed before paid evolution')
