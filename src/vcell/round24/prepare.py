"""Read-only Round23 lineage and source baseline freezing; no new splits or downloads."""
import hashlib
from pathlib import Path
import numpy as np
from ..model_agent.artifacts import read,verify
from ..model_agent.contracts import protected
from ..round11 import freeze
from ..round22.evaluator import inputs,fixed_predictions
from ..utils import write_json,file_sha256
from .modules import WINNER


def parent_run(work,explicit=None):
    if explicit:return Path(explicit)
    candidates=[]
    for p in (Path(work)/'runs').glob('round23*'):
        if (p/'COMPLETE.json').exists() and (p/'confirmation_frozen.json').exists():
            w=read(p/'confirmation_frozen.json').get('repeat_expanded',{}).get('winner')
            if w and w['code_sha256']==WINNER:candidates.append(p)
    if not candidates:raise ValueError('No completed Round23 winner found; pass --round23-run pointing to the full server run, not its light archive')
    return max(candidates,key=lambda p:(p/'COMPLETE.json').stat().st_mtime)


def prepare(parent,root,cfg):
    parent=Path(parent);root=Path(root);verify(parent)
    source=read(parent/'confirmation_frozen.json')['repeat_expanded']['winner']
    if source is None or source['code_sha256']!=cfg['winner_sha256'] or cfg['winner_sha256']!=WINNER:raise ValueError('This Round24 ablation release requires the frozen Round23 winner')
    path=parent/'pools/repeat_expanded/evaluations/search'/WINNER/'candidate.py'
    code=path.read_text(encoding='utf-8')
    if hashlib.sha256(code.encode()).hexdigest()!=WINNER:raise ValueError('Round23 winner code changed')
    frozen=root/'frozen_winner.py'
    if frozen.exists() and frozen.read_text(encoding='utf-8')!=code:raise ValueError('Frozen winner changed')
    frozen.write_text(code,encoding='utf-8')
    tasks={};references={}
    for task in ('repeat','transfer'):
        pool=parent/'pools'/f'{task}_expanded';plan=read(pool/'data_plan.json');tasks[task]={}
        selected=read(pool/'baseline_selection.json')[task]
        if len(plan['gene_order'])!=cfg['panel_genes']:raise ValueError('Round24 frozen panel mismatch')
        for phase in ('search','confirmation'):
            records=[]
            for fold in plan['folds'][task][phase]:
                d,g,m=inputs(fold)
                if not np.array_equal(d['genes'],np.asarray(plan['gene_order'],dtype='U')):raise ValueError('Source fold gene order differs from frozen panel')
                if any(protected(str(c)) for key in ('train_source_context','calibration_source_context','query_source_context') for c in d[key]):raise ValueError('Reserved context found in a source-only fold')
                dest=root/'private'/task/phase/fold['name'];dest.mkdir(parents=True,exist_ok=True)
                baseline=dest/'baseline.npz'
                if not baseline.exists():np.savez_compressed(baseline,prediction=fixed_predictions(d)[selected])
                records.append({**fold,'baseline':str(baseline),'baseline_sha256':file_sha256(baseline)})
                if task=='repeat':
                    prediction=pool/'evaluations'/phase/WINNER/'full/workers'/fold['name']/'correct/output/prediction.npz'
                    if not prediction.is_file():raise ValueError('Original replay needs full Round23 worker predictions: '+str(prediction))
                    references[phase+'/'+fold['name']]={'path':str(prediction),'sha256':file_sha256(prediction)}
            tasks[task][phase]=records
    freeze(root/'folds.json',{'tasks':tasks,'outer_included':False,'panel_genes':cfg['panel_genes']})
    freeze(root/'lineage.json',{'parent':str(parent),'parent_complete_sha256':file_sha256(parent/'COMPLETE.json'),
        'winner_sha256':WINNER,'reference_predictions':references,'selection_history':'Round23 winner and confirmation have already been inspected; all Round24 confirmation metrics are development evidence',
        'new_datasets':cfg['new_source_datasets'],'raw_data_changed':False,'HT29_evaluated':False,'test_evaluated':False})
    return code


def verify_replay(root,records):
    root=Path(root);refs=read(root/'lineage.json')['reference_predictions'];checks=[]
    for phase,r in records.items():
        if r['status']!='valid':raise RuntimeError('Frozen winner replay failed: '+r.get('error',''))
        for fold in read(root/'folds.json')['tasks']['repeat'][phase]:
            old=refs[phase+'/'+fold['name']];new=Path(r['directory'])/(fold['name']+'.npz')
            if file_sha256(old['path'])!=old['sha256']:raise ValueError('Round23 reference prediction changed')
            with np.load(old['path'],allow_pickle=False) as f:a={k:f[k] for k in f.files}
            with np.load(new,allow_pickle=False) as f:b={k:f[k] for k in f.files}
            ok=np.array_equal(a['row_ids'],b['row_ids']) and np.array_equal(a['genes'],b['genes']) and np.allclose(a['prediction'],b['prediction'],rtol=1e-5,atol=1e-7)
            checks.append({'phase':phase,'fold':fold['name'],'passed':bool(ok),'max_absolute_prediction_difference':float(np.max(np.abs(a['prediction']-b['prediction'])))})
    write_json(root/'replay.json',{'passed':all(c['passed'] for c in checks),'checks':checks,'tolerance':{'rtol':1e-5,'atol':1e-7}})
    if not all(c['passed'] for c in checks):raise ValueError('Original winner did not reproduce; paid evolution blocked; inspect replay.json')
