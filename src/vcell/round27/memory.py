"""Evidence-linked source research memory; reconstructed from committed experiments."""
from pathlib import Path
from dataclasses import asdict
from ..model_agent.artifacts import read
from ..model_agent.contracts import digest
from ..evolution.journal import append, events
from ..utils import write_json, file_sha256


def remember(root, ticket, kind, **fields):
    path=Path(root)/'memory/events.jsonl'
    body={'ticket':ticket,'scope':'source_search_only',**fields}
    old=next((r for r in events(path) if r['ticket']==ticket),None)
    if old:
        if old['kind']!=kind or any(old.get(k)!=v for k,v in body.items()):
            raise ValueError('Committed research memory changed: '+ticket)
        return old
    return append(path,kind,**body)


def outcome(root,arm,task,experiment_id,action,result,revision=None):
    """Atomic hypothesis/measurement pair; a duplicate resume cannot add credit."""
    return remember(root,experiment_id,'experiment',arm=arm,task=task,
        hypothesis_id=experiment_id+'/hypothesis' if action else None,
        action=asdict(action) if action else None,revision=revision,
        children=result.get('children',[]),latest=result.get('latest'),
        prediction=result.get('prediction'),probe=result.get('probe'),
        attribution=result.get('attribution'),
        measurement_status='diagnostic_complete' if result.get('probe') else
            'measured' if any(c.get('status')=='valid' for c in result.get('children',[])) else
            'invalid_execution' if result.get('children') else 'no_new_assessment',
        result_kind='diagnostic' if result.get('probe') else 'numerical')


def experience(root,arm,task):
    rows=[r for r in events(Path(root)/'memory/events.jsonl') if r.get('arm')==arm and r.get('task')==task]
    used={r['revision']['predecessor_experiment_id'] for r in rows if r.get('revision')}
    hypotheses=[r for r in rows if r.get('action')]
    # Pending hypotheses survive arbitrary numbers of routine native mutations.
    pending=[r for r in hypotheses if r['ticket'] not in used and
             r['action']['action']!='stop_unproductive_branch' and
             (r.get('probe') or r.get('measurement_status')=='no_new_assessment' or
              (r.get('prediction') or {}).get('status')=='not_supported')]
    def card(r):
        return {k:r.get(k) for k in ('ticket','hypothesis_id','action','revision','prediction','children','latest','probe','result_kind','measurement_status')}
    return {'pending_revisions':[card(r) for r in pending[-3:]],
        'hypothesis_history':[card(r) for r in hypotheses[-6:]],
        'observed_experiments':len(rows),
        'interpretation':'A measured action sequence, not causal proof that the critic improved prediction.'}


def library(root,cfg):
    """Export only source cards, never confirmation scores or summaries."""
    from .runtime import source_records
    root=Path(root);methods=[]
    for arm in cfg['arms']:
        for task in cfg['tasks']:
            pool=root/'arms'/arm/task
            if not (pool/'run_plan.json').exists():continue
            folds=read(pool/'folds.json')['search']
            for r in source_records(pool):
                methods.append({'arm':arm,'task':task,'module_sha256':r['module_sha256'],
                    'status':r['status'],'source_metrics':r['metrics']['public'],
                    'source_inputs':[f['input_sha256'] for f in folds],
                    'record':str(Path(r['directory'])/'assessment.json'),
                    'record_sha256':file_sha256(Path(r['directory'])/'assessment.json')})
    return {'version':1,'scope':'source_search_only','methods':methods,
        'research_events':events(root/'memory/events.jsonl'),
        'confirmation_included':False,'test_included':False}


def import_history(root,parent):
    """Freeze historical source experience, available equally to every arm."""
    from ..round11 import freeze
    root=Path(root);parent=Path(parent)
    path=parent/'memory_library.json'
    if path.exists():
        value=read(path)
        if value.get('scope')!='source_search_only' or value.get('confirmation_included') or value.get('test_included'):
            raise ValueError('Memory import must contain source-only evidence')
        # Histories are not executable instructions. Old proposals do not become facts.
        value={'parent':str(parent),'sha256':file_sha256(path),'scope':'source_search_only',
            'source_lessons':[{'task':r.get('task'),'action':r.get('action'),'prediction':r.get('prediction')}
                for r in value.get('research_events',[]) if r.get('scope')=='source_search_only' and r.get('action')][-12:]}
    else:
        value={'parent':str(parent),'scope':'source_search_only','source_lessons':[],
            'note':'Round26 has no typed persistent hypothesis memory; source model starts are imported separately.'}
    freeze(root/'historical_memory.json',value)
    return value


def publish(root,work,cfg):
    """Immutable cross-campaign export plus a small external index."""
    from ..round11 import freeze
    root=Path(root);work=Path(work);value=library(root,cfg)
    freeze(root/'memory_library.json',value)
    index=work/'agent_memory'/root.name
    index.mkdir(parents=True,exist_ok=True)
    freeze(index/'source_library.json',{'campaign':str(root),'library':str(root/'memory_library.json'),
        'sha256':file_sha256(root/'memory_library.json'),'scope':'source_search_only'})
    return value
