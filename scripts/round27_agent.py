"""Read status/memory, request pause, clear pause, or run a frozen Virtual Cell method."""
import argparse
import json
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vcell.model_agent.artifacts import read
from vcell.utils import write_json
from vcell.round27.manager import control_path


def status(root):
    if not (root/'campaign_plan.json').exists():return {'state':'not_started','run':str(root)}
    heartbeat=read(root/'heartbeat.json') if (root/'heartbeat.json').exists() else {}
    age=time.time()-heartbeat.get('updated_at',0)
    error=read(root/'INCOMPLETE.json') if (root/'INCOMPLETE.json').exists() else None
    if error and (root/'stage.json').exists():error['older_than_current_stage']=(root/'INCOMPLETE.json').stat().st_mtime<(root/'stage.json').stat().st_mtime
    return {'run':str(root),'execution_complete':(root/'COMPLETE.json').exists(),
        'stage':read(root/'stage.json') if (root/'stage.json').exists() else {},
        'process':{'heartbeat':heartbeat,'age_seconds':round(age,1),
            'fresh':heartbeat.get('state')=='running' and age<45},
        'paused':read(root/'PAUSED.json') if (root/'PAUSED.json').exists() else None,
        'error':error,
        'tasks':[read(p) for p in sorted((root/'tasks').glob('*.json'))],
        'worker':[{'ledger':str(p.parent.relative_to(root)),**read(p)} for p in root.glob('*/meta/worker_progress.json')]+[
            {'ledger':str(p.parent.relative_to(root)),**read(p)} for p in root.glob('arms/*/worker_progress.json')],
        'usage':[{ 'arm':p.parent.name,'requests':len(read(p)),
            'roles':{role:sum(r.get('role')==role for r in read(p)) for role in sorted({str(r.get('role')) for r in read(p)})}}
            for p in root.glob('arms/*/api_ledger.json')],
        'summary':read(root/'summary.json') if (root/'summary.json').exists() else None}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['status','pause','clear-pause','memory','catalog','models','predict','knowledge'])
    p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round27_agent_01')
    p.add_argument('--method');p.add_argument('--task',choices=['repeat','transfer']);p.add_argument('--input');p.add_argument('--output-name')
    p.add_argument('--gene',action='append',default=[])
    a=p.parse_args();work=Path(a.work_dir).resolve();control=control_path(work,a.name);root=work/'runs'/a.name
    if a.action in ('pause','clear-pause'):
        write_json(control,{'pause_requested':a.action=='pause','requested_at':time.time()})
        value={'pause_requested':a.action=='pause','note':'Pause takes effect after the current committed experiment. Clear-pause does not restart a process; launch with --resume.'}
    elif a.action=='status':value=status(root)
    elif a.action=='knowledge':
        from vcell.round27.knowledge import lookup
        if not a.gene:p.error('knowledge requires at least one --gene')
        value=lookup(root,a.gene)
    elif a.action=='predict':
        if not all((a.method,a.task,a.input,a.output_name)):p.error('predict requires --method --task --input --output-name')
        from vcell.round27.predict import predict
        value={'prediction':str(predict(root,work,a.method,a.task,a.input,a.output_name))}
    else:
        name={'memory':'memory_library.json','catalog':'domain_catalog.json','models':'model_registry.json'}[a.action]
        if (root/name).exists():value=read(root/name)
        elif a.action=='memory':
            from vcell.evolution.journal import events
            value={'scope':'source_search_only','events':events(root/'memory/events.jsonl')}
        else:value={'state':'not_ready','file':name}
    print(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
