"""Read source-only round progress, worker steps and API accounting."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round19_dual_100_01')
a=p.parse_args();root=Path(a.work_dir)/'runs'/a.name
if not root.exists():print('Run not started:',root)
for name in ('stage.json','budget.json','iteration_progress.json','worker_progress.json','INCOMPLETE.json'):
    path=root/name
    if path.exists():
        v=json.loads(path.read_text())
        if name=='iteration_progress.json':v={k:value for k,value in v.items() if k!='best_source'}
        print(name+(' (last failure; may remain during resume)' if name=='INCOMPLETE.json' else ''),json.dumps(v,indent=2))
worker=root/'worker_progress.json'
if worker.exists():
    for name in json.loads(worker.read_text()).get('active',[]):
        cfg=root/'worker_configs'/(name+'.json')
        if cfg.exists():
            out=Path(json.loads(cfg.read_text())['output_dir']);progress=out/'progress.json'
            print('Worker:',name,json.loads(progress.read_text()) if progress.exists() else 'loading model / before first checkpoint')
for role in ('training','review'):
    api=root/'api'/role;cfg=api/'configuration.json'
    if cfg.exists():
        attempts=sorted(api.glob('attempt_*.json'));last=json.loads(attempts[-1].read_text()) if attempts else {}
        print('API',role,'attempts=',len(attempts),'/',json.loads(cfg.read_text())['max_api_calls'],
              'last=',last.get('status'),last.get('failure_category'))
if (root/'COMPLETE.json').exists():print('ROUND19 COMPLETE:',root/'round19_review_light.tar.gz')
