"""Source signals, raw audit, worker steps, search budgets and actual completion."""
import argparse,json
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round21_signal_01');a=p.parse_args()
root=Path(a.work_dir)/'runs'/a.name
if not root.exists():print('Not started:',root)
statuses=[json.loads(p.read_text())['status'] for p in (root/'workers').rglob('training_status.json')]
if statuses:print('Correction heads:',dict(Counter(statuses)))
for name in ('stage.json','budget.json','raw_audit/RAW_STATUS.json','raw_audit/progress.json','search_progress.json','worker_progress.json','INCOMPLETE.json'):
    path=root/name
    if path.exists():print(name+(' (last failure; may remain during resume)' if name=='INCOMPLETE.json' else ''),json.dumps(json.loads(path.read_text()),indent=2))
queue=root/'worker_progress.json'
if queue.exists():
    for name in json.loads(queue.read_text()).get('active',[]):
        path=root/'worker_configs'/(name+'.json')
        if path.exists():
            progress=Path(json.loads(path.read_text())['output_dir'])/'progress.json'
            print('Active:',name,json.loads(progress.read_text()) if progress.exists() else 'loading / before checkpoint')
for path in sorted((root/'selections').glob('*.json')):
    d=json.loads(path.read_text());print('Gate:',path.stem,d['gate'])
for path in sorted((root/'search').glob('*/*/*api/attempt_*.json')):
    d=json.loads(path.read_text());print('API',str(path.relative_to(root)),d['status'],d.get('failure_category'),d.get('output_tokens'))
if (root/'COMPLETE.json').exists():print('ROUND21 COMPLETE:',root/'round21_review_light.tar.gz')
