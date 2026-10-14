"""Report real candidate, isolation, source-confirmation and API states."""
import argparse
import json
from pathlib import Path
from collections import Counter
p=argparse.ArgumentParser();p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round22_evolve_01');a=p.parse_args()
root=Path(a.work_dir)/'runs'/a.name
for name in ('stage.json','sandbox_verified.json','worker_progress.json','INCOMPLETE.json'):
    path=root/name
    if path.exists():print(name,json.dumps(json.loads(path.read_text()),indent=2))
for task in ('repeat','transfer'):
    records=[json.loads(p.read_text()) for p in (root/'evolution'/task).glob('gen_*/results/correct.json')]
    print(task,'Shinka evaluated:',dict(Counter('valid' if r.get('correct') else 'failed' for r in records)))
ledger=root/'api_ledger.json'
if ledger.exists():
    d=json.loads(ledger.read_text());print('API:',len(d),'requests;',dict(Counter(r['status'] for r in d)), 'output tokens charged:',sum(r.get('charged_output_tokens',0) for r in d))
budget=root/'evaluation_budget.json'
if budget.exists():
    d=json.loads(budget.read_text());print('Evaluation seconds:',sum(v['charged_seconds'] for v in d['workers'].values()),'/',d['cap_seconds'],'worker attempts:',len(d['workers']))
if (root/'COMPLETE.json').exists():print('ROUND22 COMPLETE:',root/'round22_review_light.tar.gz')
