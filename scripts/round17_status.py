"""Read status without API requests or changing a completed experiment."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round17_oof_01')
a=p.parse_args();root=Path(a.work_dir)/'runs'/a.name
for name in ('stage.json','budget.json','INCOMPLETE.json','summary.json'):
    path=root/name
    if path.exists():print(name,json.dumps(json.loads(path.read_text()),indent=2))
path=root/'progress.json'
if path.exists():
    value=json.loads(path.read_text());counts={}
    for name,job in value.get('jobs',{}).items():
        counts[job['state']]=counts.get(job['state'],0)+1
        if job['state'] in ('running','failed'):print(name,job)
    print('Current queue:',counts)
if (root/'COMPLETE.json').exists():print('ROUND17 COMPLETE:',root/'round17_review_light.tar.gz')
