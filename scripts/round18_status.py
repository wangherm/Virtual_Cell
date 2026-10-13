"""Read Round18 stages, bounded budgets and source search decisions."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round18_01')
a=p.parse_args();root=Path(a.work_dir)/'runs'/a.name
for name in ('stage.json','budget.json','INCOMPLETE.json','summary.json'):
    path=root/name
    if path.exists():print(name+(' (last failure; can remain while resuming)' if name=='INCOMPLETE.json' else ''),json.dumps(json.loads(path.read_text()),indent=2))
path=root/'progress.json'
if path.exists():
    counts={}
    for n,v in json.loads(path.read_text()).get('jobs',{}).items():
        counts[v['state']]=counts.get(v['state'],0)+1
        if v['state'] in ('running','failed'):print(n,v)
    print('Current queue (not whole experiment):',counts)
for path in root.glob('search/*/*/selection.json'):
    value=json.loads(path.read_text());print(path.parent.relative_to(root),len(value['trials']),'trials; selected=',
        value['selected_candidate']['expert_id'] if value['selected_candidate'] else 'baseline retained')
for path in sorted(root.glob('search/*/*/api/configuration.json')):
    api=path.parent;attempts=[json.loads(p.read_text()) for p in sorted(api.glob('attempt_*.json'))]
    config=json.loads(path.read_text());last=attempts[-1] if attempts else {}
    print(api.parent.relative_to(root),'API attempts=',len(attempts),'/',config['max_api_calls'],
          'last=',last.get('status'),'category=',last.get('failure_category'))
    if last:
        diag=api/f'validation_{last["attempt"]:04d}.json'
        if diag.exists():print('Validation:',json.loads(diag.read_text())['validation_rule'])
for path in sorted(root.glob('search/*/*/trial_*/result.json')):
    result=json.loads(path.read_text())
    if result.get('proposal_executed') is False:print(path.parent.relative_to(root),'rejected:',result['reason'])
if (root/'COMPLETE.json').exists():print('ROUND18 COMPLETE:',root/'round18_review_light.tar.gz')
