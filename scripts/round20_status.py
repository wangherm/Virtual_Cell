"""Progress and gate decisions, not just the currently active queue."""
import argparse,json
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round20_signal_01')
a=p.parse_args();root=Path(a.work_dir)/'runs'/a.name
if not root.exists():print('Run not started:',root)
if (root/'progress.json').exists():
    progress=json.loads((root/'progress.json').read_text());print('Source expert queue (not whole experiment):',dict(Counter(v['state'] for v in progress.get('jobs',{}).values())))
for name in ('stage.json','budget.json','search_progress.json','worker_progress.json','INCOMPLETE.json'):
    path=root/name
    if path.exists():print(name+(' (last failure; may remain during resume)' if name=='INCOMPLETE.json' else ''),json.dumps(json.loads(path.read_text()),indent=2))
if (root/'worker_progress.json').exists():
    for name in json.loads((root/'worker_progress.json').read_text()).get('active',[]):
        config=root/'worker_configs'/(name+'.json')
        if config.exists():
            progress=Path(json.loads(config.read_text())['output_dir'])/'progress.json'
            print('Active:',name,json.loads(progress.read_text()) if progress.exists() else 'loading / before checkpoint')
for path in sorted((root/'qc').glob('*/source_signal.json')):
    d=json.loads(path.read_text());print('QC',path.parent.name,'pairs=',d['paired_units'],'strong_genes=',d['strong_genes'],'gate=',d['signal_gate'])
for path in sorted((root/'selections').glob('*.json')):
    d=json.loads(path.read_text());print('Selection',path.stem,d['gate'])
for path in sorted((root/'search').glob('*/*/*api/attempt_*.json')):
    d=json.loads(path.read_text());print('API',str(path.relative_to(root)),d['status'],d.get('failure_category'),d.get('output_tokens'))
if (root/'COMPLETE.json').exists():print('ROUND20 COMPLETE:',root/'round20_review_light.tar.gz')
