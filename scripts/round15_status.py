"""Round15 core progress and readiness, with no numerical dependencies."""
import argparse
from collections import Counter, deque
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',nargs='?',default='/root/autodl-tmp/vcell-work/runs/round15_01')
root=Path(p.parse_args().run)
for name in ('stage.json','INCOMPLETE.json','screen_summary.json'):
    if (root/name).exists():print(name,(root/name).read_text())
states=[]
for path in (root/'models').glob('*/head_status.json'):states+=json.loads(path.read_text()).get('heads',[])
print('Heads:',dict(Counter(s['status'] for s in states)))
if (root/'progress.json').exists():
    for name,job in json.loads((root/'progress.json').read_text()).get('jobs',{}).items():
        if job['state'] in ('running','failed'):
            print(name,job['state'],'exit=',job.get('exit_code'));path=Path(job['log'])
            if path.exists():
                with path.open(errors='replace') as stream:print(''.join(deque(stream,maxlen=4)))
if (root/'CORE_COMPLETE.json').exists():
    print('CORE SCREEN COMPLETE; P3/P4/P5/P6/FS NOT RUN:',root/'round15_review_light.tar.gz')
