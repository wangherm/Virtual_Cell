"""Compact Round13 progress; no network or heavy imports."""
import argparse
from collections import deque
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',nargs='?',default='/root/autodl-tmp/vcell-work/runs/round13_01')
root=Path(p.parse_args().run)
for name in ('stage.json','INCOMPLETE.json'):
    if (root/name).exists():print(name,(root/name).read_text())
if (root/'COMPLETE.json').exists():print('ROUND13 COMPLETE:',root/'round13_review_light.tar.gz')
elif (root/'progress.json').exists():
    jobs=json.loads((root/'progress.json').read_text()).get('jobs',{})
    print('Completed:',sum(j['state']=='complete' for j in jobs.values()),'/',len(jobs))
    for name,job in jobs.items():
        if job['state'] in ('running','failed'):
            print(name,job['state'],'exit=',job.get('exit_code'))
            path=Path(job['log'])
            if path.exists():
                with path.open(errors='replace') as stream:print(''.join(deque(stream,maxlen=4)))
else:print('Preparing/auditing inputs; inspect the round13_pipeline log.')
