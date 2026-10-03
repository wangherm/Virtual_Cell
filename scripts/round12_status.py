"""Compact Round12 stage and active head progress, without heavy imports."""
import argparse
from collections import deque
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',nargs='?',default='/root/autodl-tmp/vcell-work/runs/round12_01')
root=Path(p.parse_args().run)
if not root.exists():
    print('No run directory; inspect the round12_pipeline log.')
else:
    for name in ('stage.json','INCOMPLETE.json'):
        if (root/name).exists():print(name,(root/name).read_text())
    if (root/'COMPLETE.json').exists():
        print('ROUND12 COMPLETE. Review:',root/'round12_review_light.tar.gz')
    elif (root/'progress.json').exists():
        jobs=json.loads((root/'progress.json').read_text()).get('jobs',{})
        print('Current stage completed:',sum(j['state']=='complete' for j in jobs.values()),'/',len(jobs))
        for name,job in jobs.items():
            if job['state'] in ('running','failed'):
                print(name,job['state'],'exit=',job.get('exit_code'))
                path=Path(job['log'])
                if path.exists():
                    with path.open(errors='replace') as f:print(''.join(deque(f,maxlen=4)))
    else:
        print('Preparing raw data/knowledge; follow the round12_pipeline log for file/cell progress.')
