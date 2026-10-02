"""Round9 stage, job counts and recent logs; no torch import or network access."""
import argparse
from collections import deque
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',nargs='?',default='/root/autodl-tmp/vcell-work/runs/round9_01')
root=Path(p.parse_args().run)
for name in ('stage.json','COMPLETE.json','INCOMPLETE.json'):
    if (root/name).exists():print(name,(root/name).read_text())
if (root/'progress.json').exists():
    state=json.loads((root/'progress.json').read_text());jobs=state.get('jobs',{})
    print('Completed:',sum(r['state']=='complete' for r in jobs.values()),'/',len(jobs))
    for name,r in jobs.items():
        if r['state'] in ('running','failed'):
            print(name,r['state'],'exit=',r.get('exit_code'))
            if Path(r['log']).exists():
                with Path(r['log']).open(errors='replace') as f:print(''.join(deque(f,maxlen=4)))
if not root.exists():print('No run directory; inspect the round9_pipeline log.')
