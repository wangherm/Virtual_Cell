"""Show P4 stage, process progress and the latest worker log tail."""
import argparse
from collections import Counter, deque
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round15_p4_01')
    a=p.parse_args();root=Path(a.work_dir)/'runs'/a.name
    for name in ('stage.json','INCOMPLETE.json','budget.json'):
        path=root/name
        if path.exists():print(name,json.dumps(json.loads(path.read_text()),indent=2))
    progress=root/'progress.json'
    if progress.exists():
        jobs=json.loads(progress.read_text())['jobs'];print('Jobs:',dict(Counter(job['state'] for job in jobs.values())))
        for name,job in jobs.items():
            if job['state']=='running':
                path=Path(job['log']);print('RUNNING:',name)
                if path.exists():
                    with path.open(encoding='utf-8',errors='replace') as f:print(''.join(deque(f,maxlen=4)).strip())
    if (root/'COMPLETE.json').exists():print('P4 COMPLETE:',root/'round15_p4_review_light.tar.gz')
    elif not root.exists():print('No run directory: launch P4 first')


if __name__=='__main__':main()
