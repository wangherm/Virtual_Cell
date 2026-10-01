"""Print Round8 status without importing torch or contacting external services."""
import argparse
import json
from collections import deque
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('run',nargs='?',default='/root/autodl-tmp/vcell-work/runs/round8_01')
args=parser.parse_args()
root=Path(args.run)
for name in ('COMPLETE.json','INCOMPLETE.json','progress.json'):
    if (root/name).exists():
        print(name)
        value=json.loads((root/name).read_text())
        if name!='progress.json' or 'jobs' not in value:
            print(json.dumps(value,indent=2));continue
        jobs=value['jobs'];print(f"Complete: {sum(j['state']=='complete' for j in jobs.values())}/{len(jobs)}")
        for job,record in jobs.items():
            if record['state'] in ('running','failed'):
                print(job,record['state'],record.get('exit_code'))
                log=Path(record['log'])
                if log.exists():
                    with log.open(errors='replace') as stream:print(''.join(deque(stream,maxlen=3)))
if not root.exists():print('Run directory does not exist; inspect the pipeline log.')
