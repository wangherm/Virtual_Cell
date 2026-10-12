"""Read live pilot progress without sending provider requests."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='deepseek_pilot_01')
    a=p.parse_args();root=Path(a.work_dir)/'runs'/a.name
    if not root.exists():print('No pilot directory; start the run first');return
    for name in ('stage.json','INCOMPLETE.json','provider_summary.json'):
        path=root/name
        if path.exists():print(name,json.dumps(json.loads(path.read_text()),indent=2))
    for audit in sorted((root/'api').glob('round_*')):
        records=[json.loads(path.read_text()) for path in sorted(audit.glob('attempt_*.json'))]
        if records:
            last=records[-1];print(audit.name,'requests=',len(records),'last=',last['status'],'started=',last.get('started_at'),
                                  'model=',last.get('returned_model',last.get('requested_model')))
    if (root/'COMPLETE.json').exists():print('PILOT COMPLETE:',root/'model_agent_review_light.tar.gz')


if __name__=='__main__':main()
