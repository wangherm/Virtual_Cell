"""Read-only progress, accepted critic actions, fallback attribution and matched caps."""
import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vcell.model_agent.artifacts import read
from vcell.round25.runtime import charged


def main():
    p=argparse.ArgumentParser();p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round26_critic_01');a=p.parse_args()
    root=Path(a.work_dir)/'runs'/a.name
    if not (root/'campaign_plan.json').exists():print('ROUND26 NOT STARTED:',root);return
    cfg=read(root/'campaign_plan.json')['config'];calls=0;seconds=charged(root/'baseline_budget')
    for name in ('stage.json','INCOMPLETE.json','LIVE_SMOKE_COMPLETE.json'):
        if (root/name).exists():
            row=read(root/name)
            if name=='LIVE_SMOKE_COMPLETE.json':row={k:v for k,v in row.items() if k!='files'}
            if name=='INCOMPLETE.json' and (root/'stage.json').exists() and (root/name).stat().st_mtime<(root/'stage.json').stat().st_mtime:row['note']='Older error; compare current stage/process/log activity'
            print(name,row)
    for arm,ar in [('pilot',root/'pilot/meta'),*[(arm,root/'arms'/arm) for arm in cfg['arms']+['meta_extension']]]:
        if not ar.exists():continue
        ledger=read(ar/'api_ledger.json') if (ar/'api_ledger.json').exists() else [];calls+=len(ledger);seconds+=charged(ar)
        reviews=[read(p) for p in ar.glob('*/decisions/*/decision.json')]
        print(arm,'requests',len(ledger),'roles',dict(Counter(r.get('role') for r in ledger)),
            'accepted critic',sum(r['attribution']=='critic' for r in reviews),'fallback',sum(r['attribution']=='fallback' for r in reviews),
            'worker minutes',round(charged(ar)/60,2),'search complete',(ar/'SEARCH_COMPLETE.json').exists())
        for task in cfg['tasks']:
            pool=ar/task
            if not (pool/'run_plan.json').exists():continue
            # Progress reads small checksummed records, without hashing large worker payloads.
            rows=[read(p) for p in (pool/'assessments').glob('*/assessment.json')]
            rows=[r for r in rows if r['identity']['phase']=='search']
            shared={read(r['record'])['behavior_sha256'] for r in read(pool/'common_sources.json')}
            new={r['behavior_sha256'] for r in rows if r['status']=='valid'}-shared
            print(' ',task,'assessments',dict(Counter(r['status'] for r in rows)),'distinct new',len(new),'source frozen',(pool/'search_frozen.json').exists())
    print('TOTAL requests including pilot',calls,'worker hours',round(seconds/3600,3))
    if (root/'COMPLETE.json').exists():print('ROUND26 SEARCH COMPLETE:',root/'round26_review_light.tar.gz','Scientific PASS:',read(root/'summary.json')['scientific_pass'])
    else:print('NOT COMPLETE. The stage is not a heartbeat; check process and latest log activity.')


if __name__=='__main__':main()
