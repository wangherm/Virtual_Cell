"""Read-only progress of the whole campaign, individual arms, repairs and provider usage."""
import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vcell.model_agent.artifacts import read
from vcell.round25.runtime import charged


def main():
    p=argparse.ArgumentParser();p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round25_self_correct_01');a=p.parse_args()
    root=Path(a.work_dir)/'runs'/a.name
    if not (root/'campaign_plan.json').exists():print('ROUND25 NOT STARTED:',root);return
    cfg=read(root/'campaign_plan.json')['config']
    for name in ('stage.json','baseline_replay.json','INCOMPLETE.json'):
        if (root/name).exists():
            r=read(root/name)
            if name=='baseline_replay.json':r={'passed':r['passed'],'folds':len(r['checks'])}
            if name=='INCOMPLETE.json' and (root/'stage.json').exists() and (root/name).stat().st_mtime<(root/'stage.json').stat().st_mtime:
                r['note']='This error predates current progress and may belong to the previous interrupted attempt'
            print(name,r)
    total_calls=0;seconds=charged(root/'baseline_budget')
    if (root/'baseline_budget/worker_progress.json').exists():print('Baseline worker:',read(root/'baseline_budget/worker_progress.json'))
    for arm in cfg['arms']:
        ar=root/'arms'/arm;calls=read(ar/'api_ledger.json') if (ar/'api_ledger.json').exists() else [];total_calls+=len(calls);seconds+=charged(ar)
        print(arm,'API',len(calls),'/',cfg['api_requests_per_arm'],dict(Counter(r['status'] for r in calls)),
            'roles',dict(Counter(r.get('role') for r in calls)),'output tokens',sum(r.get('charged_output_tokens',0) for r in calls),
            'worker minutes',round(charged(ar)/60,1),'search complete',(ar/'SEARCH_COMPLETE.json').exists())
        novel_total=0
        for task in cfg['tasks']:
            pool=ar/task;records=[read(p) for p in (pool/'assessments').glob('*/assessment.json')]
            records=[r for r in records if r['identity']['phase']=='search']
            unique={r['behavior_sha256'] for r in records if r['status']=='valid'}
            novel_total+=max(0,len(unique)-1)
            # The source comparator/seed is not a newly evolved prediction behavior.
            print(' ',task,'assessments',dict(Counter(r['status'] for r in records)),
                  'distinct new',max(0,len(unique)-1),'source frozen',(pool/'search_frozen.json').exists())
        print(' ',arm,'distinct new total',novel_total,'/',cfg['valid_target_per_arm'],'(extension is evidence-gated)')
        if (ar/'worker_progress.json').exists():print(' ',read(ar/'worker_progress.json'))
    print('TOTAL API',total_calls,'/',len(cfg['arms'])*cfg['api_requests_per_arm'],
        '; worker hours',round(seconds/3600,3),'/',cfg['baseline_hours']+len(cfg['arms'])*cfg['arm_hours'])
    if (root/'COMPLETE.json').exists():print('ROUND25 COMPLETE:',root/'round25_review_light.tar.gz')
    else:print('Campaign not complete. Stage and process/log activity determine whether it is running.')


if __name__=='__main__':main()
