"""Read-only Round24 stage, worker, module-yield and paid-budget status."""
import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vcell.model_agent.artifacts import read


def main():
    p=argparse.ArgumentParser();p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round24_mechanism_01');a=p.parse_args();root=Path(a.work_dir)/'runs'/a.name
    if not root.exists():print('ROUND24 NOT STARTED:',root);return
    for name in ('stage.json','replay.json','pilot.json','seed_selection.json','INCOMPLETE.json','worker_progress.json'):
        if (root/name).exists():
            value=read(root/name)
            if name=='replay.json':value={'passed':value['passed'],'folds':len(value['checks'])}
            if name=='INCOMPLETE.json' and (root/'stage.json').exists() and (root/name).stat().st_mtime<(root/'stage.json').stat().st_mtime:value={**value,'note':'This error predates the current stage; it may describe the previous interrupted attempt'}
            print(name,value)
    records=[read(p) for p in (root/'assessments').glob('*/assessment.json')]
    for scope in ('fixed','evolved','random'):
        block=[r for r in records if r['identity']['scope']==scope]
        print(scope,'assessments:',dict(Counter(r['status'] for r in block)),'failure categories:',dict(Counter(r.get('category') for r in block if r['status']=='invalid')))
    if (root/'api_ledger.json').exists():
        rows=read(root/'api_ledger.json');print('API requests:',len(rows),'/ 60; charged output tokens:',sum(r.get('charged_output_tokens',0) for r in rows))
    if (root/'search_frozen.json').exists():
        d=read(root/'search_frozen.json');print('Distinct valid new:',d['valid_new'],'/ 20; source search frozen:',d['reason'])
    if (root/'evaluation_budget.json').exists():
        d=read(root/'evaluation_budget.json');print('Worker seconds charged:',round(sum(r['charged_seconds'] for r in d['workers'].values()),1))
    if (root/'COMPLETE.json').exists():print('ROUND24 COMPLETE:',root/'round24_review_light.tar.gz')


if __name__=='__main__':main()
