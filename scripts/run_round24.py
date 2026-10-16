"""Round24: frozen replay, mechanism ablations, and bounded source-only module evolution."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import subprocess
import sys
from importlib.metadata import version
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import numpy as np
import pandas as pd
from vcell.clean_background import run_lock
from vcell.model_agent.artifacts import read,verify,seal
from vcell.round11 import freeze
from vcell.round22.gateway import Gateway
from vcell.round22.evaluator import worker_env
from vcell.round24 import modules
from vcell.round24.prepare import parent_run,prepare,verify_replay
from vcell.round24.runtime import assess,fingerprint
from vcell.round24.stats import cluster_gain
from vcell.evolution.journal import feedback
from vcell.utils import write_json,file_sha256
from run_round22 import preflight,package


def stage(root,name,**extra):
    write_json(root/'stage.json',{'stage':name,'test_evaluated':False,**extra});print('ROUND24 STAGE:',name,extra,flush=True)


def used(root):
    p=root/'evaluation_budget.json'
    return sum(r['charged_seconds'] for r in read(p)['workers'].values()) if p.exists() else 0.


def export_attempts(root):
    for p in (root/'evolution').glob('*.sqlite'):
        con=sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'attempt_log' in tables:
                con.row_factory=sqlite3.Row;records=[dict(r) for r in con.execute('SELECT * FROM attempt_log ORDER BY rowid')]
                write_json(root/'attempt_log_export.json',json.loads(json.dumps(records,default=str)))
        finally:con.close()


def novel(root):
    proposed={hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest() for p in (root/'evolution').glob('gen_*/main.py')}
    rows=[read(p) for p in (root/'assessments').glob('*/assessment.json')]
    rows=[r for r in rows if r['status']=='valid' and r['identity']['scope']=='evolved' and r['identity']['phase']=='search']
    seed_hash=hashlib.sha256((root/'seed.py').read_text(encoding='utf-8').encode()).hexdigest()
    seen={r['behavior_sha256'] for r in rows if r['module_sha256']==seed_hash};result=[]
    for r in sorted(rows,key=lambda r:r['metrics']['combined_score'],reverse=True):
        if r['module_sha256']!=seed_hash and r['module_sha256'] in proposed and r['behavior_sha256'] not in seen:
            seen.add(r['behavior_sha256']);result.append(r)
    return result


def source_memory(root):
    records=feedback(root/'journal.jsonl',limit=12)
    # Existing helper emits safe fields only. Attach source metrics/errors explicitly.
    safe=[]
    for p in sorted((root/'assessments').glob('*/assessment.json'),key=lambda p:p.stat().st_mtime)[-12:]:
        r=read(p)
        if r['identity']['phase']=='search':safe.append({'scope':r['identity']['scope'],'scores':r['metrics']['public'],'error':r.get('error'),'module_sha256':r['module_sha256']})
    write_json(root/'source_feedback.json',{'history':records,'assessments':safe,'scope':'source search only'})


def select_seed(records,definitions):
    source=[r for r in records if r['task']=='repeat' and r['phase']=='search' and r['arm']=='all' and not r['name'].endswith('_factorial')]
    best=min(source,key=lambda r:r['metrics']['public']['mse'])
    near=[r for r in source if r['metrics']['public']['mse']<=best['metrics']['public']['mse']*1.002]
    return min(near,key=lambda r:len(definitions[r['name']].splitlines())),best


def diagnostics(root,code):
    definitions=modules.methods(code);records=[];modules_dir=root/'methods';modules_dir.mkdir(exist_ok=True)
    for name,body in definitions.items():
        p=modules_dir/(name+'.py')
        if p.exists() and p.read_text(encoding='utf-8')!=body:raise ValueError('Frozen method changed')
        p.write_text(body,encoding='utf-8')
    replay={}
    for phase in ('search','confirmation'):
        stage(root,'replay original winner',phase=phase)
        replay[phase]=assess(root,definitions['winner_original'],phase=phase,audit=True,controls=False)
    verify_replay(root,replay)
    for name,body in definitions.items():
        arms=('none','historical','new','all') if name in ('winner_original','winner_repaired') else ('all',)
        for arm in arms:
            for phase in ('search','confirmation'):
                stage(root,'fixed mechanism and data ablation',method=name,data_arm=arm,phase=phase)
                r=assess(root,body,phase=phase,arm=arm,controls=False,audit=True)
                if r['status']!='valid':raise RuntimeError('Fixed method failed; fix before paid evolution: '+r.get('error',''))
                records.append({'name':name,'arm':arm,'task':'repeat','phase':phase,**r})
    for name in ('winner_original','winner_repaired','simple_support'):
        for phase in ('search','confirmation'):
            stage(root,'separate support and identity controls',method=name,phase=phase)
            r=assess(root,definitions[name],phase=phase,audit=True,controls=True)
            if r['status']!='valid':raise RuntimeError('Factorial controls failed: '+r.get('error',''))
            records.append({'name':name+'_factorial','arm':'all','task':'repeat','phase':phase,**r})
    for name in ('winner_original','winner_repaired','simple_support'):
        for phase in ('search','confirmation'):
            stage(root,'separate transfer development diagnostic',method=name,phase=phase)
            r=assess(root,definitions[name],task='transfer',phase=phase,controls=False,audit=True)
            records.append({'name':name,'arm':'all','task':'transfer','phase':phase,**r})
    freeze(root/'diagnostics_frozen.json',records)
    # Selection reads SOURCE rows only; confirmation is never used to choose a seed.
    seed,best=select_seed(records,definitions)
    seed_path=root/'seed.py';body=definitions[seed['name']]
    if seed_path.exists() and seed_path.read_text(encoding='utf-8')!=body:raise ValueError('Source seed changed')
    seed_path.write_text(body,encoding='utf-8')
    freeze(root/'seed_selection.json',{'method':seed['name'],'best_fixed_comparator':best['name'],'selection':'SOURCE MSE only; shortest method within 0.2% of best source MSE','confirmation_used':False})
    reference={}
    for phase in ('search','confirmation'):
        r=next(r for r in records if r['task']=='repeat' and r['phase']==phase and r['arm']=='all' and r['name']==best['name'])
        reference[phase]={}
        for fold in read(root/'folds.json')['tasks']['repeat'][phase]:
            path=Path(r['directory'])/(fold['name']+'.npz');reference[phase][fold['name']]={'path':str(path),'sha256':file_sha256(path)}
    freeze(root/'search_baseline.json',reference)
    return records


def evolve(root,cfg,shinka):
    if (root/'search_frozen.json').exists():return read(root/'search_frozen.json')
    seed=assess(root,(root/'seed.py').read_text(encoding='utf-8'),scope='evolved',audit=False)
    if seed['status']!='valid':raise RuntimeError('Seed module failed its source contract: '+seed.get('error',''))
    key=os.environ.get('DEEPSEEK_API_KEY','').strip()
    if not key:raise RuntimeError('Export DEEPSEEK_API_KEY; numeric diagnostics are cached and will not be repeated')
    if not Path(shinka).is_file():raise ValueError('Pinned Shinka environment missing; reuse the Round22 installer')
    relay=secrets.token_hex(24)
    with Gateway(root/'api_ledger.json',key,cfg['model'],relay,cfg['api_requests'],cfg['api_output_tokens'],cfg['port'],max_tokens=cfg['api_max_tokens'],reasoning_effort=cfg['reasoning_effort'],failure_limit=cfg['api_failure_limit']) as gateway:
        pilot_done=(root/'pilot.json').exists();gateway.requests_max=cfg['api_requests'] if pilot_done and read(root/'pilot.json')['passed'] else cfg['pilot_requests']
        if pilot_done and not read(root/'pilot.json')['passed']:return freeze_search(root,novel(root),cfg,'pilot validity gate failed')
        empty=0
        for cycle in range(cfg['max_cycles']):
            stamp=root/'cycles'/f'{cycle:03d}.json'
            if stamp.exists():
                empty=read(stamp)['empty_streak'];continue
            rows=novel(root)
            if len(rows)>=cfg['valid_new_target']:break
            if used(root)+cfg['worker_seconds']>cfg['evaluation_hours']*3600*cfg['search_fraction']:break
            if len(gateway.records)>=gateway.requests_max or sum(r['charged_output_tokens'] for r in gateway.records)+cfg['api_max_tokens']>cfg['api_output_tokens']:break
            if len(gateway.records)>=cfg['api_failure_limit'] and all(r['status']!='complete' for r in gateway.records[-cfg['api_failure_limit']:]):break
            source_memory(root);stage(root,'Shinka module evolution',cycle=cycle+1,valid_new=len(rows),target=cfg['valid_new_target'])
            env=worker_env();env.update(ROUND24_ROOT=str(root),ROUND22_GATEWAY_KEY=relay,SHINKA_PRICING_MODE='offline')
            path=root/'logs'/f'shinka_{cycle:03d}.log';path.parent.mkdir(exist_ok=True)
            target=1+(cycle+1)*cfg['generations_per_cycle']
            with path.open('a',encoding='utf-8') as log:
                r=subprocess.run([str(shinka),str(REPO/'scripts/run_shinka_round24.py'),'--root',str(root),'--generations',str(target)],env=env,stdout=log,stderr=subprocess.STDOUT)
            now=novel(root);empty=empty+1 if len(now)==len(rows) else 0
            # Reaching the planned pilot ceiling is a defined stop, not a quota reset.
            pilot_ceiling_hit=not pilot_done and len(gateway.records)>=cfg['pilot_requests']
            if not pilot_done and (pilot_ceiling_hit or cycle>=1):
                passed=len(now)>=cfg['pilot_min_valid'] and len(now)/max(len(gateway.records),1)>=cfg['pilot_min_yield']
                freeze(root/'pilot.json',{'passed':passed,'requests':len(gateway.records),'distinct_valid_new':len(now),'yield':len(now)/max(len(gateway.records),1),'source_only':True})
                pilot_done=True
                if passed:gateway.requests_max=cfg['api_requests']
                else:return freeze_search(root,now,cfg,'pilot validity gate failed')
            if r.returncode and not pilot_ceiling_hit and len(gateway.records)<gateway.requests_max:raise RuntimeError('Shinka module runner failed; inspect '+str(path)+'; resume retains ledger/cache')
            freeze(stamp,{'valid_new':len(now),'empty_streak':empty,'source_only':True})
            if empty>=cfg['zero_yield_patience']:break
    return freeze_search(root,novel(root),cfg,'valid target, cycle, provider, no-yield or worker budget reached')


def freeze_search(root,rows,cfg,reason):
    if any(r['identity']['phase']!='search' for r in rows):raise ValueError('Confirmation cannot select a source finalist')
    finalists=[]
    for tier,field in (('target_response','target_response_candidate'),('overall_error','overall_error_candidate')):
        legal=[r for r in rows if r['metrics']['public'].get(field)]
        if legal:finalists.append({'tier':tier,**max(legal,key=lambda r:r['metrics']['combined_score'])})
    value={'finalists':finalists,'valid_new':len(rows),'goal_reached':len(rows)>=cfg['valid_new_target'],'reason':reason,'selection':'Source-only winners frozen before any evolved confirmation; no outer evaluation'}
    freeze(root/'search_frozen.json',value);return value


def random_control(root,cfg,target):
    path=root/'random_frozen.json'
    if path.exists():return read(path)
    rng=np.random.default_rng(cfg['seed']);rows=[];seen=set()
    for trial in range(max(1,target)*4):
        if len(rows)>=target or used(root)+cfg['worker_seconds']>cfg['evaluation_hours']*3600*cfg['random_fraction']:break
        body=modules.simple_module(alpha_grid=tuple(sorted({0.,float(rng.choice([.03,.1,.3,.5,.7,1.,1.5]))})),ridge=bool(rng.integers(2)),penalty=float(rng.choice([.1,1.,10.,100.])))
        stage(root,'matched valid-count random comparator',trial=trial+1,valid=len(rows),target=target)
        r=assess(root,body,scope='random',audit=False)
        if r['status']=='valid' and r['behavior_sha256'] not in seen:seen.add(r['behavior_sha256']);rows.append(r)
    value={'records':rows,'target':target,'matched':len(rows)==target,'scope':'Disclosed narrow alpha/shared-ridge grammar; not the same code search space as Shinka'}
    freeze(path,value);return value


def report(root,fixed,selection,random):
    rows=[]
    for r in fixed:rows.append({'method':r['name'],'arm':r['arm'],'task':r['task'],'phase':r['phase'],'status':r['status'],**r['metrics']['public']})
    confirmation=[]
    if selection is not None:
        for r in novel(root):rows.append({'method':'shinka_source','module_sha256':r['module_sha256'],'arm':'all','task':'repeat','phase':'search','status':r['status'],**r['metrics']['public']})
        for r in random.get('records',[]) if random else []:rows.append({'method':'random_source','module_sha256':r['module_sha256'],'arm':'all','task':'repeat','phase':'search','status':r['status'],**r['metrics']['public']})
        for item in selection['finalists']:
            stage(root,'source-frozen evolved development confirmation',tier=item['tier'])
            body=Path(item['module']).read_text(encoding='utf-8')
            if hashlib.sha256(body.encode()).hexdigest()!=item['module_sha256']:raise ValueError('Source-frozen finalist code changed')
            r=assess(root,body,phase='confirmation',scope='evolved',audit=False)
            confirmation.append({'tier':item['tier'],'source_module_sha256':item['module_sha256'],**r})
            rows.append({'method':'shinka_'+item['tier'],'arm':'all','task':'repeat','phase':'confirmation','status':r['status'],**r['metrics']['public']})
        if random and random['records']:
            item=max(random['records'],key=lambda r:r['metrics']['combined_score'])
            r=assess(root,Path(item['module']).read_text(encoding='utf-8'),phase='confirmation',scope='random',audit=False)
            rows.append({'method':'random_source_best','arm':'all','task':'repeat','phase':'confirmation','status':r['status'],**r['metrics']['public']})
    write_json(root/'evolved_confirmation.json',confirmation)
    pd.DataFrame(rows).to_csv(root/'comparison.csv',index=False)
    sensitivities=[]
    for r in fixed:
        if r['status']=='valid' and r['task']=='repeat' and r['phase']=='confirmation' and r['arm']=='all':
            folder=Path(r['directory']);sensitivities.append({'method':r['name'],**cluster_gain(pd.read_csv(folder/'scores.csv'),pd.read_csv(folder/'baseline.csv'))})
    write_json(root/'paired_sensitivity.json',sensitivities)
    summary={'stage':'complete','source_search':selection,'confirmed_target_response':any(r['tier']=='target_response' and r['status']=='valid' and r['metrics']['public']['target_response_candidate'] for r in confirmation),
        'confirmed_overall_error':any(r['status']=='valid' and r['metrics']['public']['overall_error_candidate'] for r in confirmation),
        'random_valid_count_matched':random.get('matched') if random else None,'worker_seconds_charged':used(root),'test_evaluated':False,'HT29_evaluated':False,'outer_evaluated':False,
        'claim':'Old confirmation is development evidence, not a fresh independent test; deterministic methods are not repeated merely with new seed labels.'}
    write_json(root/'summary.json',summary)
    text=['# Round24 mechanism and evolution report','',summary['claim'],'',
        'Original frozen replay: '+str(read(root/'replay.json')['passed']),
        'New confirmed target-response candidate: '+str(summary['confirmed_target_response']),
        'See comparison.csv for fixed-data/structure ablations and separate transfer diagnostics.',
        'Branch diagnostics and actual auxiliary use: assessments/*/branch_audit.json.',
        'Support-only and identity-only controls: assessments/*/controls.csv.','',
        'No outer or sealed external cohort was opened. See lineage.json for the original data boundary.']
    (root/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def launch(a):
    work=Path(a.work_dir);root=work/'runs'/a.name;cfg=read(a.config);parent=parent_run(work,a.round23_run)
    if root.exists() and any(root.iterdir()) and not a.resume:raise ValueError('Use --resume or a new run name')
    root.mkdir(parents=True,exist_ok=True)
    if not 0<cfg['diagnostic_fraction']<cfg['search_fraction']<cfg['random_fraction']<1:raise ValueError('Reserve computation for search, random and confirmation')
    if not 1<=cfg['api_requests']<=60 or not 1<=cfg['pilot_requests']<=cfg['api_requests'] or not 1<=cfg['valid_new_target']<=20:raise ValueError('This Round24 release bounds search to 60 requests and 20 new behaviors')
    if cfg['max_cycles']<2 or cfg['generations_per_cycle']<1 or cfg['worker_seconds']<1 or cfg['steps']<1:raise ValueError('Positive worker/cycle budgets required')
    budget={'seed':cfg['seed'],'device':a.device,'steps':cfg['steps'],'seconds':cfg['worker_seconds'],'ram_bytes':12*1024**3,'evaluation_seconds_max':cfg['evaluation_hours']*3600,'root':str(root)}
    plan={'repo':str(REPO),'python':sys.executable,'framework_sha256':fingerprint(REPO),'parent':str(parent),'parent_complete_sha256':file_sha256(parent/'COMPLETE.json'),
        'config':cfg,'budget':budget,'versions':{n:version(n) for n in ('numpy','pandas','torch','scipy')},'test_evaluated':False}
    freeze(root/'run_plan.json',plan)
    if (root/'COMPLETE.json').exists():verify(root);print('ROUND24 COMPLETE VERIFIED',root);return root
    try:
        if shutil.disk_usage(root).free<10*1024**3:raise ValueError('Need 10 GiB reserve for deduplicated worker inputs; no artifacts deleted')
        stage(root,'freeze full Round23 lineage and source-only folds',parent=str(parent));code=prepare(parent,root,cfg)
        if a.mode=='prepare':return root
        stage(root,'verify Linux kernel isolation');preflight(root,budget,sys.executable)
        fixed=diagnostics(root,code)
        if a.mode=='diagnostics':stage(root,'numeric diagnostics complete; paid evolution not run');report(root,fixed,None,None);return root
        selection=evolve(root,cfg,a.shinka_python or work/'envs/shinka/bin/python')
        random=random_control(root,cfg,selection['valid_new'])
        report(root,fixed,selection,random);(root/'INCOMPLETE.json').unlink(missing_ok=True);stage(root,'complete')
        return root
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error_type':type(exc).__name__,'error':str(exc),'test_evaluated':False});raise
    finally:
        export_attempts(root)
        if (root/'stage.json').exists() and read(root/'stage.json')['stage']=='complete' and not (root/'INCOMPLETE.json').exists():seal(root,test_evaluated=False)
        path=package(root);target=root/'round24_review_light.tar.gz';path.replace(target);print('REVIEW PACKAGE:',target,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round24_mechanism_01');p.add_argument('--round23-run')
    p.add_argument('--config',default=str(REPO/'configs/round24.json'));p.add_argument('--shinka-python');p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--mode',choices=['prepare','diagnostics','all'],default='all');p.add_argument('--resume',action='store_true');a=p.parse_args()
    work=Path(a.work_dir)
    with run_lock(work/'runs'/('.'+a.name+'.lock')):
        with run_lock(work/'runs'/'.evolution_gpu.lock'):launch(a)


if __name__=='__main__':main()
