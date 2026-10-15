"""Run a resumable research campaign; numerical workers remain kernel-isolated."""
import argparse
import copy
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import pandas as pd
from vcell.clean_background import run_lock
from vcell.model_agent.artifacts import read,verify,seal
from vcell.round11 import freeze
from vcell.round22.evaluator import inputs,fixed_predictions,worker_env
from vcell.round22.gateway import Gateway
from vcell.round22.metrics import score,macro
from vcell.utils import file_sha256,write_json
from vcell.evolution.catalog import inventory,fetch,inspect_h5ad,expanded_catalog
from vcell.evolution.grader import assess,valid_candidates
from vcell.evolution.journal import append,feedback,fingerprint,events
from vcell.evolution.pools import compile_pools
from vcell.evolution.seeds import seed,random_method
from run_round22 import preflight,stage,package
from run_shinka_round22 import check_code_probe


def novel(pool):
    records=valid_candidates(pool);initial=hashlib.sha256((Path(pool)/'seed.py').read_text(encoding='utf-8').encode()).hexdigest()
    proposed={hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest() for p in (Path(pool)/'evolution').glob('gen_*/main.py')}
    base=next((r.get('behavior_sha256') for r in records if r['code_sha256']==initial),None)
    seen={base};out=[]
    for r in sorted(records,key=lambda r:r['metrics']['combined_score'],reverse=True):
        behavior=r.get('behavior_sha256',r['code_sha256'])
        if r['code_sha256']==initial or r['code_sha256'] not in proposed or behavior in seen:continue
        seen.add(behavior);out.append(r)
    return out


def freeze_baseline(pool):
    plan=read(pool/'run_plan.json');task=plan['task'];frames=[]
    for fold in read(pool/'data_plan.json')['folds'][task]['search']:
        d,g,m=inputs(fold)
        for name,pred in fixed_predictions(d).items():
            f=score(pred,g['truth'],g['background'],g['strong_mask'],m);f['method']=name;frames.append(f)
    frame=pd.concat(frames);best=min(frame.method.unique(),key=lambda m:macro(frame[frame.method.eq(m)],'mse'))
    freeze(pool/'baseline_selection.json',{task:best});frame.to_csv(pool/'fixed_source_scores.csv',index=False)


def provider_room(root,cfg):
    rows=read(root/'api_ledger.json') if (root/'api_ledger.json').exists() else []
    if len(rows)>=cfg['api_requests'] or sum(r.get('charged_output_tokens',0) for r in rows)+cfg['api_max_tokens']>cfg['api_output_tokens']:return False
    failed=0
    for r in reversed(rows):
        if r['status']=='complete':break
        failed+=1
    return failed<cfg['api_failure_limit']


def used_seconds(root):
    ledger=read(root/'evaluation_budget.json') if (root/'evaluation_budget.json').exists() else {'workers':{}}
    return sum(r['charged_seconds'] for r in ledger['workers'].values())


def preserve_data_evidence(root,work,catalog):
    ledger=catalog.get('data_agent_audit',{}).get('ledger')
    if not ledger:return
    folder=Path(ledger).resolve().parent
    if not folder.is_relative_to((Path(work)/'data_agents').resolve()):raise ValueError('Data-agent evidence must be inside the work data_agents directory')
    allowed={'api_ledger.json','summary.json','data_agent_plan.json','ready_catalog.json','data_agent_complete.json','inspect.json','adapter_report.json','gene_mapping.csv'}
    for p in sorted(folder.rglob('*')):
        if p.is_file() and (p.name in allowed or p.name.startswith(('proposal_','blocked_')) and p.suffix=='.json'):
            freeze(root/'data_evidence'/p.relative_to(folder).with_suffix(p.suffix+'.snapshot.json'),{'sha256':file_sha256(p),'content':p.read_text(encoding='utf-8')})


def export_database(pool):
    """Keep failed proposal evidence, not merely valid algorithm snapshots."""
    for path in sorted([*(pool/'evolution').rglob('*.db'),*(pool/'evolution').rglob('*.sqlite')]):
        con=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'attempt_log' in tables:
                con.row_factory=sqlite3.Row
                records=[dict(r) for r in con.execute('SELECT * FROM attempt_log ORDER BY rowid')]
                write_json(pool/'attempt_log_export.json',json.loads(json.dumps(records,default=str)))
        finally:con.close()


def source_memory(pool):
    history=feedback(pool/'events.jsonl')
    for item in history:
        if item['kind']=='invalid_candidate':
            p=pool/'evaluations/search'/item['code_sha256']/'candidate.py'
            if p.exists():item['failed_code_excerpt']=p.read_text(encoding='utf-8')[-6000:]
    write_json(pool/'source_feedback.json',history)


def search_pool(root,pool,cfg,shinka,relay,one_cycle=False):
    if (pool/'STOP.json').exists():return novel(pool)
    for cycle in range(1,cfg['max_cycles_per_pool']+1):
        stamp=pool/'cycles'/f'{cycle:03d}.json'
        if stamp.exists():continue
        count=len(novel(pool))
        if count>=cfg['valid_new_candidates_per_pool'] or not provider_room(root,cfg):break
        ledger=read(root/'evaluation_budget.json')
        if sum(r['charged_seconds'] for r in ledger['workers'].values())+cfg['worker_seconds']>cfg['evaluation_hours']*3600*cfg['source_compute_fraction']:break
        source_memory(pool);stage(root,'Shinka source evolution',pool=pool.name,cycle=cycle,valid_new=count,target=cfg['valid_new_candidates_per_pool'])
        env=worker_env();env.update(ROUND22_GATEWAY_KEY=relay,ROUND23_POOL=str(pool),SHINKA_PRICING_MODE='offline')
        target=1+cycle*cfg['generations_per_cycle']
        log=pool/'logs'/f'cycle_{cycle:03d}.log';log.parent.mkdir(exist_ok=True)
        with log.open('a',encoding='utf-8') as f:
            run=subprocess.run([str(shinka),str(REPO/'scripts/run_shinka_round23.py'),'--root',str(root),'--pool',str(pool),'--generations',str(target)],env=env,stdout=f,stderr=subprocess.STDOUT)
        export_database(pool)
        if run.returncode:raise RuntimeError('Shinka cycle failed; resume preserves paid ledger and evaluations: '+str(log))
        after=len(novel(pool));freeze(stamp,{'cycle':cycle,'valid_new_before':count,'valid_new_after':after,'framework_generation_target':target,'source_only':True})
        append(root/'events.jsonl','cycle_complete',phase='search',pool=pool.name,cycle=cycle,valid_new=after)
        patience=cfg['zero_yield_patience_cycles'];recent=[pool/'cycles'/f'{c:03d}.json' for c in range(max(1,cycle-patience+1),cycle+1)]
        if len(recent)==patience and all(p.exists() and read(p)['valid_new_after']==read(p)['valid_new_before'] for p in recent):
            write_json(pool/'STOP.json',{'reason':f'{patience} source cycles produced no new valid behavior; inspect failure memory before spending more','goal_reached':False});break
        if one_cycle:break
    return novel(pool)


def finalize(root,pools,cfg):
    frozen=read(root/'search_frozen.json');comparisons=[];confirmations={}
    for name,selection in frozen['pools'].items():
        pool=pools[name];task=read(pool/'run_plan.json')['task']
        freeze(pool/'search_frozen.json',selection)
        results=[]
        for record in selection['finalists']:
            stage(root,'frozen source confirmation',pool=name,arm=record['arm'])
            outcome=assess(Path(record['code']).read_text(encoding='utf-8'),pool,'confirmation')
            results.append({**record,'confirmation':outcome})
        contenders=[r for r in results if r['arm']=='shinka' and r['confirmation']['status']=='valid' and r['confirmation']['metrics']['public']['signal_screen_passed']]
        winner=max(contenders,key=lambda r:r['confirmation']['metrics']['combined_score']) if contenders else None
        freeze(pool/'confirmation_frozen.json',{'results':results,'winner':winner,'outer_used':False})
        confirmations[name]={'results':results,'winner':winner}
    freeze(root/'confirmation_frozen.json',confirmations)
    for name,selection in frozen['pools'].items():
        pool=pools[name];plan=read(pool/'run_plan.json');task=plan['task'];arm=plan['arm']
        for r in selection['source_records']:comparisons.append({'phase':'search','pool':name,'task':task,'data_arm':arm,'method_arm':r['arm'],'code_sha256':r['code_sha256'],**r['metrics']['public']})
        for r in confirmations[name]['results']:
            if r['confirmation']['status']=='valid':comparisons.append({'phase':'confirmation','pool':name,'task':task,'data_arm':arm,'method_arm':r['arm'],**r['confirmation']['metrics']['public']})
        winner=confirmations[name]['winner']
        if task!='transfer' or winner is None:
            write_json(pool/'OUTER_NOT_RUN.json',{'reason':'Repeat is supported prediction, or no evolved source-confirmed promotion; no outer score exposed to search'});continue
        for r in confirmations[name]['results']:
            if r['arm']=='shinka' and r['code_sha256']!=winner['code_sha256']:continue
            outcome=assess(Path(r['code']).read_text(encoding='utf-8'),pool,'outer')
            if outcome['status']=='valid':comparisons.append({'phase':'development_outer','pool':name,'task':task,'data_arm':arm,'method_arm':r['arm'],**outcome['metrics']['public']})
    pd.DataFrame(comparisons).to_csv(root/'comparison.csv',index=False)
    ledger=read(root/'evaluation_budget.json') if (root/'evaluation_budget.json').exists() else {'workers':{}}
    cost={name:sum(r['charged_seconds'] for key,r in ledger['workers'].items() if '/evaluations/'+name+'/' in key.replace('\\','/')) for name in ('search','random','confirmation','outer')}
    write_json(root/'cost_report.json',{'charged_worker_seconds_by_phase':cost,'total_charged_worker_seconds':used_seconds(root),'scope':'Worker wall time, not GPU FLOPs or full end-to-end duration. LLM and trusted preprocessing time separate.'})
    summary={'framework':'ShinkaEvolve 0.0.7, persistent per-pool database plus source-only repair memory',
        'valid_new_goal_per_pool':cfg['valid_new_candidates_per_pool'],'pools':{n:{'valid_new_source_candidates':s['valid_count'],'random_valid_comparator_count':s['random_count'],
        'candidate_count_matched':s['valid_count']==s['random_count'],'goal_reached':s['valid_count']>=cfg['valid_new_candidates_per_pool'],'search_stop':s.get('stop_reason','frozen checkpoint'),'confirmed_promotion':confirmations[n]['winner'] is not None} for n,s in frozen['pools'].items()},
        'output_panel':read(root/'campaign.json')['catalog'].get('output_panel',{'policy':'Legacy reference panel'}),
        'new_external_source_bundles':sum(e.get('role')=='source' and bool(e.get('bundle')) for e in read(root/'campaign.json')['catalog']['sources']),
        'test_evaluated':False,'HT29_evaluated':False,'sealed_external_evaluated':False,'raw_dataset_changed':False,
        'claim':'Framework completion is distinct from attaining candidate targets or predictive gain; confirmation targets have historical development exposure.'}
    write_json(root/'summary.json',summary);(root/'INCOMPLETE.json').unlink(missing_ok=True);stage(root,'complete')


def campaign(a):
    work=Path(a.work_dir);root=work/'runs'/a.name;cfg=read(a.config);catalog=read(a.catalog);parent=Path(a.round21_run or work/'runs/round21_signal_01')
    if cfg['tasks']!=['repeat','transfer'] or set(cfg['arms'])!={'core','expanded'}:raise ValueError('This release fixes the two-task/two-data-arm comparison')
    if not 1<=cfg['valid_new_candidates_per_pool']<=100 or not 1<=cfg['max_cycles_per_pool']<=100:raise ValueError('Bounded candidate and cycle counts required')
    for key in ('generations_per_cycle','max_patch_resamples','max_patch_attempts','api_requests','api_output_tokens','api_max_tokens','api_failure_limit','evaluation_hours','worker_seconds','steps','source_finalists_per_pool','zero_yield_patience_cycles'):
        if not isinstance(cfg[key],(int,float)) or cfg[key]<=0:raise ValueError('Positive campaign budget required: '+key)
    if not 0<cfg['source_compute_fraction']<cfg['random_compute_fraction']<1:raise ValueError('Reserve compute for random comparison and confirmation')
    if root.exists() and any(root.iterdir()) and not a.resume:raise ValueError('Use --resume or a new run name')
    root.mkdir(parents=True,exist_ok=True);framework=fingerprint(REPO)
    freeze(root/'campaign.json',{'repo':str(REPO),'framework_sha256':framework,'config':cfg,'catalog':catalog,'parent':str(parent),'parent_sha256':file_sha256(parent/'COMPLETE.json'),
        'python':sys.executable,'training_versions':{n:version(n) for n in ('numpy','torch','pandas','scipy')},'device':a.device,'test_evaluated':False})
    if (root/'COMPLETE.json').exists():verify(root);print('ROUND23 COMPLETE VERIFIED',root);return root
    worker={'steps':cfg['steps'],'seconds':cfg['worker_seconds'],'seed':cfg['seed'],'device':a.device,'ram_bytes':12*1024**3,'evaluation_seconds_max':cfg['evaluation_hours']*3600,
        'source_seconds_max':cfg['evaluation_hours']*3600*cfg['source_compute_fraction'],'random_seconds_max':cfg['evaluation_hours']*3600*cfg['random_compute_fraction'],'root':str(root)}
    try:
        preserve_data_evidence(root,work,catalog)
        stage(root,'compile and audit core/expanded source pools')
        pools=compile_pools(parent,root,cfg['tasks'],cfg['arms'],{'repo':str(REPO),'outer_context':cfg['outer_context'],'seed':cfg['seed'],'worker':worker},sys.executable,framework,catalog)
        if a.mode=='prepare':stage(root,'prepared; no code execution or API');return root
        stage(root,'kernel boundary and strong-baseline contract smoke');preflight(root,worker,sys.executable)
        for name,pool in pools.items():
            freeze_baseline(pool);selected=read(pool/'baseline_selection.json')[read(pool/'run_plan.json')['task']];code=seed(REPO,selected);path=pool/'seed.py'
            if path.exists() and path.read_text(encoding='utf-8')!=code:raise ValueError('Seed changed')
            path.write_text(code,encoding='utf-8');result=assess(code,pool)
            if result['status']!='valid':raise RuntimeError('Strong seed contract failed: '+result.get('error',''))
        if a.mode=='smoke':stage(root,'numeric smoke complete; no paid calls');return root
        if not (root/'search_frozen.json').exists():
            key=os.environ.get('DEEPSEEK_API_KEY','').strip()
            if not key:raise RuntimeError('Set DEEPSEEK_API_KEY in server orchestrator environment')
            shinka=Path(a.shinka_python or work/'envs/shinka/bin/python')
            if not shinka.is_file():raise ValueError('Install pinned Shinka with scripts/install_round22_autodl.sh')
            relay=secrets.token_hex(24)
            with Gateway(root/'api_ledger.json',key,cfg['model'],relay,cfg['api_requests'],cfg['api_output_tokens'],cfg['port'],max_tokens=cfg['api_max_tokens'],reasoning_effort=cfg['reasoning_effort'],failure_limit=cfg['api_failure_limit']) as gateway:
                if not (root/'provider_verified.json').exists():
                    status,response=gateway.forward({'model':cfg['model'],'messages':[{'role':'user','content':'Return only Python code: def round22_probe(): followed by an indented return 22. No explanation.'}],'max_tokens':cfg['api_max_tokens']})
                    if status!=200:raise RuntimeError('Code-output provider smoke failed; inspect api_ledger.json')
                    check_code_probe(response['choices'][0]['message']['content']);write_json(root/'provider_verified.json',{'final_code_parsed':True})
                # Round-robin scheduling prevents the first task consuming all API budget.
                for _ in range(cfg['max_cycles_per_pool']):
                    if not provider_room(root,cfg) or used_seconds(root)+cfg['worker_seconds']>worker['source_seconds_max']:break
                    for name,pool in pools.items():search_pool(root,pool,cfg,shinka,relay,one_cycle=True)
                    if all((p/'STOP.json').exists() or len(novel(p))>=cfg['valid_new_candidates_per_pool'] for p in pools.values()):break
            source_seconds_at_stop=used_seconds(root)
            provider_room_at_stop=provider_room(root,cfg)
            selections={}
            # Match random search to the number of valid distinct evolved behaviors;
            # retain attempt/time caps and disclose the narrower random grammar.
            for name,pool in pools.items():
                shinka_records=novel(pool);chosen=[];random_records=[];seen=set()
                stage(root,'valid-count random comparator',pool=name,target=len(shinka_records))
                for i in range(max(1,len(shinka_records))*4):
                    if len(random_records)>=len(shinka_records):break
                    if used_seconds(root)+cfg['worker_seconds']>worker['random_seconds_max']:break
                    r=assess(random_method(REPO,cfg['seed']+i),pool,scope='random')
                    if r['status']=='valid' and r['behavior_sha256'] not in seen:
                        seen.add(r['behavior_sha256']);random_records.append({**r,'arm':'random'})
                    write_json(root/'random_progress.json',{'pool':name,'trial':i+1,'valid':len(random_records),'target':len(shinka_records)})
                seed_record=assess((pool/'seed.py').read_text(encoding='utf-8'),pool)
                all_source=[{**r,'arm':'shinka'} for r in shinka_records]+random_records+[{**seed_record,'arm':'fixed'}]
                candidates=[{**r,'arm':'shinka'} for r in shinka_records if r['metrics']['public']['signal_screen_passed']][:cfg['source_finalists_per_pool']]
                chosen.extend(candidates)
                if random_records:chosen.append(max(random_records,key=lambda r:r['metrics']['combined_score']))
                chosen.append({**seed_record,'arm':'fixed'})
                reason='target_met' if len(shinka_records)>=cfg['valid_new_candidates_per_pool'] else 'zero_yield' if (pool/'STOP.json').exists() else 'provider_budget_or_circuit' if not provider_room_at_stop else 'source_compute_cap' if source_seconds_at_stop+cfg['worker_seconds']>worker['source_seconds_max'] else 'cycle_limit'
                selections[name]={'finalists':chosen,'source_records':all_source,'valid_count':len(shinka_records),'random_count':len(random_records),'stop_reason':reason,
                    'scope':'Valid behavior-count matched where attainable; random grammar narrower than arbitrary code; source-only selection'}
            freeze(root/'search_frozen.json',{'pools':selections,'confirmation_used':False,'outer_used':False})
        finalize(root,pools,cfg);return root
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error_type':type(exc).__name__,'error':str(exc),'test_evaluated':False});raise
    finally:
        for pool in (root/'pools').glob('*'):
            if (pool/'events.jsonl').exists():write_json(pool/'journal_export.json',events(pool/'events.jsonl'))
            export_database(pool)
        if (root/'events.jsonl').exists():write_json(root/'journal_export.json',events(root/'events.jsonl'))
        if (root/'stage.json').exists() and read(root/'stage.json')['stage']=='complete' and not (root/'INCOMPLETE.json').exists():seal(root,test_evaluated=False)
        archive=package(root);target=root/'round23_review_light.tar.gz';archive.replace(target)
        print('REVIEW PACKAGE:',target,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','inventory','expand-catalog','public-data','data-agent','adapt','status']);p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round23_evolve_01')
    p.add_argument('--config',default=str(REPO/'configs/round23_campaign.json'));p.add_argument('--catalog',default=str(REPO/'configs/round23_public_sources.json'));p.add_argument('--round21-run');p.add_argument('--shinka-python');p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--mode',choices=['prepare','smoke','all'],default='all');p.add_argument('--resume',action='store_true');p.add_argument('--download',action='store_true');p.add_argument('--h5ad');p.add_argument('--adapter-config');p.add_argument('--panel');p.add_argument('--dest');a=p.parse_args()
    work=Path(a.work_dir)
    if a.command=='inventory':
        target=work/'catalog/round23_inventory.json';write_json(target,inventory(work));print(target);return
    if a.command=='expand-catalog':
        if not a.dest:raise ValueError('--dest is required')
        parent=Path(a.round21_run or work/'runs/round21_signal_01');canonical=read(parent/'lineage.json')['data_dir']
        freeze(a.dest,expanded_catalog(work,read(a.catalog),canonical));print('ALL COMPATIBLE SOURCES:',a.dest);return
    if a.command=='public-data':
        for e in read(a.catalog)['sources']:
            dest=work/'external'/e['id'];path=dest/e['filename']
            if a.download:path=fetch(e,dest)
            if path.exists():write_json(dest/'inspect.json',inspect_h5ad(path));print(dest/'inspect.json')
            else:print('NOT_DOWNLOADED',e['id'])
        return
    if a.command=='adapt':
        from vcell.evolution.adapter import adapt
        import numpy as np
        if not all((a.h5ad,a.adapter_config,a.panel,a.dest)):raise ValueError('adapt requires --h5ad --adapter-config --panel --dest')
        with np.load(a.panel,allow_pickle=False) as f:genes=f['genes']
        entry=adapt(a.h5ad,read(a.adapter_config),genes,Path(a.dest))
        target=Path(a.dest)/'ready_catalog.json';freeze(target,{'sources':[entry]});print('READY CATALOG:',target);return
    if a.command=='data-agent':
        from vcell.evolution.data_agent import prepare_sources
        key=os.environ.get('DEEPSEEK_API_KEY','').strip()
        if not key:raise ValueError('Set DEEPSEEK_API_KEY for the metadata-only data agent')
        parent=Path(a.round21_run or work/'runs/round21_signal_01');lineage=read(parent/'lineage.json')
        panel=a.panel or str(Path(lineage['data_dir'])/'dataset.npz')
        target=prepare_sources(work,read(a.catalog),panel,Path(a.dest or work/'data_agents/round23_data_01'),key)
        print('READY CATALOG:',target);print(json.dumps(read(target.parent/'summary.json'),indent=2));return
    if a.command=='status':
        root=work/'runs'/a.name
        if not root.exists():
            print('Campaign not created yet; check data-agent acquisition logs before model search')
            progress=work/'data_agents/round23_data_01/progress.json'
            if progress.exists():print('Data agent:',json.dumps(read(progress),indent=2))
        for name in ('stage.json','summary.json','INCOMPLETE.json','worker_progress.json','random_progress.json'):
            if (root/name).exists():
                info=read(root/name)
                if name=='summary.json' and 'genes' in info.get('output_panel',{}):info['output_panel']['genes']='<ordered IDs saved in summary.json>'
                print(name,json.dumps(info,indent=2))
        for pool in sorted((root/'pools').glob('*')):
            if (pool/'seed.py').exists():
                results=[read(p) for p in (pool/'evaluations/search').glob('*/assessment.json')]
                print(pool.name,'distinct valid new source behaviors:',len(novel(pool)),'assessments:',dict(Counter(r['status'] for r in results)),
                    'failure categories:',dict(Counter(r.get('category') for r in results if r['status']!='valid')),'cycles:',len(list((pool/'cycles').glob('*.json'))))
        if (root/'api_ledger.json').exists():
            rows=read(root/'api_ledger.json');print('API requests',len(rows),'charged output tokens',sum(r.get('charged_output_tokens',0) for r in rows))
        if (root/'COMPLETE.json').exists():print('ROUND23 COMPLETE',root/'round23_review_light.tar.gz')
        return
    with run_lock(work/'runs'/('.'+a.name+'.lock')):
        if a.device=='cuda':
            with run_lock(work/'runs'/'.evolution_gpu.lock'):campaign(a)
        else:campaign(a)


if __name__=='__main__':main()
