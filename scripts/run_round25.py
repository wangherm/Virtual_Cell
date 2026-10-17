"""Round25: matched native Shinka campaigns with bounded repair and source-only strategy."""
import argparse
from collections import Counter
import copy
from dataclasses import asdict
import difflib
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import subprocess
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import numpy as np
import pandas as pd
from vcell.clean_background import run_lock
from vcell.evolution.journal import append,events
from vcell.evolution.self_correction import (ResearchAction,critic_prompt,distinct,evidence,frontier,
    parse_json,prediction_outcome,source_records,quota_failure)
from vcell.model_agent.artifacts import read,seal,verify
from vcell.round11 import freeze
from vcell.round22.evaluator import worker_env
from vcell.round24.stats import cluster_gain
from vcell.round25.prepare import inherited,methods,freeze_baselines,replay_parent
from vcell.round25.provider import ResearchGateway,critic_request
from vcell.round25.runtime import assess,charged,fingerprint
from vcell.utils import file_sha256,write_json
from run_round22 import preflight


def stage(root,name,**fields):
    write_json(root/'stage.json',{'stage':name,'test_evaluated':False,**fields})
    print('ROUND25 STAGE:',name,fields,flush=True)


def event(root,ticket,kind,**fields):
    path=Path(root)/'research_decisions.jsonl'
    if not any(r.get('ticket')==ticket for r in events(path)):
        append(path,kind,phase='search',ticket=ticket,**fields)


def cfg_check(cfg):
    if cfg['arms']!=['shinka','repair','meta'] or set(cfg['tasks'])!={'repeat','transfer'}:
        raise ValueError('This matched release requires three arms and two separate tasks')
    if not 1<=cfg['api_requests_per_arm']<=46 or not 1<=cfg['initial_valid_per_arm']<=cfg['valid_target_per_arm']<=20:
        raise ValueError('Freeze at most 138 total API requests and 60 distinct behaviors')
    if cfg['max_repairs_per_candidate']!=2 or cfg['critic_every_valid']<1 or cfg['stagnation_turns']<2:
        raise ValueError('Bounded repair/checkpoint policy required')
    if not .4<=cfg['repeat_share']<=.7 or cfg['maximum_task_search_share']!=.7:
        raise ValueError('Source allocation must remain within the declared bounds')
    if cfg['baseline_hours']+3*cfg['arm_hours']>14 or min(cfg['baseline_hours'],cfg['arm_hours'])<=0:
        raise ValueError('Positive aggregate evaluation cap no larger than 14 hours')
    if not 0<cfg['search_fraction']<1 or cfg['worker_seconds']<=0 or not 1<=cfg['steps']<=500:
        raise ValueError('Invalid uniform evaluation budget')
    if cfg['independent_confirmation'] or cfg['confirmation_scope']!='historically_viewed_development':
        raise ValueError('Inherited confirmation cannot be relabelled independent')


def budget(root,cfg,hours,device):
    return {'root':str(root),'seed':cfg['seed'],'device':device,'steps':cfg['steps'],
        'seconds':cfg['worker_seconds'],'ram_bytes':12*1024**3,'evaluation_seconds_max':hours*3600}


def pool_plan(pool,root,task,arm,cfg,folds,parent,device,baseline=False):
    pool.mkdir(parents=True,exist_ok=True)
    b=budget(root,cfg,cfg['baseline_hours'] if baseline else cfg['arm_hours'],device)
    freeze(pool/'run_plan.json',{'repo':str(REPO),'python':sys.executable,'framework_sha256':fingerprint(REPO),
        'task':task,'arm':arm,'config':cfg,'budget':b,'parent':str(parent),
        'parent_sha256':file_sha256(parent/'COMPLETE.json'),'test_evaluated':False})
    freeze(pool/'folds.json',{phase:folds[task][phase] for phase in ('search','confirmation')})


def prepare(root,parent,cfg,device):
    stage(root,'verify full historical source lineage; no new independent confirmation')
    marker=root/'lineage.json'
    if marker.exists():
        old=read(marker)
        if old['parent_sha256']!=file_sha256(parent/'COMPLETE.json'):raise ValueError('Parent changed')
        folds=read(parent/'folds.json')['tasks']
    else:
        folds=inherited(parent,cfg)
        freeze(marker,{'parent':str(parent),'parent_sha256':file_sha256(parent/'COMPLETE.json'),
            'confirmation_scope':cfg['confirmation_scope'],'independent_confirmation':False,
            'HT29_evaluated':False,'outer_evaluated':False,'test_evaluated':False,
            'memory_policy':'Only new campaign search assessments enter mutation/critic memory. Historical confirmation is development evidence.'})
    definitions=methods(parent)
    for task in cfg['tasks']:
        p=root/'baselines'/task
        pool_plan(p,root/'baseline_budget',task,'fixed',cfg,folds,parent,device,True)
        chosen=freeze_baselines(p,definitions)
        for arm in cfg['arms']:
            pool=root/'arms'/arm/task;pool_plan(pool,root/'arms'/arm,task,arm,cfg,folds,parent,device)
            freeze(pool/'baseline_selection.json',chosen)
            body=(p/'seed.py').read_text(encoding='utf-8');seed=pool/'seed.py'
            if seed.exists() and seed.read_text(encoding='utf-8')!=body:raise ValueError('Seed changed')
            seed.write_text(body,encoding='utf-8')
    replay_parent(root,parent)
    return folds


def novel(pool):
    body=(pool/'seed.py').read_text(encoding='utf-8');sha=hashlib.sha256(body.encode()).hexdigest()
    rows=source_records(pool)
    seed=next((r for r in rows if r['module_sha256']==sha),None)
    if seed is None:seed=assess(pool,body);rows=source_records(pool)
    if seed['status']!='valid':raise RuntimeError('Source seed failed: '+seed.get('error',''))
    return distinct(rows,excluded=[seed['behavior_sha256']])


def next_generation(pool):
    path=pool/'evolution/programs.sqlite';highest=0
    if path.exists():
        con=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'programs' in tables:highest=max(highest,con.execute('SELECT COALESCE(MAX(generation),0) FROM programs').fetchone()[0])
            if 'attempt_log' in tables:
                cols={r[1] for r in con.execute('PRAGMA table_info(attempt_log)')}
                if 'generation' in cols:highest=max(highest,con.execute('SELECT COALESCE(MAX(generation),0) FROM attempt_log').fetchone()[0])
        finally:con.close()
    # Native num_generations includes generation zero. Reuse this absolute target on interruption.
    return int(highest)+2


def default_action(rows,reason):
    valid=[r for r in rows if r['status']=='valid']
    parent=max(valid,key=lambda r:r['metrics']['combined_score'])
    return ResearchAction('explore_new_mechanism',reason,'Separate common response from target-specific signal using a different mechanism',
        'new response mechanism',parent['module_sha256'],None,'high','none',
        {'metric':'mse','direction':'decrease','minimum_change':.00001},'Change measured archive parent',
        reason,[parent['module_sha256']],.6)


def choose_action(pool,arm,state,task,gateway,cfg,ticket):
    rows=source_records(pool);pending=state['pending_repairs'].get(task)
    if arm!='shinka' and pending and state['repair_counts'].get(pending['origin'],0)<cfg['max_repairs_per_candidate']:
        origin=pending['origin'];state['repair_counts'][origin]=state['repair_counts'].get(origin,0)+1
        action=ResearchAction('repair_candidate','measured runtime failure','Correct the reported implementation failure without changing the scientific objective',
            'local code repair',pending['parent'],None,'high','none',{'metric':'valid','direction':'increase','minimum_change':1},
            'Stop repairing this candidate after two attempts','Use bounded traceback and the real two-row contract',[pending['parent']],state['repeat_share'])
        return action,origin,None
    count=len(novel(pool));due=(count-state['critic_at'][task]>=cfg['critic_every_valid'] or state['empty'][task]>=cfg['stagnation_turns'])
    if arm=='meta' and due:
        info=evidence(pool);decision=pool/'decisions'/f'{ticket}.json'
        if decision.exists(): action=ResearchAction.parse(read(decision)['action'],info)
        else:
            gateway.scope=f'{task}/{ticket}/critic';gateway.role='critic'
            payload=''
            try:
                payload=critic_request(gateway,critic_prompt(info),cfg['critic_max_tokens'])
                action=ResearchAction.parse(parse_json(payload),info)
            except (RuntimeError,ValueError,TypeError,KeyError) as exc:
                action=default_action(rows,'Critic schema rejected; bounded source-based exploration fallback')
                write_json(pool/'decisions'/f'{ticket}_rejected.json',{'error':str(exc),'response_sha256':hashlib.sha256(payload.encode()).hexdigest()})
            freeze(decision,{'action':asdict(action),'source_evidence':info})
        state['critic_at'][task]=count;state['repeat_share']=action.repeat_share
        # A bounded runtime repair remains at most two attempts, including critic requests.
        if action.action=='repair_candidate':
            r=next(r for r in rows if r['module_sha256']==action.parent_code_sha256)
            origin=state['repair_origins'].get(action.parent_code_sha256,action.parent_code_sha256)
            if r['status']=='valid' or state['repair_counts'].get(origin,0)>=2:
                action=default_action(rows,'Invalid or exhausted critic repair request; switch mechanism')
            else:
                state['repair_counts'][origin]=state['repair_counts'].get(origin,0)+1
                return action,origin,info
        return action,None,info
    return None,None,None


def run_probe(pool,action):
    if action.requested_source_diagnostic=='none':return None
    info=evidence(pool);selected=[]
    for c in info['candidates']:
        if 'diagnosis' not in c:continue
        d=c['diagnosis'];key={'reference_coverage':'reference_coverage','context_losses':'by_context',
            'target_losses':'worst_targets','strong_response':None}[action.requested_source_diagnostic]
        selected.append({'module_sha256':c['module_sha256'],'result':d[key] if key else c['metrics']})
    return {'probe':action.requested_source_diagnostic,'scope':'source_search_only','measured':selected}


def search_arm(root,arm,cfg,shinka):
    armroot=root/'arms'/arm;done=armroot/'SEARCH_COMPLETE.json'
    if done.exists():return read(done)
    if any(read(p).get('state')=='reserved' for p in (armroot/'provider_responses').glob('*.json')):
        raise RuntimeError('Unresolved provider ticket; stopped before issuing any further paid request')
    pools={task:armroot/task for task in cfg['tasks']}
    for pool in pools.values():novel(pool)
    state={'pending_repairs':{},'repair_counts':{},'repair_origins':{},'critic_at':{t:0 for t in pools},'empty':{t:0 for t in pools},
           'banned':{t:[] for t in pools},'repeat_share':cfg['repeat_share'],'stopped':[],'last_results':{},'extended':False}
    relay=secrets.token_hex(24);key=os.environ.get('DEEPSEEK_API_KEY','').strip()
    if not key:raise RuntimeError('Export DEEPSEEK_API_KEY in this terminal before starting screen')
    with ResearchGateway(armroot/'api_ledger.json',key,cfg['model'],relay,cfg['api_requests_per_arm'],
        cfg['api_output_tokens_per_arm'],cfg['port'],max_tokens=cfg['api_max_tokens'],reasoning_effort=cfg['reasoning_effort'],failure_limit=cfg['api_failure_limit']) as gateway:
        reason='turn ceiling'
        for i in range(cfg['max_turns_per_arm']):
            folder=armroot/'turns'/f'{i:03d}';marker=folder/'done.json'
            if marker.exists():state=read(marker)['state'];continue
            continuing=(folder/'plan.json').exists()
            allnovel={t:novel(p) for t,p in pools.items()};n=sum(map(len,allnovel.values()))
            if not continuing and n>=cfg['valid_target_per_arm']:reason='distinct behavior target reached';break
            if not continuing and n>=cfg['initial_valid_per_arm'] and not state['extended']:
                can_extend=any(r['metrics']['public']['gain_over_fixed']>=cfg['extension_minimum_source_gain'] or
                    (r['metrics']['public'].get('strong_mse_ratio') is not None and r['metrics']['public']['strong_mse_ratio']<=.99
                     and (r['metrics']['public'].get('specificity_gap') or 0)>(r['metrics']['public'].get('baseline_specificity_gap') or 0)+.005)
                    for rs in allnovel.values() for r in rs)
                event(root,arm+'/extension','tranche_decision',arm=arm,extend=can_extend,distinct_valid=n)
                if not can_extend:reason='first tranche has no qualifying source gain';break
                state['extended']=True
            if not continuing and (len(gateway.records)>=cfg['api_requests_per_arm'] or sum(r.get('charged_output_tokens',0) for r in gateway.records)+cfg['api_max_tokens']>cfg['api_output_tokens_per_arm']):
                reason='provider request/token ceiling';break
            if not continuing and charged(armroot)+cfg['worker_seconds']>cfg['arm_hours']*3600*cfg['search_fraction']:
                reason='source evaluation budget reserved for confirmation';break
            legal=[t for t in pools if t not in state['stopped'] and charged(armroot,pools[t])+cfg['worker_seconds']<=cfg['arm_hours']*3600*cfg['search_fraction']*.7]
            if not continuing and not legal:reason='no productive task/quota remains';break
            shares={'repeat':state['repeat_share'],'transfer':1-state['repeat_share']}
            task=read(folder/'plan.json')['task'] if continuing else min(legal,key=lambda t:charged(armroot,pools[t])/shares[t])
            pool=pools[task];ticket=f'turn_{i:03d}'
            planned=folder/'plan.json'
            if planned.exists():
                plan=read(planned);task=plan['task'];pool=pools[task];state=plan['state']
                action=ResearchAction(**plan['action']) if plan['action'] else None;origin=plan['repair_origin']
            else:
                action,origin,_=choose_action(pool,arm,state,task,gateway,cfg,ticket)
                if action and action.action=='stop_unproductive_branch':
                    state['banned'][task].append(action.parent_code_sha256)
                    remaining=[r for r in source_records(pool) if r['status']=='valid' and r['module_sha256'] not in state['banned'][task]]
                    if not remaining:state['stopped'].append(task)
                probe=run_probe(pool,action) if action else None
                plan={'task':task,'action':asdict(action) if action else None,'repair_origin':origin,
                    'state':copy.deepcopy(state),'probe':probe,'generation_target':next_generation(pool),
                    'prediction_reference_sha256':action.parent_code_sha256 if action and action.parent_code_sha256 else hashlib.sha256((pool/'seed.py').read_text(encoding='utf-8').encode()).hexdigest(),
                    'before_ids':[r['module_sha256'] for r in source_records(pool)],'before_novel':len(allnovel[task])}
                freeze(planned,plan)
            write_json(pool/'active_action.json',{'action':plan['action'],'probe_result':plan.get('probe'),'banned':state['banned'][task]})
            snapshot=folder/'source_feedback.json'
            if not snapshot.exists():freeze(snapshot,evidence(pool))
            write_json(pool/'source_feedback.json',read(snapshot))
            event(root,arm+'/'+ticket,'research_action',arm=arm,task=task,action=plan['action'],probe=plan.get('probe'))
            if task in state['stopped']:
                write_json(marker,{'state':state,'stopped_without_new_request':True});continue
            role='repair' if action and action.action=='repair_candidate' else 'mutation'
            gateway.scope=f'{task}/{ticket}/{role}';gateway.role=role
            stage(root,'native Shinka generation',arm=arm,task=task,turn=i+1,role=role,distinct_valid=n,
                  api_used=len(gateway.records),api_max=cfg['api_requests_per_arm'])
            env=worker_env();env.update(ROUND25_POOL=str(pool),ROUND22_GATEWAY_KEY=relay,
                SHINKA_PRICING_MODE='offline',PYTHONIOENCODING='utf-8')
            log=folder/'shinka.log';folder.mkdir(parents=True,exist_ok=True)
            with log.open('a',encoding='utf-8') as f:
                result=subprocess.run([str(shinka),str(REPO/'scripts/run_shinka_round25.py'),'--pool',str(pool),
                    '--generations',str(plan['generation_target'])],env=env,stdout=f,stderr=subprocess.STDOUT)
            rows=source_records(pool);new=[r for r in rows if r['module_sha256'] not in plan['before_ids']]
            after=len(novel(pool));state['empty'][task]=state['empty'][task]+1 if after==plan['before_novel'] else 0
            outcome=None
            if new:
                latest=max(new,key=lambda r:r['finished_at']);state['last_results'][task]=latest['module_sha256']
                if latest['status']=='invalid' and not quota_failure(latest):
                    state['pending_repairs'][task]={'parent':latest['module_sha256'],'origin':origin or latest['module_sha256']}
                    state['repair_origins'][latest['module_sha256']]=origin or latest['module_sha256']
                else:state['pending_repairs'].pop(task,None)
                if action:
                    before=next(r for r in rows if r['module_sha256']==plan['prediction_reference_sha256'])
                    outcome=prediction_outcome(action,before,latest)
                    outcome['reference_sha256']=before['module_sha256']
                    outcome['reference_kind']='declared_parent' if action.parent_code_sha256 else 'predeclared_fixed_seed'
                    diff=''.join(difflib.unified_diff(Path(before['module']).read_text().splitlines(True),Path(latest['module']).read_text().splitlines(True),fromfile='parent',tofile='child'))
                    (folder/'code.diff').write_text(diff,encoding='utf-8')
                if latest['status']=='invalid' and quota_failure(latest):state['stopped'].append(task)
            if state['empty'][task]>=cfg['stagnation_turns'] and (arm!='meta' or action is not None):
                state['stopped'].append(task)
            event(root,arm+'/'+ticket+'/outcome','measured_action_outcome',arm=arm,task=task,role=role,
                repair_origin=origin,repair_attempt=state['repair_counts'].get(origin,0) if origin else None,
                children=[{'module_sha256':r['module_sha256'],'status':r['status']} for r in new],prediction=outcome,
                distinct_before=plan['before_novel'],distinct_after=after,runner_exit=result.returncode)
            write_json(marker,{'state':state,'runner_exit':result.returncode,'prediction':outcome})
            if any(read(p).get('state')=='reserved' for p in (armroot/'provider_responses').glob('*.json')):
                raise RuntimeError('Unresolved provider ticket after interruption; stopped to prevent duplicate billing')
            # A charged failed call is recorded; repeated failures trip the shared provider circuit.
            if result.returncode and not new:
                if len(gateway.records)>=cfg['api_failure_limit'] and all(r['status']!='complete' for r in gateway.records[-cfg['api_failure_limit']:]):
                    reason='provider circuit opened';break
        value={'arm':arm,'reason':reason,'distinct_valid':{t:len(novel(p)) for t,p in pools.items()},
               'api_requests':len(gateway.records),'state':state,'test_evaluated':False}
        freeze(done,value);return value


def freeze_finalists(pool):
    path=pool/'search_frozen.json'
    if path.exists():return read(path)
    rows=novel(pool);task=read(pool/'run_plan.json')['task'];selected=[]
    for tier,key in [('overall','overall_error_candidate'),('response','target_response_candidate')]:
        eligible=[r for r in rows if r['metrics']['public'].get(key) and (task!='transfer' or r['metrics']['public'].get('transfer_candidate'))]
        if eligible:
            r=min(eligible,key=lambda r:r['metrics']['public']['mse'])
            if not any(s['module_sha256']==r['module_sha256'] for s in selected):selected.append({'tier':tier,**r})
    if not selected and rows:
        selected=[{'tier':'diagnostic_only_no_promotion',**min(rows,key=lambda r:r['metrics']['public']['mse'])}]
    value={'finalists':selected[:2],'source_only':True,'distinct_valid':len(rows),'task':task}
    freeze(path,value);return value


def report(root,cfg,confirm=True):
    comparison=[];trace=[];source=[];later=[];sensitivity=[];efficiency=[];library=[]
    for arm in cfg['arms']:
        armroot=root/'arms'/arm
        calls=read(armroot/'api_ledger.json') if (armroot/'api_ledger.json').exists() else []
        valid_count=0
        for task in cfg['tasks']:
            pool=armroot/task
            if not (pool/'baseline_selection.json').exists():continue
            selected=read(pool/'search_frozen.json') if (pool/'search_frozen.json').exists() else {'finalists':[]}
            base=read(pool/'baseline_selection.json')
            base_metrics=copy.deepcopy(base['confirmation_record']['metrics']['public'])
            base_metrics.update(baseline_mse=base_metrics['mse'],gain_over_fixed=0.,baseline_strong_mse=base_metrics['strong_mse'],
                baseline_specificity_gap=base_metrics['specificity_gap'],strong_mse_ratio=1. if base_metrics['strong_mse'] is not None else None,
                overall_error_candidate=False,target_response_candidate=False,transfer_candidate=False)
            comparison.append({'arm':arm,'task':task,'method':'fixed/'+base['method'],'phase':'development_confirmation',**base_metrics})
            rows=source_records(pool);distinct_rows=novel(pool);valid_count+=len(distinct_rows)
            for r in rows:
                trace.append({'arm':arm,'task':task,'module_sha256':r['module_sha256'],'status':r['status'],
                    'category':r.get('category'),'code_lines':r['code_lines'],'worker_seconds':r['worker_seconds'],
                    'peak_vram_bytes':r['peak_vram_bytes'],**r['metrics']['public']})
                if r['status']=='valid':source.append(pd.read_csv(Path(r['directory'])/'scores.csv').assign(arm=arm,task=task,module_sha256=r['module_sha256']))
            write_json(pool/'source_frontier.json',[{'module_sha256':r['module_sha256'],'metrics':r['metrics']['public']} for r in frontier(rows)])
            for r in frontier(rows):
                library.append({'arm':arm,'task':task,'phase':'search','module':r['module'],'module_sha256':r['module_sha256'],
                    'source_metrics':r['metrics']['public'],'source_fold_inputs':[f['input_sha256'] for f in read(pool/'folds.json')['search']],
                    'eligibility':'Source-tested code only. A later campaign must audit its new folds before importing this method; confirmation is not reusable search memory.'})
            if confirm:
                for r in selected['finalists']:
                    body=Path(r['module']).read_text(encoding='utf-8')
                    if hashlib.sha256(body.encode()).hexdigest()!=r['module_sha256']:raise ValueError('Frozen finalist changed')
                    stage(root,'frozen finalist development retest',arm=arm,task=task,module_sha256=r['module_sha256'])
                    result=assess(pool,body,phase='confirmation')
                    comparison.append({'arm':arm,'task':task,'method':r['module_sha256'],'tier':r['tier'],'phase':'development_confirmation','status':result['status'],**result['metrics']['public']})
                    if result['status']=='valid':
                        folder=Path(result['directory']);frame=pd.read_csv(folder/'scores.csv');baseline=pd.read_csv(folder/'baseline.csv')
                        later.append(frame.assign(arm=arm,task=task,module_sha256=r['module_sha256']))
                        sensitivity.append({'arm':arm,'task':task,'module_sha256':r['module_sha256'],**cluster_gain(frame,baseline)})
        seconds=charged(armroot)
        efficiency.append({'arm':arm,'provider_requests':len(calls),'roles':dict(Counter(r.get('role') for r in calls)),
            'charged_output_tokens':sum(r.get('charged_output_tokens',0) for r in calls),'provider_latency_seconds':sum(r.get('latency_seconds',0) for r in calls),
            'distinct_valid':valid_count,'valid_per_api':valid_count/len(calls) if calls else None,
            'worker_seconds':seconds,'valid_per_worker_hour':valid_count/max(seconds/3600,1e-12),
            'currency_cost':None,'currency_note':'Usage measured; no unverified provider price assumed'})
    pd.DataFrame(comparison).to_csv(root/'comparison.csv',index=False);pd.DataFrame(trace).to_csv(root/'evolution_trace.csv',index=False)
    for name,frames in [('source_scores.csv',source),('confirmation_scores.csv',later)]:
        (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=['arm','task','module_sha256','row_id','context','target','mse'])).to_csv(root/name,index=False)
    outcomes=[r for r in events(root/'research_decisions.jsonl') if r['kind']=='measured_action_outcome']
    repairs=[{'arm':r['arm'],'task':r['task'],'ticket':r['ticket'],'origin':r.get('repair_origin'),
        'attempt':r.get('repair_attempt'),'successful':any(c['status']=='valid' for c in r.get('children',[]))} for r in outcomes if r['role']=='repair']
    pd.DataFrame(repairs,columns=['arm','task','ticket','origin','attempt','successful']).to_csv(root/'repair_audit.csv',index=False)
    write_json(root/'efficiency.json',efficiency);write_json(root/'paired_sensitivity.json',sensitivity)
    write_json(root/'source_memory_library.json',{'scope':'source_search_only','methods':library,
        'automatic_cross_campaign_import':False,'confirmation_metrics_included':False})
    write_json(root/'summary.json',{'scope':'three-arm, two-task SOURCE research and historical development retest',
        'efficiency':efficiency,'repair_attempts':len(repairs),'successful_repairs':sum(r['successful'] for r in repairs),
        'hypothesis_outcomes':dict(Counter((r.get('prediction') or {}).get('status','unmeasured') for r in outcomes)),
        'independent_confirmation':False,'test_evaluated':False,'outer_evaluated':False,'HT29_evaluated':False,
        'budget_scope':{'provider_requests_max':len(cfg['arms'])*cfg['api_requests_per_arm'],
            'aggregate_worker_hours_max':cfg['baseline_hours']+len(cfg['arms'])*cfg['arm_hours'],
            'all_roles_charged':True,'cpu_gpu_wall_time_not_an_elapsed_time_estimate':True},
        'interpretation':'One seed is a comparative pilot. Old confirmation is development evidence. Algorithm labels are hypotheses; behavior diversity is quantized at 1e-5.'})
    lines=['# Round25 research campaign','',
        'Historical confirmation is development retest, not a fresh independent test. HT29/outer tests were not opened.',
        'One seed is a comparative pilot. Completion does not imply predictive or framework superiority.','',
        '| Arm | Requests | Novel behaviors | Repairs attempted | Worker hours |',
        '| --- | ---: | ---: | ---: | ---: |']
    for e in efficiency:
        lines.append(f"| {e['arm']} | {e['provider_requests']} | {e['distinct_valid']} | {sum(r['arm']==e['arm'] for r in repairs)} | {e['worker_seconds']/3600:.3f} |")
    lines+=['','| Arm | Task | Method | Development MSE | Gain over matched fixed |',
            '| --- | --- | --- | ---: | ---: |']
    for r in comparison:
        mse=r.get('mse');gain=r.get('gain_over_fixed')
        lines.append(f"| {r['arm']} | {r['task']} | {r['method']} | {mse if mse is not None else 'undefined'} | {gain if gain is not None else 'undefined'} |")
    lines+=['','See repair_audit.csv, evolution_trace.csv, paired_sensitivity.json and research_decisions.jsonl for measured failures and hypothesis outcomes.',
            'Source recipe exports require an explicit fold-scope audit before a later campaign imports them.']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def export_native_archives(root):
    for path in (root/'arms').glob('*/*/evolution/programs.sqlite'):
        con=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
        try:
            tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'programs' in tables:
                rows=[]
                for row in con.execute('SELECT id,code,parent_id,generation,correct,combined_score,island_idx,complexity FROM programs ORDER BY generation,timestamp'):
                    r=dict(row);r['module_sha256']=hashlib.sha256(r.pop('code').encode()).hexdigest();rows.append(r)
                write_json(path.parent/'archive_export.json',rows)
            if 'attempt_log' in tables:
                write_json(path.parent/'attempt_log_export.json',[dict(r) for r in con.execute('SELECT id,generation,stage,status,details,created_at FROM attempt_log ORDER BY id')])
        finally:con.close()


def package(root):
    import io,tarfile
    from collections import deque
    path=root/'round25_review_light.tar.gz';entries={}
    with tarfile.open(path,'w:gz') as tar:
        for p in sorted(root.rglob('*')):
            rel=p.relative_to(root)
            if not p.is_file() or p.suffix not in ('.json','.jsonl','.csv','.md','.py','.log','.diff'):continue
            if set(rel.parts)&{'private','inputs','workers','contract','contract2','preflight','input_cache','provider_responses'}:continue
            payload=p.read_bytes()
            if p.suffix=='.log':
                with p.open(encoding='utf-8',errors='replace') as f:payload=''.join(deque(f,maxlen=60)).encode()
            info=tarfile.TarInfo(rel.as_posix());info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
            entries[rel.as_posix()]={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
        payload=json.dumps({'files':entries,'scope':'Source research decisions, generated modules, scores, audits and log tails; no matrices, weights or secrets'}).encode()
        info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
    print('REVIEW PACKAGE:',path,flush=True);return path


def launch(a):
    work=Path(a.work_dir);root=work/'runs'/a.name;parent=Path(a.round24_run or work/'runs/round24_mechanism_01');cfg=read(a.config);cfg_check(cfg)
    if root.exists() and any(root.iterdir()) and not a.resume:raise ValueError('Use --resume with identical frozen options or a new run name')
    root.mkdir(parents=True,exist_ok=True)
    freeze(root/'campaign_plan.json',{'repo':str(REPO),'python':sys.executable,'framework_sha256':fingerprint(REPO),'config':cfg,'device':a.device,
        'parent':str(parent),'parent_sha256':file_sha256(parent/'COMPLETE.json'),
        'versions':{n:version(n) for n in ('numpy','pandas','scipy','torch')},'test_evaluated':False})
    if (root/'COMPLETE.json').exists():verify(root);print('ROUND25 COMPLETE VERIFIED:',root);return root
    shinka=Path(a.shinka_python or work/'envs/shinka/bin/python')
    try:
        if shutil.disk_usage(root).free<cfg['disk_reserve_gib']*1024**3:raise ValueError('Need frozen disk reserve; no artifacts deleted')
        if not shinka.is_file():raise ValueError('Pinned Shinka environment missing')
        env=worker_env();env.update(SHINKA_PRICING_MODE='offline',PYTHONIOENCODING='utf-8')
        capabilities=subprocess.run([str(shinka),str(REPO/'scripts/run_shinka_round25.py'),'--check-api'],env=env,text=True,capture_output=True)
        if capabilities.returncode:raise RuntimeError('Pinned Shinka API preflight failed: '+capabilities.stderr[-1600:])
        write_json(root/'shinka_capabilities.json',parse_json(capabilities.stdout))
        stage(root,'kernel isolation preflight')
        preflight(root,budget(root/'baseline_budget',cfg,cfg['baseline_hours'],a.device),sys.executable)
        prepare(root,parent,cfg,a.device)
        if a.mode=='preflight':stage(root,'numeric preflight complete; no paid evolution');return root
        for arm in cfg['arms']:search_arm(root,arm,cfg,shinka)
        # Freeze every task/arm before opening any evolved development confirmation.
        selections={arm:{task:freeze_finalists(root/'arms'/arm/task) for task in cfg['tasks']} for arm in cfg['arms']}
        freeze(root/'search_frozen.json',selections)
        report(root,cfg);(root/'INCOMPLETE.json').unlink(missing_ok=True);stage(root,'complete')
        export_native_archives(root);seal(root,test_evaluated=False);return root
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error_type':type(exc).__name__,'error':str(exc),'test_evaluated':False});raise
    finally:
        if not (root/'COMPLETE.json').exists():export_native_archives(root)
        package(root)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round25_self_correct_01')
    p.add_argument('--round24-run');p.add_argument('--config',default=str(REPO/'configs/round25.json'));p.add_argument('--shinka-python')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--mode',choices=['preflight','all'],default='all');p.add_argument('--resume',action='store_true');a=p.parse_args()
    with run_lock(Path(a.work_dir)/'runs'/('.'+a.name+'.lock')):
        with run_lock(Path(a.work_dir)/'runs'/'.evolution_gpu.lock'):launch(a)


if __name__=='__main__':main()
