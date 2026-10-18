"""Round27: persistent Virtual Cell agent, balanced tasks and measured hypothesis revisions."""
import argparse
from collections import Counter
import copy
from dataclasses import asdict
import difflib
from importlib.metadata import version
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import numpy as np
from vcell.clean_background import run_lock
from vcell.model_agent.artifacts import read,seal,verify
from vcell.evolution.journal import append,events
from vcell.evolution.self_correction import ResearchAction,prediction_outcome,quota_failure,frontier
from vcell.round11 import freeze
from vcell.round22.evaluator import worker_env
from vcell.round27.provider import ResearchGateway
from vcell.round25.runtime import charged
from vcell.round27.runtime import assess,fingerprint,source_records,novel
from vcell.round27.prepare import prepare,pool_plan,protocol_parent
from vcell.round27.critic import decide,evidence
from vcell.round27 import memory,domain,manager
from vcell.round27.tools import source_probe
from vcell.utils import file_sha256,write_json
from run_round22 import preflight
from run_round25 import next_generation,default_action,export_native_archives


def stage(root,name,**fields):
    write_json(root/'stage.json',{'stage':name,'test_evaluated':False,**fields})
    print('ROUND27 STAGE:',name,fields,flush=True)


def event(root,ticket,kind,**fields):
    path=root/'research_decisions.jsonl'
    if not any(r.get('ticket')==ticket for r in events(path)):append(path,kind,ticket=ticket,phase='search',**fields)


def cfg_check(cfg):
    if cfg['arms']!=['shinka','repair','meta'] or cfg['tasks']!=['repeat','transfer']:raise ValueError('Three matched arms and two separate tasks required')
    if not 1<=cfg['api_requests_per_arm']<=30 or not 1<=cfg['valid_target_per_arm']<=15:raise ValueError('Bounded matched tranche required')
    if cfg['max_repairs_per_candidate']!=2 or not .4<=cfg['repeat_share']<=.7:raise ValueError('Invalid bounded action policy')
    if cfg['critic_every_valid']<1 or cfg['maximum_consecutive_critic_fallbacks']!=2:raise ValueError('Explicit critic stop policy required')
    if not 1<=cfg['steps']<=500 or cfg['maximum_task_search_share']!=.7 or not 0<cfg['search_fraction']<1:raise ValueError('Invalid numerical budget')
    if not 1<=cfg['pilot_requests']<=6 or cfg['extension_requests']!=0:raise ValueError('No automatic budget extension in this balanced tranche')
    if cfg['baseline_hours']+cfg['pilot_hours']+3*cfg['arm_hours']+cfg['extension_hours']>14:raise ValueError('Aggregate worker cap exceeds 14 hours')
    if min(cfg['baseline_hours'],cfg['pilot_hours'],cfg['arm_hours'])<=0 or not 1<=cfg['worker_seconds']<=120:
        raise ValueError('Positive bounded worker allowances required')
    if not 1<=cfg['critic_max_tokens']<=8192 or not cfg['critic_max_tokens']<=cfg['api_max_tokens']<=32768:
        raise ValueError('Bounded critic and mutation output caps required')
    if cfg['independent_confirmation'] or cfg['confirmation_scope']!='historically_viewed_development':raise ValueError('No new independent confirmation is claimed')
    if set(cfg['task_targets'])!=set(cfg['tasks']) or any(type(v)!=int or not 1<=v<=8 for v in cfg['task_targets'].values()):raise ValueError('Positive per-task exploration targets required')
    if sum(cfg['task_targets'].values())!=cfg['valid_target_per_arm']:raise ValueError('Per-task goals must sum to the arm goal')
    if not 1<=cfg['minimum_branch_exploration']<=min(cfg['task_targets'].values()):raise ValueError('Branch-stop exploration allowance is invalid')
    if not 2<=cfg['heartbeat_seconds']<=60:raise ValueError('Heartbeat interval must be bounded')


def unresolved(armroot):
    if any(read(p).get('state')=='reserved' for p in (armroot/'provider_responses').glob('*.json')):
        raise RuntimeError('Unresolved provider ticket; stopped before any further paid request')


def gateway_for(root,cfg,relay,requests=None,tokens=None):
    unresolved(root);key=os.environ.get('DEEPSEEK_API_KEY','').strip()
    if not key:raise RuntimeError('Export DEEPSEEK_API_KEY before starting screen')
    return ResearchGateway(root/'api_ledger.json',key,cfg['model'],relay,requests or cfg['api_requests_per_arm'],
        tokens or cfg['api_output_tokens_per_arm'],cfg['port'],max_tokens=cfg['api_max_tokens'],
        reasoning_effort=cfg['reasoning_effort'],failure_limit=cfg['api_failure_limit'])


def probe(pool,action):
    if action is None or action.requested_source_diagnostic=='none':return None
    cards=evidence(pool)['candidates'];key=action.requested_source_diagnostic
    field={'reference_coverage':'reference_coverage','context_losses':'by_context','target_losses':'worst_targets'}
    return {'probe':key,'scope':'source_search_only','measured':[{'module_sha256':r['module_sha256'],
        'result':r.get('diagnosis',{}).get(field[key]) if key in field else r['metrics']} for r in cards if r['status']=='valid']}


def generation(root,pool,folder,action,attribution,banned,turn,gateway,relay,shinka):
    marker=folder/'done.json'
    if marker.exists():return read(marker)
    planned=folder/'plan.json'
    if not planned.exists():
        rows=source_records(pool)
        reference=action.parent_code_sha256 if action and action.parent_code_sha256 else read(pool/'baseline_selection.json')['module_sha256']
        freeze(planned,{'action':asdict(action) if action else None,'attribution':attribution,'banned':list(banned),
            'generation_target':next_generation(pool),'before_ids':[r['module_sha256'] for r in rows],
            'prediction_reference_sha256':reference,'before_novel':len(novel(pool)),'turn':turn,'probe':probe(pool,action)})
    plan=read(planned);action=ResearchAction(**plan['action']) if plan['action'] else None
    snapshot=folder/'source_feedback.json'
    if not snapshot.exists():
        info=evidence(pool);first=read(pool/'folds.json')['search'][0]
        info['declared_revision']=read(folder/'controller_plan.json').get('revision') if (folder/'controller_plan.json').exists() else None
        info['native_frontier']=[{'module_sha256':r['module_sha256'],'status':r['status']} for r in frontier(source_records(pool))]
        with np.load(first['input'],allow_pickle=False) as data:
            info['input_contract']={k:{'shape':list(data[k].shape),'dtype':str(data[k].dtype)} for k in data.files}
        freeze(snapshot,info)
    write_json(pool/'source_feedback.json',read(snapshot));write_json(pool/'active_action.json',plan)
    task=read(pool/'run_plan.json')['task'];arm=read(pool/'run_plan.json')['arm'];role='repair' if action and action.action=='repair_candidate' else 'mutation'
    gateway.scope=f'{task}/{folder.name}/{role}';gateway.role=role
    stage(root,'native generation',arm=arm,task=task,turn=turn+1,action_attribution=plan['attribution'],api_used=len(gateway.records),api_max=gateway.requests_max)
    env=worker_env();env.update(ROUND27_POOL=str(pool),ROUND22_GATEWAY_KEY=relay,SHINKA_PRICING_MODE='offline',PYTHONIOENCODING='utf-8')
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'shinka.log').open('a',encoding='utf-8') as log:
        result=subprocess.run([str(shinka),str(REPO/'scripts/run_shinka_round27.py'),'--pool',str(pool),
            '--generations',str(plan['generation_target'])],env=env,stdout=log,stderr=subprocess.STDOUT)
    unresolved(gateway.path.parent)
    rows=source_records(pool);new=[r for r in rows if r['module_sha256'] not in plan['before_ids']]
    measured=None;latest=max(new,key=lambda r:r['finished_at']) if new else None
    if latest and action:
        before=next(r for r in rows if r['module_sha256']==plan['prediction_reference_sha256'])
        measured=prediction_outcome(action,before,latest);measured['reference_sha256']=before['module_sha256']
        (folder/'code.diff').write_text(''.join(difflib.unified_diff(Path(before['module']).read_text(encoding='utf-8').splitlines(True),
            Path(latest['module']).read_text(encoding='utf-8').splitlines(True),fromfile='parent',tofile='child')),encoding='utf-8')
    value={'runner_exit':result.returncode,'children':[{'module_sha256':r['module_sha256'],'status':r['status']} for r in new],
        'latest':latest['module_sha256'] if latest else None,'prediction':measured,'attribution':plan['attribution'],
        'before_novel':plan['before_novel'],'after_novel':len(novel(pool))}
    event(root,arm+'/'+str(turn)+'/outcome','measured_action_outcome',arm=arm,task=task,role=role,action=plan['action'],
        attribution=plan['attribution'],**{k:v for k,v in value.items() if k!='attribution'})
    freeze(marker,value);return value


def live_smoke(root,cfg,shared,shinka):
    marker=root/'LIVE_SMOKE_COMPLETE.json'
    if marker.exists():
        done=read(marker)
        for path,sha in done['files'].items():
            if file_sha256(root/path)!=sha:raise ValueError('Live smoke evidence changed')
        child=read(root/done['child_record'])
        for name,sha in child['files'].items():
            if file_sha256(Path(child['directory'])/name)!=sha:raise ValueError('Live smoke payload changed')
        return done
    ar=root/'pilot/meta';pool=ar/'repeat';item=shared['repeat']
    pilot_cfg={**cfg,'search_fraction':1.,'maximum_task_search_share':1.}
    required=(2+len(item['folds']['search'])*len(cfg['counterfactual_variants']))*cfg['worker_seconds']
    if cfg['pilot_hours']*3600<required:raise ValueError('Pilot allowance insufficient before any API call: '+str(required))
    freeze(root/'pilot_budget.json',{'maximum_candidate_seconds':required,'dedicated_seconds':cfg['pilot_hours']*3600})
    pool_plan(pool,ar,'repeat','pilot',pilot_cfg,item['folds'],item['baseline'],item['common'],REPO,sys.executable,cfg['pilot_hours'])
    relay=secrets.token_hex(24)
    with gateway_for(ar,cfg,relay,cfg['pilot_requests'],cfg['pilot_requests']*cfg['api_max_tokens']) as gateway:
        allowed={'refine_parent','explore_new_mechanism'}
        stage(root,'live domain-aware critic and measured revision gate')
        first=decide(pool,gateway,'pilot_initial',cfg['critic_max_tokens'],allowed_actions=allowed)
        if first['attribution']!='critic':raise RuntimeError('Live critic schema gate failed; inspect pilot decisions')
        action=ResearchAction(**first['action'])
        result=generation(root,pool,ar/'turns/000',action,'critic',[],0,gateway,relay,shinka)
        memory.outcome(root,'pilot','repeat','pilot/0',action,result,first.get('revision'))
        child=next((r for r in source_records(pool) if r['module_sha256']==result['latest']),None)
        if not child or child['status']!='valid' or not result.get('prediction') or result['prediction']['status']=='undefined':
            raise RuntimeError('Pilot child invalid or undefined: '+str((child or {}).get('error','No new native assessment')))
        update=decide(pool,gateway,'pilot_after_result',cfg['critic_max_tokens'],allowed_actions=allowed,required_evidence=child['module_sha256'])
        if update['attribution']!='critic':raise RuntimeError('Post-result structured revision gate failed')
        paths=[pool/'decisions/pilot_initial/decision.json',pool/'decisions/pilot_after_result/decision.json',
            ar/'turns/000/done.json',Path(child['directory'])/'assessment.json']
        value={'accepted_decisions':2,'measured_child':child['module_sha256'],'hypothesis_outcome':result['prediction'],
            'api_requests':len(gateway.records),'child_record':(Path(child['directory'])/'assessment.json').relative_to(root).as_posix(),
            'files':{p.relative_to(root).as_posix():file_sha256(p) for p in paths},'scope':'source capability gate, excluded from arms',
            'test_evaluated':False}
        freeze(marker,value);return value

def choose_action(pool,arm,state,task,gateway,cfg,ticket):
    pending=state['pending'].get(task)
    if arm!='shinka' and pending and state['repairs'].get(pending['origin'],0)<2:
        origin=pending['origin'];state['repairs'][origin]=state['repairs'].get(origin,0)+1
        return ResearchAction('repair_candidate','measured runtime error','Fix the actual failing numerical implementation','runtime repair',
            pending['parent'],None,'high','none',{'metric':'valid','direction':'increase','minimum_change':1},
            'Stop this repair lineage after two attempts','Use the measured traceback',[pending['parent']],state['share']), 'automatic_repair',origin,None
    count=len(novel(pool));pending_science=memory.experience(pool.parents[2],arm,task)['pending_revisions']
    due=bool(pending_science) or state['critic_at'][task]<0 or count-state['critic_at'][task]>=cfg['critic_every_valid'] or state['empty'][task]>=cfg['stagnation_turns']
    if arm.startswith('meta') and due:
        allowed={'repair_candidate','refine_parent','crossover_methods','explore_new_mechanism','run_source_probe'}
        if count>=cfg['minimum_branch_exploration'] and not pending_science:allowed.add('stop_unproductive_branch')
        review=decide(pool,gateway,ticket,cfg['critic_max_tokens'],allowed_actions=allowed,banned=state['banned'][task],
            repair_counts=state['repairs'],repair_origins=state['origins'])
        state['critic_at'][task]=count
        if review['attribution']=='critic':
            action=ResearchAction(**review['action']);state['fallbacks']=0;state['share']=action.repeat_share
            if action.action=='repair_candidate':
                origin=state['origins'].get(action.parent_code_sha256,action.parent_code_sha256)
                state['repairs'][origin]=state['repairs'].get(origin,0)+1
                return action,'critic',origin,review.get('revision')
            return action,'critic',None,review.get('revision')
        state['fallbacks']+=1
        fallback=default_action(source_records(pool),'Critic rejected; explicitly attributed fallback')
        fallback=ResearchAction(**{**asdict(fallback),'repeat_share':state['share']})
        return fallback,'fallback',None,None
    return None,'native_pareto',None,None


def search_arm(root,arm,cfg,shinka):
    ar=root/'arms'/arm;marker=ar/'SEARCH_COMPLETE.json'
    if marker.exists():return read(marker)
    unresolved(ar);pools={t:ar/t for t in cfg['tasks']}
    state={'pending':{},'repairs':{},'origins':{},'share':cfg['repeat_share'],'critic_at':{t:-1 for t in pools},
        'empty':{t:0 for t in pools},'banned':{t:[] for t in pools},'blocked':[],'fallbacks':0}
    relay=secrets.token_hex(24);reason='turn ceiling'
    with gateway_for(ar,cfg,relay) as gateway:
        for turn in range(cfg['max_turns_per_arm']):
            manager.checkpoint(root.parents[1],root.name)
            folder=ar/'turns'/f'{turn:03d}';done=folder/'state.json';planned=folder/'controller_plan.json'
            if done.exists():state=read(done);continue
            counts={t:len(novel(pool)) for t,pool in pools.items()}
            manager.task_queue(root,cfg,arm,counts)
            if not planned.exists():
                if all(counts[t]>=cfg['task_targets'][t] for t in pools):reason='both task targets reached';break
                if len(gateway.records)>=gateway.requests_max or sum(r.get('charged_output_tokens',0) for r in gateway.records)+cfg['api_max_tokens']>gateway.output_tokens_max:
                    reason='provider cap; unmet tasks remain coverage-incomplete';break
                if state['fallbacks']>=cfg['maximum_consecutive_critic_fallbacks']:reason='critic circuit opened';break
                seconds={t:charged(ar,pool) for t,pool in pools.items()}
                blocked=set(state['blocked'])|{t for t in pools if seconds[t]+cfg['worker_seconds']>cfg['arm_hours']*3600*cfg['search_fraction']*cfg['maximum_task_search_share']}
                task=manager.choose_task(cfg,counts,seconds,blocked)
                if task is None or charged(ar)+cfg['worker_seconds']>cfg['arm_hours']*3600*cfg['search_fraction']:
                    reason='worker cap; unmet tasks remain coverage-incomplete';break
                pool=pools[task]
                # A critic decision needs room for its implementation, not just valid JSON.
                if arm=='meta' and len(gateway.records)+2>gateway.requests_max:
                    reason='reserve for critic plus implementation unavailable';break
                action,origin,repair_origin,revision=choose_action(pool,arm,state,task,gateway,cfg,f'turn_{turn:03d}')
                freeze(planned,{'task':task,'action':asdict(action) if action else None,'attribution':origin,
                    'repair_origin':repair_origin,'revision':revision,'state':copy.deepcopy(state)})
            current=read(planned);state=current['state'];task=current['task'];pool=pools[task]
            manager.task_queue(root,cfg,arm,counts,active=task)
            action=ResearchAction(**current['action']) if current['action'] else None
            event(root,arm+'/'+str(turn),'research_action',arm=arm,task=task,action=current['action'],attribution=current['attribution'])
            if action and action.action=='run_source_probe':
                measured=source_probe(pool,action.requested_source_diagnostic,folder/'source_probe.json')
                result={'latest':None,'children':[],'prediction':None,'attribution':current['attribution'],
                    'probe':{'kind':measured['probe'],'path':str(folder/'source_probe.json'),'sha256':file_sha256(folder/'source_probe.json'),
                        'summary':[{'module_sha256':r['module_sha256'],'values':r['values'][:3]} for r in measured['methods']]}}
                freeze(folder/'done.json',result)
                event(root,arm+'/'+str(turn)+'/probe','executed_source_probe',arm=arm,task=task,probe=result['probe'])
            elif action and action.action=='stop_unproductive_branch':
                state['banned'][task].append(action.parent_code_sha256)
                result={'latest':None,'children':[],'prediction':None,'attribution':current['attribution']}
                freeze(folder/'done.json',result)
                event(root,arm+'/'+str(turn)+'/stop','branch_stopped',arm=arm,task=task,reason='Only this parent is stopped; task quota remains active')
            else:
                result=generation(root,pool,folder,action,current['attribution'],state['banned'][task],turn,gateway,relay,shinka)
            memory.outcome(root,arm,task,arm+'/'+str(turn),action,result,current.get('revision'))
            latest=next((r for r in source_records(pool) if r['module_sha256']==result['latest']),None)
            if latest and latest['status']=='invalid' and not quota_failure(latest):
                origin=current['repair_origin'] or latest['module_sha256']
                state['pending'][task]={'parent':latest['module_sha256'],'origin':origin};state['origins'][latest['module_sha256']]=origin
            elif latest:state['pending'].pop(task,None)
            if latest and quota_failure(latest):state['blocked'].append(task)
            # Stagnation requests scientific reconsideration; it cannot silently kill a whole task.
            after=len(novel(pool));state['empty'][task]=state['empty'][task]+1 if after==result.get('before_novel',counts[task]) else 0
            freeze(done,state)
            if result.get('runner_exit') and len(gateway.records)>=cfg['api_failure_limit'] and all(r['status']!='complete' for r in gateway.records[-cfg['api_failure_limit']:]):
                reason='provider circuit opened';break
        counts={t:len(novel(pool)) for t,pool in pools.items()}
        queue=manager.task_queue(root,cfg,arm,counts,reason)
        value={'arm':arm,'reason':reason,'distinct_valid':counts,'coverage_complete':all(r['status']=='complete' for r in queue),
            'api_requests':len(gateway.records),'requests_max':cfg['api_requests_per_arm'],'valid_target':cfg['valid_target_per_arm'],
            'task_targets':cfg['task_targets'],'worker_hours_max':cfg['arm_hours'],
            'unspent_requests':cfg['api_requests_per_arm']-len(gateway.records),'state':state,'test_evaluated':False}
        freeze(marker,value);return value

def finalists(pool):
    path=pool/'search_frozen.json'
    if path.exists():return read(path)
    rows=novel(pool);selected=[]
    if rows:selected.append({'tier':'source_mse',**min(rows,key=lambda r:r['metrics']['public']['mse'])})
    signal=[r for r in frontier(rows) if r['metrics']['public'].get('strong_mse_ratio') is not None
        and r['metrics']['public']['strong_mse_ratio']<1 and (r['metrics']['public'].get('specificity_gap') or 0)>0
        and r['metrics']['public']['amplitude_ratio']>=.05]
    if signal:
        r=min(signal,key=lambda r:r['metrics']['public']['strong_mse'])
        if not any(s['module_sha256']==r['module_sha256'] for s in selected):selected.append({'tier':'signal_diagnostic_not_promotion',**r})
    value={'finalists':selected,'source_only':True,'task':read(pool/'run_plan.json')['task']}
    freeze(path,value);return value


def launch(a):
    from vcell.round27.report import report,package
    work=Path(a.work_dir).resolve();manager.control_path(work,a.name);root=work/'runs'/a.name;parent=Path(a.parent_run or work/'runs/round26_critic_02').resolve()
    cfg=read(a.config);cfg['device']=a.device;cfg_check(cfg)
    if root.exists() and any(root.iterdir()) and not a.resume:raise ValueError('Use a new name, or --resume with identical frozen options')
    root.mkdir(parents=True,exist_ok=True)
    freeze(root/'campaign_plan.json',{'repo':str(REPO),'python':sys.executable,'framework_sha256':fingerprint(REPO),'config':cfg,
        'parent':str(parent),'protocol_parent':str(protocol_parent(parent)),'parent_sha256':file_sha256(parent/'COMPLETE.json'),
        'versions':{n:version(n) for n in ('numpy','pandas','scipy','torch')},'test_evaluated':False})
    if (root/'COMPLETE.json').exists():verify(root);print('ROUND27 COMPLETE VERIFIED:',root);return root
    shinka=Path(a.shinka_python or work/'envs/shinka/bin/python')
    heartbeat=manager.Heartbeat(root,cfg['heartbeat_seconds']).__enter__()
    try:
        manager.checkpoint(work,a.name)
        (root/'PAUSED.json').unlink(missing_ok=True)
        if shutil.disk_usage(root).free<cfg['disk_reserve_gib']*1024**3:raise ValueError('Need frozen data-disk reserve')
        if not shinka.is_file():raise ValueError('Pinned Shinka environment missing')
        env=worker_env();env.update(SHINKA_PRICING_MODE='offline',PYTHONIOENCODING='utf-8')
        checked=subprocess.run([str(shinka),str(REPO/'scripts/run_shinka_round27.py'),'--check-api'],env=env,text=True,capture_output=True)
        if checked.returncode:raise RuntimeError('Pinned Shinka API check failed: '+checked.stderr[-1200:])
        import json
        write_json(root/'shinka_capabilities.json',json.loads(checked.stdout))
        stage(root,'kernel isolation preflight')
        preflight(root,{'root':str(root/'baseline_budget'),'seed':cfg['seed'],'device':a.device,'steps':cfg['steps'],
            'seconds':cfg['worker_seconds'],'ram_bytes':12*1024**3,'evaluation_seconds_max':cfg['baseline_hours']*3600},sys.executable)
        shared=prepare(root,parent,cfg,REPO,sys.executable,stage)
        domain.build(root,shared,cfg);memory.import_history(root,parent)
        manager.checkpoint(work,a.name)
        if a.mode=='preflight':stage(root,'numeric preflight complete; live gate and searches not run');return root
        live_smoke(root,cfg,shared,shinka)
        if a.mode=='live-smoke':stage(root,'live critic generation and result update verified; matched search not run');return root
        for arm in cfg['arms']:search_arm(root,arm,cfg,shinka)
        manager.checkpoint(work,a.name)
        freeze(root/'matched_search_frozen.json',{arm:{task:finalists(root/'arms'/arm/task) for task in cfg['tasks']} for arm in cfg['arms']})
        freeze(root/'search_frozen.json',{'matched':read(root/'matched_search_frozen.json'),'extension_executed':False})
        report(root,cfg,stage);export_native_archives(root)
        memory.publish(root,work,cfg)
        (root/'INCOMPLETE.json').unlink(missing_ok=True)
        heartbeat.__exit__();stage(root,'complete');seal(root,test_evaluated=False);return root
    except manager.PauseRequested as exc:
        write_json(root/'PAUSED.json',{'reason':str(exc),'resume':'Clear the pause request and repeat identical options with --resume'})
        stage(root,'paused at committed experiment boundary');return root
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error_type':type(exc).__name__,'error':str(exc),'test_evaluated':False});raise
    finally:
        heartbeat.__exit__()
        if not (root/'COMPLETE.json').exists():export_native_archives(root)
        package(root)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work')
    p.add_argument('--name',default='round27_agent_01');p.add_argument('--parent-run','--round26-run',dest='parent_run');p.add_argument('--config',default=str(REPO/'configs/round27.json'))
    p.add_argument('--shinka-python');p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--mode',choices=['preflight','live-smoke','all'],default='all');p.add_argument('--resume',action='store_true');a=p.parse_args()
    with run_lock(Path(a.work_dir)/'runs'/('.'+a.name+'.lock')):
        with run_lock(Path(a.work_dir)/'runs'/'.evolution_gpu.lock'):launch(a)


if __name__=='__main__':main()
