"""Round26: real critic gate, matched native search, separately labelled optional Meta extension."""
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
from vcell.round26.provider import ResearchGateway
from vcell.round25.runtime import charged
from vcell.round26.runtime import assess,fingerprint,source_records,novel
from vcell.round26.prepare import prepare,pool_plan
from vcell.round26.critic import decide,evidence
from vcell.utils import file_sha256,write_json
from run_round22 import preflight
from run_round25 import next_generation,default_action,export_native_archives


def stage(root,name,**fields):
    write_json(root/'stage.json',{'stage':name,'test_evaluated':False,**fields})
    print('ROUND26 STAGE:',name,fields,flush=True)


def event(root,ticket,kind,**fields):
    path=root/'research_decisions.jsonl'
    if not any(r.get('ticket')==ticket for r in events(path)):append(path,kind,ticket=ticket,phase='search',**fields)


def cfg_check(cfg):
    if cfg['arms']!=['shinka','repair','meta'] or cfg['tasks']!=['repeat','transfer']:raise ValueError('Three matched arms and two separate tasks required')
    if not 1<=cfg['api_requests_per_arm']<=30 or not 1<=cfg['valid_target_per_arm']<=15:raise ValueError('Bounded matched tranche required')
    if cfg['max_repairs_per_candidate']!=2 or not .4<=cfg['repeat_share']<=.7:raise ValueError('Invalid bounded action policy')
    if cfg['critic_every_valid']<1 or cfg['maximum_consecutive_critic_fallbacks']!=2:raise ValueError('Explicit critic stop policy required')
    if not 1<=cfg['steps']<=500 or cfg['maximum_task_search_share']!=.7 or not 0<cfg['search_fraction']<1:raise ValueError('Invalid numerical budget')
    if not 1<=cfg['pilot_requests']<=6 or not 1<=cfg['extension_requests']<=20 or not 1<=cfg['extension_valid_target']<=10:raise ValueError('Pilot/extension caps exceeded')
    if cfg['baseline_hours']+cfg['pilot_hours']+3*cfg['arm_hours']+cfg['extension_hours']>14:raise ValueError('Aggregate worker cap exceeds 14 hours')
    if min(cfg['baseline_hours'],cfg['pilot_hours'],cfg['arm_hours'],cfg['extension_hours'])<=0 or not 1<=cfg['worker_seconds']<=120:
        raise ValueError('Positive bounded worker allowances required')
    if not 1<=cfg['critic_max_tokens']<=8192 or not cfg['critic_max_tokens']<=cfg['api_max_tokens']<=32768:
        raise ValueError('Bounded critic and mutation output caps required')
    if cfg['independent_confirmation'] or cfg['confirmation_scope']!='historically_viewed_development':raise ValueError('No new independent confirmation is claimed')


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
        info['native_frontier']=[{'module_sha256':r['module_sha256'],'status':r['status']} for r in frontier(source_records(pool))]
        with np.load(first['input'],allow_pickle=False) as data:
            info['input_contract']={k:{'shape':list(data[k].shape),'dtype':str(data[k].dtype)} for k in data.files}
        freeze(snapshot,info)
    write_json(pool/'source_feedback.json',read(snapshot));write_json(pool/'active_action.json',plan)
    task=read(pool/'run_plan.json')['task'];arm=read(pool/'run_plan.json')['arm'];role='repair' if action and action.action=='repair_candidate' else 'mutation'
    gateway.scope=f'{task}/{folder.name}/{role}';gateway.role=role
    stage(root,'native generation',arm=arm,task=task,turn=turn+1,action_attribution=plan['attribution'],api_used=len(gateway.records),api_max=gateway.requests_max)
    env=worker_env();env.update(ROUND26_POOL=str(pool),ROUND22_GATEWAY_KEY=relay,SHINKA_PRICING_MODE='offline',PYTHONIOENCODING='utf-8')
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'shinka.log').open('a',encoding='utf-8') as log:
        result=subprocess.run([str(shinka),str(REPO/'scripts/run_shinka_round26.py'),'--pool',str(pool),
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
            if file_sha256(Path(child['directory'])/name)!=sha:raise ValueError('Live smoke numerical payload changed')
        return done
    armroot=root/'pilot/meta';pool=armroot/'repeat';item=shared['repeat']
    pool_plan(pool,armroot,'repeat','pilot',cfg,item['folds'],item['baseline'],item['common'],REPO,sys.executable,cfg['pilot_hours'])
    relay=secrets.token_hex(24)
    with gateway_for(armroot,cfg,relay,cfg['pilot_requests'],cfg['pilot_requests']*cfg['api_max_tokens']) as gateway:
        stage(root,'authenticated scientific critic smoke; main searches gated')
        allowed={'refine_parent','run_source_probe','explore_new_mechanism'}
        first=decide(pool,gateway,'pilot_initial',cfg['critic_max_tokens'],allowed_actions=allowed)
        if first['attribution']!='critic':raise RuntimeError('Live critic gate failed; no main search launched. Inspect pilot decisions and API ledger')
        action=ResearchAction(**first['action'])
        outcome=generation(root,pool,armroot/'turns/000',action,'critic',[],0,gateway,relay,shinka)
        child=next((r for r in source_records(pool) if r['module_sha256']==outcome['latest']),None)
        if child is None or child['status']!='valid' or not outcome['prediction'] or outcome['prediction']['status']=='undefined':
            raise RuntimeError('Live smoke did not produce a valid measured child and hypothesis outcome; main searches blocked')
        update=decide(pool,gateway,'pilot_after_result',cfg['critic_max_tokens'],allowed_actions=allowed,required_evidence=child['module_sha256'])
        if update['attribution']!='critic':raise RuntimeError('Post-experiment critic update failed; main searches blocked')
        checked=[pool/'decisions/pilot_initial/decision.json',pool/'decisions/pilot_after_result/decision.json',
            armroot/'turns/000/done.json',Path(child['directory'])/'assessment.json']
        value={'accepted_decisions':2,'measured_child':child['module_sha256'],'hypothesis_outcome':outcome['prediction'],
            'child_record':(Path(child['directory'])/'assessment.json').relative_to(root).as_posix(),
            'api_requests':len(gateway.records),'scope':'source_only_capability_gate; excluded from matched search',
            'files':{p.relative_to(root).as_posix():file_sha256(p) for p in checked},'test_evaluated':False}
        freeze(marker,value);return value


def choose_action(pool,arm,state,task,gateway,cfg,ticket):
    pending=state['pending'].get(task)
    if arm!='shinka' and pending and state['repairs'].get(pending['origin'],0)<2:
        origin=pending['origin'];state['repairs'][origin]=state['repairs'].get(origin,0)+1
        return ResearchAction('repair_candidate','measured runtime error','Fix the actual failing numerical implementation','runtime repair',
            pending['parent'],None,'high','none',{'metric':'valid','direction':'increase','minimum_change':1},
            'Stop this repair lineage after two attempts','Use the measured traceback',[pending['parent']],state['share']), 'automatic_repair',origin
    count=len(novel(pool));due=state['critic_at'][task]<0 or count-state['critic_at'][task]>=cfg['critic_every_valid'] or state['empty'][task]>=cfg['stagnation_turns']
    if arm.startswith('meta') and due:
        review=decide(pool,gateway,ticket,cfg['critic_max_tokens'],banned=state['banned'][task],
            repair_counts=state['repairs'],repair_origins=state['origins'])
        state['critic_at'][task]=count
        if review['attribution']=='critic':
            action=ResearchAction(**review['action']);state['fallbacks']=0;state['share']=action.repeat_share
            if action.action=='repair_candidate':
                origin=state['origins'].get(action.parent_code_sha256,action.parent_code_sha256)
                state['repairs'][origin]=state['repairs'].get(origin,0)+1
                return action,'critic',origin
            return action,'critic',None
        state['fallbacks']+=1
        fallback=default_action(source_records(pool),'Critic rejected; explicitly attributed fallback')
        fallback=ResearchAction(**{**asdict(fallback),'repeat_share':state['share']})
        return fallback,'fallback',None
    return None,'native_pareto',None


def search_arm(root,arm,cfg,shinka):
    armroot=root/'arms'/arm;marker=armroot/'SEARCH_COMPLETE.json'
    if marker.exists():return read(marker)
    unresolved(armroot);pools={t:armroot/t for t in cfg['tasks']}
    state={'pending':{},'repairs':{},'origins':{},'share':cfg['repeat_share'],'critic_at':{t:-1 for t in pools},
        'empty':{t:0 for t in pools},'banned':{t:[] for t in pools},'stopped':[],'fallbacks':0}
    relay=secrets.token_hex(24);reason='turn ceiling'
    with gateway_for(armroot,cfg,relay) as gateway:
        for turn in range(cfg['max_turns_per_arm']):
            folder=armroot/'turns'/f'{turn:03d}';done=folder/'state.json';plan=folder/'controller_plan.json'
            if done.exists():state=read(done);continue
            continuing=plan.exists()
            if not continuing:
                n=sum(len(novel(p)) for p in pools.values())
                if n>=cfg['valid_target_per_arm']:reason='matched distinct-behavior target reached';break
                if len(gateway.records)>=gateway.requests_max or sum(r.get('charged_output_tokens',0) for r in gateway.records)+cfg['api_max_tokens']>gateway.output_tokens_max:
                    reason='matched provider cap';break
                if state['fallbacks']>=cfg['maximum_consecutive_critic_fallbacks']:reason='scientific critic repeatedly rejected';break
                shares={'repeat':state['share'],'transfer':1-state['share']}
                legal=[t for t in pools if t not in state['stopped'] and charged(armroot,pools[t])+cfg['worker_seconds']<=cfg['arm_hours']*3600*cfg['search_fraction']*.7]
                if not legal or charged(armroot)+cfg['worker_seconds']>cfg['arm_hours']*3600*cfg['search_fraction']:reason='matched source worker cap';break
                task=min(legal,key=lambda t:charged(armroot,pools[t])/shares[t]);pool=pools[task]
                action,origin,repair_origin=choose_action(pool,arm,state,task,gateway,cfg,f'turn_{turn:03d}')
                if action and action.action=='stop_unproductive_branch':
                    state['banned'][task].append(action.parent_code_sha256)
                    if not any(r['status']=='valid' and r['module_sha256'] not in state['banned'][task] for r in source_records(pool)):
                        state['stopped'].append(task)
                freeze(plan,{'task':task,'action':asdict(action) if action else None,'attribution':origin,
                    'repair_origin':repair_origin,'state':copy.deepcopy(state)})
            current=read(plan);state=current['state'];task=current['task'];pool=pools[task]
            action=ResearchAction(**current['action']) if current['action'] else None
            event(root,arm+'/'+str(turn),'research_action',arm=arm,task=task,action=current['action'],attribution=current['attribution'])
            if task in state['stopped']:
                event(root,arm+'/'+str(turn)+'/stop','branch_stopped',arm=arm,task=task,attribution=current['attribution'],
                    action=current['action'],reason='No unblocked measured parent remains; no new provider mutation requested')
                freeze(done,state);continue
            result=generation(root,pool,folder,action,current['attribution'],state['banned'][task],turn,gateway,relay,shinka)
            state['empty'][task]=state['empty'][task]+1 if result['after_novel']==result['before_novel'] else 0
            latest=next((r for r in source_records(pool) if r['module_sha256']==result['latest']),None)
            if latest and latest['status']=='invalid' and not quota_failure(latest):
                origin=current['repair_origin'] or latest['module_sha256']
                state['pending'][task]={'parent':latest['module_sha256'],'origin':origin};state['origins'][latest['module_sha256']]=origin
            elif latest:state['pending'].pop(task,None)
            if latest and quota_failure(latest):state['stopped'].append(task)
            if state['empty'][task]>=cfg['stagnation_turns'] and (not arm.startswith('meta') or action is not None):state['stopped'].append(task)
            freeze(done,state)
            if result['runner_exit'] and len(gateway.records)>=cfg['api_failure_limit'] and all(r['status']!='complete' for r in gateway.records[-cfg['api_failure_limit']:]):
                reason='provider circuit opened';break
        value={'arm':arm,'reason':reason,'distinct_valid':{t:len(novel(p)) for t,p in pools.items()},
            'api_requests':len(gateway.records),'requests_max':cfg['api_requests_per_arm'],'valid_target':cfg['valid_target_per_arm'],
            'worker_hours_max':cfg['arm_hours'],'unspent_requests':cfg['api_requests_per_arm']-len(gateway.records),
            'state':state,'test_evaluated':False}
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


def extension(root,cfg,shared,shinka):
    eligible=any(r['metrics']['public']['gain_over_fixed']>=cfg['extension_minimum_source_gain'] or
        (r['metrics']['public'].get('strong_mse_ratio') is not None and r['metrics']['public']['strong_mse_ratio']<=.99
         and (r['metrics']['public'].get('specificity_gap') or 0)>(r['metrics']['public'].get('baseline_specificity_gap') or 0)+.005)
        for task in cfg['tasks'] for r in novel(root/'arms/meta'/task))
    freeze(root/'extension_gate.json',{'eligible':eligible,'scope':'matched Meta SOURCE evidence only'})
    if not eligible:return False
    extra=copy.deepcopy(cfg);extra.update(api_requests_per_arm=cfg['extension_requests'],api_output_tokens_per_arm=cfg['extension_output_tokens'],
        valid_target_per_arm=cfg['extension_valid_target'],arm_hours=cfg['extension_hours'])
    for task,item in shared.items():
        common=list(item['common'])
        for r in frontier(source_records(root/'arms/meta'/task),limit=3):
            p=Path(r['directory'])/'assessment.json'
            ref={'record':str(p),'sha256':file_sha256(p)}
            if ref not in common:common.append(ref)
        pool_plan(root/'arms/meta_extension'/task,root/'arms/meta_extension',task,'meta_extension',extra,
            item['folds'],item['baseline'],common,REPO,sys.executable,extra['arm_hours'])
    search_arm(root,'meta_extension',extra,shinka);return True


def launch(a):
    from vcell.round26.report import report,package
    work=Path(a.work_dir);root=work/'runs'/a.name;parent=Path(a.round25_run or work/'runs/round25_self_correct_02')
    cfg=read(a.config);cfg['device']=a.device;cfg_check(cfg)
    if root.exists() and any(root.iterdir()) and not a.resume:raise ValueError('Use a new name, or --resume with identical frozen options')
    root.mkdir(parents=True,exist_ok=True)
    freeze(root/'campaign_plan.json',{'repo':str(REPO),'python':sys.executable,'framework_sha256':fingerprint(REPO),'config':cfg,
        'parent':str(parent),'parent_sha256':file_sha256(parent/'COMPLETE.json'),'extend_meta':a.extend_meta,
        'versions':{n:version(n) for n in ('numpy','pandas','scipy','torch')},'test_evaluated':False})
    if (root/'COMPLETE.json').exists():verify(root);print('ROUND26 COMPLETE VERIFIED:',root);return root
    shinka=Path(a.shinka_python or work/'envs/shinka/bin/python')
    try:
        if shutil.disk_usage(root).free<cfg['disk_reserve_gib']*1024**3:raise ValueError('Need frozen data-disk reserve')
        if not shinka.is_file():raise ValueError('Pinned Shinka environment missing')
        env=worker_env();env.update(SHINKA_PRICING_MODE='offline',PYTHONIOENCODING='utf-8')
        checked=subprocess.run([str(shinka),str(REPO/'scripts/run_shinka_round26.py'),'--check-api'],env=env,text=True,capture_output=True)
        if checked.returncode:raise RuntimeError('Pinned Shinka API check failed: '+checked.stderr[-1200:])
        import json
        write_json(root/'shinka_capabilities.json',json.loads(checked.stdout))
        stage(root,'kernel isolation preflight')
        preflight(root,{'root':str(root/'baseline_budget'),'seed':cfg['seed'],'device':a.device,'steps':cfg['steps'],
            'seconds':cfg['worker_seconds'],'ram_bytes':12*1024**3,'evaluation_seconds_max':cfg['baseline_hours']*3600},sys.executable)
        shared=prepare(root,parent,cfg,REPO,sys.executable,stage)
        if a.mode=='preflight':stage(root,'numeric preflight complete; live gate and searches not run');return root
        live_smoke(root,cfg,shared,shinka)
        if a.mode=='live-smoke':stage(root,'live critic generation and result update verified; matched search not run');return root
        for arm in cfg['arms']:search_arm(root,arm,cfg,shinka)
        # Freeze the matched comparison before any extension; confirmation remains unopened here.
        freeze(root/'matched_search_frozen.json',{arm:{task:finalists(root/'arms'/arm/task) for task in cfg['tasks']} for arm in cfg['arms']})
        extended=extension(root,cfg,shared,shinka) if a.extend_meta else False
        if extended:
            for task in cfg['tasks']:finalists(root/'arms/meta_extension'/task)
        freeze(root/'search_frozen.json',{'matched':read(root/'matched_search_frozen.json'),'extension_executed':extended})
        report(root,cfg,stage);export_native_archives(root)
        (root/'INCOMPLETE.json').unlink(missing_ok=True);stage(root,'complete');seal(root,test_evaluated=False);return root
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error_type':type(exc).__name__,'error':str(exc),'test_evaluated':False});raise
    finally:
        if not (root/'COMPLETE.json').exists():export_native_archives(root)
        package(root)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work')
    p.add_argument('--name',default='round26_critic_01');p.add_argument('--round25-run');p.add_argument('--config',default=str(REPO/'configs/round26.json'))
    p.add_argument('--shinka-python');p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--mode',choices=['preflight','live-smoke','all'],default='all');p.add_argument('--resume',action='store_true');p.add_argument('--extend-meta',action='store_true');a=p.parse_args()
    with run_lock(Path(a.work_dir)/'runs'/('.'+a.name+'.lock')):
        with run_lock(Path(a.work_dir)/'runs'/'.evolution_gpu.lock'):launch(a)


if __name__=='__main__':main()
