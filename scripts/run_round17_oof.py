"""Real nested OOF experts, bounded source-only code experiments, paired routing."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from time import perf_counter
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import numpy as np
import pandas as pd
from run_round7 import run_jobs
from vcell.background_training import pretrain
from vcell.clean_background import run_lock
from vcell.data import load_prepared
from vcell.model_agent.artifacts import load_prediction, read, seal, verify
from vcell.model_agent.contracts import digest, load_query, safe_name
from vcell.model_agent.deepseek import DEFAULT_MODEL, DeepSeekPlanner
from vcell.model_agent.evidence import fit_profiles
from vcell.model_agent.experts import Registry, export_p4
from vcell.model_agent.nested import DEVELOPMENT, audit_public_cache, cases, holdout_targets, lines, partition, source_summary
from vcell.model_agent.research import ResearchPlanner, candidate_worker, export_candidate, proposal_observation
from vcell.model_agent.routing import GatePlanner, OneShotPlanner, fit_gate
from vcell.model_agent.runtime import code_identity, execute
from vcell.model_agent.workflows import combine, query_from_prepared
from vcell.round11 import freeze, verify_files
from vcell.round15_p4 import METHODS, worker
from vcell.utils import file_sha256, write_json


def package(root):
    """Reviewable small reports, patches and log tails; no matrices or weights."""
    entries={};path=root/'round17_review_light.tar.gz'
    with tarfile.open(path,'w:gz') as tar:
        for p in sorted(root.rglob('*')):
            rel=p.relative_to(root)
            if not p.is_file() or p.suffix not in ('.json','.csv','.html','.py','.log'):continue
            if 'queries' in rel.parts or p.name=='state.json' or 'decisions' in rel.parts or 'configs' in rel.parts:continue
            if p.suffix=='.log':
                from collections import deque
                with p.open(encoding='utf-8',errors='replace') as stream:value=''.join(deque(stream,maxlen=60)).encode()
            else:value=p.read_bytes()
            name=rel.as_posix();member=tarfile.TarInfo(name);member.size=len(value);tar.addfile(member,io.BytesIO(value))
            entries[name]={'sha256':hashlib.sha256(value).hexdigest(),'bytes':len(value)}
        value=json.dumps({'files':entries,'scope':'Reports and head patches; no control/truth arrays, weights or API credentials'},indent=2).encode()
        m=tarfile.TarInfo('ARCHIVE_MANIFEST.json');m.size=len(value);tar.addfile(m,io.BytesIO(value))
    return path


def dispatch(specs,root,parallel,label,stage):
    jobs={}
    for name,cfg in specs.items():
        path=root/'configs'/(name+'.json');freeze(path,{**cfg,'stage':stage})
        jobs[name]={'cmd':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-config',str(path),'--resume'],
                    'resource':'gpu','complete':str(Path(cfg['output_dir'])/'COMPLETE.json')}
    outcome=run_jobs(jobs,root,parallel,False,label=label)
    write_json(root/(label+'_job_results.json'),outcome)
    if any(outcome.values()):raise RuntimeError(label+' failed; inspect logs and resume identical command')


def prepared_inputs(a,parent,root,freeze_inputs=True):
    verify_files(parent)
    template=parent/'models'/f'context_expanded_seed{a.parent_seed}'
    verify_files(template);cfg=read(template/'manifest.json')['config']
    bg=read(Path(cfg['encoder']).parent/'manifest.json')['config']
    data=load_prepared(cfg['data_dir'])
    old=load_prepared(read(parent/'models'/f'context_reference_seed{a.parent_seed}'/'manifest.json')['config']['data_dir'])
    n=len(old['meta'])
    if data['audit'].get('historical_rows',n)!=n or n>=len(data['meta']):raise ValueError('Historical extension boundary unavailable')
    for field in ('baseline','delta','genes'):
        x=data[field] if field=='genes' else data[field][:n]
        if not np.array_equal(x,old[field]):raise ValueError('Historical data changed')
    from vcell.model_agent.contracts import NORMALIZATION
    if data['audit'].get('normalization')!=NORMALIZATION:raise ValueError('Response units changed')
    public=audit_public_cache(bg['background_cache'],DEVELOPMENT)
    htargets=holdout_targets(data['meta']);plans=[];bs={};ws={}
    for context in a.outer_contexts:
        inner=[c for c in DEVELOPMENT if c!=context][:a.inner_folds]
        for seed in a.seeds:
            label=f'{context}_seed{seed}';folds=[]
            for number,held in enumerate([context]+inner):
                final=number==0;excluded=[context] if final else [context,held]
                count=a.targets_per_context if final else getattr(a,'profile_targets',a.targets_per_context)
                if count==0:count=int(data['meta'].loc[data['meta'].context.eq(held),'perturbation'].nunique())
                queries=cases(data['meta'],held,htargets,count)
                foldname=label+('_outer' if final else f'_inner{number}')
                configs={}
                for variant in ('reference','expanded'):
                    name=foldname+'_'+variant;part=partition(data,n,excluded,queries,htargets,variant,final)
                    bc={**bg,'data_dir':cfg['data_dir'],'partition':part,'seed':seed,'device':a.device,
                        'pretrain_steps':a.pretrain_steps or bg['pretrain_steps'],'output_dir':str(root/'backgrounds'/name),
                        'save_every':a.save_every}
                    wc={**cfg,'data_dir':cfg['data_dir'],'partition':part,'seed':seed,'device':a.device,
                        'methods':list(METHODS),'protocol':'nested_cell_line','variant':variant,
                        'max_steps':a.max_steps,'batch_size':a.batch_size,'save_every':a.save_every,
                        'encoder':str(Path(bc['output_dir'])/'encoder.pt'),'output_dir':str(root/'models'/name)}
                    bs['bg_'+name]=bc;ws['model_'+name]=wc;configs[variant]=wc
                folds.append({'name':foldname,'held':held,'final':final,'query_rows':queries,'configs':configs})
            plans.append({'name':label,'outer_context':context,'seed':seed,'folds':folds})
    inputs={'data_dir':cfg['data_dir'],'data':data['audit']['fingerprint'],'metadata':file_sha256(Path(cfg['data_dir'])/'metadata.csv'),
        'dataset_sha256':file_sha256(Path(cfg['data_dir'])/'dataset.npz'),'relations':file_sha256(cfg['relations']),
        'historical_rows':n,'held_targets':htargets,'public_cache':public,'background_cache_complete':file_sha256(Path(bg['background_cache'])/'COMPLETE.json'),
        'folds':plans,'gene_order':data['genes'].tolist(),'normalization':NORMALIZATION,
        'query_selection':'identifier-hash targets balanced held-target/seen-target; all technical rows retained',
        'outer_training':'Final refits use only source fit/calibration/diagnostic rows; held-out controls used only at inference',
        'test_evaluated':False}
    if freeze_inputs:freeze(root/'inputs.json',inputs)
    return data,plans,bs,ws


def pool_for(fold,root):
    paths=[]
    for variant,cfg in fold['configs'].items():
        dest=root/'pools'/fold['name']/variant
        if (dest/'COMPLETE.json').exists():verify(dest);path=dest/'registry.json'
        else:path=export_p4(cfg['output_dir'],dest)
        paths.append(path)
    dest=root/'pools'/fold['name']/'registry.json'
    if not dest.exists():combine(paths,dest)
    Registry(dest);return dest


def query_path(data,row,root,cfg):
    dest=root/'queries'/('q_'+digest(str(data['meta'].iloc[row].row_id))[:16]+'.json')
    if not dest.exists():query_from_prepared(cfg['data_dir'],str(data['meta'].iloc[row].row_id),dest)
    return dest


def evidence_for(plan,data,root,pools,extra=None,suffix='base',include_zero=False):
    dest=root/'evidence'/plan['name']/suffix;dest.mkdir(parents=True,exist_ok=True)
    episodes=[]
    for fold in plan['folds'][1:]:
        registry=Registry(pools[fold['name']]);models={n:e.path for n,e in registry.experts.items() if e.card['kind']=='p4' or include_zero and e.card['kind']=='zero'}
        if extra:models.update(extra[fold['name']])
        for row in fold['query_rows']:
            qp=query_path(data,row,root,fold['configs']['expanded']);q=load_query(qp)
            truth=dest/('truth_'+digest(q.query_id)[:16]+'.npz')
            if not truth.exists():
                np.savez_compressed(truth,genes=data['genes'],delta=data['delta'][row],query_fingerprint=np.array(q.fingerprint))
            for bundle in models.values():episodes.append({'bundle':str(bundle.resolve()),'query':str(qp.resolve()),
                'truth':str(truth.resolve()),'fold_id':fold['name']})
    spec={'source_contexts':[f['held'] for f in plan['folds'][1:]],'excluded_contexts':[plan['outer_context']],
          'excluded_cell_lines':[plan['outer_context'].removesuffix('_IFNG')],'episodes':episodes}
    freeze(dest/'spec.json',spec);profile=dest/'profiles'
    if (profile/'COMPLETE.json').exists():verify(profile)
    else:
        if profile.exists() and any(profile.iterdir()):
            import uuid
            # Preserve interrupted writes. Retraining is unnecessary; only this
            # cheap derived profile is regenerated from sealed OOF bundles.
            profile.rename(dest/('profiles_incomplete_'+uuid.uuid4().hex[:8]))
        fit_profiles(dest/'spec.json',profile)
    frame=pd.read_csv(profile/'private_oof_scores.csv');summary=source_summary(frame)
    rowlookup={str(r.row_id):i for i,r in data['meta'].iterrows()}
    zero=frame.loc[frame.expert_id.eq(sorted(set(frame.expert_id))[0]),['query_id','context','target']].copy()
    zero['mse']=[float(np.square(data['delta'][rowlookup[q]]).mean()) for q in zero.query_id]
    summary['zero_macro_mse']=float(zero.groupby(['context','target']).mse.mean().groupby(level='context').mean().mean())
    freeze(dest/'source_summary.json',summary);freeze(dest/'gate.json',fit_gate(frame))
    return profile,frame,summary


def admit(code,rank,dest):
    # Separate CPU process, bounded timeout, no provider credentials in its environment.
    env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN','SECRET','KEY','PASSWORD','CREDENTIAL'))}
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(REPO/'src'))
    result=subprocess.run([sys.executable,str(REPO/'scripts/check_agent_head.py'),'--code',str(code),'--rank',str(rank),
        '--output',str(dest/'admission.json')],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
    (dest/'admission.log').write_text(result.stdout[-10000:],encoding='utf-8')
    if result.returncode:raise ValueError('Candidate failed bounded CPU admission; inspect admission.log')
    return read(dest/'admission.json')


def improve(a,plan,data,root,pools,summary):
    folder=root/'research'/plan['name'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETE.json').exists():verify(folder);return read(folder/'selection.json')
    planner=ResearchPlanner(folder/'api',a.model,reasoning_effort=a.research_effort,
        max_api_calls=a.trials,max_output_tokens=a.max_output_tokens,timeout=a.api_timeout)
    previous=[];best=summary['fixed_source_mse'][summary['best_fixed']];chosen=None
    for i in range(a.trials):
        dest=folder/f'trial_{i+1:02d}';dest.mkdir(parents=True,exist_ok=True)
        obs=proposal_observation(summary,previous);freeze(dest/'observation.json',obs)
        if (dest/'proposal.json').exists():
            from vcell.model_agent.research import parse_proposal
            proposal=parse_proposal(json.dumps(read(dest/'proposal.json')))
        else:proposal=planner.decide(obs);write_json(dest/'proposal.json',asdict(proposal))
        if proposal.tool=='stop':break
        source=dest/'head.py'
        if not source.exists():source.write_text(proposal.code,encoding='utf-8')
        if source.read_text(encoding='utf-8')!=proposal.code:raise ValueError('Candidate source changed')
        if (dest/'result.json').exists():value=read(dest/'result.json')
        else:
            try:
                admit(source,proposal.rank,dest);specs={};name=f'agent_head_{i+1:02d}'
                for fold in plan['folds'][1:]:
                    cfg={**fold['configs']['expanded'],'stage':'candidate','recipe':asdict(proposal),'head_code':str(source),
                        'template_worker':fold['configs']['expanded']['output_dir'],'output_dir':str(dest/'workers'/fold['name'])}
                    specs[name+'_'+fold['name']]=cfg
                dispatch(specs,root,a.parallel_jobs,plan['name']+f'_trial{i+1}','candidate')
                extra={}
                for fold in plan['folds'][1:]:
                    bundle=export_candidate(dest/'workers'/fold['name'],dest/'bundles'/fold['name'],
                        Registry(pools[fold['name']]).experts['expanded_factor_rank16'].path,name)
                    extra[fold['name']]={name:bundle}
                _,frame,ss=evidence_for(plan,data,root,pools,extra,suffix=f'trial{i+1}')
                value={'trial':i+1,'expert_id':name,'status':'evaluated','source_oof_mse':ss['fixed_source_mse'][name],
                    'code_sha256':file_sha256(source),'reason':proposal.reason,'outer_used':False}
            except Exception as exc:
                # Admission/training errors are explicit trial failures, never fictitious scores.
                value={'trial':i+1,'status':'failed','error_type':type(exc).__name__,'reason':str(exc),'outer_used':False}
            write_json(dest/'result.json',value)
        previous.append(value)
        if value['status']=='evaluated' and value['source_oof_mse']<best:
            best=value['source_oof_mse'];chosen={'trial':i+1,'expert_id':value['expert_id'],'source_oof_mse':best,
                'proposal':asdict(proposal),'head_code':str(source),'trial_dir':str(dest)}
    selection={'selected_candidate':chosen,'baseline_fixed':summary['best_fixed'],'baseline_mse':summary['fixed_source_mse'][summary['best_fixed']],
        'trials':previous,'provider_summary':planner.summary(),'criterion':'strictly lower macro-source-inner-OOF MSE; retain baseline on tie/worse',
        'limitation':'Adaptive inner reuse is optimistic; only untouched outer evaluates the complete search process','outer_used':False}
    write_json(folder/'selection.json',selection);seal(folder);return selection


def activate(plan,selection,data,root,pools,a):
    selected=selection['selected_candidate']
    if selected is None:return None
    final=plan['folds'][0];dest=root/'research'/plan['name']/'final_candidate'
    cfg={**final['configs']['expanded'],'recipe':selected['proposal'],'head_code':selected['head_code'],
        'template_worker':final['configs']['expanded']['output_dir'],'output_dir':str(dest/'worker')}
    dispatch({plan['name']+'_selected_head':cfg},root,a.parallel_jobs,plan['name']+'_final_candidate','candidate')
    bundle=export_candidate(dest/'worker',dest/'bundle',Registry(pools[final['name']]).experts['expanded_factor_rank16'].path,selected['expert_id'])
    registry=dest/'registry.json'
    if not registry.exists():
        old=Registry(pools[final['name']]);write_json(registry,{'schema_version':1,'experts':[
            {**r,'bundle':str(old.experts[r['expert_id']].path)} for r in old.config['experts']]+[
            {'expert_id':selected['expert_id'],'bundle':str(bundle),'cost_units':1.,'cost_source':'unmeasured'}]})
    return registry


def replay(a,plan,data,root,pool,profile,frame,summary,label):
    final=plan['folds'][0];registry=Registry(pool);gate=fit_gate(frame);records=[]
    source_q=query_path(data,plan['folds'][1]['query_rows'][0],root,final['configs']['expanded'])
    timings=[]
    for name,e in registry.experts.items():
        q=load_query(source_q)
        if not e.compatible(q)[0]:continue
        for repeat in range(2):
            tick=perf_counter();e.predict(q);timings.append({'expert_id':name,'measurement':'first_observed' if repeat==0 else 'repeat',
                'seconds':perf_counter()-tick,'note':'Model tensors reloaded each call; OS caches uncontrolled; source-control query only'})
    timing_path=root/'evaluation'/plan['name']/label;timing_path.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(timings).to_csv(timing_path/'inference_times.csv',index=False)
    live=a.live_routing and summary['oracle_relative_gap']>=a.oracle_gap and (summary['leave_one_target_out_gap_min'] or 0.)>0.
    policies=['fixed','rule','risk','workflow','softmax_gate']
    if live:policies += [f'{p}_{effort}' for p in ('one_shot','adaptive') for effort in a.routing_efforts]
    for row in final['query_rows']:
        qp=query_path(data,row,root,final['configs']['expanded']);q=load_query(qp);qid='q_'+digest(q.query_id)[:16]
        for policy in policies:
            out=root/'episodes'/plan['name']/label/qid/policy;planner=None
            if policy=='softmax_gate':planner=GatePlanner(gate,registry,q)
            elif policy.startswith(('one_shot_','adaptive_')):
                effort=policy.rsplit('_',1)[1];cls=OneShotPlanner if policy.startswith('one_shot') else DeepSeekPlanner
                planner=cls(root/'api'/plan['name']/label/qid/policy,a.model,reasoning_effort=effort,
                    max_api_calls=1 if cls is OneShotPlanner else a.routing_calls,max_output_tokens=a.max_output_tokens,timeout=a.api_timeout)
            result=execute(pool,qp,out,policy=policy if planner is None else 'rule',profile_path=profile,
                budget_units=3.1,max_steps=8,resume=a.resume,planner=planner,
                planner_identity=getattr(planner,'identity',None),formal_outer=True)
            # Decisions for EVERY query/policy are sealed before the private evaluator is entered.
            records.append({'query_id':q.query_id,'query_path':str(qp),'row':row,'policy':policy,'episode':str(out),
                            'selected_expert':result['selected_expert'],'result':result})
    freeze(timing_path/'decisions_sealed.json',{'episodes':{r['episode']:file_sha256(Path(r['episode'])/'COMPLETE.json') for r in records},
        'live_routing_executed':live,'source_oracle_gap':summary['oracle_relative_gap'],'threshold':a.oracle_gap,'no_outer_selection':True})
    return records


def evaluate_records(plan,records,data,root,label,pool):
    """Private evaluator called only after all source search and routing decisions."""
    from vcell.round13 import retrieval
    registry=Registry(pool);results=[];preds={};rows={}
    for r in records:
        out=Path(r['episode']);verify(out);q=load_query(r['query_path']);truth=data['delta'][r['row']]
        value=r['result'];provider=value.get('provider_summary',{})
        if value['selected_expert'] is None:
            results.append({'policy':r['policy'],'context':q.context,'target':q.target,'query_id':q.query_id,'available':False,
                'selected_expert':None,'mse':np.nan,'zero_mse':float(np.square(truth).mean())});continue
        prediction=load_prediction(out/'selected.npz',q);error=float(np.square(prediction-truth).mean())
        results.append({'policy':r['policy'],'context':q.context,'target':q.target,'query_id':q.query_id,'available':True,
            'selected_expert':value['selected_expert'],'mse':error,'zero_mse':float(np.square(truth).mean()),
            'predicted_norm':float(np.linalg.norm(prediction)),'true_norm':float(np.linalg.norm(truth)),
            'cosine_direction':float(prediction@truth/max(np.linalg.norm(prediction)*np.linalg.norm(truth),1e-12)),
            'tool_wall_seconds':sum(e['wall_seconds'] for e in read(out/'trace.json')),
            'api_seconds':provider.get('known_api_latency_seconds',0.),
            'end_to_end_this_session':value['wall_seconds_this_session'],'tool_units':value['spent_units'],
            'input_tokens':provider.get('known_input_tokens',0),'output_tokens':provider.get('known_output_tokens',0),
            'reasoning_tokens':provider.get('known_reasoning_tokens',0),'requests':provider.get('attempted_requests',0),
            'fit_target_seen':registry.experts[value['selected_expert']].features(q)['fit_target_seen'],
            'comparable_reference_available':any(e.card['kind']=='reference' and e.compatible(q)[0] for e in registry.experts.values())})
        preds.setdefault(r['policy'],[]).append(prediction);rows.setdefault(r['policy'],[]).append(r['row'])
    dest=root/'evaluation'/plan['name']/label;frame=pd.DataFrame(results);frame.to_csv(dest/'per_query.csv',index=False)
    context=frame.groupby(['policy','context','target'],dropna=False).agg(mse=('mse','mean'),coverage=('available','mean'),zero_mse=('zero_mse','mean')).reset_index()
    context.groupby(['policy','context']).agg(mse=('mse','mean'),coverage=('coverage','mean'),zero_mse=('zero_mse','mean')).reset_index().to_csv(dest/'comparison.csv',index=False)
    fixed=frame.loc[frame.policy.eq('fixed'),['query_id','mse','available']].rename(columns={'mse':'fixed_mse','available':'fixed_available'})
    pairs=frame.merge(fixed,on='query_id',validate='many_to_one')
    pairs['paired_available']=pairs.available & pairs.fixed_available
    pairs['mse_gain_vs_fixed']=np.where(pairs.paired_available,pairs.fixed_mse-pairs.mse,np.nan)
    pairs[['policy','context','target','query_id','paired_available','mse_gain_vs_fixed']].to_csv(dest/'paired_vs_fixed.csv',index=False)
    retrieval_rows=[]
    for p,pred in preds.items():
        ix=rows[p];retrieval_rows.extend(retrieval(data['meta'].iloc[ix].reset_index(drop=True),data['delta'][ix],np.array(pred),p))
    pd.DataFrame(retrieval_rows).to_csv(dest/'specificity.csv',index=False)
    # Matched separate reference/zero expert diagnostics are posthoc, never forwarded to planners.
    expert_rows=[]
    for row in plan['folds'][0]['query_rows']:
        qp=query_path(data,row,root,plan['folds'][0]['configs']['expanded']);q=load_query(qp)
        for name,e in registry.experts.items():
            available=e.compatible(q)[0]
            prediction=e.predict(q) if available else None
            if available:
                from vcell.model_agent.artifacts import save_prediction
                folder=dest/'private_predictions'/name;folder.mkdir(parents=True,exist_ok=True)
                save_prediction(folder/('q_'+digest(q.query_id)[:16]+'.npz'),q,prediction)
            expert_rows.append({'query_id':q.query_id,'context':q.context,'target':q.target,'expert_id':name,'available':available,
                'mse':float(np.square(prediction-data['delta'][row]).mean()) if available else np.nan})
    pd.DataFrame(expert_rows).to_csv(dest/'private_expert_scores.csv',index=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker-config')
    p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round17_oof_01')
    p.add_argument('--p4-run');p.add_argument('--parent-seed',type=int,default=17)
    p.add_argument('--outer-contexts',nargs='+',choices=DEVELOPMENT,default=['A549_IFNG'])
    p.add_argument('--inner-folds',type=int,choices=[2,3],default=2);p.add_argument('--targets-per-context',type=int,default=10)
    p.add_argument('--seeds',nargs='+',type=int,default=[17]);p.add_argument('--parallel-jobs',type=int,choices=[1,2,3,4],default=3)
    p.add_argument('--max-steps',type=int,default=2000);p.add_argument('--pretrain-steps',type=int)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--resume',action='store_true')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--auto-improve',action='store_true')
    p.add_argument('--trials',type=int,choices=[1,2,3],default=2);p.add_argument('--model',default=DEFAULT_MODEL)
    p.add_argument('--research-effort',choices=['low','high','max'],default='high');p.add_argument('--max-output-tokens',type=int,default=8192)
    p.add_argument('--api-timeout',type=float,default=180.);p.add_argument('--live-routing',action='store_true')
    p.add_argument('--routing-efforts',nargs='+',choices=['low','high','max'],default=['low','max'])
    p.add_argument('--routing-calls',type=int,default=5);p.add_argument('--oracle-gap',type=float,default=.01)
    a=p.parse_args(argv)
    if a.worker_config:
        for key in list(os.environ):
            if any(s in key.upper() for s in ('TOKEN','SECRET','KEY','PASSWORD','CREDENTIAL')):os.environ.pop(key,None)
        cfg=read(a.worker_config);stage=cfg.pop('stage')
        return pretrain(cfg,a.resume) if stage=='background' else candidate_worker(cfg,a.resume) if stage=='candidate' else worker(cfg,a.resume)
    safe_name(a.name)
    if min(a.max_steps,a.batch_size,a.save_every,a.targets_per_context)<1 or (a.pretrain_steps is not None and a.pretrain_steps<1):p.error('Positive training/query budgets required')
    if len(set(a.outer_contexts))!=len(a.outer_contexts) or len(set(a.seeds))!=len(a.seeds) or min(a.seeds)<0:p.error('Unique contexts/seeds required')
    if not 0<=a.oracle_gap<1 or not 1<=a.routing_calls<=8:p.error('Invalid routing gate/budget')
    work=Path(a.work_dir).resolve();root=work/'runs'/a.name
    with run_lock(work/'runs'/('.'+a.name+'.lock')):return launch(a,root,work)


def launch(a,root,work):
    parent=Path(a.p4_run or work/'runs/round15_p4_02').resolve();verify_files(parent)
    plan={'experiment':'nested_source_oof_with_bounded_head_search','source':code_identity(),'launcher':file_sha256(__file__),
        'parent':str(parent),'parent_complete':file_sha256(parent/'COMPLETE.json'),
        **{k:v for k,v in vars(a).items() if k not in ('resume','prepare_only','worker_config')},
        'scope':'Cell-line-held development benchmark; shared public primary-cell priors; HT29/Jurkat sealed',
        'test_evaluated':False,'code_scope':'bounded CandidateHead in isolated run directories; no repository edits'}
    if root.exists() and any(root.iterdir()) and not a.resume:raise ValueError('Use --resume or new name')
    root.mkdir(parents=True,exist_ok=True);freeze(root/'plan.json',plan)
    if (root/'COMPLETE.json').exists():verify(root);print('ROUND17 COMPLETE VERIFIED:',root);return root
    def stage(s):write_json(root/'stage.json',{'stage':s,'test_evaluated':False});print('ROUND17 STAGE:',s,flush=True)
    try:
        stage('freeze nested cell-line folds, target holdout and lineage')
        data,plans,bs,ws=prepared_inputs(a,parent,root)
        freeze(root/'budget.json',{'background_fits':len(bs),'numerical_heads':len(ws)*3,'parallel_jobs':a.parallel_jobs,
            'max_research_calls':len(plans)*a.trials if a.auto_improve else 0,'max_routing_calls_per_query_per_pool':
            len(a.routing_efforts)*(1+a.routing_calls) if a.live_routing else 0,
            'encoder':'Source-trained Qwen-sized expression projector; NOT pretrained Qwen language-model inference',
            'candidate_budget':'At most 3 trials, <=200k head parameters, fixed source folds/steps/batch'})
        if a.prepare_only:stage('prepared; no training or API calls');return root
        stage('source-only B2 background refits');dispatch(bs,root,a.parallel_jobs,'background','background')
        stage('six numerical experts per outer/inner fold');dispatch(ws,root,a.parallel_jobs,'experts','expert')
        pools={f['name']:pool_for(f,root) for pl in plans for f in pl['folds']};pending=[]
        for pl in plans:
            stage('inner OOF profiles and complementarity: '+pl['name'])
            profile,frame,summary=evidence_for(pl,data,root,pools)
            final_pool=pools[pl['folds'][0]['name']]
            pending.append((pl,'baseline_pool',final_pool,profile,frame,summary))
            if a.auto_improve:
                stage('isolated source-only head proposals: '+pl['name'])
                selected=improve(a,pl,data,root,pools,summary)
                improved=activate(pl,selected,data,root,pools,a)
                if improved:
                    i=selected['selected_candidate']['trial'];profile2=root/'evidence'/pl['name']/f'trial{i}'/'profiles';verify(profile2)
                    f2=pd.read_csv(profile2/'private_oof_scores.csv');pending.append((pl,'improved_pool',improved,profile2,f2,source_summary(f2)))
        # Freeze ALL search decisions across folds before any outer truth evaluation.
        stage('paired routing on sealed outer queries')
        decisions=[]
        for pl,label,pool,profile,frame,summary in pending:
            records=replay(a,pl,data,root,pool,profile,frame,summary,label);decisions.append((pl,label,pool,records))
        freeze(root/'search_and_routing_sealed.json',{'pools':[{'fold':pl['name'],'pool':label,'registry':Registry(pool).fingerprint}
            for pl,label,pool,_ in decisions],'outer_scores_used_for_search':False})
        stage('private outer evaluator; no further model or policy selection')
        for pl,label,pool,records in decisions:evaluate_records(pl,records,data,root,label,pool)
        comparisons=[]
        for pl,label,_,_ in decisions:
            df=pd.read_csv(root/'evaluation'/pl['name']/label/'comparison.csv');df['seed']=pl['seed'];df['pool']=label;comparisons.append(df)
        comparison=pd.concat(comparisons,ignore_index=True);comparison.to_csv(root/'comparison.csv',index=False)
        # Average optimizer seeds before the macro-background comparison.
        per_context=comparison.groupby(['pool','policy','context']).agg(mse=('mse','mean'),coverage=('coverage','mean'),zero_mse=('zero_mse','mean')).reset_index()
        per_context.groupby(['pool','policy']).agg(macro_mse=('mse','mean'),coverage=('coverage','mean'),zero_mse=('zero_mse','mean')).reset_index().to_csv(root/'comparison_macro.csv',index=False)
        html='<html><meta charset="utf-8"><title>Nested source OOF agent evaluation</title><body><h1>Development cell-line holdout</h1>'
        html+='<p>Source-only search and routing sealed before private outer scores. MSE is conditional on availability; compare coverage and paired contrasts. Seeds are optimizer repetitions, not biological replicates. HT29/Jurkat were not evaluated.</p>'
        html+=comparison.to_html(index=False,escape=True)+'</body></html>'
        (root/'report.html').write_text(html,encoding='utf-8')
        write_json(root/'summary.json',{'outer_contexts':a.outer_contexts,'optimizer_seeds':a.seeds,
            'no_significance_claim':True,'test_evaluated':False,'source_search_frozen_before_outer_scores':True,
            'interpretation':'Compare pools for predictor improvements; compare policies within identical pools for routing gains'})
        (root/'INCOMPLETE.json').unlink(missing_ok=True);stage('complete');seal(root,test_evaluated=False)
        print('ROUND17 COMPLETE:',root/'comparison.csv',flush=True)
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error_type':type(exc).__name__,'error':str(exc),'test_evaluated':False});raise
    finally:print('REVIEW PACKAGE:',package(root),flush=True)
    return root


if __name__=='__main__':main()
