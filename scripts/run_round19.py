"""100 source-only candidate experiments, dual scientists and a local DeepSeek predictor."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import numpy as np
import pandas as pd
import torch
from vcell.clean_background import run_lock
from vcell.data import load_prepared
from vcell.model_agent.artifacts import read,arrays,seal,verify
from vcell.model_agent.contracts import digest,safe_name
from vcell.model_agent.deepseek import DEFAULT_MODEL
from vcell.round11 import freeze
from vcell.round19.agents import TrainingAgent,IterationAgent,DEFAULT_RECIPE,TEACHERS,validate_recipe,fixed_recipes
from vcell.round19.cache import build_cache,source_lines
from vcell.round19.predictor import train_worker
from vcell.utils import write_json,file_sha256
from prepare_qwen import prepare,matches
import run_round17_oof as review_tools
import round19_recovery as recovery


def identity():
    paths=[Path(__file__),Path(recovery.__file__),REPO/'scripts/prepare_qwen.py',*sorted((REPO/'src/vcell/round19').glob('*.py'))]
    return digest({p.relative_to(REPO).as_posix():file_sha256(p) for p in paths})


def worker_environment():
    env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL'))}
    env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    return env


def dispatch(specs,root,parallel):
    """Separate processes and RNG; exact worker resume is verified by the worker."""
    pending=list(specs);active={};failed=[]
    (root/'logs').mkdir(parents=True,exist_ok=True)
    try:
        while pending or active:
            while pending and len(active)<parallel:
                name=pending.pop(0);cfg=specs[name];out=Path(cfg['output_dir'])
                # Completed output must match the immutable worker config.
                if (out/'COMPLETE.json').exists():
                    verify(out)
                    if read(out/'manifest.json')['config']!=cfg:raise ValueError('Completed worker config changed')
                    continue
                path=root/'worker_configs'/(name+'.json');freeze(path,cfg)
                log=(root/'logs'/(name+'.log')).open('a',encoding='utf-8')
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker-config',str(path)],
                    env=worker_environment(),stdout=log,stderr=subprocess.STDOUT)
                active[name]=(proc,log);print('ROUND19 WORKER START',name,proc.pid,flush=True)
            for name,(proc,log) in list(active.items()):
                if proc.poll() is not None:
                    code=proc.returncode;log.close();del active[name]
                    if code or not (Path(specs[name]['output_dir'])/'COMPLETE.json').exists():failed.append(name)
                    print('ROUND19 WORKER EXIT',name,'code=',code,flush=True)
            write_json(root/'worker_progress.json',{'active':list(active),'pending':pending,'failed':failed})
            if active:time.sleep(1)
    finally:
        for proc,log in active.values():
            if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
            log.close()
    if failed:raise RuntimeError('Round19 workers failed; inspect logs and resume: '+','.join(failed))


def worker_configs(a,recipe,caches,root,label,models):
    if recipe['backbone'] not in models:raise ValueError('Recipe requests an unavailable backbone')
    return {label+'_'+name:{'cache':str(path),'recipe':recipe,'model':models[recipe['backbone']],
        'output_dir':str(root/'workers'/label/name),'device':a.device,'seed':a.seed,
        'max_steps':a.max_steps,'batch_size':a.batch_size,'save_every':a.save_every}
        for name,path in caches.items()}


def score_rows(pred,truth,meta,method):
    error=np.square(pred-truth).mean(1);zero=np.square(truth).mean(1)
    norm=np.linalg.norm(pred,axis=1);tn=np.linalg.norm(truth,axis=1)
    cosine=(pred*truth).sum(1)/np.maximum(norm*tn,1e-12)
    # Other targets only within identical measured conditions.
    keys=[k for k in ('context','dataset','stimulus','time','modality','batch') if k in meta]
    groups=meta[keys].astype(str).agg('|'.join,axis=1).to_numpy();target=meta.perturbation.to_numpy()
    similarities=(pred@truth.T)/np.maximum(norm[:,None]*tn[None,:],1e-12)
    specificity=[]
    for i in range(len(meta)):
        negative=(groups==groups[i])&(target!=target[i])&(tn>1e-8)
        specificity.append(float(cosine[i]-similarities[i,negative].max()) if negative.any() and tn[i]>1e-8 else None)
    return pd.DataFrame({'row_id':meta.row_id.to_numpy(),'context':meta.context.to_numpy(),'target':target,
        'method':method,'mse':error,'zero_mse':zero,'prediction_norm':norm,'truth_norm':tn,
        'cosine':cosine,'specificity_gap':specificity})


def source_scores(data,caches,specs,method):
    frames=[]
    for name,path in caches.items():
        cfg=next(c for c in specs.values() if c['cache']==str(path));out=Path(cfg['output_dir']);verify(out)
        d=arrays(path/'features.npz');p=arrays(out/'prediction.npz');rows=d['query'].astype(int)
        if not np.array_equal(p['row_ids'],data['meta'].iloc[rows].row_id.to_numpy(dtype='U')) or not np.array_equal(p['genes'],data['genes']):raise ValueError('Prediction alignment changed')
        frames.append(score_rows(p['prediction'],data['delta'][rows],data['meta'].iloc[rows].reset_index(drop=True),method))
    frame=pd.concat(frames,ignore_index=True)
    units=frame.groupby(['context','target']).agg(mse=('mse','mean'),zero=('zero_mse','mean'),norm=('prediction_norm','mean'),truth_norm=('truth_norm','mean'),specificity=('specificity_gap','mean'))
    contexts=units.groupby('context').mean();score=float(contexts.mse.mean())
    summary={'source_mse':score,'zero_mse':float(contexts.zero.mean()),'contexts':{},
        'source_distinct_targets':int(frame.target.nunique()),'source_query_rows':len(frame),'outer_used':False}
    for c,row in contexts.iterrows():
        summary['contexts'][c]={'mse':float(row.mse),'zero_mse':float(row.zero),
            'prediction_to_truth_norm':float(row.norm/max(row.truth_norm,1e-12)),
            'specificity_gap':float(row.specificity) if np.isfinite(row.specificity) else None}
    return summary,frame


def source_baselines(data,caches):
    frames=[]
    for cache in caches.values():
        d=arrays(cache/'features.npz');rows=d['query'].astype(int);meta=data['meta'].iloc[rows].reset_index(drop=True)
        p=arrays(cache/'teacher_query_predictions.npz')['prediction']
        for j,name in enumerate(TEACHERS):frames.append(score_rows(p[:,j],data['delta'][rows],meta,name))
        frames.append(score_rows(np.zeros_like(data['delta'][rows]),data['delta'][rows],meta,'zero'))
        frames.append(score_rows(p.mean(1),data['delta'][rows],meta,'teacher_mean'))
        # Isolate gains due simply to projecting the teacher average into the
        # student's source-fit response basis, before any language backbone.
        for requested in (16,32):
            rank=min(requested,len(d['basis']))
            projected=d['teachers'][rows].mean(1)[:,:rank]@d['basis'][:rank]*float(d['response_scale'])
            frames.append(score_rows(projected,data['delta'][rows],meta,'teacher_mean_rank'+str(requested)))
    f=pd.concat(frames,ignore_index=True)
    units=f.groupby(['method','context','target']).mse.mean().groupby(['method','context']).mean().groupby('method').mean()
    return units.to_dict(),f


def propose_random(index,seed,available):
    rng=np.random.default_rng(seed+index*193)
    chosen=[name for name in TEACHERS if rng.random()>.4]
    backbone=str(rng.choice(['deepseek','mlp']))
    return validate_recipe({**DEFAULT_RECIPE,'backbone':backbone,
        'adaptation':str(rng.choice(['frozen','lora'])) if backbone!='mlp' else 'frozen',
        'teachers':chosen,'structure':str(rng.choice(['direct','teacher_residual','gate'])) if chosen else 'direct',
        'rank':int(rng.choice([16,32])),'width':int(rng.choice([32,64,128])),
        'learning_rate':float(rng.choice([.0001,.0003,.0005,.001])),'use_string':bool(rng.integers(2)),
        'specificity_weight':float(rng.choice([0.,.03,.1])),'kd_weight':float(rng.choice([0.,.1,.3])) if chosen else 0.})


def model_specs(a,work):
    lock=read(REPO/'configs/deepseek_predictor.lock.json');folder=Path(a.model_dir or work/'models/DeepSeek-R1-Distill-Qwen-1.5B')
    prepare(folder,a.endpoint,lock=lock,label='DEEPSEEK PREDICTOR')
    spec={'model_dir':str(folder.resolve()),'dtype':'bfloat16' if a.device=='cuda' else 'float32',
        'identity':{'model_id':lock['model_id'],'revision':lock['revision'],'files':lock['files']}}
    models={'deepseek':spec,'mlp':spec}
    if a.qwen_model_dir:
        qlock=read(REPO/'configs/qwen_backbone.lock.json');qfolder=Path(a.qwen_model_dir)
        if not all(matches(qfolder/name,item) for name,item in qlock['files'].items()):raise ValueError('Qwen control snapshot is incomplete or changed')
        models['qwen']={**spec,'model_dir':str(qfolder.resolve()),'identity':qlock}
    return models


def run(a):
    work=Path(a.work_dir).resolve();root=work/'runs'/a.name;parent=Path(a.source_run or work/'runs/round18_pilot_01').resolve()
    verify(parent);inputs=read(parent/'inputs.json')
    plans=[p for p in inputs['folds'] if p['outer_context']==a.outer_context and p['seed']==a.seed]
    if len(plans)!=1 or len(plans[0]['folds'])!=4:raise ValueError('Completed source run must contain one matched outer and three inner folds')
    pl=plans[0];data=load_prepared(inputs['data_dir'])
    if data['audit']['fingerprint']!=inputs['data_fingerprint']:raise ValueError('Source data changed')
    if root.exists() and any(root.iterdir()) and not a.resume:raise ValueError('Use --resume or a new name')
    root.mkdir(parents=True,exist_ok=True)
    data_dir=Path(inputs['data_dir'])
    plan={'experiment':'round19_dual_agent_local_predictor','code':identity(),'parent_complete':file_sha256(parent/'COMPLETE.json'),
        'dataset_sha256':file_sha256(data_dir/'dataset.npz'),'metadata_sha256':file_sha256(data_dir/'metadata.csv'),
        **{k:v for k,v in vars(a).items() if k not in ('resume','prepare_only','repair_truncated_responses')},
        'truncation_recovery':{'version':1,'model_namespace':recovery.namespace_identity(),
            'scope':'length-truncated API responses consume attempts; no retry or partial decision execution'},
        'test_evaluated':False,'outer_role':'Previously observed development cell line, not independent external test'}
    recovery.freeze_plan(root,plan,getattr(a,'repair_truncated_responses',False),a.resume)
    if (root/'COMPLETE.json').exists():verify(root);print('ROUND19 COMPLETE VERIFIED',root);return root
    def stage(value,**details):write_json(root/'stage.json',{'stage':value,'test_evaluated':False,**details});print('ROUND19 STAGE:',value,details,flush=True)
    budget={'candidate_rounds':a.iterations,'training_agent_requests_max':a.iterations if a.search_mode!='random' else 0,
        'review_agent_requests_max':a.iterations if a.search_mode=='dual' else 0,
        'candidate_inner_fits_max':a.iterations*3,'steps_per_fit':a.max_steps,
        'candidate_optimizer_updates_max':a.iterations*3*a.max_steps,'parallel_jobs':a.parallel_jobs,
        'outer_evaluation':'Only after all candidate search and final recipes freeze',
        'invalid_and_duplicate_trials':'Count as attempted rounds; no unbounded retries; successful fit count reported separately',
        'currency_cost':'Not estimated; actual request, token, time usage retained'}
    freeze(root/'budget.json',budget)
    if a.prepare_only:stage('prepared; no API, download or training');return root
    try:
        if a.search_mode!='random' and not os.environ.get('DEEPSEEK_API_KEY','').strip():raise RuntimeError('Set DEEPSEEK_API_KEY in the server environment')
        stage('verify/download pinned local backbone');models=model_specs(a,work);freeze(root/'models.json',models)
        stage('nested teacher response-head cross-fitting and feature caches')
        caches={}
        for fold in pl['folds'][1:]:
            pool=parent/'pools'/fold['name']/'registry_with_zero.json'
            cache=root/'caches'/fold['name'];build_cache(data,fold,pool,cache,a.teacher_steps,models['deepseek']['identity']);caches[fold['name']]=cache
        baseline,baseline_frame=source_baselines(data,caches);baseline_frame.to_csv(root/'source_numerical_baselines.csv',index=False)
        controls=fixed_recipes()
        if 'qwen' in models:controls['qwen_matched_fusion']={**DEFAULT_RECIPE,'backbone':'qwen'}
        control_summary={};control_specs={}
        stage('fixed predictor and same-input controls')
        for name,recipe in controls.items():
            specs=worker_configs(a,recipe,caches,root,name,models);dispatch(specs,root,a.parallel_jobs)
            result,frame=source_scores(data,caches,specs,name);frame.to_csv(root/(name+'_source_scores.csv'),index=False)
            control_summary[name]=result;control_specs[name]=recipe
        freeze(root/'fixed_source_summary.json',{'numerical':baseline,'predictors':control_summary})
        training=TrainingAgent(root/'api/training',a.agent_model,reasoning_effort=a.effort,max_api_calls=a.iterations,
            max_output_tokens=a.max_output_tokens,timeout=a.api_timeout) if a.search_mode!='random' else None
        reviewer=IterationAgent(root/'api/review',a.agent_model,reasoning_effort=a.effort,max_api_calls=a.iterations,
            max_output_tokens=a.max_output_tokens,timeout=a.api_timeout) if a.search_mode=='dual' else None
        history=[];seen={digest(r) for r in controls.values()};best=None
        # Source-only abstract context; never read the parent's outer comparisons.
        papers=read(parent/'literature_snapshot.json').get('papers',[]) if (parent/'literature_snapshot.json').exists() else []
        for i in range(1,a.iterations+1):
            folder=root/'trials'/f'trial_{i:03d}';folder.mkdir(parents=True,exist_ok=True)
            stage('training agent proposal and source OOF',iteration=i,total=a.iterations)
            obs={'iteration':i,'max_iterations':a.iterations,'available_backbones':list(models),
                'available_teachers':list(TEACHERS),'starting_recipe':DEFAULT_RECIPE,
                'fixed_source_results':{'numerical':baseline,'predictors':control_summary},
                'recent_source_experiments':history[-8:],'best_source_experiment':best,
                'literature_abstracts':papers[:4],
                'fixed_steps':a.max_steps,'outer_results_available':False}
            freeze(folder/'training_observation.json',obs)
            if (folder/'decision.json').exists():decision=read(folder/'decision.json')
            else:
                if training:
                    response,failure=recovery.call(training,obs,'training');decision=asdict(response) if response else {'status':'rejected','validation':failure}
                    if failure:print('ROUND19 TRAINING DECISION REJECTED',i,failure,flush=True)
                else:decision={'tool':'train','expert_id':None,'reason':'budget-matched seeded random recipe','recipe':propose_random(i,a.seed,models)}
                freeze(folder/'decision.json',decision)
            if (folder/'result.json').exists():result=read(folder/'result.json')
            else:
                result={'iteration':i,'status':'rejected','outer_used':False,'reason':'Invalid training proposal','validation':decision.get('validation')}
                if 'recipe' in decision:
                    recipe=validate_recipe(decision['recipe']);signature=digest(recipe)
                    if recipe['backbone'] not in models:result['reason']='Unavailable backbone'
                    elif signature in seen:result['reason']='Duplicate complete recipe; no repeated training'
                    else:
                        specs=worker_configs(a,recipe,caches,root,'trial_'+str(i),models)
                        # Worker failures pause, preserving step checkpoints; never mark them complete.
                        dispatch(specs,root,a.parallel_jobs)
                        summary,frame=source_scores(data,caches,specs,'trial_'+str(i));frame.to_csv(folder/'source_scores.csv',index=False)
                        result={'iteration':i,'status':'evaluated','recipe':recipe,**summary}
                freeze(folder/'result.json',result)
            if 'recipe' in decision:seen.add(digest(decision['recipe']))
            if result['status']=='evaluated' and result['recipe']['backbone']=='deepseek' and (best is None or result['source_mse']<best['source_mse']):best=result
            stage('iteration agent source-only review',iteration=i,total=a.iterations)
            review_obs={'iteration':i,'experiment':result,'training_rationale':decision.get('reason'),
                'best_source_experiment':best,'fixed_source_results':{'numerical':baseline,'predictors':control_summary},
                'allowed_recipe':DEFAULT_RECIPE,'outer_results_available':False}
            freeze(folder/'review_observation.json',review_obs)
            if not (folder/'review.json').exists():
                if reviewer:
                    response,failure=recovery.call(reviewer,review_obs,'review');rev=asdict(response) if response else {'status':'rejected','validation':failure}
                    if failure:print('ROUND19 REVIEW UNAVAILABLE',i,failure,flush=True)
                else:rev={'role':'no_second_agent','next_hypothesis':None}
                freeze(folder/'review.json',rev)
            history.append({'result':result,'review':read(folder/'review.json')})
            seal(folder);write_json(root/'iteration_progress.json',{'completed_rounds':i,'total_rounds':a.iterations,
                'evaluated_candidates':sum(r['result']['status']=='evaluated' for r in history),'best_source':best,
                'completed_reviews':sum(r['review'].get('tool')=='review' for r in history),
                'missing_review_rounds':sum(r['review'].get('status')=='rejected' for r in history),
                'training_api':training.summary() if training else None,'review_api':reviewer.summary() if reviewer else None})
            print('ROUND19 ITERATION COMPLETE',i,'/',a.iterations,result['status'],flush=True)
        source_candidates={**baseline,**{name:r['source_mse'] for name,r in control_summary.items()}}
        if best:source_candidates['searched_deepseek']=best['source_mse']
        deployment_choice=min(source_candidates,key=lambda name:(source_candidates[name],name))
        selection={'best_deepseek_source_candidate':best,'fixed_source_results':control_summary,
            'source_deployment_choice':deployment_choice,'source_deployment_mse':source_candidates[deployment_choice],
            'searched_deepseek_beats_best_fixed':bool(best and best['source_mse']<min(v for k,v in source_candidates.items() if k!='searched_deepseek')),
            'candidate_rounds':a.iterations,'evaluated_candidates':sum(r['result']['status']=='evaluated' for r in history),
            'selection_rule':'Minimum source macro MSE among valid local DeepSeek candidates; no outer scores',
            'outer_used':False,'no_significance_claim':True}
        freeze(root/'search_frozen.json',selection)
        stage('source-selected final refits; outer responses still sealed')
        final=pl['folds'][0];outer_cache=root/'caches'/final['name'];build_cache(data,final,parent/'pools'/final['name']/'registry_with_zero.json',outer_cache,a.teacher_steps,models['deepseek']['identity'])
        final_recipes=dict(control_specs)
        if best:final_recipes['dual_agent_selected' if a.search_mode=='dual' else a.search_mode+'_selected']=best['recipe']
        final_specs={}
        for name,recipe in final_recipes.items():final_specs.update(worker_configs(a,recipe,{final['name']:outer_cache},root,'final_'+name,models))
        dispatch(final_specs,root,a.parallel_jobs)
        freeze(root/'final_decisions_sealed.json',{'source_selection_sha256':file_sha256(root/'search_frozen.json'),
            'final_recipes':final_recipes,'worker_completions':{n:file_sha256(Path(c['output_dir'])/'COMPLETE.json') for n,c in final_specs.items()},'outer_used':False})
        stage('private outer evaluation; no further search')
        frames=[]
        for name,recipe in final_recipes.items():
            specs={n:c for n,c in final_specs.items() if n=='final_'+name+'_'+final['name']}
            _,frame=source_scores(data,{final['name']:outer_cache},specs,name);frames.append(frame)
        _,numerical=source_baselines(data,{final['name']:outer_cache});frames.append(numerical)
        frame=pd.concat(frames,ignore_index=True);frame.to_csv(root/'outer_per_query.csv',index=False)
        units=frame.groupby(['method','context','target']).agg(mse=('mse','mean'),zero=('zero_mse','mean'),specificity=('specificity_gap','mean'),prediction_norm=('prediction_norm','mean'),truth_norm=('truth_norm','mean'))
        comparison=units.groupby(['method','context']).mean().reset_index();comparison.to_csv(root/'comparison.csv',index=False)
        write_json(root/'summary.json',{'rounds':a.iterations,'search_mode':a.search_mode,'selected':best,
            'evaluated_candidates':selection['evaluated_candidates'],'source_run':str(parent),
            'held_development_context':a.outer_context,'seed':a.seed,'pretrained_predictor_model':'DeepSeek-R1-Distill-Qwen-1.5B',
            'api_model_weights_updated':False,'student_weights_updated':True,'test_evaluated':False,
            'backbone_pretraining_overlap':'Not independently audited; controlled source supervision is not a guarantee about public language pretraining',
            'external_native_teachers':'NOT_CONNECTED; no public checkpoint assumed leakage-free',
            'qwen_control':'matched pretrained Qwen fusion' if 'qwen' in models else 'NOT_RUN; B2 projector is not a Qwen language-model control',
            'training_agent':training.summary() if training else None,'iteration_agent':reviewer.summary() if reviewer else None,
            'missing_review_rounds':sum(r['review'].get('status')=='rejected' for r in history),
            'search_vs_controls':'Single/random equal-round search requires separate --search-mode runs; fixed controls alone cannot prove dual-agent superiority'})
        (root/'report.html').write_text('<html><meta charset="utf-8"><h1>Round19 local predictor development benchmark</h1><p>100 rounds are source-only candidate attempts. Outer scores are read after search and final refits freeze. API scientists are not fine-tuned. Native State ST/TxPert are not connected.</p>'+comparison.to_html(index=False,escape=True)+'</html>',encoding='utf-8')
        (root/'INCOMPLETE.json').unlink(missing_ok=True);stage('complete');seal(root,test_evaluated=False)
        print('ROUND19 COMPLETE:',root/'comparison.csv',flush=True)
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error_type':type(exc).__name__,'error':str(exc),'test_evaluated':False});raise
    finally:
        path=review_tools.package(root);path.replace(root/'round19_review_light.tar.gz');print('REVIEW PACKAGE:',root/'round19_review_light.tar.gz',flush=True)
    return root


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-config')
    p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round19_dual_100_01')
    p.add_argument('--source-run');p.add_argument('--outer-context',default='BXPC3_IFNG');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--iterations',type=int,default=100);p.add_argument('--search-mode',choices=['dual','single','random'],default='dual')
    p.add_argument('--max-steps',type=int,default=500);p.add_argument('--teacher-steps',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--save-every',type=int,default=50)
    p.add_argument('--parallel-jobs',type=int,choices=[1,2,3],default=3);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--model-dir');p.add_argument('--qwen-model-dir');p.add_argument('--endpoint',default=os.environ.get('HF_ENDPOINT','https://hf-mirror.com'))
    p.add_argument('--agent-model',default=DEFAULT_MODEL);p.add_argument('--effort',choices=['low','high','max'],default='high')
    p.add_argument('--max-output-tokens',type=int,default=8192);p.add_argument('--api-timeout',type=float,default=180)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--repair-truncated-responses',action='store_true',help='Audit a supported old source-search run and retain completed fits/paid calls')
    a=p.parse_args(argv)
    if a.worker_config:return train_worker(read(a.worker_config),True)
    safe_name(a.name)
    if a.repair_truncated_responses and not a.resume:p.error('Recovery requires --resume')
    if not 1<=a.iterations<=100 or min(a.max_steps,a.teacher_steps,a.batch_size,a.save_every,a.max_output_tokens)<1 or a.api_timeout<=0 or a.seed<0:p.error('Invalid bounded experiment budgets')
    with run_lock(Path(a.work_dir)/'runs'/('.'+a.name+'.lock')):return run(a)


if __name__=='__main__':main()
