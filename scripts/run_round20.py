"""Round20: measured signal, paired ablations, bounded experimental reference and gated agent search."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import numpy as np
import pandas as pd
from vcell.clean_background import run_lock
from vcell.data import load_prepared
from vcell.model_agent.artifacts import arrays,read,seal,verify
from vcell.model_agent.contracts import digest
from vcell.model_agent.nested import DEVELOPMENT,holdout_targets
from vcell.model_agent.candidate_code import BASE_HEAD
from vcell.model_agent.research import parse_proposal
from vcell.round11 import freeze
from vcell.round19.agents import DEFAULT_RECIPE
from vcell.round19.cache import build_cache
from vcell.round19.predictor import train_worker
from vcell.round20.reference import worker as reference_worker,references
from vcell.round20.signal import ALPHAS,fit_signal,replicate_qc,exposure_weights,macro,summarize,promotion
from vcell.round20.agents import SearchAgent,ShortReviewer
from vcell.utils import write_json,file_sha256
import run_round17_oof as core
import run_round18 as source
import run_round19 as previous
import round19_recovery as recovery

DIRECT={**DEFAULT_RECIPE,'teachers':[],'use_string':False,'structure':'direct','rank':32,'width':128,
    'learning_rate':5e-5,'weight_decay':.05,'loss':'mse','kd_weight':0.,'specificity_weight':.1}
ARMS=('fixed','random','single','dual')
INFORMATIVE_CONTROLS={'direct_mlp_string','direct_pretrained','direct_pretrained_string','direct_random_frozen',
    'direct_scale_0.2','direct_scale_1.0','direct_scale_learned','matched','correction_only'}


def recovery_gate(results,baseline):
    # Negative retrieval/text controls cannot justify a biological search branch.
    return any(promotion(result,baseline) for method,result in results.items() if method in INFORMATIVE_CONTROLS)


def verify_models(models):
    lock=read(REPO/'configs/deepseek_predictor.lock.json');spec=models['deepseek'];identity=spec['identity']
    if identity!={k:lock[k] for k in ('model_id','revision','files')}:raise ValueError('Round19 backbone identity differs from the pinned DeepSeek snapshot')
    if not all(previous.matches(Path(spec['model_dir'])/name,item) for name,item in lock['files'].items()):
        raise ValueError('Local DeepSeek snapshot changed or is incomplete; restore the pinned Round19 files')


def dispatch(specs,root,parallel):
    pending=list(specs);active={};failed=[];(root/'logs').mkdir(parents=True,exist_ok=True)
    try:
        while pending or active:
            while pending and len(active)<parallel:
                name=pending.pop(0);cfg=specs[name];out=Path(cfg['output_dir'])
                if (out/'COMPLETE.json').exists():
                    verify(out)
                    if read(out/'manifest.json')['config']!=cfg:raise ValueError('Completed worker configuration changed')
                    continue
                path=root/'worker_configs'/(name+'.json');freeze(path,cfg)
                stream=(root/'logs'/(name+'.log')).open('a',encoding='utf-8')
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker-config',str(path)],
                    env=previous.worker_environment(),stdout=stream,stderr=subprocess.STDOUT)
                active[name]=(proc,stream);print('ROUND20 WORKER START',name,proc.pid,flush=True)
            for name,(proc,stream) in list(active.items()):
                if proc.poll() is not None:
                    stream.close();del active[name]
                    if proc.returncode or not (Path(specs[name]['output_dir'])/'COMPLETE.json').exists():failed.append(name)
                    print('ROUND20 WORKER EXIT',name,'code=',proc.returncode,flush=True)
            write_json(root/'worker_progress.json',{'active':list(active),'pending':pending,'failed':failed})
            if active:time.sleep(1)
    finally:
        for proc,stream in active.values():
            if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
            stream.close()
    if failed:raise RuntimeError('Worker failed; resume identical command after inspecting '+','.join(failed))


def score(data,cache,pred,method,strong,available):
    d=arrays(cache/'features.npz');ix=d['query'].astype(int);truth=data['delta'][ix]
    f=previous.score_rows(pred,truth,data['meta'].iloc[ix].reset_index(drop=True),method)
    f['strong_mse']=np.square(pred[:,strong]-truth[:,strong]).mean(1) if strong.any() else np.nan
    f['reference_available']=available;f['globally_unknown']=f.target.isin(holdout_targets(data['meta']))
    for k in ('dataset','stimulus','time','modality','batch','replicate'):
        if k in data['meta']:f[k]=data['meta'].iloc[ix][k].to_numpy()
    return f


def availability(data,cache,held):
    d=arrays(cache/'features.npz');s=arrays(cache/'source_labels.npz')
    return references(data['meta'],s['fit'],s['fit_delta'],d['control'],held)[2][d['query']]


def configs(a,caches,root,label,method,models,metadata,held,alpha=1.,proposal=None):
    specs={}
    for name,path in caches.items():
        base={'cache':str(path),'output_dir':str(root/'workers'/label/name),'seed':a.seed,'device':a.device,
            'max_steps':a.max_steps,'batch_size':a.batch_size,'save_every':a.save_every}
        if method.startswith('direct_'):
            model=dict(models['deepseek']);recipe=dict(DIRECT)
            if method in ('direct_mlp','direct_mlp_string'):recipe['backbone']='mlp'
            if method.endswith('_string'):recipe['use_string']=True
            if method=='direct_random_frozen':model['random_frozen']=True
            if method=='direct_shuffled_target':model['prompt_mode']='shuffled_target'
            if method.startswith('direct_scale_'):model['correction_scale']=method.removeprefix('direct_scale_') if method.endswith('learned') else float(method.removeprefix('direct_scale_'))
            weights=root/'exposure'/(name+'.npz')
            base.update(kind='direct',recipe=recipe,model=model,exposure_weights=str(weights))
        else:
            base.update(kind='reference',metadata=str(metadata),held_targets=held,mode=method,alpha=float(alpha),
                radius=.2,learning_rate=.0005,weight_decay=.01,loss='mse',rank=32,code=BASE_HEAD)
            if proposal:base.update({k:proposal[k] for k in ('code','rank','learning_rate','weight_decay','loss')})
        specs[label+'_'+name]=base
    return specs


def evaluate_workers(data,caches,specs,method,masks,held):
    frames=[]
    for name,path in caches.items():
        cfg=next(c for c in specs.values() if c['cache']==str(path));out=Path(cfg['output_dir']);verify(out)
        p=arrays(out/'prediction.npz');d=arrays(path/'features.npz')
        if not np.array_equal(p['genes'],data['genes']) or not np.array_equal(p['row_ids'],d['row_ids'][d['query']]):raise ValueError('Prediction row/gene alignment changed')
        f=score(data,path,p['prediction'],method,masks[name],p.get('reference_available',availability(data,path,held)))
        f['fold']=name;frames.append(f)
    return pd.concat(frames,ignore_index=True)


def numerical(data,caches,masks,held):
    frames=[]
    for name,path in caches.items():
        d=arrays(path/'features.npz');p=arrays(path/'teacher_query_predictions.npz')['prediction'];av=availability(data,path,held)
        predictions=dict(zip(previous.TEACHERS,[p[:,j] for j in range(3)]));predictions['teacher_mean']=p.mean(1)
        predictions['zero']=np.zeros_like(p[:,0])
        labels=arrays(path/'source_labels.npz');ref=references(data['meta'],labels['fit'],labels['fit_delta'],d['control'],held)[0]
        predictions['reference_only']=ref[d['query']]@d['basis'].T@d['basis']*float(d['response_scale'])
        for family,pred in predictions.items():
            for alpha in ((0.,) if family=='zero' else ALPHAS):
                f=score(data,path,alpha*pred,family,masks[name],av);f['alpha']=alpha;f['fold']=name;frames.append(f)
    return pd.concat(frames,ignore_index=True)


def choose_alphas(frame):
    chosen={}
    for family,f in frame.groupby('method'):
        chosen[family]=min(sorted(f.alpha.unique()),key=lambda x:(macro(f.loc[f.alpha.eq(x)]),x))
    selected=pd.concat([frame.loc[frame.method.eq(k)&frame.alpha.eq(v)] for k,v in chosen.items()],ignore_index=True)
    return chosen,selected


def admission(code,rank,caches,root):
    path=root/'head.py';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(code,encoding='utf-8')
    env=previous.worker_environment();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONPATH']=str(REPO/'src')
    for name,cache in caches.items():
        d=arrays(cache/'features.npz');actual=min(rank,len(d['basis']))
        out=root/(name+'_admission.json')
        result=subprocess.run([sys.executable,str(REPO/'scripts/check_agent_head.py'),'--code',str(path),'--rank',str(rank),
            '--target-dim',str(d['target'].shape[1]+len(d['basis'])),'--background-dim',str(2*d['control'].shape[1]+1),
            '--output',str(out)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        (root/(name+'_admission.log')).write_text(result.stdout[-10000:],encoding='utf-8')
        if result.returncode:raise ValueError('Isolated CPU admission failed')
        # Actual source basis may have fewer than sixteen dimensions in fixtures.
        from vcell.model_agent.candidate_code import check_head
        if actual!=rank:check_head(code,d['target'].shape[1]+len(d['basis']),2*d['control'].shape[1]+1,actual)


def search(a,pl,data,caches,root,masks,models,metadata,held,alpha,baseline,slots):
    records=[];selected={};api=[]
    for arm in ARMS:
        folder=root/'search'/pl['name']/arm;folder.mkdir(parents=True,exist_ok=True);history=[];seen=set();best=None
        agent=SearchAgent(folder/'api',a.agent_model,reasoning_effort='low',max_api_calls=slots,max_output_tokens=4096) if arm in ('single','dual') else None
        reviewer=ShortReviewer(folder/'review_api',a.agent_model,reasoning_effort='low',max_api_calls=slots,max_output_tokens=4096) if arm=='dual' else None
        for i in range(1,slots+1):
            dest=folder/f'trial_{i:02d}';dest.mkdir(parents=True,exist_ok=True)
            obs={'iteration':i,'candidate_slots':slots,'source_baseline':baseline,'previous_source_trials':history,
                'starting_head':BASE_HEAD,'outer_results_available':False,'fixed_radius':.2,
                'steps_per_fit':a.max_steps,'bounded_family':'existing CandidateHead AST <=200k parameters, width<=128; rank16/32; source-fit reference inputs'}
            freeze(dest/'observation.json',obs)
            if (dest/'result.json').exists():result=read(dest/'result.json')
            else:
                p=None;failure=None
                if agent:
                    decision,failure=recovery.call(agent,obs,'training');p=asdict(decision) if decision else None
                else:
                    rng=np.random.default_rng(a.seed+i*101+len(records))
                    width=[32,64][i-1] if arm=='fixed' else int(rng.choice([32,64,128]))
                    lr=[.0005,.001][i-1] if arm=='fixed' else float(rng.choice([.0001,.0005,.001]))
                    p={'tool':'propose_trial','expert_id':None,'reason':arm+' bounded design','code':BASE_HEAD.replace('32',str(width)),
                        'rank':([32,16][i-1] if arm=='fixed' else int(rng.choice([16,32]))),'learning_rate':lr,
                        'weight_decay':(.01 if arm=='fixed' else float(rng.choice([0.,.01,.05]))),'loss':('mse' if arm=='fixed' else str(rng.choice(['mse','huber'])))}
                if p and p['tool']=='stop':result={'status':'stopped','slot':i,'outer_used':False}
                elif not p:result={'status':'rejected','slot':i,'validation':failure,'outer_used':False}
                else:
                    recipe={k:p[k] for k in ('code','rank','learning_rate','weight_decay','loss')};fingerprint=digest(recipe)
                    if fingerprint in seen:result={'status':'duplicate','slot':i,'proposal':p,'outer_used':False}
                    else:
                        try:
                            parse_proposal(json.dumps(p));admission(p['code'],p['rank'],caches,dest)
                        except (ValueError,SyntaxError,subprocess.TimeoutExpired) as exc:
                            result={'status':'rejected','slot':i,'proposal':p,'validation':str(exc),'outer_used':False}
                        else:
                            label=pl['name']+'_'+arm+'_'+str(i);specs=configs(a,caches,root,label,'matched',models,metadata,held,alpha,p)
                            try:
                                dispatch(specs,root,a.parallel_jobs)
                            except RuntimeError:
                                result={'status':'training_failed','slot':i,'proposal':p,'worker_configs':list(specs),'outer_used':False}
                            else:
                                f=evaluate_workers(data,caches,specs,label,masks,held);f.to_csv(dest/'source_scores.csv',index=False)
                                s=summarize(f);result={'status':'evaluated','slot':i,'proposal':p,'source':s,
                                    'promoted':promotion(s,baseline),'label':label,'outer_used':False}
                freeze(dest/'result.json',result)
            if result.get('proposal'):seen.add(digest({k:result['proposal'][k] for k in ('code','rank','learning_rate','weight_decay','loss')}))
            review=None
            if reviewer and result['status']!='stopped':
                response,failure=recovery.call(reviewer,{'iteration':i,'source_result':result,'outer_results_available':False},'review')
                review=asdict(response) if response else {'status':'rejected','validation':failure};freeze(dest/'review.json',review)
            history.append({'result':result,'review':review});records.append({'outer_context':pl['outer_context'],'arm':arm,**result})
            if result.get('promoted') and (best is None or result['source']['mse']<best['source']['mse']):best=result
            write_json(root/'search_progress.json',{'context':pl['outer_context'],'arm':arm,'slot':i,'slots':slots,'status':result['status']})
            if result['status']=='stopped':break
        selected[arm]=best
        for planner in (agent,reviewer):
            if planner:api.append(planner.summary())
    return records,selected,api


def launch(a):
    work=Path(a.work_dir);root=work/'runs'/a.name;parent=Path(a.round19_run or work/'runs/round19_dual_100_01');verify(parent)
    oldplan=read(parent/'plan.json');src=Path(oldplan['source_run'] or work/'runs/round18_pilot_01');verify(src)
    source_args=read(src/'plan.json');p4=Path(source_args.get('p4_run') or work/'runs/round15_p4_02')
    if root.exists() and any(root.iterdir()) and not a.resume:raise ValueError('Use --resume or a new run name')
    root.mkdir(parents=True,exist_ok=True)
    code_paths=[Path(__file__),REPO/'scripts/check_agent_head.py',*sorted((REPO/'src/vcell/round20').glob('*.py')),REPO/'src/vcell/round19/predictor.py']
    freeze(root/'run_conditions.json',{'experiment':'round20_signal_and_bounded_reference','code':digest({p.relative_to(REPO).as_posix():file_sha256(p) for p in code_paths}),
        'parent_complete_sha256':file_sha256(parent/'COMPLETE.json'),'source_complete_sha256':file_sha256(src/'COMPLETE.json'),
        **{k:v for k,v in vars(a).items() if k not in ('resume','prepare_only','worker_config')},
        'test_evaluated':False,'outer_role':'previously observed development lines; HT29/Jurkat unscored',
        'API_code_scope':'isolated CandidateHead only; controller fixes reference library, folds, scaling, panel and evaluation',
        'technical_replicates':'not independent biological samples','exposure':'50:50 historical/IFNG when both available',
        'promotion':'>=1% source OOF gain against strongest simple baseline; strong-gene MSE and specificity nondegrade; amplitude>=5% truth norm'})
    if (root/'COMPLETE.json').exists():verify(root);print('ROUND20 COMPLETE VERIFIED',root);return root
    def stage(label):write_json(root/'stage.json',{'stage':label,'test_evaluated':False});print('ROUND20 STAGE:',label,flush=True)
    try:
        stage('P0 freeze parent data and source-fit replicate signal')
        prep=SimpleNamespace(**{**source_args,'outer_contexts':a.outer_contexts,'seeds':[a.seed],'inner_folds':3,
            'profile_targets':0,'targets_per_context':10,'device':a.device,'reuse_round17':str(src)})
        data,plans,bs,ws=core.prepared_inputs(prep,p4,root,freeze_inputs=False);reused=source.reuse_completed(prep,plans,bs,ws)
        freeze(root/'data_audit_snapshot.json',data['audit'])
        metadata=Path(plans[0]['folds'][0]['configs']['expanded']['data_dir'])/'metadata.csv';held=holdout_targets(data['meta'])
        freeze(root/'inputs.json',{'folds':plans,'data_dir':str(metadata.parent),'dataset_sha256':file_sha256(metadata.parent/'dataset.npz'),
            'metadata_sha256':file_sha256(metadata),'gene_order':data['genes'].tolist(),'held_targets':held,'reused_workers':reused,
            'normalization':data['audit']['normalization'],'parent_models':read(parent/'models.json'),'test_evaluated':False})
        qc={};masks={}
        for pl in plans:
            for fold in pl['folds']:
                name=fold['name'];out=root/'qc'/name;out.mkdir(parents=True,exist_ok=True)
                report,pairs,genes,mask=fit_signal(data,fold['configs']['expanded']['partition']['fit'])
                freeze(out/'source_signal.json',report);pairs.to_csv(out/'replicates.csv',index=False);genes.to_csv(out/'genes.csv',index=False)
                qc[name]=report;masks[name]=mask
        freeze(root/'budget.json',{'candidate_slots_max':12,'slots_per_arm_per_outer':[2,1],'arms':list(ARMS),'source_folds_per_candidate':3,
            'candidate_inner_fits_max':36,'candidate_final_fits_max':8,'steps_per_fit':a.max_steps,'parallel_jobs':a.parallel_jobs,
            'new_background_workers':len(bs),'new_expert_workers':len(ws),'api_requests_max':9,'output_token_cap_per_request':4096,
            'candidate_budget_matched_within_each_outer':True,'actual_api_costs_not_equal':'dual includes independent reviews; all usage reported',
            'currency_cost':'not estimated','search_gate':'source signal + nonzero/specificity recovery + >=1% gain; otherwise NOT_RUN'})
        if a.prepare_only:stage('P0 prepared; no new training or API calls');return root
        stage('prepare missing legal source OOF experts')
        core.dispatch(bs,root,a.parallel_jobs,'background','background');core.dispatch(ws,root,a.parallel_jobs,'numerical','response')
        models=read(parent/'models.json');verify_models(models);allcaches={};pending=[];candidate_records=[];api=[];source_frames=[];gates={}
        for pi,pl in enumerate(plans):
            caches={}
            for fold in pl['folds']:
                name=fold['name'];path=root/'caches'/name;old=parent/'caches'/name
                if (old/'COMPLETE.json').exists():
                    verify(old);m=read(old/'manifest.json')
                    if m['data']==data['audit']['fingerprint'] and m['partition']==fold['configs']['expanded']['partition'] and m['query_rows']==fold['query_rows'] and m['genes']==data['genes'].tolist():path=old
                    else:build_cache(data,fold,core.pool_for(fold,root),path,a.teacher_steps,models['deepseek']['identity'])
                else:build_cache(data,fold,core.pool_for(fold,root),path,a.teacher_steps,models['deepseek']['identity'])
                caches[name]=path;allcaches[name]={'path':str(path),'complete_sha256':file_sha256(path/'COMPLETE.json'),'manifest_sha256':file_sha256(path/'manifest.json')}
                labels=arrays(path/'source_labels.npz');weights=root/'exposure'/(name+'.npz');weights.parent.mkdir(parents=True,exist_ok=True)
                fw=exposure_weights(data['meta'].iloc[labels['fit']]);cw=exposure_weights(data['meta'].iloc[labels['calibration']])
                if not weights.exists():np.savez_compressed(weights,fit_weights=fw,calibration_weights=cw)
                else:
                    w=arrays(weights)
                    if not np.array_equal(w['fit_weights'],fw) or not np.array_equal(w['calibration_weights'],cw):raise ValueError('Frozen exposure weights changed')
            inner={f['name']:caches[f['name']] for f in pl['folds'][1:]};outer={pl['folds'][0]['name']:caches[pl['folds'][0]['name']]}
            stage('P1 source-only amplitude controls: '+pl['name'])
            grid=numerical(data,inner,masks,held);grid.to_csv(root/(pl['name']+'_alpha_grid_source.csv'),index=False)
            alphas,num=choose_alphas(grid);num['experiment']=pl['name'];source_frames.append(num)
            baseline_name=min(alphas,key=lambda m:macro(num.loc[num.method.eq(m)]));baseline=summarize(num.loc[num.method.eq(baseline_name)])
            signal=all(qc[f['name']]['signal_gate'] for f in pl['folds'][1:])
            methods=['direct_mlp','direct_mlp_string','matched','correction_only','wrong','shuffled','none']
            if signal:methods+=['direct_pretrained','direct_pretrained_string','direct_random_frozen','direct_shuffled_target','direct_scale_0.2','direct_scale_1.0','direct_scale_learned']
            control_results={}
            for method in methods:
                label=pl['name']+'_'+method;specs=configs(a,inner,root,label,method,models,metadata,held,alphas['reference_only'])
                dispatch(specs,root,a.parallel_jobs);f=evaluate_workers(data,inner,specs,method,masks,held);f['experiment']=pl['name'];source_frames.append(f)
                result=summarize(f);control_results[method]=result
                pending.append((pl,outer,method,None,alphas['reference_only']))
            recovered=recovery_gate(control_results,baseline)
            search_baseline=min([baseline,*control_results.values()],key=lambda s:s['mse'])
            gate=signal and recovered and not a.no_agent;gates[pl['name']]={'signal_gate':signal,'nonzero_recovery_promotion':recovered,'agent_search':gate,
                'transformer_controls':'RUN' if signal else 'NOT_RUN: source replicate gate failed'}
            selected=None
            if gate:
                stage('P3 isolated autonomous code search: '+pl['name'])
                records,selected,ledger=search(a,pl,data,inner,root,masks,models,metadata,held,alphas['reference_only'],search_baseline,2 if pi==0 else 1)
                candidate_records+=records;api+=ledger
                for arm,best in selected.items():
                    if best:pending.append((pl,outer,'search_'+arm,best['proposal'],alphas['reference_only']))
            freeze(root/'selections'/(pl['name']+'.json'),{'alphas':alphas,'baseline':baseline_name,'controls':control_results,
                'selected_candidates':selected,'gate':gates[pl['name']],'outer_used':False})
            pending.append((pl,outer,'numerical',None,alphas))
        freeze(root/'cache_lineage.json',allcaches)
        freeze(root/'search_frozen.json',{'gates':gates,'candidate_records':candidate_records,'outer_used':False,
            'selections_sha256':{p.name:file_sha256(p) for p in sorted((root/'selections').glob('*.json'))}})
        pd.concat(source_frames,ignore_index=True).to_csv(root/'source_scores.csv',index=False)
        candidate_table=pd.DataFrame([{'context':r['outer_context'],'arm':r['arm'],'slot':r['slot'],'status':r['status'],**r.get('source',{})} for r in candidate_records])
        if not len(candidate_table):candidate_table=pd.DataFrame(columns=['context','arm','slot','status','mse','strong_mse','amplitude_ratio','specificity_gap'])
        candidate_table.to_csv(root/'candidate_scores.csv',index=False)
        write_json(root/'api_ledger.json',api)
        stage('P4 private held development evaluation; no further selection')
        outer_frames=[]
        for pl,outer,method,proposal,alpha in pending:
            if method=='numerical':
                f=numerical(data,outer,masks,held);f=pd.concat([f.loc[f.method.eq(k)&f.alpha.eq(v)] for k,v in alpha.items()],ignore_index=True)
            else:
                label=pl['name']+'_outer_'+method;specs=configs(a,outer,root,label,'matched' if proposal else method,models,metadata,held,alpha,proposal)
                dispatch(specs,root,a.parallel_jobs);f=evaluate_workers(data,outer,specs,method,masks,held)
            f['experiment']=pl['name'];outer_frames.append(f)
        of=pd.concat(outer_frames,ignore_index=True);of.to_csv(root/'outer_scores.csv',index=False)
        summaries=[]
        for (experiment,method),f in of.groupby(['experiment','method']):summaries.append({'experiment':experiment,'method':method,**{k:v for k,v in summarize(f).items() if not isinstance(v,dict)}})
        pd.DataFrame(summaries).to_csv(root/'comparison.csv',index=False)
        for pl in plans:
            fold=pl['folds'][0];pairs,effects,noise=replicate_qc(data,fold['query_rows']);out=root/'qc'/fold['name']
            pairs.to_csv(out/'held_development_replicates.csv',index=False)
            if len(effects):pd.DataFrame({'gene':data['genes'],'mean_absolute_replicate_average':np.abs(effects).mean(0),
                'technical_sigma':np.sqrt(np.median(noise,axis=0)),'source_strong_mask':masks[fold['name']]}).to_csv(out/'held_gene_distribution.csv',index=False)
        write_json(root/'summary.json',{'gates':gates,'attempted_candidate_slots':len(candidate_records),'API':api,'test_evaluated':False,
            'agent_result':'source-gated bounded code search; NOT_RUN is a valid negative diagnostic outcome',
            'interpretation':'Compare reference-available and global unknown subsets separately; technical replicates are not biological replicates; no significance claim',
            'raw_cell_variance':'Not estimated; gene noise derives from prepared Rep1/Rep2 responses; control matching inherited from adapter and retained in data audit'})
        (root/'report.html').write_text('<html><meta charset="utf-8"><title>Round20</title><h1>Development signal and reference controls</h1><p>All thresholds and selections frozen before held development scoring. See source/outer rows for reference availability. HT29 and Jurkat unscored. Gates may stop API search.</p>'+pd.DataFrame(summaries).to_html(index=False,escape=True),encoding='utf-8')
        (root/'INCOMPLETE.json').unlink(missing_ok=True);stage('complete');seal(root,test_evaluated=False);return root
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error_type':type(exc).__name__,'error':str(exc),'test_evaluated':False});raise
    finally:
        path=core.package(root);target=root/'round20_review_light.tar.gz';path.replace(target);print('REVIEW PACKAGE:',target,flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker-config')
    p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round20_signal_01')
    p.add_argument('--round19-run');p.add_argument('--outer-contexts',nargs='+',choices=DEVELOPMENT,default=['BXPC3_IFNG','HAP1_IFNG'])
    p.add_argument('--seed',type=int,default=17);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--parallel-jobs',type=int,choices=[1,2,3],default=3);p.add_argument('--max-steps',type=int,default=500)
    p.add_argument('--teacher-steps',type=int,default=200);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--save-every',type=int,default=50)
    p.add_argument('--agent-model',default='deepseek-v4-pro');p.add_argument('--no-agent',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args(argv)
    if a.worker_config:
        cfg=read(a.worker_config)
        # Workers never receive provider credentials or the full expression dataset.
        if cfg['kind']=='direct':return train_worker(cfg,True)
        return reference_worker(cfg,True)
    if len(a.outer_contexts)!=2 or len(set(a.outer_contexts))!=2:raise ValueError('Debug protocol requires two distinct development backgrounds')
    if min(a.max_steps,a.teacher_steps,a.batch_size,a.save_every)<1:raise ValueError('Positive training budgets required')
    with run_lock(Path(a.work_dir)/'runs'/('.'+a.name+'.lock')):return launch(a)


if __name__=='__main__':main()
