"""Separate supported technical-repeat prediction from source-only cell-line transfer."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import numpy as np
import pandas as pd
from vcell.clean_background import run_lock
from vcell.data import load_prepared
from vcell.model_agent.artifacts import read,arrays,seal,verify
from vcell.model_agent.contracts import digest
from vcell.model_agent.candidate_code import BASE_HEAD,check_head
from vcell.model_agent.research import parse_proposal
from vcell.round11 import freeze
from vcell.round20.signal import ALPHAS,macro
from vcell.round21.agents import SignatureScientist as SearchAgent,SignatureReviewer as ShortReviewer
from vcell.round21 import transfer,repeat,audit
from vcell.round21.signal import POLICY,signature_table
from vcell.utils import write_json,file_sha256
import run_round19 as previous
import round19_recovery as recovery


def stage(root,value,**details):
    write_json(root/'stage.json',{'stage':value,'test_evaluated':False,**details});print('ROUND21 STAGE:',value,details,flush=True)


def dispatch(specs,root,parallel):
    pending=list(specs);active={};failed=[];(root/'logs').mkdir(parents=True,exist_ok=True)
    try:
        while pending or active:
            while pending and len(active)<parallel:
                name=pending.pop(0);cfg=specs[name];out=Path(cfg['output_dir'])
                if (out/'COMPLETE.json').exists():
                    verify(out)
                    if read(out/'manifest.json')['config']!=cfg:raise ValueError('Worker configuration changed')
                    continue
                path=root/'worker_configs'/(name+'.json');freeze(path,cfg);log=(root/'logs'/(name+'.log')).open('a',encoding='utf-8')
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker-config',str(path)],env=previous.worker_environment(),stdout=log,stderr=subprocess.STDOUT)
                active[name]=(proc,log);print('ROUND21 WORKER START',name,proc.pid,flush=True)
            for name,(proc,log) in list(active.items()):
                if proc.poll() is not None:
                    log.close();del active[name]
                    if proc.returncode or not (Path(specs[name]['output_dir'])/'COMPLETE.json').exists():failed.append(name)
                    print('ROUND21 WORKER EXIT',name,'code=',proc.returncode,flush=True)
            write_json(root/'worker_progress.json',{'active':list(active),'pending':pending,'failed':failed})
            if active:time.sleep(1)
    finally:
        for proc,log in active.values():
            if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
            log.close()
    if failed:raise RuntimeError('Workers failed; inspect logs and resume: '+','.join(failed))


def score(pred,truth,meta,method,strong,available,common,unknown,task):
    f=previous.score_rows(pred,truth,meta.reset_index(drop=True),method)
    strong_error=np.square(pred-truth)*strong;counts=strong.sum(1)
    f['strong_mse']=np.divide(strong_error.sum(1),counts,out=np.full(len(pred),np.nan),where=counts>0)
    f['strong_genes']=counts;f['reference_available']=available;f['common_support']=common;f['globally_unknown']=unknown
    f['task']=task
    for k in ('dataset','stimulus','time','modality','batch','replicate'):f[k]=meta[k].to_numpy() if k in meta else 'unknown'
    return f


def summary(f):
    values={k:macro(f,k) for k in ('mse','zero_mse','prediction_norm','truth_norm','strong_mse','specificity_gap')}
    values['amplitude_ratio']=values['prediction_norm']/max(values['truth_norm'],1e-12)
    values.update(rows=len(f),targets=int(f.target.nunique()),reference_rows=int(f.reference_available.sum()),common_rows=int(f.common_support.sum()))
    return {k:float(v) if np.isfinite(v) else None for k,v in values.items()}


def numerical(data,caches,held,alphas=None):
    frames=[]
    for name,path in caches.items():
        d=arrays(path/'features.npz');ix=d['query'].astype(int);m=data['meta'].iloc[ix]
        vectors={'zero':np.zeros_like(d['reference'][ix]),'source_target_mean':d['reference'][ix],'nearest_background':d['nearest'][ix],
            'background_mean':d['background_mean'][ix],'string_ridge':d['ridge'],'wrong_reference':d['wrong'][ix],'shuffled_reference':d['shuffled'][ix]}
        for method,p in vectors.items():
            grid=ALPHAS if alphas is None and method not in ('zero','wrong_reference','shuffled_reference') else [0. if method=='zero' else (alphas.get(method,alphas['source_target_mean']) if alphas else 1.)]
            for alpha in grid:
                av=d['available'][ix] if method not in ('wrong_reference','shuffled_reference') else d['common_support'][ix]
                f=score(alpha*p*float(d['response_scale']),data['delta'][ix],m,method,d['strong_mask'][ix],av,d['common_support'][ix],m.perturbation.isin(held).to_numpy(),'cross_cell_transfer')
                f['alpha']=alpha;f['fold']=name;f['method_status']='FROZEN_BASELINE';frames.append(f)
    return pd.concat(frames,ignore_index=True)


def choose(grid):
    result={}
    for method,f in grid.groupby('method'):
        if method in ('wrong_reference','shuffled_reference'):continue
        result[method]=float(min(sorted(f.alpha.unique()),key=lambda a:(macro(f.loc[f.alpha.eq(a)]),a)))
    return result


def head_configs(a,caches,root,label,alpha,proposal=None):
    recipe={'code':BASE_HEAD,'rank':32,'learning_rate':.0005,'weight_decay':.01,'loss':'mse'}
    if proposal:recipe.update({k:proposal[k] for k in recipe})
    return {label+'_'+name:{'kind':'transfer','cache':str(path),'output_dir':str(root/'workers'/label/name),
        'alpha':alpha,'seed':a.seed,'device':a.device,'max_steps':a.max_steps,'batch_size':a.batch_size,'save_every':a.save_every,**recipe} for name,path in caches.items()}


def head_scores(data,caches,specs,held,label):
    frames=[]
    for name,path in caches.items():
        cfg=next(c for c in specs.values() if c['cache']==str(path));out=Path(cfg['output_dir']);verify(out);p=arrays(out/'prediction.npz');d=arrays(path/'features.npz')
        ix=d['query'].astype(int)
        if not np.array_equal(p['row_ids'],data['meta'].iloc[ix].row_id.to_numpy(dtype='U')) or not np.array_equal(p['genes'],data['genes']):raise ValueError('Head prediction alignment changed')
        f=score(p['prediction'],data['delta'][ix],data['meta'].iloc[ix],label,d['strong_mask'][ix],d['available'][ix],d['common_support'][ix],data['meta'].iloc[ix].perturbation.isin(held).to_numpy(),'cross_cell_transfer')
        f['fold']=name;f['method_status']=read(out/'training_status.json')['status'];frames.append(f)
    return pd.concat(frames,ignore_index=True)


def gates(frame,repeat_frame,raw_status):
    positive=[];facts=[]
    for context,f in frame.groupby('context'):
        common=f.loc[f.common_support & ~f.globally_unknown];references=common.loc[common.method.eq('source_target_mean')];shuffled=common.loc[common.method.eq('shuffled_reference')]
        baseline=min(macro(g) for method,g in f.groupby('method') if method in ('zero','background_mean','string_ridge'))
        candidate=min((summary(g) for method,g in f.groupby('method') if method in ('source_target_mean','nearest_background','mlp_correction') and not (g.method_status=='BLOCKED_NO_RELIABLE_TRAIN_REFERENCE').any()),key=lambda s:s['mse'])
        paired=bool(len(references) and references.target.nunique()>=4 and len(references)==len(shuffled) and macro(references)<.99*macro(shuffled) and macro(references,'specificity_gap')>macro(shuffled,'specificity_gap'))
        margin=bool(candidate['mse']<.99*baseline and candidate['amplitude_ratio']>=.05 and candidate['strong_mse'] is not None and candidate['specificity_gap'] is not None and candidate['specificity_gap']>0)
        facts.append({'context':context,'common_targets':int(references.target.nunique()),'matched_beats_shuffled':paired,'positive_transfer_margin':margin})
        if paired and margin:positive.append(context)
    within=False
    for context,f in repeat_frame.groupby('context') if len(repeat_frame) else []:
        proper=f.loc[f.method.eq('calibrated_rep1')];wrong=f.loc[f.method.eq('wrong_rep1')]
        if len(proper) and len(wrong) and proper.target.nunique()>=4 and macro(proper)<.99*macro(wrong) and macro(proper,'specificity_gap')>macro(wrong,'specificity_gap'):within=True
    correction=frame.loc[frame.method.eq('mlp_correction')]
    trainable=bool(len(correction) and correction.fold.nunique()>=2 and correction.method_status.eq('TRAINED').all())
    enabled=bool(within and len(positive)>=2 and trainable and raw_status.get('independent_repeat_controls_verified',False))
    return {'agent_search_allowed':enabled,'within_source_repeat_specificity':within,'positive_source_contexts':positive,'source_context_evidence':facts,
        'raw_controls_verified':raw_status.get('independent_repeat_controls_verified',False),'outer_used':False,
        'source_correction_folds_trainable':trainable,
        'rule':'Within-source calibrated repeat beats wrong; >=2 source-held contexts have >=1% transferable gain, nonzero amplitude, strong-gene metric and positive specificity; >=4 common targets with correct-reference advantage; raw NT pools independent; every source correction fold has trainable support'}


def promote(s,baseline):
    return bool(s['strong_mse'] is not None and s['specificity_gap'] is not None and baseline['strong_mse'] is not None and baseline['specificity_gap'] is not None and
        s['mse']<.99*baseline['mse'] and s['strong_mse']<=baseline['strong_mse'] and s['specificity_gap']>=baseline['specificity_gap'] and s['amplitude_ratio']>=.05)


def admit(code,rank,caches,dest):
    dest.mkdir(parents=True,exist_ok=True);path=dest/'head.py';path.write_text(code,encoding='utf-8');env=previous.worker_environment();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONPATH']=str(REPO/'src')
    for name,cache in caches.items():
        d=arrays(cache/'features.npz');target_dim=d['target'].shape[1];background_dim=d['background'].shape[1]
        result=subprocess.run([sys.executable,str(REPO/'scripts/check_agent_head.py'),'--code',str(path),'--rank',str(rank),
            '--target-dim',str(target_dim),'--background-dim',str(background_dim),'--output',str(dest/(name+'_admission.json'))],env=env,capture_output=True,text=True,timeout=60)
        (dest/(name+'_admission.log')).write_text((result.stdout+result.stderr)[-10000:],encoding='utf-8')
        if result.returncode:raise ValueError('Isolated code admission failed')
        if len(d['basis'])<rank:check_head(code,target_dim,background_dim,len(d['basis']))


def search(a,pl,data,caches,root,held,alpha,baseline,slots):
    records=[];selected={};ledger=[]
    for arm in ('fixed','random','single','dual'):
        folder=root/'search'/pl['name']/arm;history=[];seen=set();best=None
        scientist=SearchAgent(folder/'api',a.agent_model,reasoning_effort='low',max_api_calls=slots,max_output_tokens=4096) if arm in ('single','dual') else None
        reviewer=ShortReviewer(folder/'review_api',a.agent_model,reasoning_effort='low',max_api_calls=slots,max_output_tokens=4096) if arm=='dual' else None
        for i in range(1,slots+1):
            dest=folder/f'trial_{i:02d}';dest.mkdir(parents=True,exist_ok=True)
            obs={'iteration':i,'candidate_slots':slots,'source_baseline':baseline,'previous_source_trials':history,'starting_head':BASE_HEAD,
                'task':'Correct a reliable, same-target experimental source signature. No query labels or fold edits.',
                'fixed_radius':.2,'steps_per_fit':a.max_steps,'outer_results_available':False}
            freeze(dest/'observation.json',obs)
            if (dest/'result.json').exists():record=read(dest/'result.json')
            else:
                if scientist:
                    response,failure=recovery.call(scientist,obs,'training');p=asdict(response) if response else None
                else:
                    rng=np.random.default_rng(a.seed+i*173);width=[32,64][i-1] if arm=='fixed' else int(rng.choice([32,64,128]));failure=None
                    p={'tool':'propose_trial','expert_id':None,'reason':arm+' permitted head','code':BASE_HEAD.replace('32',str(width)),
                        'rank':32 if arm=='fixed' else int(rng.choice([16,32])),'learning_rate':.0005 if arm=='fixed' else float(rng.choice([.0001,.0005,.001])),'weight_decay':.01,'loss':'mse'}
                recipe={k:p[k] for k in ('code','rank','learning_rate','weight_decay','loss')} if p else None
                if not p:record={'status':'rejected','validation':failure}
                elif p['tool']=='stop':record={'status':'stopped'}
                elif digest(recipe) in seen:record={'status':'duplicate','proposal':p}
                else:
                    try:
                        parse_proposal(json.dumps(p));admit(p['code'],p['rank'],caches,dest)
                    except (ValueError,SyntaxError,subprocess.TimeoutExpired) as exc:record={'status':'rejected','proposal':p,'validation':str(exc)}
                    else:
                        label=pl['name']+'_'+arm+'_'+str(i);specs=head_configs(a,caches,root,label,alpha,p)
                        try:dispatch(specs,root,a.parallel_jobs)
                        except RuntimeError:record={'status':'training_failed','proposal':p}
                        else:
                            f=head_scores(data,caches,specs,held,label);f.to_csv(dest/'source_scores.csv',index=False);s=summary(f)
                            trained=not f.method_status.eq('BLOCKED_NO_RELIABLE_TRAIN_REFERENCE').any()
                            record={'status':'evaluated' if trained else 'blocked','proposal':p,'source':s,'promoted':trained and promote(s,baseline)}
                record.update(slot=i,arm=arm,outer_context=pl['outer_context'],outer_used=False);freeze(dest/'result.json',record)
            if record.get('proposal'):seen.add(digest({k:record['proposal'][k] for k in ('code','rank','learning_rate','weight_decay','loss')}))
            review=None
            if reviewer and record['status']!='stopped':
                response,failure=recovery.call(reviewer,{'iteration':i,'source_result':record,'outer_results_available':False},'review');review=asdict(response) if response else {'status':'rejected','validation':failure};freeze(dest/'review.json',review)
            history.append({'result':record,'review':review});records.append(record)
            if record.get('promoted') and (best is None or record['source']['mse']<best['source']['mse']):best=record
            write_json(root/'search_progress.json',{'context':pl['outer_context'],'arm':arm,'slot':i,'slots':slots,'status':record['status']})
            if record['status']=='stopped':break
        selected[arm]=best
        for agent in (scientist,reviewer):
            if agent:ledger.append(agent.summary())
    return records,selected,ledger


def launch(a):
    work=Path(a.work_dir);parent=Path(a.round20_run or work/'runs/round20_signal_01');verify(parent);inputs=read(parent/'inputs.json');data=load_prepared(inputs['data_dir']);metadata=Path(inputs['data_dir'])/'metadata.csv'
    if file_sha256(metadata)!=inputs['metadata_sha256'] or file_sha256(metadata.parent/'dataset.npz')!=inputs['dataset_sha256'] or not np.array_equal(data['genes'],inputs['gene_order']):raise ValueError('Parent data changed')
    root=work/'runs'/a.name
    if root.exists() and any(root.iterdir()) and not a.resume:raise ValueError('Use --resume or a new name')
    root.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),REPO/'scripts/audit_round21_counts.R',REPO/'scripts/check_agent_head.py',Path(previous.__file__),Path(recovery.__file__),REPO/'src/vcell/round20/signal.py',REPO/'src/vcell/model_agent/candidate_code.py',
        REPO/'src/vcell/model_agent/deepseek.py',REPO/'src/vcell/round20/agents.py',REPO/'src/vcell/round19/agents.py',*sorted((REPO/'src/vcell/round21').glob('*.py'))]
    freeze(root/'run_plan.json',{'experiment':'round21_supported_repeat_and_cell_transfer','parent_complete_sha256':file_sha256(parent/'COMPLETE.json'),
        'code':digest({p.relative_to(REPO).as_posix():file_sha256(p) for p in paths}),**{k:v for k,v in vars(a).items() if k not in ('prepare_only','resume','worker_config')},
        'signal_policy':POLICY,'outer_evaluation':'all method choices frozen before any held-development response diagnosis or scoring',
        'test_evaluated':False,'scope':'Previously observed development; technical repeat prediction is not cell-line or unsupported-target generalization'})
    if (root/'COMPLETE.json').exists():verify(root);print('ROUND21 COMPLETE VERIFIED',root);return root
    try:
        stage(root,'freeze original outer/inner folds, target and gene order')
        plans=inputs['folds'];held=inputs['held_targets']
        if len(plans)!=2 or any(p['seed']!=a.seed or len(p['folds'])!=4 for p in plans):raise ValueError('Requires original two development backgrounds, one matching seed, three source folds')
        original=read(parent/'cache_lineage.json');parents={}
        for pl in plans:
            for fold in pl['folds']:
                record=original[fold['name']];path=Path(record['path']);verify(path)
                if file_sha256(path/'COMPLETE.json')!=record['complete_sha256']:raise ValueError('Inherited feature cache changed')
                parents[fold['name']]=path
        freeze(root/'lineage.json',{'parent':str(parent),'data_dir':inputs['data_dir'],'gene_order':inputs['gene_order'],'held_targets':held,
            'folds':plans,'inherited_caches':original,'test_evaluated':False})
        freeze(root/'budget.json',{'candidate_slots_max':12,'slots_per_arm_per_outer':[2,1],'candidate_source_fits_max':36,'api_requests_max':9,
            'output_token_cap':4096,'parallel_jobs':a.parallel_jobs,'steps':a.max_steps,'new_large_backbone_fits':0,'raw_R_processes':1,
            'cost_note':'candidate counts and training steps matched; actual API cost differs by review calls and is reported'})
        if a.prepare_only:stage(root,'prepared; raw audit, training and API not started');return root
        stage(root,'P0 raw library sizes and exact control memberships')
        raw=audit.raw_audit(a.adapter_dir or work/'prepared/round15_mixscale_ifng_04',a.raw_dir or work/'raw/round15_D1/IFNG',root/'raw_audit',REPO,a.rscript,a.skip_raw_audit)
        caches={};qc=[];mis=[];repeat_scores=[];source_scores=[];records=[];ledger=[];pending=[];gate_records={}
        for pi,pl in enumerate(plans):
            stage(root,'source-only signatures and reference support',context=pl['outer_context'])
            for fold in pl['folds']:
                name=fold['name'];caches[name]=transfer.cache(parents[name],metadata,held,root/'caches'/name,a.seed)
                q=pd.read_csv(caches[name]/'replicate_qc_by_target.csv');q['fold']=name;q['experiment']=pl['name'];qc.append(q)
                mm=pd.read_csv(caches[name]/'signal_vs_mismatch.csv');mm['fold']=name;mm['experiment']=pl['name'];mis.append(mm)
            inner={f['name']:caches[f['name']] for f in pl['folds'][1:]};outer={pl['folds'][0]['name']:caches[pl['folds'][0]['name']]}
            stage(root,'supported Rep1 -> Rep2 source target folds',context=pl['outer_context'])
            final=arrays(parents[pl['folds'][0]['name']]/'source_labels.npz');scale=float(arrays(parents[pl['folds'][0]['name']]/'features.npz')['response_scale']);fit=final['fit'].astype(int)
            specs=repeat.repeat_plan(data['meta'],fit,final['fit_delta']*scale,data['baseline']);jobs={};rp={}
            for j,s in enumerate(specs):
                name=pl['name']+'_repeat_'+str(j);cache=repeat.pack(s,data['baseline'],root/'repeat_inputs'/name);rp[name]=s
                jobs[name]={'kind':'repeat','cache':str(cache),'output_dir':str(root/'workers'/name),'seed':a.seed,'device':a.device,
                    'max_steps':a.max_steps,'batch_size':a.batch_size,'save_every':a.save_every}
            dispatch(jobs,root,a.parallel_jobs);rframes=[]
            for name,s in rp.items():
                out=Path(jobs[name]['output_dir']);p=arrays(out/'prediction.npz');ix=[int(u['rep2_rows'][0]) for u in s['query']];truth=np.array([u['rep2'] for u in s['query']]);m=data['meta'].iloc[ix]
                for method in sorted(set(p)-{'strong_mask','targets'}):
                    f=score(p[method],truth,m,method,p['strong_mask'],np.ones(len(ix),bool),np.ones(len(ix),bool),np.zeros(len(ix),bool),'supported_technical_repeat')
                    f['experiment']=pl['name'];f['fold']=name;f['method_status']=read(out/'lineage.json')['status'];rframes.append(f)
                    f['independent_controls_verified']=raw.get('independent_repeat_controls_verified',False)
            rf=pd.concat(rframes,ignore_index=True) if rframes else pd.DataFrame();repeat_scores.append(rf)
            stage(root,'source OOF baseline calibration and small correction heads',context=pl['outer_context'])
            grid=numerical(data,inner,held);grid.to_csv(root/(pl['name']+'_source_alpha_grid.csv'),index=False);alphas=choose(grid);f=numerical(data,inner,held,alphas)
            heads=head_configs(a,inner,root,pl['name']+'_mlp',alphas['source_target_mean']);dispatch(heads,root,a.parallel_jobs);hf=head_scores(data,inner,heads,held,'mlp_correction');f=pd.concat([f,hf],ignore_index=True)
            f['experiment']=pl['name'];source_scores.append(f);gate=gates(f,rf,raw);gate_records[pl['name']]=gate;selected=None
            if gate['agent_search_allowed'] and not a.no_agent:
                stage(root,'limited autonomous head search from source evidence',context=pl['outer_context'])
                baseline=min((summary(g) for method,g in f.groupby('method') if method not in ('wrong_reference','shuffled_reference') and not g.method_status.eq('BLOCKED_NO_RELIABLE_TRAIN_REFERENCE').any()),key=lambda s:s['mse'])
                trials,selected,api=search(a,pl,data,inner,root,held,alphas['source_target_mean'],baseline,2 if pi==0 else 1);records+=trials;ledger+=api
            freeze(root/'selections'/(pl['name']+'.json'),{'alphas':alphas,'gate':gate,'agent_disabled_by_option':a.no_agent,'selected_candidates':selected,'outer_used':False})
            pending.append((pl,outer,alphas,selected))
        pd.concat(qc,ignore_index=True).to_csv(root/'replicate_qc_by_target.csv',index=False);pd.concat(mis,ignore_index=True).to_csv(root/'signal_vs_mismatch.csv',index=False)
        repeat_all=pd.concat(repeat_scores,ignore_index=True);repeat_all.to_csv(root/'within_line_repeat_scores.csv',index=False)
        sf=pd.concat(source_scores,ignore_index=True);sf.to_csv(root/'source_cross_cell_transfer_scores.csv',index=False)
        freeze(root/'search_frozen.json',{'gates':gate_records,'candidates':records,'outer_used':False,'selections_sha256':{p.name:file_sha256(p) for p in (root/'selections').glob('*.json')}})
        write_json(root/'api_ledger.json',ledger);pd.DataFrame([{'context':r['outer_context'],'arm':r['arm'],'slot':r['slot'],'status':r['status'],**r.get('source',{})} for r in records],columns=None if records else ['context','arm','slot','status','mse']).to_csv(root/'candidate_scores.csv',index=False)
        stage(root,'private held-development evaluation; all choices sealed')
        outer_frames=[]
        for pl,outer,alphas,selected in pending:
            f=numerical(data,outer,held,alphas);heads=head_configs(a,outer,root,pl['name']+'_outer_mlp',alphas['source_target_mean']);dispatch(heads,root,a.parallel_jobs)
            f=pd.concat([f,head_scores(data,outer,heads,held,'mlp_correction')],ignore_index=True)
            for arm,best in (selected or {}).items():
                if best:
                    jobs=head_configs(a,outer,root,pl['name']+'_outer_'+arm,alphas['source_target_mean'],best['proposal']);dispatch(jobs,root,a.parallel_jobs)
                    f=pd.concat([f,head_scores(data,outer,jobs,held,'agent_'+arm)],ignore_index=True)
            f['experiment']=pl['name'];outer_frames.append(f)
        of=pd.concat(outer_frames,ignore_index=True);of.to_csv(root/'cross_cell_transfer_scores.csv',index=False)
        summaries=[]
        for task,frame in [('supported_technical_repeat',repeat_all),('source_cross_cell_transfer',sf),('held_development_cross_cell_transfer',of)]:
            for (ex,method),g in frame.groupby(['experiment','method']) if len(frame) else []:
                subsets={'all':g,'common_reference_support':g.loc[g.common_support],'reference_available':g.loc[g.reference_available],
                    'reference_unavailable':g.loc[~g.reference_available],'globally_unknown':g.loc[g.globally_unknown],'known_target':g.loc[~g.globally_unknown]}
                for subset,part in subsets.items():
                    if len(part):summaries.append({'task':task,'experiment':ex,'method':method,'subset':subset,**summary(part)})
        pd.DataFrame(summaries).to_csv(root/'comparison.csv',index=False);audit.response_audit(data,root/'raw_audit')
        if (root/'raw_audit/batch_audit.csv').exists():
            import shutil
            shutil.copyfile(root/'raw_audit/batch_audit.csv',root/'batch_audit.csv')
        else:pd.DataFrame(columns=['cell_line','replicate','target','raw_audit_status']).to_csv(root/'batch_audit.csv',index=False)
        from collections import Counter
        head_statuses=dict(Counter(read(p)['status'] for p in (root/'workers').rglob('training_status.json')))
        write_json(root/'summary.json',{'gates':gate_records,'candidate_slots_attempted':len(records),'api':ledger,'raw_audit':raw,'transfer_head_statuses':head_statuses,
            'test_evaluated':False,'task_boundary':'supported repeat is not unknown-target or cell-line generalization',
            'claims':'development diagnostic; top response genes are not statistically tested DEGs; two technical repeats do not establish biological reproducibility',
            'prepared_data_modified':False,'large_backbone_fits':0})
        (root/'report.html').write_text('<html><meta charset="utf-8"><h1>Round21</h1><p>Supported repeats and cross-cell transfer are separate tasks. Source choices sealed before development scoring; HT29/Jurkat unscored. Common reference support and global unknown targets are reported separately.</p>'+pd.DataFrame(summaries).to_html(index=False,escape=True),encoding='utf-8')
        (root/'README.md').write_text('''# Round21 review

Read summary.json and comparison.csv first. The task and subset columns separate
supported technical-repeat prediction, source cross-cell OOF and held development
evaluation. Raw controls must be verified before interpreting repeat prediction
as an independent-repeat result. Gates and untrained fallback statuses are explicit.

replicate_qc_by_target.csv and signal_vs_mismatch.csv use declared source fit labels.
batch_audit.csv and raw_audit/ contain observed library/control facts; no causal
batch explanation or normalization repair is claimed. hap1_response_audit is
post-freeze development diagnosis. Top response genes are not inferential DEGs.

No raw expression, per-cell membership, matrix or model weights belong in this light
archive. Retain the full server run, parent runs and raw adapter to reproduce it.
''',encoding='utf-8')
        (root/'INCOMPLETE.json').unlink(missing_ok=True);stage(root,'complete');seal(root,test_evaluated=False);return root
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error_type':type(exc).__name__,'error':str(exc),'test_evaluated':False});raise
    finally:
        print('REVIEW PACKAGE:',package(root),flush=True)


def package(root):
    import tarfile,io,hashlib
    from collections import deque
    path=root/'round21_review_light.tar.gz';entries={}
    with tarfile.open(path,'w:gz') as tar:
        for p in sorted(root.rglob('*')):
            rel=p.relative_to(root)
            if not p.is_file() or p.suffix not in ('.json','.csv','.html','.py','.log','.md'):continue
            if p.name=='selected_cells.csv' or 'worker_configs' in rel.parts:continue
            if p.suffix=='.log':
                with p.open(encoding='utf-8',errors='replace') as stream:payload=''.join(deque(stream,maxlen=60)).encode()
            else:payload=p.read_bytes()
            name=rel.as_posix();m=tarfile.TarInfo(name);m.size=len(payload);tar.addfile(m,io.BytesIO(payload));entries[name]={'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
        payload=json.dumps({'files':entries,'scope':'reports, patches and log tails; no per-cell membership, expression arrays, weights or credentials'}).encode();m=tarfile.TarInfo('ARCHIVE_MANIFEST.json');m.size=len(payload);tar.addfile(m,io.BytesIO(payload))
    return path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker-config');p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work')
    p.add_argument('--name',default='round21_signal_01');p.add_argument('--round20-run');p.add_argument('--adapter-dir');p.add_argument('--raw-dir');p.add_argument('--rscript')
    p.add_argument('--seed',type=int,default=17);p.add_argument('--parallel-jobs',type=int,choices=[1,2,3],default=3);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--max-steps',type=int,default=500);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--save-every',type=int,default=50)
    p.add_argument('--agent-model',default='deepseek-v4-pro');p.add_argument('--no-agent',action='store_true');p.add_argument('--skip-raw-audit',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args(argv)
    if a.worker_config:
        cfg=read(a.worker_config)
        return repeat.worker(cfg,True) if cfg['kind']=='repeat' else transfer.worker(cfg,True)
    if min(a.max_steps,a.batch_size,a.save_every)<1 or a.seed<0:raise ValueError('Positive budgets and nonnegative seed required')
    with run_lock(Path(a.work_dir)/'runs'/('.'+a.name+'.lock')):return launch(a)


if __name__=='__main__':main()
