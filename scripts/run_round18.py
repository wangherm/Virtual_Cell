"""Scoped scientific memory, literature-grounded search controls and robust routing."""
import argparse
from dataclasses import asdict
import copy
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import numpy as np
import pandas as pd
import run_round17_oof as core
from vcell.clean_background import run_lock
from vcell.model_agent.artifacts import read,seal,verify
from vcell.model_agent.contracts import digest,load_query,safe_name
from vcell.model_agent.deepseek import DEFAULT_MODEL,DeepSeekPlanner
from vcell.model_agent.experts import Expert,Registry
from vcell.model_agent.iteration import IterationBudget,feedback
from vcell.model_agent.nested import DEVELOPMENT,cell_line_key,source_summary
from vcell.model_agent.reliability import ReliabilityPlanner,fit_reliability,macro
from vcell.model_agent.research import Proposal,ResearchPlanner,export_candidate,parse_proposal
from vcell.model_agent.routing import GatePlanner,OneShotPlanner,fit_gate
from vcell.model_agent.runtime import code_identity,execute
from vcell.model_agent.scientific_memory import ScientificMemory,collect_literature,retrieve_papers
from vcell.model_agent.scientist import ScientistPlanner,control_proposal,observation
from vcell.round11 import freeze,verify_files
from vcell.utils import file_sha256,write_json


def zero_pool(path):
    registry=Registry(path);dest=registry.path.parent/'zero';template=registry.experts['expanded_factor_rank16']
    if (dest/'COMPLETE.json').exists():verify(dest)
    else:
        dest.mkdir(parents=True,exist_ok=True)
        stats={k:v for k,v in template.stats.items() if not k.startswith('reference_')}
        np.savez_compressed(dest/'input.npz',**stats)
        card={**template.card,'expert_id':'zero','kind':'zero','method':'zero','recipe':{
            'method':'zero','genes':stats['genes'].tolist(),'normalization':template.card['normalization']},
            'response_fit':'No learned response parameters; fit-only control statistics support the router covariates'}
        card['recipe_fingerprint']=digest(card['recipe']);write_json(dest/'expert.json',card);seal(dest)
    records=[{**r,'bundle':str(registry.experts[r['expert_id']].path)} for r in registry.config['experts']]
    records.append({'expert_id':'zero','bundle':str(dest),'cost_units':.01,'cost_source':'unmeasured'})
    new=registry.path.parent/'registry_with_zero.json';freeze(new,{'schema_version':1,'experts':records});Registry(new);return new


def reuse_completed(a,plans,bs,ws):
    """Read-only exact fit/cal/budget reuse. Query count may expand independently."""
    if not a.reuse_round17:return []
    parent=Path(a.reuse_round17).resolve();verify(parent);old=read(parent/'inputs.json');records=[]
    for plan in plans:
        for fold in plan['folds']:
            for variant,cfg in list(fold['configs'].items()):
                for oldplan in old['folds']:
                    if oldplan['seed']!=plan['seed']:continue
                    for oldfold in oldplan['folds']:
                        oc=oldfold['configs'][variant]
                        exact=['data_dir','relations','model','seed','device','methods','ridge_alphas','max_steps','batch_size','save_every','variant']
                        if any(cfg[k]!=oc[k] for k in exact):continue
                        if any(cfg['partition'][k]!=oc['partition'][k] for k in ('fit','calibration')):continue
                        bgold=read(Path(oc['encoder']).parent/'manifest.json')['config']
                        bgnew=bs['bg_'+fold['name']+'_'+variant]
                        if any(bgold[k]!=bgnew[k] for k in ('pretrain_steps','pretrain_batch','background_cache','background','family')):continue
                        verify_files(oc['output_dir']);verify_files(Path(oc['encoder']).parent)
                        if read(Path(oc['output_dir'])/'manifest.json')['encoder']!=file_sha256(oc['encoder']):raise ValueError('Reused encoder changed')
                        fold['configs'][variant]=oc
                        bs.pop('bg_'+fold['name']+'_'+variant,None);ws.pop('model_'+fold['name']+'_'+variant,None)
                        records.append({'fold':fold['name'],'variant':variant,'worker':oc['output_dir'],
                            'complete_sha256':file_sha256(Path(oc['output_dir'])/'COMPLETE.json'),
                            'scope':'same fit/cal/seed/budgets; original diagnostic partition preserved; expanded source queries inferred with no refit'})
                        break
                    else:continue
                    break
    return records


def candidate_acceptance(frame,name,baseline,margin):
    group=frame.loc[frame.expert_id.eq(name)];base=frame.loc[frame.expert_id.eq(baseline)]
    left=base.groupby(['context','target']).mse.mean();right=group.groupby(['context','target']).mse.mean()
    if not left.index.equals(right.index):raise ValueError('Candidate matched source units changed')
    bm=macro(base);cm=macro(group);context_gain=(left-right).groupby(level='context').mean()
    base_context=left.groupby(level='context').mean();relative=context_gain/np.maximum(base_context,1e-12)
    gains=[]
    for target in sorted(set(group.target)):
        b=base.loc[~base.target.eq(target)];c=group.loc[~group.target.eq(target)]
        if len(c):gains.append((macro(b)-macro(c))/max(macro(b),1e-12))
    accepted=(cm<bm*(1-margin) and int((relative>0).sum())>=2 and float(relative.min())>=-.01 and min(gains,default=-1)>0)
    return {'source_oof_mse':cm,'baseline_mse':bm,'relative_gain':(bm-cm)/max(bm,1e-12),
        'context_relative_gain':{k:float(v) for k,v in relative.items()},'leave_one_target_out_gain_min':min(gains,default=None),
        'accepted':bool(accepted),'criterion':'>=margin relative macro gain vs best fixed including zero; >=2 contexts improve; no context >1% worse; every leave-target-out gain positive',
        'margin':margin,'confidence_interval_claim':False}


def search_arm(a,pl,data,root,pools,summary,arm,papers,memory_snapshot,memory):
    folder=root/'search'/pl['name']/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETE.json').exists():verify(folder);return read(folder/'selection.json')
    planner=None
    if arm.startswith('agent'):
        cls=ScientistPlanner if arm=='agent_literature' else ResearchPlanner
        planner=cls(folder/'api',a.model,reasoning_effort=a.research_effort,max_api_calls=a.trials*a.cycles,
                    max_output_tokens=a.max_output_tokens,timeout=a.api_timeout)
    previous=[];best=summary['fixed_source_mse'][summary['best_fixed']];chosen=None;best_candidate=None;best_observed=float('inf')
    iteration=IterationBudget(a.cycles,a.trials,a.patience_cycles)
    for i in range(1,a.trials*a.cycles+1):
        cycle=(i-1)//a.trials+1
        dest=folder/f'trial_{i:02d}';dest.mkdir(parents=True,exist_ok=True)
        active_papers=retrieve_papers(papers,'cross-background perturbation prediction regularization low rank '+(previous[-1].get('reason','') if previous else '')) if arm=='agent_literature' else None
        obs=observation(summary,previous,memory_snapshot,active_papers);obs['self_iteration']=feedback(previous,cycle,a.cycles,best_candidate)
        freeze(dest/'observation.json',obs)
        if (dest/'proposal.json').exists():
            value=read(dest/'proposal.json');p=parse_proposal(json.dumps(value['proposal']))
            if arm=='agent_literature':
                from vcell.model_agent.scientific_memory import validate_citations
                validate_citations(value['citations'],active_papers,p.tool=='propose_trial')
        else:
            generated=planner.decide(obs) if planner else control_proposal(arm,i,pl['seed'],pl['outer_context'])
            p=Proposal(**{k:getattr(generated,k) for k in Proposal.__dataclass_fields__})
            value={'proposal':asdict(p),'hypothesis':getattr(generated,'hypothesis',p.reason),
                'citations':getattr(generated,'citations',[]),'arm':arm,'cycle':cycle}
            write_json(dest/'proposal.json',value)
        if p.tool=='stop':
            stop={'cycle':cycle,'stop_reason':'agent_requested_stop','outer_used':False,'next_cycle_allowed':False}
            iteration.records.append(stop);freeze(folder/'cycles'/('cycle_'+str(cycle)+'.json'),stop);break
        source=dest/'head.py'
        if not source.exists():source.write_text(p.code,encoding='utf-8')
        if source.read_text(encoding='utf-8')!=p.code:raise ValueError('Frozen head code changed')
        name=arm+'_head_'+str(i);suffix=arm+'_trial'+str(i)
        if (dest/'result.json').exists():result=read(dest/'result.json')
        else:
            try:
                core.admit(source,p.rank,dest);jobs={}
                for fold in pl['folds'][1:]:
                    cfg={**fold['configs']['expanded'],'recipe':asdict(p),'head_code':str(source),
                        'template_worker':fold['configs']['expanded']['output_dir'],'output_dir':str(dest/'workers'/fold['name'])}
                    jobs[name+'_'+fold['name']]=cfg
                core.dispatch(jobs,root,a.parallel_jobs,pl['name']+'_'+suffix,'candidate')
                extra={}
                for fold in pl['folds'][1:]:
                    bundle=export_candidate(dest/'workers'/fold['name'],dest/'bundles'/fold['name'],
                        Registry(pools[fold['name']]).experts['expanded_factor_rank16'].path,name)
                    extra[fold['name']]={name:bundle}
                _,frame,_=core.evidence_for(pl,data,root,pools,extra,suffix,include_zero=True)
                result={'trial':i,'expert_id':name,'status':'evaluated','outer_used':False,
                        **candidate_acceptance(frame,name,summary['best_fixed'],a.minimum_gain)}
            except Exception as exc:
                result={'trial':i,'status':'failed','error_type':type(exc).__name__,'reason':str(exc),'outer_used':False}
            write_json(dest/'result.json',result)
        previous.append(result)
        if result.get('status')=='evaluated' and result['source_oof_mse']<best_observed:
            best_observed=result['source_oof_mse'];best_candidate={'proposal':asdict(p),'source_mse':best_observed,
                'accepted':result['accepted'],'hypothesis':value['hypothesis']}
        if result.get('accepted') and result['source_oof_mse']<best:
            best=result['source_oof_mse'];chosen={'trial':i,'expert_id':name,'source_oof_mse':best,'proposal':asdict(p),
                'head_code':str(source),'evidence_suffix':suffix}
        used=set()
        for fold in pl['folds'][1:]:
            e=Registry(pools[fold['name']]).experts['expanded_factor_rank16'];used.update(e.provenance['fit_cell_lines']);used.update(e.provenance['calibration_cell_lines']);used.add(cell_line_key(fold['held']))
        memory.add_experiment(digest([str(root),pl['name'],arm,i]),{'evaluation_role':'source_inner_oof','outer_used':False,
            'used_cell_lines':sorted(used),'data_fingerprint':data['audit']['fingerprint'],'arm':arm,'status':result['status'],
            'source_mse':result.get('source_oof_mse'),'baseline_mse':summary['fixed_source_mse'][summary['best_fixed']],
            'reason':p.reason,'recipe_sha256':digest(asdict(p))})
        if i%a.trials==0:
            decision=iteration.complete(cycle,previous,summary['fixed_source_mse'][summary['best_fixed']])
            freeze(folder/'cycles'/('cycle_'+str(cycle)+'.json'),{**decision,'feedback':feedback(previous,cycle,a.cycles,best_candidate)})
            if not decision['next_cycle_allowed']:break
    selection={'selected_candidate':chosen,'baseline_fixed':summary['best_fixed'],'baseline_mse':summary['fixed_source_mse'][summary['best_fixed']],
        'trials':previous,'outer_used':False,'provider_summary':planner.summary() if planner else {'attempted_requests':0},
        'cycles':iteration.records,
        'budget':{'candidate_limit':a.trials*a.cycles,'max_cycles':a.cycles,'trials_per_cycle':a.trials,
            'steps_each':a.max_steps,'batch_each':a.batch_size,'inner_folds':a.inner_folds},
        'claim':'Source-search result only; rejects are retained; whole search process evaluated on sealed outer'}
    write_json(folder/'selection.json',selection);seal(folder);return selection


def replay(a,pl,data,root,pool,raw_profile,frame,summary,label):
    final=pl['folds'][0];registry=Registry(pool);gate=fit_gate(frame)
    calibrated=fit_reliability(frame,raw_profile,root/'reliability'/pl['name']/label)
    policies=['zero','fixed','rule','risk','workflow','softmax_gate','risk_shrunk','conservative','legacy_fixed']
    # Exact same identifier-chosen query subset for every policy in a live benchmark.
    selected_targets=[]
    for i in final['query_rows']:
        t=str(data['meta'].iloc[i].perturbation)
        if t not in selected_targets:selected_targets.append(t)
    if a.live_routing:selected_targets=selected_targets[:a.live_targets]
    selected_rows=[i for i in final['query_rows'] if data['meta'].iloc[i].perturbation in selected_targets]
    if a.live_routing:policies += [p+'_'+effort for p in ('one_shot','adaptive') for effort in a.routing_efforts]
    records=[];numerical={n:m for n,m in summary['fixed_source_mse'].items() if n!='zero'}
    legacy=min(numerical,key=lambda n:(numerical[n],n));profile_value=read(Path(raw_profile)/'profile.json')
    from vcell.model_agent.runtime import BaselinePlanner
    for row in selected_rows:
        qp=core.query_path(data,row,root,final['configs']['expanded']);q=load_query(qp);qid='q_'+digest(q.query_id)[:16]
        for policy in policies:
            out=root/'episodes'/pl['name']/label/qid/policy;planner=None;profile=raw_profile
            if policy in ('zero','risk_shrunk','conservative'):
                planner=ReliabilityPlanner(calibrated,registry,q,policy);profile=calibrated
            elif policy=='legacy_fixed':planner=BaselinePlanner('fixed',legacy);planner.identity='fixed_without_zero:'+legacy
            elif policy=='softmax_gate':planner=GatePlanner(gate,registry,q)
            elif policy.startswith(('one_shot_','adaptive_')):
                effort=policy.rsplit('_',1)[1];cls=OneShotPlanner if policy.startswith('one_shot') else DeepSeekPlanner
                planner=cls(root/'api'/pl['name']/label/qid/policy,a.model,reasoning_effort=effort,
                    max_api_calls=1 if cls is OneShotPlanner else a.routing_calls,max_output_tokens=a.max_output_tokens,timeout=a.api_timeout)
                profile=calibrated
            result=execute(pool,qp,out,policy=policy if planner is None else 'rule',profile_path=profile,
                budget_units=3.1,max_steps=8,resume=a.resume,planner=planner,planner_identity=getattr(planner,'identity',None),formal_outer=True)
            records.append({'query_id':q.query_id,'query_path':str(qp),'row':row,'policy':policy,'episode':str(out),
                'selected_expert':result['selected_expert'],'result':result})
    dest=root/'evaluation'/pl['name']/label;dest.mkdir(parents=True,exist_ok=True)
    freeze(dest/'decisions_sealed.json',{'episodes':{r['episode']:file_sha256(Path(r['episode'])/'COMPLETE.json') for r in records},
        'live_routing_executed':a.live_routing,'paired_query_ids':sorted({r['query_id'] for r in records}),
        'source_expert_gap_diagnostic':summary['oracle_relative_gap'],'oracle_exposed_to_planner':False,
        'profile_scope':'raw risk comparator; all new guard/LLM policies share the same shrunk source-only profile',
        'cost_warning':'Live benchmark is permitted even if oracle gap is small; inspect actual tokens and matched accuracy'} )
    return records


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work')
    p.add_argument('--name',default='round18_01');p.add_argument('--p4-run');p.add_argument('--parent-seed',type=int,default=17)
    p.add_argument('--reuse-round17');p.add_argument('--outer-contexts',nargs='+',choices=DEVELOPMENT,default=['BXPC3_IFNG','HAP1_IFNG','MCF7_IFNG'])
    p.add_argument('--seeds',nargs='+',type=int,default=[17,29,43]);p.add_argument('--inner-folds',type=int,choices=[3],default=3)
    p.add_argument('--targets-per-context',type=int,default=10);p.add_argument('--profile-targets',type=int,default=0,help='0 = all source IFNG targets')
    p.add_argument('--max-steps',type=int,default=2000);p.add_argument('--pretrain-steps',type=int);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--save-every',type=int,default=100);p.add_argument('--parallel-jobs',type=int,choices=[1,2,3,4],default=3)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--search-arms',nargs='+',choices=['fixed','random','agent_plain','agent_literature'],default=['fixed','random','agent_plain','agent_literature'])
    p.add_argument('--trials',type=int,choices=[1,2],default=2);p.add_argument('--cycles',type=int,choices=[1,2,3],default=2)
    p.add_argument('--patience-cycles',type=int,choices=[1,2,3],default=2);p.add_argument('--minimum-gain',type=float,default=.01)
    p.add_argument('--model',default=DEFAULT_MODEL);p.add_argument('--research-effort',choices=['low','high','max'],default='high')
    p.add_argument('--max-output-tokens',type=int,default=8192);p.add_argument('--api-timeout',type=float,default=180.)
    p.add_argument('--literature-cutoff',default='2026-10-12');p.add_argument('--literature-cache');p.add_argument('--memory-db')
    p.add_argument('--live-routing',action='store_true');p.add_argument('--live-targets',type=int,default=3)
    p.add_argument('--routing-efforts',nargs='+',choices=['low','high','max'],default=['low']);p.add_argument('--routing-calls',type=int,default=4)
    a=p.parse_args(argv);safe_name(a.name)
    if min(a.max_steps,a.batch_size,a.save_every,a.targets_per_context,a.live_targets)<1 or a.profile_targets<0 or not 0<=a.minimum_gain<1:p.error('Invalid budgets')
    if a.pretrain_steps is not None and a.pretrain_steps<1 or not 1<=a.routing_calls<=8:p.error('Invalid optimizer/API budget')
    for field in ('outer_contexts','seeds','search_arms','routing_efforts'):
        if not getattr(a,field) or len(set(getattr(a,field)))!=len(getattr(a,field)):p.error('Unique '+field+' required')
    if min(a.seeds)<0:p.error('Nonnegative seeds required')
    work=Path(a.work_dir).resolve();root=work/'runs'/a.name
    with run_lock(work/'runs'/('.'+a.name+'.lock')):return launch(a,root,work)


def launch(a,root,work):
    parent=Path(a.p4_run or work/'runs/round15_p4_02').resolve();verify_files(parent)
    plan={'experiment':'round18_scoped_virtual_cell_scientist','source':code_identity(),'launcher':file_sha256(__file__),
        'core_launcher':file_sha256(core.__file__),'parent':str(parent),'parent_complete':file_sha256(parent/'COMPLETE.json'),
        **{k:v for k,v in vars(a).items() if k not in ('resume','prepare_only')},
        'test_evaluated':False,'scientific_claim':'Development benchmark; literature inspiration is not method reproduction; source memory only'}
    if root.exists() and any(root.iterdir()) and not a.resume:raise ValueError('Use --resume or new name')
    root.mkdir(parents=True,exist_ok=True);freeze(root/'plan.json',plan)
    if (root/'COMPLETE.json').exists():verify(root);print('ROUND18 COMPLETE VERIFIED:',root);return root
    def stage(label):write_json(root/'stage.json',{'stage':label,'test_evaluated':False});print('ROUND18 STAGE:',label,flush=True)
    try:
        stage('freeze extended source queries and cell-line scopes')
        data,plans,bs,ws=core.prepared_inputs(a,parent,root,freeze_inputs=False);reuse=reuse_completed(a,plans,bs,ws)
        freeze(root/'inputs.json',{'folds':plans,'data_dir':next(iter(plans[0]['folds'][0]['configs'].values()))['data_dir'],
            'data_fingerprint':data['audit']['fingerprint'],'metadata_sha256':file_sha256(Path(plans[0]['folds'][0]['configs']['expanded']['data_dir'])/'metadata.csv'),
            'genes':data['genes'].tolist(),'reuse':reuse,'profile_targets':'all eligible' if a.profile_targets==0 else a.profile_targets,
            'source_conditions':'same frozen normalization/target aliases/public priors as Round17; whole-line exclusion; HT29/Jurkat unscored'})
        max_queries=sum(len({str(data['meta'].iloc[i].perturbation) for i in pl['folds'][0]['query_rows']}) for pl in plans)
        possible_pools=1+len(a.search_arms)
        query_rows=sum(sum(data['meta'].iloc[pl['folds'][0]['query_rows']].perturbation.isin(
            list(dict.fromkeys(data['meta'].iloc[pl['folds'][0]['query_rows']].perturbation))[:a.live_targets])) for pl in plans)
        freeze(root/'budget.json',{'new_background_fits':len(bs),'new_numerical_heads':len(ws)*3,'reused_workers':len(reuse),
            'candidate_head_fits_max':len(plans)*len(a.search_arms)*a.trials*a.cycles*a.inner_folds,
            'final_candidate_fits_max':len(plans)*len(a.search_arms),'parallel_jobs':a.parallel_jobs,
            'research_requests_max':len(plans)*sum(s.startswith('agent') for s in a.search_arms)*a.trials*a.cycles,
            'self_iteration_cycles':a.cycles,'trials_per_cycle':a.trials,'stop_patience_cycles':a.patience_cycles,
            'routing_requests_max':int(query_rows*possible_pools*len(a.routing_efforts)*(1+a.routing_calls)) if a.live_routing else 0,
            'matched_outer_distinct_targets':max_queries,'currency_cost':'Not estimated; bounded request/token usage recorded'})
        if a.prepare_only:stage('prepared; no network/API/training');return root
        stage('public literature snapshot and scoped memory')
        memory=ScientificMemory(a.memory_db or work/'knowledge/round18/scientific_memory.sqlite');papers=[]
        if 'agent_literature' in a.search_arms:
            cache=Path(a.literature_cache or work/'knowledge/round18'/('literature_'+a.literature_cutoff))
            papers=collect_literature(cache,a.literature_cutoff)
            for paper in papers:memory.add_paper(paper)
            freeze(root/'literature_snapshot.json',{'papers':papers,'complete_sha256':file_sha256(cache/'COMPLETE.json'),
                'question':'Cross-background perturbation prediction, low-rank interactions and regularization','full_text_read':False})
        # Freeze every arm/seed's memory inputs before writing any new trial results.
        snapshots={}
        for pl in plans:
            path=root/'memory_snapshots'/(pl['name']+'.json')
            if not path.exists():freeze(path,{'experiments':memory.retrieve([pl['outer_context']],data['audit']['fingerprint']),
                'excluded_cell_line':cell_line_key(pl['outer_context']),'outer_used':False})
            snapshots[pl['name']]=read(path)['experiments']
        stage('source-only B2 and six-expert refits')
        if bs:core.dispatch(bs,root,a.parallel_jobs,'R18_background','background')
        if ws:core.dispatch(ws,root,a.parallel_jobs,'R18_experts','expert')
        pools={f['name']:zero_pool(core.pool_for(f,root)) for pl in plans for f in pl['folds']};pending=[];selections=[]
        for pl in plans:
            stage('source capabilities, zero evidence and matched search: '+pl['name'])
            profile,frame,summary=core.evidence_for(pl,data,root,pools,include_zero=True)
            pending.append((pl,'baseline_pool',pools[pl['folds'][0]['name']],profile,frame,summary))
            docs=papers
            for arm in a.search_arms:
                selection=search_arm(a,pl,data,root,pools,summary,arm,docs,snapshots[pl['name']],memory)
                selections.append({'fold':pl['name'],'arm':arm,'selected':selection['selected_candidate'],
                    'best_candidate_source_mse':min([r['source_oof_mse'] for r in selection['trials'] if r.get('source_oof_mse') is not None],default=None),
                    'baseline_source_mse':summary['fixed_source_mse'][summary['best_fixed']]})
                armroot=root/'arms'/arm;armroot.mkdir(parents=True,exist_ok=True)
                improved=core.activate(pl,selection,data,armroot,pools,a)
                if improved:
                    selected=selection['selected_candidate'];p=root/'evidence'/pl['name']/selected['evidence_suffix']/'profiles'
                    f=pd.read_csv(p/'private_oof_scores.csv');pending.append((pl,arm+'_pool',improved,p,f,source_summary(f)))
        write_json(root/'search_comparison.json',selections)
        stage('paired formal routing; source search frozen')
        decisions=[]
        for pl,label,pool,profile,frame,summary in pending:
            records=replay(a,pl,data,root,pool,profile,frame,summary,label);decisions.append((pl,label,pool,records))
        freeze(root/'search_and_routing_sealed.json',{'pools':[{'fold':pl['name'],'label':label,'registry':Registry(pool).fingerprint} for pl,label,pool,_ in decisions],
            'outer_scores_used_for_search':False,'memory_updated_with_outer_scores':False})
        stage('private outer evaluation after all decisions seal')
        comparisons=[]
        for pl,label,pool,records in decisions:
            core.evaluate_records(pl,records,data,root,label,pool)
            f=pd.read_csv(root/'evaluation'/pl['name']/label/'comparison.csv');f['seed']=pl['seed'];f['pool']=label;comparisons.append(f)
        comparison=pd.concat(comparisons,ignore_index=True);comparison.to_csv(root/'comparison.csv',index=False)
        averages=comparison.groupby(['pool','policy','context']).agg(mse=('mse','mean'),coverage=('coverage','mean'),zero_mse=('zero_mse','mean')).reset_index()
        averages.groupby(['pool','policy']).agg(macro_mse=('mse','mean'),coverage=('coverage','mean'),zero_mse=('zero_mse','mean')).reset_index().to_csv(root/'comparison_macro.csv',index=False)
        (root/'report.html').write_text('<html><meta charset="utf-8"><h1>Round18 development benchmark</h1><p>Zero is selectable. Compare search arms separately from within-pool routing; optimizer seeds are not biological samples. Literature uses abstract evidence only.</p>'+comparison.to_html(index=False,escape=True)+'</html>',encoding='utf-8')
        write_json(root/'summary.json',{'outer_contexts':a.outer_contexts,'optimizer_seeds':a.seeds,'live_routing':a.live_routing,
            'search_arms':a.search_arms,'zero_registered':True,'source_profiles_all_targets':a.profile_targets==0,
            'no_significance_claim':True,'test_evaluated':False,'outer_memory_imported':False,
            'literature_scope':'Traceable retrieved abstracts, not unrestricted autonomous browsing or faithful full-text reproduction'})
        (root/'INCOMPLETE.json').unlink(missing_ok=True);stage('complete');seal(root,test_evaluated=False)
        print('ROUND18 COMPLETE:',root/'comparison.csv',flush=True)
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error_type':type(exc).__name__,'error':str(exc),'test_evaluated':False});raise
    finally:
        path=core.package(root);path.replace(root/'round18_review_light.tar.gz');print('REVIEW PACKAGE:',root/'round18_review_light.tar.gz',flush=True)
    return root


if __name__=='__main__':main()
