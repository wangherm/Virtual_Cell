"""Budgeted tool execution; a planner can be injected without access to truth."""
from dataclasses import asdict
import copy
from pathlib import Path
import time

import numpy as np

from .artifacts import arrays, load_prediction, read, save_prediction, seal, verify
from .contracts import Action, digest, load_query
from .evidence import Profiles
from .experts import Registry
from ..utils import file_sha256, write_json


def code_identity():
    return digest({p.name:file_sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


class BaselinePlanner:
    """Deterministic comparators, never labelled as an LLM agent."""
    def __init__(self,policy='rule',fixed=None):
        if policy not in ('fixed','rule','risk','workflow'):raise ValueError('Unknown baseline policy')
        self.policy,self.fixed=policy,fixed

    def decide(self,observation):
        candidates=[c for c in observation['candidates'].values() if c['supported']]
        # Never interpret missing risk as zero risk.
        if self.policy in ('fixed','risk','workflow'):
            candidates=[c for c in candidates if c['estimated_mse'] is not None]
        if self.policy=='fixed':candidates=[c for c in candidates if c['expert_id']==self.fixed]
        if not candidates:return Action('abstain',reason='No supported expert with required source evidence')
        if self.policy=='rule':
            # Predeclared capability rule; names alone do not imply empirical accuracy.
            def priority(c):
                name=c['expert_id']
                return (c['cost_units'],0 if name.endswith('string_ridge') else 1 if name.endswith('factor_rank16') else 2,name)
            order=sorted(candidates,key=priority)
        else:order=sorted(candidates,key=lambda c:(c['estimated_mse'],c['expert_id']))
        predicted=observation['predictions'];available=[c for c in order if c['expert_id'] in predicted]
        if available and (self.policy!='workflow' or len(available)>=2):
            return Action('finish',available[0]['expert_id'],'Select by source-only risk (or declared cheap-model rule)')
        for candidate in order:
            name=candidate['expert_id']
            if name in predicted or name in observation['failed_experts']:continue
            if name not in observation['inspected']:
                if observation['remaining_units']>=observation['inspection_cost']:return Action('inspect',name,'Read allowed model features')
            if candidate['cost_units']<=observation['remaining_units']:return Action('predict',name,'Execute a supported expert')
        if available:return Action('finish',available[0]['expert_id'],'Stop at the declared budget')
        return Action('abstain',reason='No affordable prediction')


def execute(registry_path,query_path,output,policy='rule',profile_path=None,budget_units=3.,max_steps=12,
            resume=False,planner=None,planner_identity=None,inspection_cost=.05,formal_outer=False):
    """Planner sees observations only; all model execution stays in this controller.

    Budget units are declared tool charges, not dollars or guaranteed wall time.
    They cover inspections and attempted predictions. Real measured latency is
    reported separately. Provider/API accounting will be added at LLM integration.
    """
    if not np.isfinite(budget_units) or budget_units<=0 or max_steps<1 or not np.isfinite(inspection_cost) or inspection_cost<=0:
        raise ValueError('Positive bounded execution budget required')
    registry=Registry(registry_path);q=load_query(query_path)
    profiles=Profiles(profile_path,registry) if profile_path else None
    if formal_outer:
        from .contracts import cell_line_key
        if profiles is None:raise ValueError('Formal outer routing requires source-only profiles')
        line=cell_line_key(q.context,q.cell_line)
        if line not in {cell_line_key(c) for c in profiles.value.get('excluded_cell_lines',profiles.value['excluded_contexts'])}:
            raise ValueError('Query cell line is not excluded from the source profiles')
        for name,e in registry.experts.items():
            if e.card['kind'] in ('p4','candidate','demo_linear') and name not in profiles.value['models']:
                raise ValueError('Formal numerical expert lacks source OOF evidence: '+name)
            d=e.diagnostics(q)
            if d['query_in_expert_fit'] or d['query_in_expert_calibration'] or d['query_cell_line_in_fit'] or d['query_cell_line_in_calibration']:
                raise ValueError('Formal query cell line/row leaked into expert fit/calibration: '+name)
    if planner is None:
        if policy in ('fixed','risk','workflow') and profiles is None:raise ValueError('This policy requires source OOF profiles')
        planner=BaselinePlanner(policy,profiles.value['fixed_expert'] if profiles else None)
        planner_identity='deterministic_baseline:'+policy
    elif not planner_identity:raise ValueError('Explicit versioned planner identity required for audited injection')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    identity={'query':q.fingerprint,'registry':registry.fingerprint,'profile':profiles.marker_sha if profiles else None,
              'budget_units':budget_units,'max_steps':max_steps,'inspection_cost':inspection_cost,
              'planner_identity':planner_identity,'source':code_identity()}
    identity['formal_outer']=formal_outer
    manifest=root/'manifest.json'
    if manifest.exists():
        if not resume or read(manifest)!=identity:raise ValueError('Resume identical model-agent inputs or use a new directory')
        if hasattr(planner,'validate_resume'):planner.validate_resume(root)
        if (root/'COMPLETE.json').exists():verify(root);return read(root/'result.json')
        state=read(root/'state.json')
    else:
        if any(root.iterdir()):raise ValueError('Output directory is not empty')
        write_json(manifest,identity)
        state={'spent_units':0.,'steps':0,'pending':None,'inspected':{},'predictions':{},'failed_experts':[],
               'trace':[],'selected':None,'finished':False,'stop_reason':None}
        write_json(root/'state.json',state)
    started=time.perf_counter()
    while not state['finished'] and state['steps']<max_steps:
        candidates=registry.candidates(q)
        for name,candidate in candidates.items():
            candidate['estimated_mse']=profiles.risk(name,registry.experts[name].features(q)) if profiles and candidate['supported'] else None
            candidate['profile_missing']=candidate['estimated_mse'] is None
            if formal_outer:candidate['source_features']=registry.experts[name].features(q)
        observation={'task':q.public,'candidates':candidates,'remaining_units':max(0.,budget_units-state['spent_units']),
            'inspection_cost':inspection_cost,'inspected':state['inspected'],'predictions':state['predictions'],
            'failed_experts':state['failed_experts'],'remaining_steps':max_steps-state['steps']}
        if state['pending'] is None:
            # Persist a decision before side effects. Pending replay avoids a new
            # planner decision; prediction outputs are atomically cached.
            try:
                decision=planner.decide(copy.deepcopy(observation))
            except Exception as exc:
                write_json(root/'PLANNER_ERROR.json',{'error_type':type(exc).__name__,
                    'provider_call':getattr(planner,'last_call',None),'note':'No deterministic fallback or success marker'})
                raise
            if isinstance(decision,dict):decision=Action(**decision)
            if not isinstance(decision,Action):raise ValueError('Planner must return an Action')
            action=asdict(decision);cost=0.
            if decision.tool!='abstain':
                if decision.expert_id not in candidates or not candidates[decision.expert_id]['supported']:
                    raise ValueError('Planner selected an unsupported/unregistered expert')
                if decision.tool=='finish' and decision.expert_id not in state['predictions']:
                    raise ValueError('Cannot finish with an unexecuted expert')
                if decision.tool=='inspect' and decision.expert_id not in state['inspected']:cost=inspection_cost
                if decision.tool=='predict' and decision.expert_id not in state['predictions']:cost=registry.costs[decision.expert_id]
            if state['spent_units']+cost>budget_units+1e-12:raise ValueError('Planner tool call exceeds declared budget')
            state['spent_units']+=cost
            state['pending']={'action':action,'charged_units':cost,'observation':copy.deepcopy(observation)}
            if getattr(planner,'last_call',None) is not None:
                state['pending']['provider_call']=copy.deepcopy(planner.last_call)
            write_json(root/'state.json',state)
            (root/'PLANNER_ERROR.json').unlink(missing_ok=True)
        pending=state['pending'];decision=Action(**pending['action']);name=decision.expert_id;result={}
        tick=time.perf_counter()
        try:
            if decision.tool=='inspect':
                features=registry.experts[name].features(q)
                identical=[other for other,value in state['inspected'].items() if value==features]
                state['inspected'][name]=features;result={**features,'inspection_information_gain':
                    {'new_feature_values':not formal_outer and not bool(identical),'identical_to':identical,'quality_evidence_added':False}}
            elif decision.tool=='predict':
                path=root/(name+'.npz')
                if not path.exists():save_prediction(path,q,registry.experts[name].predict(q))
                prediction=load_prediction(path,q)
                summary={'norm':float(np.linalg.norm(prediction)),'mean':float(prediction.mean()),'finite':True}
                # Disagreement is a diagnostic covariate, not a quality ranking.
                summary['disagreement_mse']={other:float(np.square(prediction-load_prediction(root/(other+'.npz'),q)).mean())
                                             for other in state['predictions'] if other!=name}
                state['predictions'][name]=summary;result=summary
            elif decision.tool=='finish':state['selected']=name;state['finished']=True;state['stop_reason']=decision.reason
            else:state['finished']=True;state['stop_reason']=decision.reason or 'abstained'
        except Exception as exc:
            # Charge failed attempts; do not treat failure as a successful cache.
            if name and name not in state['failed_experts']:state['failed_experts'].append(name)
            result={'error_type':type(exc).__name__,'message':str(exc)}
        state['steps']+=1
        state['trace'].append({'step':state['steps'],**pending,'result':result,'wall_seconds':time.perf_counter()-tick})
        state['pending']=None;write_json(root/'state.json',state)
    if not state['finished']:
        # Do not silently pick a model the planner did not explicitly select.
        state['finished']=True;state['stop_reason']='step budget exhausted without explicit finish'
    selected=state['selected']
    if selected:save_prediction(root/'selected.npz',q,load_prediction(root/(selected+'.npz'),q))
    result={'status':'selected' if selected else 'abstained','selected_expert':selected,'reason':state['stop_reason'],
            'spent_units':state['spent_units'],'budget_units':budget_units,'steps':state['steps'],
            'wall_seconds_this_session':time.perf_counter()-started,'planner_identity':planner_identity,
            'scope':'Nested development cell-line holdout' if formal_outer else 'Architecture validation; not a formal agent effectiveness experiment','test_evaluated':False,
            'synthetic':any(e.card['synthetic'] for e in registry.experts.values()),
            'cost_semantics':'Declared tool units; interrupted uncached execution can repeat physical compute. No hard wall-time guarantee.'}
    result['evaluation_role']='outer_cell_line_holdout' if formal_outer else 'deployment_smoke'
    result['diagnostics']={name:{**e.diagnostics(q),'profile_missing':profiles is None or name not in profiles.value['models']}
                           for name,e in registry.experts.items()}
    if hasattr(planner,'summary'):
        result['provider_summary']=planner.summary()
        result['scope']='Live nested development routing benchmark' if formal_outer else 'Live API/model-execution pilot; not a formal agent effectiveness experiment'
    write_json(root/'state.json',state);write_json(root/'trace.json',state['trace']);write_json(root/'result.json',result)
    seal(root);return result


def evaluate(run,query_path,truth_path,registry_path):
    """Offline evaluator only: oracle computations occur after decision sealing."""
    root=Path(run);verify(root);q=load_query(query_path);registry=Registry(registry_path)
    manifest=read(root/'manifest.json')
    if manifest['query']!=q.fingerprint or manifest['registry']!=registry.fingerprint:raise ValueError('Evaluation inputs changed')
    f=arrays(truth_path)
    if set(f)!={'genes','delta','query_fingerprint'} or str(f['query_fingerprint'])!=q.fingerprint or not np.array_equal(f['genes'],q.genes):
        raise ValueError('Private evaluation truth alignment mismatch')
    truth=f['delta']
    if truth.shape!=(len(q.genes),) or not np.isfinite(truth).all():raise ValueError('Invalid evaluation truth')
    scores={}
    for name,expert in registry.experts.items():
        if expert.compatible(q)[0]:scores[name]=float(np.square(expert.predict(q)-truth).mean())
    result=read(root/'result.json');selected=result['selected_expert']
    value={'selected_mse':float(np.square(load_prediction(root/'selected.npz',q)-truth).mean()) if selected else None,
           'zero_mse':float(np.square(truth).mean()),'posthoc_oracle_mse':min(scores.values()) if scores else None,
           'all_expert_scores':scores,'selected_expert':selected,'test_evaluated':False,
           'note':'Private development labels; oracle extra inference is analysis-only and excluded from agent cost. No action trace is changed.'}
    return value
