"""Paid malformed proposals are source feedback, never executed or budget resets."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from vcell.model_agent.artifacts import read, verify
from vcell.model_agent.contracts import digest
from vcell.model_agent.research import ResearchPlanner
from vcell.model_agent.scientist import control_proposal
from vcell.utils import write_json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import round18_recovery as recovery
import run_round18 as runner
from test_deepseek_agent import Client, reply


def test_rejection_diagnostics_and_budget(tmp_path,monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','SECRET_MUST_NOT_APPEAR')
    value=asdict(control_proposal('fixed',1,17,'BXPC3'))
    value['rank']=64
    client=Client([reply(content=json.dumps(value))])
    planner=recovery.install_diagnostics(ResearchPlanner(tmp_path/'api',client=client,max_api_calls=1))
    obs={'source':'only'}
    with pytest.raises(RuntimeError,match='invalid_action'):planner.decide(obs)
    failure=recovery.rejected_proposal(planner,obs)
    assert failure['reason']=='Unsupported proposal recipe' and failure['candidate_fits']==0
    assert planner.summary()['attempted_requests']==1
    assert recovery.rejected_proposal(planner,obs)==failure and len(client.calls)==1
    assert recovery.rejected_proposal(planner,{'source':'next'})['error_type']=='RequestBudgetExhausted'
    text=''.join(p.read_text() for p in (tmp_path/'api').rglob('*.json'))
    assert 'SECRET_MUST_NOT_APPEAR' not in text and 'PRIVATE_REASONING' not in text
    assert value['code'] not in text and not (tmp_path/'head.py').exists()
    # HTTP failure is not treated as a design rejection.
    planner=ResearchPlanner(tmp_path/'network',client=Client([reply()]),max_api_calls=1)
    # Explicit legacy ledger, containing neither raw response nor rule diagnostic.
    write_json(planner.audit/'counter.json',{'reserved_attempts':1})
    write_json(planner.audit/'attempt_0001.json',{'status':'failed','attempt':1,
        'observation_sha256':digest(obs),'http_status':401,'failure_category':'http_error'})
    assert recovery.rejected_proposal(planner,obs) is None
    # The real user's old audit has no retained validation rule. Do not invent one.
    write_json(planner.audit/'attempt_0001.json',{'status':'failed','attempt':1,
        'observation_sha256':digest(obs),'http_status':200,'finish_reason':'stop',
        'failure_category':'invalid_action','error_type':'ValueError'})
    legacy=recovery.rejected_proposal(planner,obs)
    assert 'old audit did not retain' in legacy['reason'] and legacy['proposal_executed'] is False


def test_invalid_code_syntax_and_safe_messages(tmp_path,monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','SECRET')
    value=asdict(control_proposal('fixed',1,17,'BXPC3'));value['code']='class CandidateHead(:'
    planner=recovery.install_diagnostics(ResearchPlanner(tmp_path/'api',client=Client([reply(content=json.dumps(value))])))
    with pytest.raises(RuntimeError,match='invalid_action'):planner.decide({'source':'only'})
    assert read(planner.audit/'validation_0001.json')['validation_rule']=='Candidate code is not valid Python syntax'
    assert recovery.safe_rule(ValueError('provider supplied SECRET'))=='Proposal rejected by local field or code validation'
    assert recovery.safe_rule(ValueError('Unknown candidate name: SECRET'))=='Unknown candidate name'


def test_legacy_plan_migration_is_narrow(tmp_path):
    old={'launcher':sorted(recovery.LEGACY_LAUNCHERS)[0],'source':'unchanged','core_launcher':'same',
         'cycles':2,'trials':2,'seed':17,'parent':'same'}
    plan={**old,'launcher':'new','proposal_recovery':{'version':1}}
    write_json(tmp_path/'plan.json',old)
    write_json(tmp_path/'search/fold/agent/api/attempt_0002.json',{
        'status':'failed','http_status':200,'finish_reason':'stop','failure_category':'invalid_action'})
    with pytest.raises(ValueError,match='Frozen plan'):recovery.freeze_plan(tmp_path,plan)
    with pytest.raises(ValueError,match='cannot change'):recovery.freeze_plan(tmp_path,{**plan,'cycles':6},True,True)
    with pytest.raises(ValueError,match='cannot change'):recovery.freeze_plan(tmp_path,{**plan,'source':'new'},True,True)
    (tmp_path/'episodes').mkdir()
    with pytest.raises(ValueError,match='before outer'):recovery.freeze_plan(tmp_path,plan,True,True)
    (tmp_path/'episodes').rmdir()
    recovery.freeze_plan(tmp_path,plan,True,True)
    assert read(tmp_path/'recovery/original_plan.json')==old
    assert read(tmp_path/'recovery/migration.json')['data_and_budgets_unchanged']
    recovery.freeze_plan(tmp_path,plan,False,True)  # subsequent ordinary resume


def test_large_budget_and_fixed_controls():
    budget=recovery.ResearchIterationBudget(6,3,3)
    for cycle,score in ((1,.97),(2,.94),(3,.91),(4,.88),(5,.85)):
        assert budget.complete(cycle,[{'status':'evaluated','source_oof_mse':score}],1.)['next_cycle_allowed']
    assert not budget.complete(6,[{'status':'evaluated','source_oof_mse':.82}],1.)['next_cycle_allowed']
    budget=recovery.ResearchIterationBudget(6,3,3)
    assert budget.complete(1,[],1.)['next_cycle_allowed'] and budget.complete(2,[],1.)['next_cycle_allowed']
    assert not budget.complete(3,[],1.)['next_cycle_allowed']
    with pytest.raises(ValueError):recovery.ResearchIterationBudget(9,3,3)
    values=[asdict(recovery.extended_control('fixed',i,17,'BXPC3')) for i in range(1,33)]
    recipes=[digest({k:v for k,v in p.items() if k not in ('reason','tool','expert_id')}) for p in values]
    assert len(set(recipes))==32


def test_search_resumes_rejected_paid_slot_without_refitting(tmp_path,monkeypatch):
    from vcell.model_agent.scientific_memory import ScientificMemory
    from test_round18 import source_frame
    monkeypatch.setenv('DEEPSEEK_API_KEY','SECRET')
    observations=[];bad=asdict(control_proposal('fixed',1,17,'BXPC3'));bad['rank']=64
    class RecordingClient(Client):
        def post(self,url,**kwargs):
            observations.append(json.loads(kwargs['json']['messages'][1]['content']))
            return super().post(url,**kwargs)
    client=RecordingClient([reply(content=json.dumps(asdict(control_proposal('fixed',1,17,'BXPC3')))),
        reply(content=json.dumps(bad)),reply(content=json.dumps(asdict(control_proposal('fixed',3,17,'BXPC3'))))])
    monkeypatch.setattr(runner,'ResearchPlanner',lambda *args,**kwargs:ResearchPlanner(*args,**kwargs,client=client))
    fake=SimpleNamespace(path=tmp_path,provenance={'fit_cell_lines':['A549','MCF7'],'calibration_cell_lines':['HAP1']})
    monkeypatch.setattr(runner,'Registry',lambda _:SimpleNamespace(experts={'expanded_factor_rank16':fake}))
    fits=[]
    monkeypatch.setattr(runner.core,'admit',lambda *args:None)
    monkeypatch.setattr(runner.core,'dispatch',lambda jobs,*args:fits.append(sorted(jobs)))
    monkeypatch.setattr(runner,'export_candidate',lambda worker,dest,*args:dest)
    base=source_frame().loc[lambda f:f.expert_id.eq('zero')]
    def evidence(pl,data,root,pools,extra,suffix,include_zero):
        name=next(iter(next(iter(extra.values()))));new=base.copy();new.expert_id=name;new.mse*=.95
        return tmp_path,pd.concat([base,new],ignore_index=True),{}
    import pandas as pd
    monkeypatch.setattr(runner.core,'evidence_for',evidence)
    cfg={'output_dir':str(tmp_path/'base')};pl={'name':'BXPC3_seed17','outer_context':'BXPC3','seed':17,
        'folds':[{'name':'final','configs':{'expanded':cfg}}]+[{'name':'inner'+c,'held':c,'configs':{'expanded':cfg}} for c in ['A','B','C']]}
    a=SimpleNamespace(model='deepseek-v4-pro',research_effort='high',trials=2,cycles=2,patience_cycles=2,
        max_output_tokens=8192,api_timeout=180,parallel_jobs=1,minimum_gain=.01,max_steps=2,batch_size=8,inner_folds=3)
    summary={'fixed_source_mse':{'zero':1.25},'best_fixed':'zero','contexts':['A','B','C'],'distinct_targets':6,'note':'source'}
    root=tmp_path/'run';root.mkdir();memory=ScientificMemory(tmp_path/'memory.sqlite')
    pools={'inner'+c:tmp_path for c in ['A','B','C']};data={'audit':{'fingerprint':'data'}}
    # Simulate interruption after the failed paid response, before result.json is written.
    original=runner.recovery.rejected_proposal
    def interrupted(planner,obs):
        result=original(planner,obs)
        if result and result.get('provider_call',{}).get('attempt')==2:raise KeyboardInterrupt()
        return result
    monkeypatch.setattr(runner.recovery,'rejected_proposal',interrupted)
    with pytest.raises(KeyboardInterrupt):runner.search_arm(a,pl,data,root,pools,summary,'agent_plain',[],[],memory)
    assert len(fits)==1 and len(client.calls)==2
    # Fourth request is an explicit stop; request two is not paid again.
    stop=asdict(control_proposal('fixed',1,17,'BXPC3'));stop.update(tool='stop',code='')
    client.responses=iter([reply(content=json.dumps(asdict(control_proposal('fixed',3,17,'BXPC3')))),reply(content=json.dumps(stop))])
    monkeypatch.setattr(runner.recovery,'rejected_proposal',original)
    result=runner.search_arm(a,pl,data,root,pools,summary,'agent_plain',[],[],memory)
    assert len(fits)==2 and len(client.calls)==4 and result['provider_summary']['attempted_requests']==4
    assert result['trials'][1]['proposal_executed'] is False
    assert observations[2]['previous_source_trials'][1]['reason']=='Unsupported proposal recipe'
    assert not (root/'search/BXPC3_seed17/agent_plain/trial_02/head.py').exists()
    verify(root/'search/BXPC3_seed17/agent_plain')
    assert runner.search_arm(a,pl,data,root,pools,summary,'agent_plain',[],[],memory)==result
    assert len(client.calls)==4 and len(fits)==2
