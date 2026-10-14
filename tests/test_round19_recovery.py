"""Length-truncated responses are accounted as missing decisions, not retried."""
import json
from pathlib import Path
import sys
import pytest
from vcell.model_agent.artifacts import read
from vcell.model_agent.contracts import digest
from vcell.round19.agents import TrainingAgent,IterationAgent,DEFAULT_RECIPE
from vcell.utils import write_json
from test_deepseek_agent import Client,reply
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import round19_recovery as recovery


@pytest.mark.parametrize('cls,role',[ (TrainingAgent,'training'),(IterationAgent,'review') ])
def test_truncation_never_retried_or_executed(tmp_path,monkeypatch,cls,role):
    monkeypatch.setenv('DEEPSEEK_API_KEY','SECRET_NOT_SAVED')
    response=reply(finish='length');response.value['usage'].update(completion_tokens=8192,
        completion_tokens_details={'reasoning_tokens':8192})
    # Even a complete-looking message cannot be accepted when the finish is length.
    client=Client([response]);p=cls(tmp_path/role,client=client,max_api_calls=100)
    obs={'iteration':6,'source_mse':.006,'outer_results_available':False}
    action,failure=recovery.call(p,obs,role)
    assert action is None and failure['feedback_available'] is False
    assert failure['output_tokens']==failure['reasoning_tokens']==8192
    assert failure['attempt']==1 and failure['retry_sent'] is False
    assert recovery.call(p,obs,role)==(None,failure) and len(client.calls)==1
    assert p.summary()['attempted_requests']==1 and p.summary()['successful_actions']==0
    assert 'SECRET_NOT_SAVED' not in ''.join(f.read_text() for f in tmp_path.rglob('*.json'))


def test_old_truncated_audit_recovers_without_api_key(tmp_path,monkeypatch):
    p=IterationAgent(tmp_path/'review',client=Client([]),max_api_calls=100)
    obs={'iteration':6,'outer_results_available':False}
    write_json(p.audit/'counter.json',{'reserved_attempts':1})
    write_json(p.audit/'attempt_0001.json',{'attempt':1,'status':'failed','http_status':200,
        'finish_reason':'length','failure_category':'incomplete_response','observation_sha256':digest(obs),
        'output_tokens':8192,'reasoning_tokens':8192})
    monkeypatch.delenv('DEEPSEEK_API_KEY',raising=False)
    assert recovery.call(p,obs,'review')[1]['paid_request_retained']
    assert p.summary()['attempted_requests']==1 and not list(p.audit.glob('decisions/*.json'))


def test_transport_and_filtering_still_stop(tmp_path,monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','SECRET')
    for i,response in enumerate((reply(finish='content_filter'),reply())):
        if i==1:response.status_code=401
        p=IterationAgent(tmp_path/str(i),client=Client([response]))
        with pytest.raises(RuntimeError):recovery.call(p,{'iteration':1},'review')
        assert p.summary()['attempted_requests']==1


def test_migration_keeps_scopes_and_worker_code(tmp_path,monkeypatch):
    # This fixture represents the known old, unchanged worker implementation.
    # Current model code is deliberately ineligible for that historical migration.
    monkeypatch.setattr(recovery,'namespace_identity',lambda:recovery.LEGACY_MODEL_NAMESPACE)
    old={'code':next(iter(recovery.LEGACY_CODES)),'iterations':100,'max_steps':500,'effort':'high','seed':17}
    new={**old,'code':'new','truncation_recovery':{'version':1}}
    write_json(tmp_path/'plan.json',old)
    write_json(tmp_path/'api/review/attempt_0006.json',{'attempt':6,'status':'failed','http_status':200,
        'finish_reason':'length','failure_category':'incomplete_response'})
    with pytest.raises(ValueError,match='Frozen'):recovery.freeze_plan(tmp_path,new)
    for update in ({'iterations':200},{'max_steps':1000},{'effort':'low'},{'seed':29}):
        with pytest.raises(ValueError,match='cannot change'):recovery.freeze_plan(tmp_path,{**new,**update},True,True)
    write_json(tmp_path/'search_frozen.json',{})
    with pytest.raises(ValueError,match='source-only'):recovery.freeze_plan(tmp_path,new,True,True)
    (tmp_path/'search_frozen.json').unlink()
    recovery.freeze_plan(tmp_path,new,True,True)
    assert read(tmp_path/'recovery/original_plan.json')==old
    assert read(tmp_path/'recovery/migration.json')['paid_attempts_retained']
    assert read(tmp_path/'recovery/migration.json')['model_namespace_sha256']==recovery.LEGACY_MODEL_NAMESPACE
    recovery.freeze_plan(tmp_path,new,False,True)


def test_changed_worker_code_cannot_use_historical_migration(tmp_path,monkeypatch):
    old={'code':next(iter(recovery.LEGACY_CODES)),'iterations':100,'max_steps':500}
    write_json(tmp_path/'plan.json',old)
    monkeypatch.setattr(recovery,'namespace_identity',lambda:'MODIFIED_WORKER_CODE')
    with pytest.raises(ValueError,match='unchanged model/cache/agent'):
        recovery.freeze_plan(tmp_path,{**old,'code':'new','truncation_recovery':{'version':1}},True,True)
