"""Mocked HTTP provider, real controller and durable paid-attempt accounting."""
from functools import partial
import json
from pathlib import Path
import sys

import pytest
import requests

from vcell.model_agent.artifacts import read, verify
from vcell.model_agent.contracts import Action
from vcell.model_agent.deepseek import DeepSeekPlanner, list_models
from vcell.model_agent.demo import make_demo
from vcell.model_agent.runtime import execute


class Response:
    def __init__(self,value,status=200):self.value=value;self.status_code=status
    def json(self):return self.value


def reply(tool='predict',expert='fast_linear',finish='stop',content=None):
    return Response({'id':'mock-request','model':'deepseek-v4-pro','choices':[{'finish_reason':finish,
        'message':{'content':content if content is not None else json.dumps({'tool':tool,'expert_id':expert,'reason':'Allowed evidence'}),
                   'reasoning_content':'PRIVATE_REASONING_MUST_NOT_BE_LOGGED'}}],
        'usage':{'prompt_tokens':50,'completion_tokens':30,'prompt_cache_hit_tokens':10,
                 'completion_tokens_details':{'reasoning_tokens':20}}})


class Client:
    def __init__(self,responses):self.responses=iter(responses);self.calls=[]
    def post(self,url,**kwargs):
        self.calls.append((url,kwargs));value=next(self.responses)
        if isinstance(value,BaseException):raise value
        return value
    def get(self,url,**kwargs):return Response({'data':[{'id':'deepseek-v4-pro','name':'Pro','effort':{'supported_levels':['high','max']}}]})


@pytest.fixture(autouse=True)
def key(monkeypatch):monkeypatch.setenv('DEEPSEEK_API_KEY','UNIT_TEST_SECRET_DO_NOT_LOG')


def obs(predicted=False):
    return {'task':{'query_id':'q','context':'A549_IFNG','target':'STAT1'},
        'candidates':{'fast_linear':{'supported':True,'cost_units':1.,'estimated_mse':None}},
        'predictions':{'fast_linear':{'norm':.1}} if predicted else {},'inspected':{},
        'failed_experts':[],'remaining_units':2.,'inspection_cost':.05,'remaining_steps':4}


def test_actual_deepseek_request_format_and_safe_usage(tmp_path):
    client=Client([reply()]);planner=DeepSeekPlanner(tmp_path/'audit',client=client)
    assert planner.decide(obs())==Action('predict','fast_linear','Allowed evidence')
    url,kwargs=client.calls[0];body=kwargs['json']
    assert url=='https://api.deepseek.com/chat/completions'
    assert body['model']=='deepseek-v4-pro' and body['thinking']=={'type':'enabled'} and body['reasoning_effort']=='max'
    assert body['response_format']=={'type':'json_object'} and 'tools' not in body and 'temperature' not in body
    assert kwargs['allow_redirects'] is False
    summary=planner.summary();assert summary['known_output_tokens']==30 and summary['known_reasoning_tokens']==20
    text=''.join(p.read_text() for p in (tmp_path/'audit').rglob('*.json'))+(tmp_path/'audit/calls.jsonl').read_text()
    assert 'UNIT_TEST_SECRET' not in text and 'PRIVATE_REASONING' not in text and 'messages' not in text


@pytest.mark.parametrize('response',[reply(finish='length'),reply(content=''),reply(expert='missing'),
    reply(tool='finish'),reply(content='{"tool":"predict","expert_id":"fast_linear","reason":"ok","shell":"bad"}'),
    reply(content='{"tool":"predict","tool":"finish","expert_id":"fast_linear","reason":"ok"}')])
def test_paid_invalid_responses_are_counted_and_never_fall_back(tmp_path,response):
    planner=DeepSeekPlanner(tmp_path/'audit',client=Client([response]),max_api_calls=1)
    with pytest.raises(RuntimeError,match='failed'):planner.decide(obs())
    assert planner.summary()['attempted_requests']==1 and planner.summary()['known_output_tokens']==30
    with pytest.raises(RuntimeError,match='budget exhausted'):
        DeepSeekPlanner(tmp_path/'audit',client=Client([]),max_api_calls=1).decide(obs())


def test_provider_errors_redacted_and_unknown_usage_is_explicit(tmp_path):
    planner=DeepSeekPlanner(tmp_path/'audit',client=Client([requests.ConnectionError('UNIT_TEST_SECRET_DO_NOT_LOG')]))
    with pytest.raises(RuntimeError) as error:planner.decide(obs())
    assert 'SECRET' not in str(error.value) and planner.summary()['unknown_usage_attempts']==1
    assert read(tmp_path/'audit/attempt_0001.json')['status']=='failed'


def test_decision_cache_replays_without_another_billed_request(tmp_path):
    client=Client([reply()]);planner=DeepSeekPlanner(tmp_path/'audit',client=client)
    action=planner.decide(obs());resumed=DeepSeekPlanner(tmp_path/'audit',client=Client([]))
    assert resumed.decide(obs())==action and resumed.last_call['cached_decision']
    assert resumed.summary()['attempted_requests']==1
    cache=next((tmp_path/'audit/decisions').glob('*.json'));value=read(cache);value['action']['tool']='finish'
    from vcell.utils import write_json
    write_json(cache,value)
    with pytest.raises(ValueError,match='checksum'):resumed.decide(obs())


def test_real_controller_records_provider_calls_and_finished_resume_is_offline(tmp_path,monkeypatch):
    base=make_demo(tmp_path/'base');client=Client([reply(),reply(tool='finish')])
    planner=DeepSeekPlanner(tmp_path/'api',client=client)
    result=execute(base/'registry.json',base/'query.json',tmp_path/'run',planner=planner,planner_identity=planner.identity)
    assert result['selected_expert']=='fast_linear' and result['provider_summary']['attempted_requests']==2
    assert read(tmp_path/'run/trace.json')[0]['provider_call']['response_id']=='mock-request'
    assert execute(base/'registry.json',base/'query.json',tmp_path/'run',resume=True,
                   planner=planner,planner_identity=planner.identity)==result
    monkeypatch.delenv('DEEPSEEK_API_KEY')
    resumed=DeepSeekPlanner(tmp_path/'api',client=Client([]))
    assert execute(base/'registry.json',base/'query.json',tmp_path/'run',resume=True,
                   planner=resumed,planner_identity=resumed.identity)==result


def test_missing_key_and_changed_configuration_do_not_send_requests(tmp_path,monkeypatch):
    client=Client([]);planner=DeepSeekPlanner(tmp_path/'api',client=client)
    monkeypatch.delenv('DEEPSEEK_API_KEY')
    with pytest.raises(RuntimeError,match='DEEPSEEK_API_KEY'):planner.decide(obs())
    assert not client.calls and planner.summary()['attempted_requests']==0
    with pytest.raises(ValueError,match='configuration changed'):DeepSeekPlanner(tmp_path/'api',reasoning_effort='high')


def test_three_round_pilot_resume_and_review_hashes(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import run_deepseek_pilot as pilot
    client=Client([reply(),reply(tool='finish')]*3);factory=partial(DeepSeekPlanner,client=client)
    args=['--work-dir',str(tmp_path),'--name','pilot','--rounds','3']
    # Preparing creates the review archive before completion; later sealing must
    # exclude that regenerable archive, so resume checksum validation still works.
    pilot.main(args+['--prepare-only'],planner_factory=factory)
    assert not client.calls
    root=pilot.main(args+['--resume'],planner_factory=factory);verify(root)
    assert len(client.calls)==6 and read(root/'provider_summary.json')['attempted_requests']==6
    assert pilot.main(args+['--resume'],planner_factory=factory)==root and len(client.calls)==6
    import hashlib,tarfile
    with tarfile.open(root/'model_agent_review_light.tar.gz') as archive:
        manifest=json.load(archive.extractfile('ARCHIVE_MANIFEST.json'))
        for name,entry in manifest['files'].items():
            data=archive.extractfile(name).read();assert hashlib.sha256(data).hexdigest()==entry['sha256']
            assert b'UNIT_TEST_SECRET' not in data and b'PRIVATE_REASONING' not in data
        assert any(name.startswith('api/') and 'attempt_' in name for name in archive.getnames())


def test_model_discovery_only_uses_metadata():
    assert list_models(client=Client([]))[0]['id']=='deepseek-v4-pro'


def test_deleted_attempt_cannot_reset_request_budget(tmp_path):
    planner=DeepSeekPlanner(tmp_path/'api',client=Client([reply()]),max_api_calls=1)
    planner.decide(obs());(tmp_path/'api/attempt_0001.json').unlink()
    with pytest.raises(ValueError,match='counter mismatch'):planner.decide({**obs(),'remaining_steps':3})


def test_missing_audit_cannot_silently_resume_with_fresh_budget(tmp_path):
    import shutil
    base=make_demo(tmp_path/'base');planner=DeepSeekPlanner(tmp_path/'api',client=Client([reply(),reply(tool='finish')]))
    execute(base/'registry.json',base/'query.json',tmp_path/'run',planner=planner,planner_identity=planner.identity)
    shutil.move(str(tmp_path/'api'),str(tmp_path/'saved_api'))
    fresh=DeepSeekPlanner(tmp_path/'api',client=Client([]))
    with pytest.raises(ValueError,match='audit missing'):
        execute(base/'registry.json',base/'query.json',tmp_path/'run',resume=True,planner=fresh,planner_identity=fresh.identity)


def test_real_p4_pilot_exports_matched_tools_and_predicts_without_private_scoring(tmp_path):
    import copy,numpy as np
    from test_round15_p4 import p4_worker_setup
    from test_round15_adapter import fake_aggregation
    from vcell.data import load_prepared
    from vcell.round15_adapter import prepare_extension,POLICY
    from vcell.round12 import make_partition
    from vcell.background_training import pretrain
    from vcell.round15_p4 import worker
    from vcell.model_agent.artifacts import arrays,seal
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import run_deepseek_pilot as pilot
    training=tmp_path/'training';training.mkdir()
    cfg=p4_worker_setup(training);old=load_prepared(cfg['data_dir'])
    fake_aggregation(old,tmp_path/'aggregation')
    expanded_path=tmp_path/'expanded';prepare_extension(old,old,tmp_path/'aggregation',expanded_path,POLICY)
    expanded=load_prepared(expanded_path);old_relations=arrays(cfg['relations'])
    rel=np.zeros((len(expanded['perturbations']),3,len(old['genes'])),np.float32)
    for i,target in enumerate(expanded['perturbations']):
        if target in old['perturbations']:rel[i]=old_relations['relations'][list(old['perturbations']).index(target)]
        else:rel[i,2]=1/len(old['genes'])
    relations=tmp_path/'all_relations.npz';np.savez_compressed(relations,genes=old['genes'],perturbations=expanded['perturbations'],relations=rel)
    parent=tmp_path/'p4'
    cfg.update(relations=str(relations),output_dir=str(parent/'models/target_reference_seed17'));worker(cfg)
    parts=make_partition(old,expanded,'target')
    bg=read(Path(cfg['encoder']).parent/'manifest.json')['config']
    bg.update(data_dir=str(expanded_path),partition=parts,output_dir=str(tmp_path/'expanded_bg'));pretrain(bg)
    cfg2={**cfg,'data_dir':str(expanded_path),'partition':parts,'encoder':str(tmp_path/'expanded_bg/encoder.pt'),
          'variant':'expanded','output_dir':str(parent/'models/target_expanded_seed17')};worker(cfg2);seal(parent)
    client=Client([reply(expert='expanded_factor_rank16'),reply(tool='finish',expert='expanded_factor_rank16')])
    root=pilot.main(['--work-dir',str(tmp_path),'--name','real_pilot','--mode','p4','--p4-run',str(parent),
                     '--protocol','target','--rounds','1'],planner_factory=partial(DeepSeekPlanner,client=client))
    summary=read(root/'summary.json')['episodes'][0]
    assert summary['status']=='selected' and summary['context']=='A549_IFNG'
    assert 'synthetic_selected_mse' not in summary and len(client.calls)==2
    assert (root/'episodes/round_001/selected.npz').exists()
    assert read(root/'inputs.json')['profiles_complete'] is None
