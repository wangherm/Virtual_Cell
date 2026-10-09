"""Tool boundaries, source OOF evidence, budget, recovery and real P4 inference."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest

from vcell.model_agent.artifacts import arrays, read, seal, verify
from vcell.model_agent.contracts import Action, Query, load_query
from vcell.model_agent.demo import make_demo, run_demo
from vcell.model_agent.evidence import fit_profiles
from vcell.model_agent.experts import Expert, Registry, export_p4
from vcell.model_agent.runtime import evaluate, execute
from vcell.model_agent.workflows import combine, query_from_prepared
from vcell.utils import write_json


def test_offline_architecture_demo_and_completed_resume(tmp_path):
    root=run_demo(tmp_path/'demo')
    assert read(root/'stage.json')['llm_called'] is False
    before={p:p.read_bytes() for p in root.glob('*/COMPLETE.json')}
    run_demo(root,True);assert all(p.read_bytes()==value for p,value in before.items())
    for policy in ('fixed','rule','risk','workflow'):
        result=read(root/policy/'result.json');assert result['status']=='selected'
        assert result['spent_units']<=result['budget_units'] and result['synthetic']
        text=json.dumps(read(root/policy/'trace.json'))
        assert 'private_truth' not in text and 'private_oof_scores' not in text and 'query_fingerprint' not in text
        for event in read(root/policy/'trace.json'):
            assert 'delta' not in event['observation'] and 'control' not in event['observation']['task']
    with tarfile.open(root/'model_agent_review_light.tar.gz') as archive:
        assert 'ARCHIVE_MANIFEST.json' in archive.getnames()
        assert not any(name.endswith(('.npz','.csv')) or 'private' in name for name in archive.getnames())


def test_query_and_tool_contract_block_truth_and_reserved_backgrounds(tmp_path):
    root=make_demo(tmp_path/'demo');value=read(root/'query.json')
    for context in ('HT29','ht29_IFNG','Jurkat_other_stimulus'):
        with pytest.raises(ValueError,match='Reserved'):Query(**{**value,'context':context})
    write_json(root/'bad.json',{**value,'mse':0.,'delta':[0,0,0]})
    with pytest.raises(ValueError,match='Unexpected'):load_query(root/'bad.json')
    with pytest.raises(ValueError,match='allowlisted'):Action('shell','fast_linear')


def test_pending_tool_resume_does_not_request_a_second_decision(tmp_path,monkeypatch):
    root=make_demo(tmp_path/'demo');calls=[]
    class Planner:
        def decide(self,obs):
            calls.append(obs)
            return Action('finish','fast_linear') if obs['predictions'] else Action('predict','fast_linear')
    original=Expert.predict
    monkeypatch.setattr(Expert,'predict',lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    args=(root/'registry.json',root/'query.json',root/'injected')
    with pytest.raises(KeyboardInterrupt):execute(*args,planner=Planner(),planner_identity='test-v1')
    assert read(root/'injected/state.json')['pending']['action']['tool']=='predict'
    monkeypatch.setattr(Expert,'predict',original)
    result=execute(*args,resume=True,planner=Planner(),planner_identity='test-v1')
    assert len(calls)==2 and result['spent_units']==1.
    trace=read(root/'injected/trace.json');assert not trace[0]['observation']['predictions']
    assert result['selected_expert']=='fast_linear'


def test_budget_and_unknown_actions_cannot_bypass_controller(tmp_path):
    root=make_demo(tmp_path/'demo')
    class Planner:
        def __init__(self,action):self.action=action
        def decide(self,obs):return self.action
    for name,action,message in [('unknown',Action('predict','missing'),'unregistered'),
                                ('expensive',Action('predict','response_linear'),'budget'),
                                ('finish',Action('finish','fast_linear'),'unexecuted')]:
        with pytest.raises(ValueError,match=message):
            execute(root/'registry.json',root/'query.json',root/name,budget_units=.5,
                    planner=Planner(action),planner_identity='test')
    result=execute(root/'registry.json',root/'query.json',root/'cheap',budget_units=.01)
    assert result['status']=='abstained' and result['spent_units']==0


def test_source_oof_profile_cannot_use_training_calibration_or_outer_context(tmp_path):
    root=make_demo(tmp_path/'demo');spec=read(root/'oof_spec.json')
    # Case variants must not evade whole-background exclusion.
    expert=root/'fast_linear';card=read(expert/'expert.json')
    card['provenance']['calibration_contexts']=['synthetic_a'];write_json(expert/'expert.json',card);seal(expert)
    with pytest.raises(ValueError,match='training or calibration'):fit_profiles(root/'oof_spec.json',root/'bad_profiles')
    card['provenance']['calibration_contexts']=[];write_json(expert/'expert.json',card);seal(expert)
    spec['source_contexts']=['SYNTHETIC_B'];spec['excluded_contexts']=['SYNTHETIC_A'];write_json(root/'bad_spec.json',spec)
    with pytest.raises(ValueError,match='source-only'):fit_profiles(root/'bad_spec.json',root/'bad_profiles')


def test_historical_validation_query_cannot_be_relabelled_by_source_context_list(tmp_path):
    root=make_demo(tmp_path/'demo');q=read(root/'oof_0.json');q['source_partition']='val'
    write_json(root/'oof_0.json',q)
    with pytest.raises(ValueError,match='training-source queries'):fit_profiles(root/'oof_spec.json',root/'bad_profiles')


def test_private_truth_poisoning_never_changes_selected_prediction(tmp_path):
    root=make_demo(tmp_path/'demo');args=(root/'registry.json',root/'query.json',root/'run')
    result=execute(*args,'risk',root/'profiles')
    before=(root/'run/selected.npz').read_bytes();trace=(root/'run/trace.json').read_bytes()
    truth=arrays(root/'private_truth.npz');truth['delta']+=1000;np.savez_compressed(root/'private_truth.npz',**truth)
    scores=evaluate(root/'run',root/'query.json',root/'private_truth.npz',root/'registry.json')
    assert scores['selected_mse']>100 and (root/'run/selected.npz').read_bytes()==before
    assert (root/'run/trace.json').read_bytes()==trace
    assert execute(*args,'risk',root/'profiles',resume=True)==result


def test_tampered_completed_artifact_and_changed_resume_inputs_are_rejected(tmp_path):
    root=make_demo(tmp_path/'demo');args=(root/'registry.json',root/'query.json',root/'run')
    execute(*args)
    with pytest.raises(ValueError,match='identical'):execute(*args,budget_units=4.,resume=True)
    (root/'run/selected.npz').write_bytes(b'bad')
    with pytest.raises(ValueError,match='checksum'):execute(*args,resume=True)


def test_real_p4_export_predicts_same_response_without_training_labels(tmp_path):
    from test_round15_p4 import p4_worker_setup
    from vcell.round15_p4 import worker
    from vcell.data import load_prepared
    from vcell.round15 import reconstruct
    cfg=p4_worker_setup(tmp_path);parent=worker(cfg)
    registry_path=export_p4(parent,tmp_path/'pool');registry=Registry(registry_path)
    data=load_prepared(cfg['data_dir']);index=cfg['partition']['outer'][0];row=data['meta'].iloc[index]
    query_path=query_from_prepared(cfg['data_dir'],str(row.row_id),tmp_path/'query.json');q=load_query(query_path)
    for method in cfg['methods']:
        expected=reconstruct(parent/method/'coefficients.npz')[0]
        # Exported inference is independent of the original prepared dataset.
        np.testing.assert_allclose(registry.experts['reference_'+method].predict(q),expected,rtol=2e-5,atol=2e-6)
    import shutil
    shutil.move(cfg['data_dir'],str(tmp_path/'quarantined_training_data'))
    result=execute(registry_path,query_path,tmp_path/'deploy',budget_units=2)
    assert result['status']=='selected'
    assert not any('per_target' in p.name for p in (tmp_path/'pool').rglob('*'))
    with pytest.raises(ValueError,match='Duplicate'):combine([registry_path,registry_path],tmp_path/'combined.json')


def test_external_planner_cannot_mutate_controller_state_through_observation(tmp_path):
    root=make_demo(tmp_path/'demo')
    class Planner:
        def decide(self,obs):
            obs['predictions']['fast_linear']={'fake':True}
            return Action('finish','fast_linear')
    with pytest.raises(ValueError,match='unexecuted'):
        execute(root/'registry.json',root/'query.json',root/'run',planner=Planner(),planner_identity='test')
