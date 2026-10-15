import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import numpy as np
import pandas as pd
import pytest
from vcell.round22 import data,evaluator,metrics
from vcell.round22.gateway import Gateway
from vcell.utils import write_json,file_sha256
from vcell.model_agent.artifacts import seal

REPO=Path(__file__).resolve().parents[1]
INITIAL=(REPO/'experiments/round22_evolve/initial.py').read_text()


def module():
    spec=importlib.util.spec_from_file_location('r22',REPO/'scripts/run_round22.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def packed(tmp_path):
    rng=np.random.default_rng(11);g=12;n=12;q=4;targets=[f'T{i}' for i in range(n)];qt=[f'Q{i}' for i in range(q)]
    record=data.pack(tmp_path,'fold','repeat',np.arange(n),np.arange(n,n+q),rng.normal(size=(n,g)),rng.normal(size=(q,g)),
        rng.normal(size=(n,5)),rng.normal(size=(q,5)),targets,qt,[f'row:{s}' for s in qt],['C']*q,np.array([f'G{i}' for i in range(g)]),
        rng.normal(size=(n,g)),rng.normal(size=(q,g)),rng.normal(size=(q,g)),rng.normal(size=(q,5)),np.ones((q,g),bool),np.ones(q,bool),np.ones(q,bool),np.zeros(q,bool),
        extra={'control':(rng.normal(size=(n,g)),rng.normal(size=(q,g))),'prior':(rng.normal(size=(n,g)),rng.normal(size=(q,g)))},wrong_targets=targets[:q])
    write_json(tmp_path/'run_plan.json',{'repo':str(REPO),'code_sha256':evaluator.namespace(REPO),'budget':{},'python':sys.executable})
    write_json(tmp_path/'data_plan.json',{'folds':{'repeat':{'search':[record],'confirmation':[record],'outer':[record]}}})
    return record


def test_mutable_whole_method_immutable_grader_wrapper():
    changed=INITIAL.replace('10.*penalty','33.*penalty');evaluator.validate_code(changed,INITIAL)
    with pytest.raises(ValueError,match='wrapper'):evaluator.validate_code(changed.replace('import numpy as np','import os'),INITIAL)
    with pytest.raises(ValueError):evaluator.validate_code(changed.replace('# EVOLVE-BLOCK-END',''),INITIAL)


def test_fixed_reproduction_compares_science_not_runtime():
    import copy
    first={'combined_score':1.,'public':{'macro_mse':.01},'private':{},
        'extra_data':{'task':'repeat','phase':'search','code_sha256':'same',
            'method_seconds':1.2,'peak_vram_bytes':100,'torch_optimizer_steps':1,'cost_scope':'runtime'},
        'text_feedback':json.dumps({'by_source_context':[{'context':'BXPC3','mse':.01}]})}
    second=copy.deepcopy(first)
    second['extra_data'].update(method_seconds=2.3,peak_vram_bytes=200,torch_optimizer_steps=2)
    m=module();m.check_fixed_reproduction(first,second)
    assert first['extra_data']['method_seconds']==1.2
    for key in ('combined_score','public','text_feedback','extra_data'):
        changed=copy.deepcopy(second)
        if key=='combined_score':changed[key]+=1e-12
        elif key=='public':changed[key]['macro_mse']+=1e-12
        elif key=='text_feedback':changed[key]='different context scores'
        else:changed[key]['code_sha256']='changed'
        with pytest.raises(ValueError,match=key):m.check_fixed_reproduction(first,changed)


def test_pack_never_exports_query_truth_and_ids_are_opaque(tmp_path):
    record=packed(tmp_path);d,g,m=evaluator.inputs(record)
    assert 'truth' not in d and 'query_y' not in d and 'query_rep2' not in d
    assert not any('Q' in s for s in d['query_ids'])
    assert set(d['train_targets']).isdisjoint(d['query_targets'])
    assert g['truth'].shape==(4,12)
    assert set(d['train_targets']).isdisjoint(d['calibration_targets'])


def test_evaluator_truth_stays_private_and_counterfactuals_use_same_method(tmp_path):
    record=packed(tmp_path);calls=[]
    def worker(code,d,dest,budget,python):
        assert 'truth' not in d;calls.append(d);return d['query_support']
    result=evaluator.evaluate(INITIAL,tmp_path,'repeat','search',tmp_path/'score',worker)
    assert len(calls)==3
    assert np.array_equal(calls[0]['train_y'],calls[1]['train_y'])
    assert not np.array_equal(calls[0]['query_support'],calls[1]['query_support'])
    assert result['private']=={} and result['extra_data']['phase']=='search'
    assert (tmp_path/'score/counterfactual_scores.csv').is_file()


def test_confirmation_and_outer_require_two_freezes(tmp_path):
    packed(tmp_path)
    with pytest.raises(ValueError,match='search freeze'):evaluator.evaluate(INITIAL,tmp_path,'repeat','confirmation',tmp_path/'bad')
    write_json(tmp_path/'search_frozen.json',{})
    with pytest.raises(ValueError,match='confirmation freeze'):evaluator.evaluate(INITIAL,tmp_path,'repeat','outer',tmp_path/'bad')


def test_evaluator_rejects_modified_input_or_trusted_code(tmp_path,monkeypatch):
    record=packed(tmp_path);monkeypatch.setattr(evaluator,'namespace',lambda repo:'changed')
    with pytest.raises(ValueError,match='grader'):evaluator.evaluate(INITIAL,tmp_path,'repeat','search',tmp_path/'bad')
    Path(record['input']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='fold'):evaluator.inputs(record)


def test_zero_and_near_zero_not_biological_success():
    truth=np.eye(5);background=np.zeros_like(truth);meta=pd.DataFrame({'context':['C']*5,'target':list('ABCDE')})
    zero=metrics.score(background,truth,background,np.ones_like(truth,bool),meta)
    _,s=metrics.fitness(zero,zero);assert not s['signal_screen_passed']
    tiny=metrics.score(truth*.001,truth,background,np.ones_like(truth,bool),meta)
    _,s=metrics.fitness(tiny,zero);assert s['gain_over_fixed']>0 and not s['signal_screen_passed']


def test_macro_target_equal_weight_not_record_count():
    f=pd.DataFrame({'context':['A','A','A','B'],'target':['x','x','y','x'],'mse':[0.,0.,2.,4.]})
    assert metrics.macro(f,'mse')==2.5


def test_worker_environment_drops_provider_and_pythonpath(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','test-secret');monkeypatch.setenv('PYTHONPATH','private-dir')
    monkeypatch.setenv('ROUND22_GATEWAY_KEY','relay-secret');env=evaluator.worker_env()
    assert not any(k in env for k in ('DEEPSEEK_API_KEY','PYTHONPATH','ROUND22_GATEWAY_KEY'))


def test_gateway_caps_records_real_usage_and_charges_failures(tmp_path,monkeypatch):
    class Response:
        ok=True;status_code=200
        def json(self):return {'id':'id','model':'m','choices':[{'finish_reason':'stop','message':{'content':'code'}}],'usage':{'completion_tokens':23,'prompt_tokens':50}}
    seen=[]
    monkeypatch.setattr('vcell.round22.gateway.requests.post',lambda *a,**k:(seen.append(k) or Response()))
    g=Gateway(tmp_path/'api.json','secret','m','relay',requests_max=1,port=0)
    try:
        assert g.forward({'model':'m','messages':[],'max_tokens':100})[0]==200
        assert g.forward({'model':'m','messages':[],'max_tokens':100})[0]==400
        assert len(seen)==1 and g.records[0]['charged_output_tokens']==23 and g.records[0]['currency_cost'] is None
        assert 'secret' not in (tmp_path/'api.json').read_text()
    finally:g.server.server_close()


def test_gateway_truncation_breaker_persists_and_bounds_paid_retries(tmp_path,monkeypatch):
    class Response:
        ok=True;status_code=200
        def json(self):return {'choices':[{'finish_reason':'length','message':{'content':'','reasoning_content':'not logged'}}],
            'usage':{'completion_tokens':32768,'completion_tokens_details':{'reasoning_tokens':32768}}}
    seen=[]
    monkeypatch.setattr('vcell.round22.gateway.requests.post',lambda *a,**k:(seen.append(k['json']) or Response()))
    path=tmp_path/'api.json';body={'model':'m','messages':[],'max_tokens':99999,'temperature':.5,'top_p':.8}
    g=Gateway(path,'secret','m','relay',port=0)
    try:
        assert g.forward(body)[1]['error']['type']=='incomplete_response'
        assert g.forward(body)[1]['error']['type']=='incomplete_response'
        assert g.forward(body)[1]['error']['type']=='provider_circuit_open'
        assert len(seen)==2 and sum(r['charged_output_tokens'] for r in g.records)==65536
        assert all(r['max_tokens']==32768 and r['reasoning_effort']=='low' for r in seen)
        assert all('temperature' not in r and 'top_p' not in r for r in seen)
        assert body['max_tokens']==99999  # Caller request must not be mutated.
        assert 'not logged' not in path.read_text() and 'secret' not in path.read_text()
    finally:g.server.server_close()
    resumed=Gateway(path,'secret','m','relay',port=0)
    try:
        assert resumed.forward(body)[1]['error']['type']=='provider_circuit_open'
        assert len(seen)==2
    finally:resumed.server.server_close()


def test_gateway_success_resets_breaker_and_total_token_cap_is_unchanged(tmp_path,monkeypatch):
    outcomes=iter(['length','stop','length','stop'])
    class Response:
        ok=True;status_code=200
        def json(self):
            finish=next(outcomes)
            return {'choices':[{'finish_reason':finish,'message':{'content':'code' if finish=='stop' else ''}}],'usage':{'completion_tokens':32768}}
    seen=[]
    monkeypatch.setattr('vcell.round22.gateway.requests.post',lambda *a,**k:(seen.append(1) or Response()))
    g=Gateway(tmp_path/'api.json','secret','m','relay',output_tokens_max=3*32768,port=0)
    try:
        body={'model':'m','messages':[],'max_tokens':32768}
        assert g.forward(body)[0]==400
        assert g.forward(body)[0]==200
        assert g.forward(body)[0]==400
        assert g.forward(body)[1]['error']['type']=='budget_exhausted'
        assert len(seen)==3
    finally:g.server.server_close()


def test_provider_smoke_requires_final_code_without_executing_it():
    spec=importlib.util.spec_from_file_location('shinka_round22',REPO/'scripts/run_shinka_round22.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.check_code_probe('```python\ndef round22_probe():\n    return 22\n```')
    for content in ('ROUND22_READY','', 'def round22_probe():\n    return 21',
            'import os\ndef round22_probe():\n    return 22', 'def round22_probe():\n    raise RuntimeError()'):
        with pytest.raises((ValueError,SyntaxError)):m.check_code_probe(content)


def test_light_archive_excludes_hidden_labels_and_matrices(tmp_path):
    for name in ('private/a/grade.npz','inputs/a/input.npz','workers/a/model.pt','private/a/metadata.csv'):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'private')
    write_json(tmp_path/'summary.json',{'complete':True})
    archive=module().package(tmp_path)
    with tarfile.open(archive) as t:
        assert set(t.getnames())=={'summary.json','ARCHIVE_MANIFEST.json'}


def test_real_linux_kernel_boundary(tmp_path):
    if sys.platform!='linux':pytest.skip('Real Landlock and seccomp require Linux; no Windows sandbox claim')
    # This is intentionally NOT skipped on Linux: the existing CPU CI must prove protection.
    budget={'seed':17,'device':'cpu','steps':1,'seconds':25,'ram_bytes':4*1024**3,'evaluation_seconds_max':100,'root':str(tmp_path)}
    module().preflight(tmp_path,budget,sys.executable)
    assert json.loads((tmp_path/'sandbox_verified.json').read_text())['verified']


def test_fresh_source_partitions_exclude_confirmation_and_outer_truth(tmp_path,monkeypatch):
    genes=np.array([f'G{i}' for i in range(1834)],dtype='U');targets=[f'T{i:02d}' for i in range(30)]
    rows=[];baseline=[];delta=[]
    for ci,context in enumerate(('A549_IFNG','BXPC3_IFNG','MCF7_IFNG','K562_IFNG','HAP1_IFNG')):
        for ti,target in enumerate(targets):
            for rep in ('Rep1','Rep2'):
                rows.append(dict(row_id=f'{context}:{target}:{rep}',context=context,cell_type=context,perturbation=target,dataset='D',stimulus='IFNG',time='24h',modality='CRISPRi',batch=rep,replicate=rep))
                baseline.append(np.sin(np.arange(1834)/51)*(ci+1)+ti*.01)
                delta.append(np.cos(np.arange(1834)/37)*((ti+1)*.01)+ci*.001)
    meta=pd.DataFrame(rows);fake={'meta':meta,'baseline':np.array(baseline,np.float32),'delta':np.array(delta,np.float32),'genes':genes}
    parent=tmp_path/'parent';parent.mkdir();prepared=tmp_path/'prepared';prepared.mkdir();(prepared/'dataset.npz').write_bytes(b'fixture')
    relations=tmp_path/'relations.npz';np.savez_compressed(relations,genes=genes,perturbations=np.array(targets,dtype='U'),relations=np.stack([np.tile(np.linspace(0,i+1,1834),(3,1)) for i in range(30)]))
    source=np.flatnonzero(meta.context.ne('HAP1_IFNG') & meta.perturbation.ne('T00'));outer=np.flatnonzero(meta.context.eq('HAP1_IFNG'))
    cfg={'partition':{'fit':source.tolist(),'calibration':[],'outer':outer.tolist()},'relations':str(relations)}
    fold={'name':'outer','held':'HAP1_IFNG','configs':{'expanded':cfg}}
    inner=[{'held':c} for c in ('A549_IFNG','BXPC3_IFNG','MCF7_IFNG')]
    write_json(parent/'lineage.json',{'data_dir':str(prepared),'held_targets':['T00'],'folds':[{'outer_context':'HAP1_IFNG','folds':[fold,*inner]}]})
    write_json(parent/'summary.json',{'raw_audit':{'independent_repeat_controls_verified':True}});seal(parent)
    monkeypatch.setattr(data,'load_prepared',lambda path:fake)
    first=tmp_path/'first';first.mkdir();p=data.prepare(parent,first,'HAP1_IFNG')
    assert len(p['folds']['transfer']['search'])==3
    confirmation=set(p['confirmation_targets'])
    for task in ('repeat','transfer'):
        for f in p['folds'][task]['search']:
            d,g,m=evaluator.inputs(f)
            assert confirmation.isdisjoint(d['train_targets']) and confirmation.isdisjoint(d['query_targets'])
            assert 'HAP1_IFNG' not in set(meta.iloc[f['source_train_rows']].context)
            d,g,_=evaluator.inputs(f)
            for variant in ('wrong','shuffled'):
                supported=g['common']
                assert np.all(g[variant+'_targets'][supported]!=d['query_targets'][supported])
    # Poison only labels that must remain invisible throughout adaptive search.
    poison=meta.context.eq('HAP1_IFNG') | meta.perturbation.isin(confirmation)
    fake['delta'][poison.to_numpy()]=10000.
    second=tmp_path/'second';second.mkdir();p2=data.prepare(parent,second,'HAP1_IFNG')
    for task in ('repeat','transfer'):
        for f,g in zip(p['folds'][task]['search'],p2['folds'][task]['search']):
            assert f['input_sha256']==g['input_sha256']
            assert f['grade_sha256']==g['grade_sha256']


def test_frozen_resume_cannot_reopen_api_or_search(tmp_path,monkeypatch):
    from argparse import Namespace
    m=module();parent=tmp_path/'parent';parent.mkdir();(parent/'COMPLETE.json').write_text('{}')
    root=tmp_path/'runs/pilot'
    def prepare_fixture(parent,root,context,seed):
        original=json.loads((root/'run_plan.json').read_text());record=packed(root)
        write_json(root/'run_plan.json',original)
        plan=json.loads((root/'data_plan.json').read_text());plan.update(raw_controls_verified=True,scope='synthetic controller fixture')
        write_json(root/'data_plan.json',plan);return plan
    monkeypatch.setattr(m,'prepare',prepare_fixture);monkeypatch.setattr(m,'preflight',lambda *a:None)
    runtimes=[]
    def evaluate_fixture(*args,**kwargs):
        result=evaluator.evaluate(*args,**kwargs,execute_fn=lambda code,d,*rest:d['query_support'])
        # Two real executions report different durations despite identical predictions.
        runtimes.append(len(runtimes)+1.)
        result['extra_data']['method_seconds']=runtimes[-1]
        return result
    monkeypatch.setattr(m,'evaluate',evaluate_fixture)
    a=Namespace(work_dir=str(tmp_path),name='pilot',round21_run=str(parent),resume=False,steps=1,worker_seconds=25,seed=17,device='cpu',evaluation_hours=1.,
        api_max_tokens=32768,reasoning_effort='low',api_failure_limit=2,
        generations=2,tasks=['repeat'],outer_context='HAP1_IFNG',api_requests=3,api_output_tokens=20000,model='m',port=18022,shinka_python=sys.executable,mode='smoke')
    m.launch(a)
    source=json.loads((root/'fixed/repeat/second/metrics.json').read_text())
    record={'arm':'shinka','task':'repeat','generation':1,'code':str(REPO/'experiments/round22_evolve/initial.py'),'sha256':file_sha256(REPO/'experiments/round22_evolve/initial.py'),'metrics':source}
    write_json(root/'search_frozen.json',{'candidates':[record],'all_records':[record],'baseline_selection':json.loads((root/'baseline_selection.json').read_text())})
    stamp=file_sha256(root/'search_frozen.json');monkeypatch.delenv('DEEPSEEK_API_KEY',raising=False)
    monkeypatch.setattr(m,'Gateway',lambda *a,**k:pytest.fail('A frozen search must never create an API relay'))
    a.resume=True;a.mode='run';m.launch(a)
    assert (root/'COMPLETE.json').exists() and file_sha256(root/'search_frozen.json')==stamp
    assert json.loads((root/'confirmation_frozen.json').read_text())['outer_used'] is False
    # Already complete resume also verifies the complete artifact without source fitting.
    m.launch(a)
