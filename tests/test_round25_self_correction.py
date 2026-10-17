import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from vcell.evolution import self_correction as sc
from vcell.model_agent.artifacts import read
from vcell.model_agent.contracts import digest
from vcell.round25 import runtime
from vcell.round25.provider import ResearchGateway
from vcell.round25.shinka_hooks import select_archive_action
from vcell.utils import write_json,file_sha256
from test_round24 import fixture_root

REPO=Path(__file__).resolve().parents[1]


def controller(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    spec=importlib.util.spec_from_file_location('round25_controller',REPO/'scripts/run_round25.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def make_pool(pool,arm='meta',task='repeat'):
    pool.mkdir(parents=True,exist_ok=True);fold=fixture_root(pool)
    cfg=read(REPO/'configs/round25.json');cfg.update(panel_genes=6,steps=5,worker_seconds=20)
    write_json(pool/'folds.json',{'search':[fold],'confirmation':[fold]})
    write_json(pool/'run_plan.json',{'repo':str(REPO),'python':sys.executable,'framework_sha256':runtime.fingerprint(REPO),
        'task':task,'arm':arm,'config':cfg,'budget':{'root':str(pool.parent),'worker_root':str(pool/'assessments'),'seed':17,'device':'cpu','steps':5,'seconds':20,
            'ram_bytes':2*1024**3,'evaluation_seconds_max':3600}})
    reference={fold['name']:{'path':fold['baseline'],'sha256':fold['baseline_sha256']}}
    write_json(pool/'baseline_selection.json',{'search':reference,'confirmation':reference})
    body='# EVOLVE-BLOCK-START\ndef fit_predict(data,budget):\n    return data["query_support"]\n# EVOLVE-BLOCK-END\n'
    (pool/'seed.py').write_text(body)
    return cfg,body


def toy_execute(code,d,*args):
    if 'BROKEN_CANDIDATE' in code:raise TypeError('actual controlled dtype failure')
    return d['query_support']*(.9 if 'REPAIRED_CANDIDATE' in code else 1.)


def production_baseline_pool(tmp_path, monkeypatch):
    m=controller(monkeypatch)
    campaign=tmp_path/'campaign';pool=campaign/'baselines/repeat'
    cfg,body=make_pool(pool,arm='fixed');folds={'repeat':read(pool/'folds.json')}
    # Recreate the production plan, including the sibling accounting directory.
    (pool/'run_plan.json').unlink();(pool/'folds.json').unlink()
    (pool/'baseline_selection.json').unlink()
    parent=tmp_path/'parent';write_json(parent/'COMPLETE.json',{'fixture':True})
    m.pool_plan(pool,campaign/'baseline_budget','repeat','fixed',cfg,folds,parent,'cpu',baseline=True)
    return pool,body


def test_production_baseline_artifacts_use_separate_shared_budget(tmp_path,monkeypatch):
    from vcell.round25.prepare import freeze_baselines
    pool,body=production_baseline_pool(tmp_path,monkeypatch)
    b=read(pool/'run_plan.json')['budget'];calls=[]
    assert not pool.is_relative_to(Path(b['root']))
    assert Path(b['worker_root'])==pool/'assessments'
    def execute(code,data,dest,budget,python):
        calls.append((Path(dest),Path(budget['root'])))
        dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'input.npz',**data)
        return toy_execute(code,data)
    monkeypatch.setattr(runtime,'execute',execute)
    selected=freeze_baselines(pool,{'uniform_ensemble':body})
    assert selected['search_record']['status']==selected['confirmation_record']['status']=='valid'
    assert calls and all(d.is_relative_to(pool/'assessments') and r==pool.parents[1]/'baseline_budget' for d,r in calls)
    assert list((Path(b['root'])/'input_cache').glob('*.npz'))
    before=len(calls)
    assert freeze_baselines(pool,{'uniform_ensemble':body})==selected and len(calls)==before
    # Repeat a worker write after deduplication; detach the input link before writing.
    dest=calls[0][0];path=dest/'input.npz';original=path.read_bytes()
    runtime.shared_execute(body,{'query_support':np.ones((1,6))},dest,b,sys.executable)
    assert any(p.read_bytes()==original for p in (Path(b['root'])/'input_cache').glob('*.npz'))
    before=len(calls)
    for outside in (pool.parent/'transfer/assessments/worker',Path(b['root'])/'worker',tmp_path/'outside'):
        with pytest.raises(ValueError,match='artifact root'):
            runtime.shared_execute(body,{},outside,b,sys.executable)
        assert not outside.exists()
    assert len(calls)==before


def action_for(r,kind='refine_parent'):
    return sc.ResearchAction(kind,'context-specific source loss','Separate shared response from residual signal',
        'hierarchical residual',r['module_sha256'],None,'high','none',
        {'metric':'mse','direction':'decrease','minimum_change':.00001},'Change archive parent',
        'Measured context disagreement',[r['module_sha256']],.6)


def test_real_assessment_source_feedback_and_confirmation_are_separate(tmp_path):
    pool=tmp_path/'arms/meta/repeat';cfg,body=make_pool(pool)
    r=runtime.assess(pool,body,execute_fn=toy_execute)
    assert r['status']=='valid'
    # Confirmation records and their scores cannot enter source/critic evidence.
    hidden=copy.deepcopy(r);hidden['identity']['phase']='confirmation';hidden['error']='CONFIRMATION_SENTINEL'
    write_json(pool/'assessments/hidden/assessment.json',hidden)
    info=sc.evidence(pool)
    assert 'CONFIRMATION_SENTINEL' not in json.dumps(info)
    assert info['scope']=='source_search_only' and info['candidates'][0]['diagnosis']['by_context']
    assert info['input_contract']['query_support']['shape']==[3,6] and 'truth' not in info['input_contract']
    assert sc.ResearchAction.parse(asdict(action_for(r)),info).parent_code_sha256==r['module_sha256']
    bad=asdict(action_for(r));bad['parent_code_sha256']='unknown'
    with pytest.raises(ValueError,match='Parent'):sc.ResearchAction.parse(bad,info)
    with pytest.raises(ValueError,match='Freeze'):runtime.assess(pool,body,phase='confirmation',execute_fn=toy_execute)
    with pytest.raises(ValueError,match='outer'):runtime.assess(pool,body,phase='outer',execute_fn=toy_execute)
    assert runtime.assess(pool,body,execute_fn=lambda *args:pytest.fail('Refitted cached candidate'))==r
    data=read(Path(r['directory'])/'assessment.json');data['metrics']['public']['mse']=0
    write_json(Path(r['directory'])/'assessment.json',data)
    with pytest.raises(ValueError,match='checksum'):sc.evidence(pool)


def test_source_metrics_keep_undefined_strong_response_and_identity_observation(tmp_path):
    pool=tmp_path/'arms/meta/repeat';cfg,body=make_pool(pool)
    r=runtime.assess(pool,body,execute_fn=toy_execute);folder=Path(r['directory'])
    a=pd.read_csv(folder/'scores.csv');b=pd.read_csv(folder/'baseline.csv');c=pd.read_csv(folder/'controls.csv')
    a['strong_mse']=np.nan;b['strong_mse']=np.nan
    m=runtime.measure(a,b,c,cfg,'repeat')['public']
    assert m['strong_mse'] is None and m['strong_mse_ratio'] is None and not m['target_response_candidate']
    assert m['correct_vs_identity_wrong_gain']==pytest.approx(0.)  # A support-only method need not use target names.


def test_counterfactuals_and_unseen_reference_subsets_are_reported(tmp_path):
    pool=tmp_path/'arms/meta/transfer';_,body=make_pool(pool,task='transfer')
    r=runtime.assess(pool,body,execute_fn=toy_execute)
    assert r['status']=='valid'
    subsets=r['metrics']['subsets']
    assert subsets['reference_available']['rows']==3
    assert subsets['globally_unseen_target']['rows']==0 and subsets['globally_unseen_target']['mse'] is None
    assert len(pd.read_csv(Path(r['directory'])/'controls.csv').variant.unique())==6


def test_native_archive_hook_repairs_actual_failed_parent_and_crosses_methods():
    def program(code,correct,score):return SimpleNamespace(code=code,correct=correct,combined_score=score,id=code)
    a=program('valid A',True,1);b=program('valid B',True,2);broken=program('failed code',False,-1)
    db=SimpleNamespace(get_all_programs=lambda:[a,b,broken])
    sha=lambda p:hashlib.sha256(p.code.encode()).hexdigest()
    original=lambda *args,**kwargs:(a,[],[],False)
    result=select_archive_action(db,original,{'action':'repair_candidate','parent_code_sha256':sha(broken)},set())
    assert result[0] is broken and result[3]
    result=select_archive_action(db,original,{'action':'crossover_methods','parent_code_sha256':sha(b),'second_parent_sha256':sha(a)},set())
    assert result[0] is b and result[1][0] is a and not result[3]
    result=select_archive_action(db,original,None,{sha(a)})
    assert result[0] is b
    with pytest.raises(ValueError,match='valid'):select_archive_action(db,original,{'action':'repair_candidate','parent_code_sha256':sha(a)},set())


def test_provider_resume_replays_saved_response_without_double_charge(tmp_path,monkeypatch):
    from vcell.round22 import gateway as base
    calls=[]
    def post(*args,**kwargs):
        calls.append(kwargs['json'])
        return SimpleNamespace(ok=True,status_code=200,json=lambda:{'id':'mock','model':'deepseek-v4-pro',
            'choices':[{'message':{'content':'answer'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':3,'completion_tokens':2}})
    monkeypatch.setattr(base.requests,'post',post)
    body={'model':'deepseek-v4-pro','messages':[{'role':'user','content':'source only'}],'max_tokens':100}
    p=tmp_path/'api.json'
    for iteration in range(2):
        relay=ResearchGateway(p,'test-not-secret','deepseek-v4-pro','local',3,1000,0,max_tokens=100)
        try:
            relay.scope='turn_1';relay.role='mutation';status,response=relay.forward(body)
            assert status==200 and response['choices'][0]['message']['content']=='answer'
            assert len(relay.records)==1 and relay.records[0]['role']=='mutation'
        finally:relay.server.server_close()
    assert len(calls)==1
    p2=tmp_path/'new.json';relay=ResearchGateway(p2,'test','deepseek-v4-pro','local',3,1000,0,max_tokens=100)
    try:
        relay.scope='ambiguous';relay.role='repair'
        canonical={**body,'reasoning_effort':'low','thinking':{'type':'enabled'},'n':1}
        key=hashlib.sha256(json.dumps({'ticket':relay.scope,'role':relay.role,'body':canonical},sort_keys=True).encode()).hexdigest()
        write_json(p2.parent/'provider_responses'/(key+'.json'),{'state':'reserved'})
        with pytest.raises(RuntimeError,match='Unresolved'):relay.forward(body)
        assert len(calls)==1
    finally:relay.server.server_close()


def test_three_arm_bounded_repair_loop_and_exact_resume(tmp_path,monkeypatch):
    m=controller(monkeypatch);root=tmp_path/'campaign';armroot=root/'arms/repair'
    cfg=None
    for task in ('repeat','transfer'):
        cfg,_=make_pool(armroot/task,arm='repair',task=task)
    cfg.update(initial_valid_per_arm=1,valid_target_per_arm=1,max_turns_per_arm=5,api_requests_per_arm=8)
    actual=runtime.assess
    monkeypatch.setattr(m,'assess',lambda *args,**kwargs:actual(*args,**kwargs,execute_fn=toy_execute))
    gateways=[];requests=[]
    class FakeGateway:
        def __init__(self,path,*args,**kwargs):self.path=Path(path);self.records=[];self.scope='';self.role='';gateways.append(self)
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(m,'ResearchGateway',FakeGateway)
    def run(argv,**kwargs):
        pool=Path(argv[argv.index('--pool')+1]);active=read(pool/'active_action.json');g=gateways[-1]
        requests.append(active.get('action'))
        g.records.append({'status':'complete','charged_output_tokens':1,'role':g.role});write_json(g.path,g.records)
        # Same failed lineage gets two repairs; only repair #2 succeeds.
        marker='REPAIRED_CANDIDATE' if len(requests)==3 else 'BROKEN_CANDIDATE'
        body=f'# {marker} {len(requests)}\ndef fit_predict(data,budget):\n    return data["query_support"]\n'
        r=m.assess(pool,body)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(m.subprocess,'run',run);monkeypatch.setenv('DEEPSEEK_API_KEY','mock-not-a-key')
    result=m.search_arm(root,'repair',cfg,'mock-shinka')
    assert len(requests)==3 and requests[0] is None
    assert [r['action'] for r in requests[1:]]==['repair_candidate','repair_candidate']
    assert max(result['state']['repair_counts'].values())==2
    outcomes=[r for r in m.events(root/'research_decisions.jsonl') if r['kind']=='measured_action_outcome']
    assert outcomes[-1]['prediction']['status']=='supported'
    m.search_arm(root,'repair',cfg,'mock-shinka');assert len(requests)==3


def test_parentless_new_mechanism_has_a_predeclared_measured_reference(tmp_path,monkeypatch):
    m=controller(monkeypatch);root=tmp_path/'campaign';armroot=root/'arms/meta'
    for task in ('repeat','transfer'):cfg,body=make_pool(armroot/task,task=task)
    cfg.update(initial_valid_per_arm=1,valid_target_per_arm=1,max_turns_per_arm=2)
    actual=runtime.assess
    monkeypatch.setattr(m,'assess',lambda *args,**kwargs:actual(*args,**kwargs,execute_fn=toy_execute))
    gateways=[]
    class FakeGateway:
        def __init__(self,path,*args,**kwargs):self.path=Path(path);self.records=[];gateways.append(self)
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(m,'ResearchGateway',FakeGateway)
    def choose(pool,*args):
        seed=m.assess(pool,(pool/'seed.py').read_text())
        action=asdict(action_for(seed,'explore_new_mechanism'));action['parent_code_sha256']=None
        return sc.ResearchAction(**action),None,None
    monkeypatch.setattr(m,'choose_action',choose)
    def run(argv,**kwargs):
        pool=Path(argv[argv.index('--pool')+1]);g=gateways[-1]
        g.records.append({'status':'complete','charged_output_tokens':1});write_json(g.path,g.records)
        m.assess(pool,'# REPAIRED_CANDIDATE\ndef fit_predict(data,budget):\n    return data["query_support"]\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(m.subprocess,'run',run);monkeypatch.setenv('DEEPSEEK_API_KEY','mock')
    m.search_arm(root,'meta',cfg,'unused')
    plan=read(armroot/'turns/000/plan.json')
    outcome=next(r for r in m.events(root/'research_decisions.jsonl') if r['kind']=='measured_action_outcome')['prediction']
    assert outcome['reference_kind']=='predeclared_fixed_seed'
    assert outcome['reference_sha256']==plan['prediction_reference_sha256']
    assert outcome['status'] in ('supported','not_supported')


def test_meta_changes_scientific_hypothesis_without_fake_runtime_repair(tmp_path,monkeypatch):
    m=controller(monkeypatch);pool=tmp_path/'campaign/arms/meta/repeat';cfg,body=make_pool(pool)
    r=runtime.assess(pool,body,execute_fn=toy_execute)
    monkeypatch.setattr(m,'novel',lambda p:[r]*5)
    state={'pending_repairs':{},'repair_counts':{},'repair_origins':{},'critic_at':{'repeat':0},'empty':{'repeat':0},'repeat_share':.6}
    response=asdict(action_for(r,'explore_new_mechanism'));calls=[]
    monkeypatch.setattr(m,'critic_request',lambda *args:calls.append(1) or json.dumps(response))
    action,origin,info=m.choose_action(pool,'meta',state,'repeat',SimpleNamespace(),cfg,'checkpoint')
    assert action.action=='explore_new_mechanism' and origin is None and calls==[1]
    assert state['critic_at']['repeat']==5
    # Re-entering the exact saved checkpoint does not make another critic call.
    state['critic_at']['repeat']=0
    action,_,_=m.choose_action(pool,'meta',state,'repeat',SimpleNamespace(),cfg,'checkpoint')
    assert calls==[1]
    probe=sc.ResearchAction(**{**asdict(action),'action':'run_source_probe','requested_source_diagnostic':'reference_coverage'})
    measured=m.run_probe(pool,probe)
    assert measured['scope']=='source_search_only' and measured['measured']


def test_two_finalist_tiers_freeze_source_only_before_confirmation(tmp_path,monkeypatch):
    m=controller(monkeypatch);pool=tmp_path/'arms/meta/repeat';make_pool(pool)
    def r(sha,mse,response):return {'module_sha256':sha,'identity':{'phase':'search'},'metrics':{'public':{
        'mse':mse,'overall_error_candidate':True,'target_response_candidate':response}}}
    source=[r('overall',.8,False),r('response',.9,True)]
    monkeypatch.setattr(m,'novel',lambda p:source)
    selected=m.freeze_finalists(pool)
    assert [r['module_sha256'] for r in selected['finalists']]==['overall','response']
    source.reverse();assert m.freeze_finalists(pool)==selected
    cfg=read(REPO/'configs/round25.json');m.cfg_check(cfg)
    cfg['independent_confirmation']=True
    with pytest.raises(ValueError,match='independent'):m.cfg_check(cfg)


def test_exhausted_repair_lineage_cannot_reset_through_critic(tmp_path,monkeypatch):
    m=controller(monkeypatch);pool=tmp_path/'campaign/arms/meta/repeat';cfg,body=make_pool(pool)
    seed=runtime.assess(pool,body,execute_fn=toy_execute)
    bad=runtime.assess(pool,'# BROKEN_CANDIDATE\ndef fit_predict(data,budget):\n    return 0\n',execute_fn=toy_execute)
    state={'pending_repairs':{'repeat':{'parent':bad['module_sha256'],'origin':'original'}},
        'repair_counts':{'original':2},'repair_origins':{bad['module_sha256']:'original'},
        'critic_at':{'repeat':0},'empty':{'repeat':5},'repeat_share':.6}
    response=asdict(action_for(bad,'repair_candidate'))
    monkeypatch.setattr(m,'critic_request',lambda *args:json.dumps(response))
    monkeypatch.setattr(m,'novel',lambda p:[])
    action,origin,_=m.choose_action(pool,'meta',state,'repeat',SimpleNamespace(),cfg,'exhausted')
    assert action.action=='explore_new_mechanism' and origin is None and state['repair_counts']=={'original':2}


def test_unresolved_provider_ticket_stops_before_new_search_request(tmp_path,monkeypatch):
    m=controller(monkeypatch);root=tmp_path/'campaign'
    write_json(root/'arms/meta/provider_responses/pending.json',{'state':'reserved'})
    monkeypatch.setattr(m,'ResearchGateway',lambda *args,**kwargs:pytest.fail('Issued another paid request'))
    with pytest.raises(RuntimeError,match='before issuing'):m.search_arm(root,'meta',read(REPO/'configs/round25.json'),'unused')


def test_runtime_code_or_comparator_drift_cannot_reuse_cached_scores(tmp_path,monkeypatch):
    pool=tmp_path/'arms/meta/repeat';_,body=make_pool(pool)
    runtime.assess(pool,body,execute_fn=toy_execute)
    monkeypatch.setattr(runtime,'fingerprint',lambda repo:'changed')
    with pytest.raises(ValueError,match='code changed'):runtime.assess(pool,body,execute_fn=toy_execute)


def test_critic_prediction_is_scored_against_measured_parent_not_self_report():
    before={'status':'valid','module_sha256':'parent','metrics':{'public':{'mse':1.}}}
    action=action_for(before)
    after={'status':'valid','metrics':{'public':{'mse':1.1}}}
    assert sc.prediction_outcome(action,before,after)['status']=='not_supported'
    after['metrics']['public']['mse']=.9
    assert sc.prediction_outcome(action,before,after)['status']=='supported'


def test_resource_failure_is_repairable_but_campaign_quota_is_not():
    assert not sc.quota_failure({'category':'resource_limit','error':'Candidate wall-time cap exceeded'})
    assert not sc.quota_failure({'category':'resource_limit','error':'Candidate RSS cap exceeded'})
    assert sc.quota_failure({'category':'resource_limit','error':'Round22 evaluation-time cap exhausted'})
    assert sc.quota_failure({'error':'Task search compute quota exhausted'})


def test_research_memory_keeps_hypotheses_across_batch_and_excludes_other_arms(tmp_path):
    from vcell.evolution.journal import append
    root=tmp_path/'campaign';pool=root/'arms/meta/repeat';_,body=make_pool(pool)
    runtime.assess(pool,body,execute_fn=toy_execute)
    path=root/'research_decisions.jsonl'
    append(path,'research_action',phase='search',arm='meta',task='repeat',action={'hypothesis':'KEEP_ME'})
    for i in range(10):append(path,'measured_action_outcome',phase='search',arm='meta',task='repeat',prediction=None)
    append(path,'research_action',phase='confirmation',arm='meta',task='repeat',action={'hypothesis':'CONFIRMATION_SENTINEL'})
    append(path,'research_action',phase='search',arm='repair',task='repeat',action={'hypothesis':'OTHER_ARM_SENTINEL'})
    text=json.dumps(sc.evidence(pool))
    assert 'KEEP_ME' in text and 'CONFIRMATION_SENTINEL' not in text and 'OTHER_ARM_SENTINEL' not in text


def test_full_controller_freezes_all_searches_and_resumes_without_paid_search(tmp_path,monkeypatch):
    m=controller(monkeypatch);cfg=read(REPO/'configs/round25.json');cfg['panel_genes']=6
    config=tmp_path/'config.json';write_json(config,cfg)
    parent=tmp_path/'parent';write_json(parent/'COMPLETE.json',{'fixture':True})
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(m.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout='{"version":"0.0.7"}',stderr=''))
    monkeypatch.setattr(m,'preflight',lambda *args:None)
    def prepare(root,*args):
        for arm in cfg['arms']:
            for task in cfg['tasks']:
                pool=root/'arms'/arm/task
                pool.mkdir(parents=True,exist_ok=True);write_json(pool/'run_plan.json',{'task':task})
    monkeypatch.setattr(m,'prepare',prepare)
    calls=[]
    def search(root,arm,*args):
        p=root/'arms'/arm/'SEARCH_COMPLETE.json'
        if p.exists():return read(p)
        calls.append(arm);write_json(p,{'mock_paid_search':True});return read(p)
    monkeypatch.setattr(m,'search_arm',search);monkeypatch.setattr(m,'novel',lambda p:[])
    attempts=[]
    def report(root,*args):
        assert set(read(root/'search_frozen.json'))==set(cfg['arms'])
        assert all((root/'arms'/arm/task/'search_frozen.json').exists() for arm in cfg['arms'] for task in cfg['tasks'])
        attempts.append(1)
        if len(attempts)==1:raise RuntimeError('interrupted after all source selections froze')
        write_json(root/'summary.json',{'test_evaluated':False})
    monkeypatch.setattr(m,'report',report)
    a=SimpleNamespace(work_dir=str(tmp_path),name='campaign',round24_run=str(parent),config=str(config),resume=False,
        device='cpu',mode='all',shinka_python=sys.executable)
    with pytest.raises(RuntimeError,match='source selections froze'):m.launch(a)
    a.resume=True;root=m.launch(a)
    assert calls==cfg['arms'] and (root/'COMPLETE.json').exists()
    m.launch(a);assert calls==cfg['arms']
    cfg['seed']+=1;write_json(config,cfg)
    with pytest.raises(ValueError):m.launch(a)


def test_all_roles_share_a_single_provider_cap_and_changed_ticket_is_not_resent(tmp_path,monkeypatch):
    from vcell.round22 import gateway as base
    calls=[]
    def post(*args,**kwargs):
        calls.append(1)
        return SimpleNamespace(ok=True,status_code=200,json=lambda:{'choices':[{'message':{'content':'ok'},'finish_reason':'stop'}],
            'usage':{'completion_tokens':2}})
    monkeypatch.setattr(base.requests,'post',post)
    relay=ResearchGateway(tmp_path/'ledger.json','mock','deepseek-v4-pro','local',1,1000,0,max_tokens=100)
    body={'model':'deepseek-v4-pro','messages':[{'role':'user','content':'source'}],'max_tokens':100}
    try:
        relay.scope='turn_1';relay.role='mutation';assert relay.forward(body)[0]==200
        with pytest.raises(ValueError,match='payload changed'):relay.forward({**body,'messages':[{'role':'user','content':'different'}]})
        relay.scope='critic_1';relay.role='critic';assert relay.forward(body)[0]==400
        assert len(calls)==1 and len(relay.records)==1
    finally:relay.server.server_close()


@pytest.mark.skipif(sys.platform!='linux',reason='Landlock/seccomp worker requires Linux')
def test_real_kernel_baseline_with_sibling_budget_directory(tmp_path,monkeypatch):
    from vcell.round25.prepare import freeze_baselines
    pool,body=production_baseline_pool(tmp_path,monkeypatch)
    selected=freeze_baselines(pool,{'uniform_ensemble':body})
    assert selected['search_record']['status']==selected['confirmation_record']['status']=='valid'
    ledger_root=pool.parents[1]/'baseline_budget'
    ledger=read(ledger_root/'evaluation_budget.json')
    assert ledger['workers'] and all(r['status']=='complete' for r in ledger['workers'].values())
    assert all(Path(k.rsplit('#',1)[0]).is_relative_to(pool/'assessments') for k in ledger['workers'])
    assert runtime.charged(ledger_root)>0 and not (pool/'evaluation_budget.json').exists()
    assert freeze_baselines(pool,{'uniform_ensemble':body})==selected


@pytest.mark.skipif(sys.platform!='linux',reason='Landlock/seccomp worker requires Linux')
def test_real_kernel_worker_and_contract_repair(tmp_path):
    pool=tmp_path/'arms/meta/repeat';_,body=make_pool(pool)
    broken='def fit_predict(data,budget):\n    raise ValueError("known shape error")\n'
    bad=runtime.assess(pool,broken);assert bad['status']=='invalid' and 'known shape error' in bad['error']
    good=runtime.assess(pool,body);assert good['status']=='valid',good.get('error')
    assert runtime.assess(pool,body)==good
    assert any((pool.parent/'input_cache').glob('*.npz'))
