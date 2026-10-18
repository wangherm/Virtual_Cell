import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from vcell.model_agent.artifacts import read,seal
from vcell.round26 import critic,runtime,prepare,native
from vcell.round25 import runtime as old_runtime
from vcell.evolution.self_correction import ResearchAction
from vcell.evolution.journal import append
from vcell.utils import write_json,file_sha256
from test_round25_self_correction import make_pool as previous_pool,toy_execute

REPO=Path(__file__).resolve().parents[1]


def controller(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    spec=importlib.util.spec_from_file_location('round26_controller',REPO/'scripts/run_round26.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def pool_fixture(pool,task='repeat',arm='meta',real=False):
    _,body=previous_pool(pool,arm,task)
    cfg=read(REPO/'configs/round26.json');cfg.update(device='cpu',panel_genes=6,steps=5,worker_seconds=20)
    p=read(pool/'run_plan.json');p.update(framework_sha256=runtime.fingerprint(REPO),config=cfg)
    write_json(pool/'run_plan.json',p)
    base=runtime.assess(pool,body,execute_fn=None if real else toy_execute)
    assert base['status']=='valid',base.get('error')
    selection={**read(pool/'baseline_selection.json'),'method':'support','module_sha256':base['module_sha256'],
        'search_record':base,'confirmation_record':base}
    write_json(pool/'baseline_selection.json',selection)
    # Reassess under the final frozen selector; the shared record must bind its
    # actual plan/folds/comparator hashes rather than a subsequently edited stub.
    base=runtime.assess(pool,body,execute_fn=None if real else toy_execute)
    assert base['status']=='valid',base.get('error')
    path=Path(base['directory'])/'assessment.json'
    write_json(pool/'common_sources.json',[{'record':str(path),'sha256':file_sha256(path)}])
    return cfg,body,base


def payload(info,kind='refine_parent'):
    a=copy.deepcopy(critic.contract(info)['example']);a['action']=kind
    if kind=='run_source_probe':a['requested_source_diagnostic']='reference_coverage'
    if kind=='explore_new_mechanism':a['parent_code_sha256']=None
    return a


class FakeGateway:
    def __init__(self,path=None,*args,**kwargs):
        self.path=Path(path) if path else None;self.records=read(path) if path and Path(path).exists() else []
        self.model='deepseek-v4-pro';self.scope='';self.role='';self.requests_max=kwargs.get('requests',20)
        self.output_tokens_max=700000;self.replies=[]
    def forward(self,body):
        self.records.append({'status':'complete','role':self.role,'charged_output_tokens':1})
        if self.path:write_json(self.path,self.records)
        if self.replies:return self.replies.pop(0)
        info=json.loads(body['messages'][1]['content']);a=payload(info)
        a['evidence_ids']=[info['candidates'][-1]['module_sha256']]
        return 200,{'choices':[{'message':{'content':json.dumps(a)},'finish_reason':'stop'}]}
    def __enter__(self):return self
    def __exit__(self,*args):pass


def response(a):return 200,{'choices':[{'message':{'content':json.dumps(a)},'finish_reason':'stop'}]}


def test_short_contract_uses_real_ids_and_six_actions(tmp_path):
    pool=tmp_path/'arms/meta/repeat';_,body,base=pool_fixture(pool)
    info=critic.evidence(pool);a=payload(info)
    second=copy.deepcopy(info['candidates'][0]);second['module_sha256']='b'*64
    broken=copy.deepcopy(second);broken.update(module_sha256='c'*64,status='invalid',error='shape mismatch')
    info['candidates'] += [second,broken]
    for kind in ('refine_parent','explore_new_mechanism','run_source_probe','stop_unproductive_branch','crossover_methods','repair_candidate'):
        item=payload(info,kind)
        if kind=='crossover_methods':item['second_parent_sha256']=second['module_sha256']
        if kind=='repair_candidate':item.update(parent_code_sha256=broken['module_sha256'],success_prediction={'metric':'valid','direction':'increase','minimum_change':1})
        action=critic.parse_action(item,info)
        assert action.action==kind
    with pytest.raises(ValueError,match='Source evidence'):critic.parse_action({**a,'evidence_ids':['unsupported']},info)
    with pytest.raises(ValueError,match='invalid parent'):critic.parse_action(payload(info,'repair_candidate'),info)
    with pytest.raises(ValueError,match='remaining'):critic.parse_action({**item,'parent_code_sha256':broken['module_sha256']},info,repair_counts={broken['module_sha256']:2})
    assert critic.contract(info)['example']['evidence_ids']==[base['module_sha256']]
    assert len(json.dumps(critic.evidence(pool)))<15000


@pytest.mark.parametrize('invalid', ['repeat_share','evidence_ids'])
def test_one_explicit_schema_correction_counted_and_exact_replay(tmp_path,invalid):
    pool=tmp_path/'arms/meta/repeat';pool_fixture(pool);info=critic.evidence(pool)
    good=payload(info);bad=copy.deepcopy(good);bad[invalid]=60 if invalid=='repeat_share' else []
    g=FakeGateway();g.replies=[response(bad),response(good)]
    result=critic.decide(pool,g,'review',8192)
    assert result['attribution']=='critic' and result['attempts']==2
    assert [r['role'] for r in g.records]==['critic','critic_schema_correction']
    assert critic.decide(pool,g,'review',8192)==result and len(g.records)==2
    assert read(pool/'decisions/review/attempt_1.json')['status']=='schema_rejected'
    assert read(pool/'decisions/review/attempt_2.json')['status']=='accepted'


def test_provider_failure_is_not_a_schema_retry_and_fallback_is_not_success(tmp_path):
    pool=tmp_path/'arms/meta/repeat';pool_fixture(pool)
    g=FakeGateway();g.replies=[(400,{'error':{'type':'incomplete_response'}})]
    result=critic.decide(pool,g,'truncated',8192)
    assert result['attribution']=='fallback' and result['action'] is None and len(g.records)==1
    assert result['error']=='incomplete_response'
    g=FakeGateway();bad=payload(critic.evidence(pool));bad['evidence_ids']=[];g.replies=[response(bad),response(bad)]
    result=critic.decide(pool,g,'twice_invalid',8192)
    assert result['status']=='schema_rejected' and result['attribution']=='fallback' and len(g.records)==2


def test_provider_cache_replay_is_audited_without_another_paid_call(tmp_path,monkeypatch):
    from vcell.round26.provider import ResearchGateway
    from vcell.round22 import gateway as base
    from vcell.evolution.journal import events
    calls=[]
    def post(*args,**kwargs):
        calls.append(1)
        return SimpleNamespace(ok=True,status_code=200,json=lambda:{'choices':[{'message':{'content':'{}'},'finish_reason':'stop'}],
            'usage':{'completion_tokens':2}})
    monkeypatch.setattr(base.requests,'post',post)
    g=ResearchGateway(tmp_path/'api.json','mock-key','deepseek-v4-pro','local',2,1000,0,max_tokens=100)
    try:
        g.scope='review';g.role='critic'
        body={'model':g.model,'messages':[{'role':'user','content':'source only'}],'max_tokens':100}
        assert g.forward(body)[0]==200 and g.forward(body)[0]==200
        assert len(calls)==len(g.records)==1
        assert len(events(tmp_path/'provider_cache_reuse.jsonl'))==1
    finally:g.server.server_close()


def test_source_evidence_excludes_confirmation_and_other_arm_history(tmp_path):
    pool=tmp_path/'arms/meta/repeat';_,body,base=pool_fixture(pool)
    hidden=copy.deepcopy(base);hidden['identity']['phase']='confirmation';hidden['error']='HIDDEN_CONFIRMATION'
    write_json(pool/'assessments/hidden/assessment.json',hidden)
    log=tmp_path/'research_decisions.jsonl'
    append(log,'research_action',phase='confirmation',arm='meta',task='repeat',action='HIDDEN_CONFIRMATION')
    append(log,'research_action',phase='search',arm='repair',task='repeat',action='OTHER_ARM')
    append(log,'research_action',phase='search',arm='meta',task='repeat',action='KEEP_SOURCE')
    text=json.dumps(critic.evidence(pool))
    assert 'KEEP_SOURCE' in text and 'HIDDEN_CONFIRMATION' not in text and 'OTHER_ARM' not in text
    assert not runtime.novel(pool)
    with pytest.raises(ValueError,match='Freeze'):runtime.assess(pool,body,phase='confirmation',execute_fn=toy_execute)


def test_source_pareto_parent_keeps_signal_with_slight_mse_cost():
    a=SimpleNamespace(id='a',code='a',correct=True,combined_score=10,island_idx=0)
    b=SimpleNamespace(id='b',code='b',correct=True,combined_score=1,island_idx=1)
    def record(p,mse,strong,gap):
        return {'status':'valid','module_sha256':hashlib.sha256(p.code.encode()).hexdigest(),
            'metrics':{'public':{'mse':mse,'strong_mse':strong,'specificity_gap':gap}},'worker_seconds':1,'code_lines':10}
    rows=[record(a,1.,2.,0),record(b,1.003,1.8,.1)]
    db=SimpleNamespace(get_all_programs=lambda:[a,b]);original=lambda *args,**kwargs:(a,[],[],False)
    parents={native.select_frontier(db,original,None,set(),rows,i)[0].id for i in range(4)}
    assert parents=={'a','b'}
    chosen=native.select_frontier(db,original,{'action':'refine_parent','parent_code_sha256':rows[1]['module_sha256']},set(),rows,0)
    assert chosen[0] is b


def test_native_imported_roots_do_not_consume_evolved_generation_slots():
    import asyncio
    async def ids():return [0,0,0,1]
    runner=SimpleNamespace(async_db=SimpleNamespace(get_persisted_generation_ids_async=ids),
        evo_config=SimpleNamespace(num_generations=3),_count_completed_generations_from_db=None)
    native.configure_generation_counter(runner)
    assert asyncio.run(runner._count_completed_generations_from_db())==2


def test_native_sampling_module_does_not_import_training_environment():
    import subprocess
    result=subprocess.run([sys.executable,'-I','-c',
        f'import sys;sys.path.insert(0,{str(REPO / "src")!r});import vcell.round26.native;assert "torch" not in sys.modules'],
        capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def mock_native(m,monkeypatch,gateways,broken_first=False):
    original=runtime.assess
    monkeypatch.setattr(m,'assess',lambda *args,**kwargs:original(*args,**kwargs,execute_fn=toy_execute))
    calls=[]
    def run(argv,**kwargs):
        pool=Path(argv[argv.index('--pool')+1]);g=gateways[-1]
        g.records.append({'status':'complete','role':g.role,'charged_output_tokens':1});write_json(g.path,g.records)
        calls.append(read(pool/'active_action.json'))
        marker='BROKEN_CANDIDATE' if broken_first and len(calls)==1 else 'REPAIRED_CANDIDATE'
        m.assess(pool,f'# {marker} {len(calls)}\ndef fit_predict(data,budget):\n    return data["query_support"]\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(m.subprocess,'run',run)
    return calls


def test_live_gate_requires_real_attribution_generation_and_result_update(tmp_path,monkeypatch):
    m=controller(monkeypatch);root=tmp_path/'run'
    pool=root/'arms/meta/repeat';cfg,body,base=pool_fixture(pool)
    shared={'repeat':{'folds':read(pool/'folds.json'),'baseline':read(pool/'baseline_selection.json'),'common':read(pool/'common_sources.json')}}
    gateways=[]
    def gateway(ar,*args):g=FakeGateway(ar/'api_ledger.json');gateways.append(g);return g
    monkeypatch.setattr(m,'gateway_for',gateway);calls=mock_native(m,monkeypatch,gateways)
    done=m.live_smoke(root,cfg,shared,'mock-native')
    assert done['accepted_decisions']==2 and done['hypothesis_outcome']['status']!='undefined'
    assert len(calls)==1 and len(gateways[-1].records)==3
    after=read(root/'pilot/meta/repeat/decisions/pilot_after_result/decision.json')
    assert done['measured_child'] in after['action']['evidence_ids']
    assert m.live_smoke(root,cfg,shared,'mock-native')==done and len(calls)==1
    badroot=tmp_path/'failed'
    def bad_gateway(ar,*args):
        g=FakeGateway(ar/'api_ledger.json');g.replies=[(400,{'error':{'type':'transport_failure'}})];return g
    monkeypatch.setattr(m,'gateway_for',bad_gateway)
    with pytest.raises(RuntimeError,match='gate failed'):m.live_smoke(badroot,cfg,shared,'mock-native')
    assert len(calls)==1 and not (badroot/'LIVE_SMOKE_COMPLETE.json').exists()


def test_matched_meta_search_repairs_with_shared_cap_and_resumes(tmp_path,monkeypatch):
    m=controller(monkeypatch);root=tmp_path/'run';gateways=[]
    for task in ('repeat','transfer'):cfg,_,_=pool_fixture(root/'arms/meta'/task,task)
    cfg.update(valid_target_per_arm=1,max_turns_per_arm=6)
    def gateway(ar,*args):g=FakeGateway(ar/'api_ledger.json');gateways.append(g);return g
    monkeypatch.setattr(m,'gateway_for',gateway);calls=mock_native(m,monkeypatch,gateways,broken_first=True)
    result=m.search_arm(root,'meta',cfg,'mock-native')
    assert result['requests_max']==cfg['api_requests_per_arm'] and sum(result['distinct_valid'].values())==1
    assert any(r['attribution']=='critic' for r in calls) and any(r['attribution']=='automatic_repair' for r in calls)
    assert max(result['state']['repairs'].values())<=2
    requests=sum(len(g.records) for g in gateways)
    m.search_arm(root,'meta',cfg,'mock-native');assert len(calls)==2 and sum(len(g.records) for g in gateways)==requests


def test_finalist_selection_keeps_signal_without_claiming_promotion(tmp_path,monkeypatch):
    m=controller(monkeypatch);pool=tmp_path/'arms/meta/transfer';_,body,_=pool_fixture(pool,task='transfer')
    rows=[]
    for sha,mse,strong,gap in [('a',1.,1.1,-.01),('b',1.003,.98,.1)]:
        rows.append({'module_sha256':sha,'status':'valid','metrics':{'public':{'mse':mse,'strong_mse':strong,
            'strong_mse_ratio':strong,'specificity_gap':gap,'amplitude_ratio':.1}}})
    monkeypatch.setattr(m,'novel',lambda p:rows)
    frozen=m.finalists(pool)
    assert [r['module_sha256'] for r in frozen['finalists']]==['a','b']
    assert frozen['finalists'][1]['tier']=='signal_diagnostic_not_promotion'


def test_three_matched_caps_and_report_separate_accepted_actions_from_fallbacks(tmp_path,monkeypatch):
    from vcell.round26 import report as reporting
    m=controller(monkeypatch);root=tmp_path/'run';gateways=[]
    for arm in ('shinka','repair','meta'):
        for task in ('repeat','transfer'):cfg,_,_=pool_fixture(root/'arms'/arm/task,task,arm)
    cfg.update(valid_target_per_arm=1,max_turns_per_arm=4)
    def gateway(ar,*args):g=FakeGateway(ar/'api_ledger.json');gateways.append(g);return g
    monkeypatch.setattr(m,'gateway_for',gateway);mock_native(m,monkeypatch,gateways)
    for arm in cfg['arms']:m.search_arm(root,arm,cfg,'mock-native')
    completed=[read(root/'arms'/arm/'SEARCH_COMPLETE.json') for arm in cfg['arms']]
    assert len({(r['requests_max'],r['valid_target'],r['worker_hours_max']) for r in completed})==1
    assert completed[-1]['api_requests']>completed[0]['api_requests']
    for arm in cfg['arms']:
        for task in cfg['tasks']:m.finalists(root/'arms'/arm/task)
    write_json(root/'search_frozen.json',{'extension_executed':False})
    write_json(root/'LIVE_SMOKE_COMPLETE.json',{'api_requests':3,'accepted_decisions':2})
    original=runtime.assess
    monkeypatch.setattr(reporting,'assess',lambda *args,**kwargs:original(*args,**kwargs,execute_fn=toy_execute))
    reporting.report(root,cfg,lambda *args,**kwargs:None)
    usage=read(root/'efficiency.json');meta=next(r for r in usage if r['arm']=='meta')
    assert meta['accepted_critic_decisions']==1 and meta['fallback_decisions']==0 and meta['critic_implemented_actions']==1
    assert read(root/'summary.json')['total_requests_including_gate']==3+sum(r['api_requests'] for r in completed)
    assert not read(root/'summary.json')['independent_confirmation']
    path=reporting.package(root)
    import tarfile
    with tarfile.open(path) as tar:
        names=tar.getnames()
        assert 'ARCHIVE_MANIFEST.json' in names
        assert any('/decisions/' in n and n.endswith('attempt_1.json') for n in names)
        assert not any('/workers/' in n or '/provider_responses/' in n or n.endswith('.npz') for n in names)


def test_full_controller_freezes_before_confirmation_and_resumes_after_report_failure(tmp_path,monkeypatch):
    from vcell.round26 import report as reporting
    m=controller(monkeypatch);parent=tmp_path/'parent';write_json(parent/'COMPLETE.json',{'fixture':True})
    cfg=read(REPO/'configs/round26.json');config=tmp_path/'config.json';write_json(config,cfg)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(m.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout='{"version":"0.0.7"}',stderr=''))
    monkeypatch.setattr(m,'preflight',lambda *args:None)
    monkeypatch.setattr(m,'prepare',lambda *args:{})
    def smoke(root,*args):write_json(root/'LIVE_SMOKE_COMPLETE.json',{'accepted_decisions':2,'files':{}})
    monkeypatch.setattr(m,'live_smoke',smoke)
    searches=[]
    def search(root,arm,*args):
        path=root/'arms'/arm/'SEARCH_COMPLETE.json'
        if path.exists():return read(path)
        searches.append(arm);write_json(path,{'paid_search_finished':True})
    monkeypatch.setattr(m,'search_arm',search)
    def finalists(pool):
        value={'finalists':[],'source_only':True};write_json(pool/'search_frozen.json',value);return value
    monkeypatch.setattr(m,'finalists',finalists)
    reports=[]
    def report(root,*args):
        assert (root/'matched_search_frozen.json').exists() and (root/'search_frozen.json').exists()
        reports.append(1)
        if len(reports)==1:raise RuntimeError('interruption after source freeze')
        write_json(root/'summary.json',{'scientific_pass':False})
    monkeypatch.setattr(reporting,'report',report)
    a=SimpleNamespace(work_dir=str(tmp_path),name='campaign',round25_run=str(parent),config=str(config),
        device='cpu',extend_meta=False,resume=False,shinka_python=sys.executable,mode='all')
    with pytest.raises(RuntimeError,match='source freeze'):m.launch(a)
    a.resume=True;root=m.launch(a)
    assert searches==cfg['arms'] and (root/'COMPLETE.json').exists()
    m.launch(a);assert searches==cfg['arms']
    a.extend_meta=True
    with pytest.raises(ValueError):m.launch(a)


def test_protocol_audit_rejects_cell_line_leakage_and_unknown_target_labels(tmp_path):
    parent=tmp_path/'parent';pool=parent/'arms/meta/repeat';cfg,_,_=pool_fixture(pool)
    write_json(parent/'campaign_plan.json',{'config':cfg})
    folds=read(pool/'folds.json')
    for task in cfg['tasks']:
        write_json(parent/'baselines'/task/'folds.json',folds)
        for arm in cfg['arms']:write_json(parent/'arms'/arm/task/'folds.json',folds)
    seal(parent)
    with pytest.raises(ValueError,match='query cell line'):prepare.audit_protocol(parent,cfg)
    # Make a legal held-line source without changing query values or strong masks.
    fold=folds['search'][0]
    with np.load(fold['input'],allow_pickle=False) as f:d={k:f[k] for k in f.files}
    for k in ('train_source_context','calibration_source_context'):d[k]=np.repeat('BXPC3_IFNG',len(d[k]))
    np.savez_compressed(fold['input'],**d);fold['input_sha256']=file_sha256(fold['input'])
    # Search and confirmation are the same toy fold, so update both frozen references.
    folds['confirmation']=[fold]
    for task in cfg['tasks']:
        write_json(parent/'baselines'/task/'folds.json',folds)
        for arm in cfg['arms']:write_json(parent/'arms'/arm/task/'folds.json',folds)
    seal(parent);_,checks=prepare.audit_protocol(parent,cfg);assert checks
    with np.load(Path(fold['private'])/'grade.npz',allow_pickle=False) as f:g={k:f[k] for k in f.files}
    g['unknown']=np.ones(len(g['unknown']),bool)
    np.savez_compressed(Path(fold['private'])/'grade.npz',**g);fold['grade_sha256']=file_sha256(Path(fold['private'])/'grade.npz')
    for task in cfg['tasks']:
        write_json(parent/'baselines'/task/'folds.json',folds)
        for arm in cfg['arms']:write_json(parent/'arms'/arm/task/'folds.json',folds)
    seal(parent)
    with pytest.raises(ValueError,match='Globally unseen'):prepare.audit_protocol(parent,cfg)


@pytest.mark.skipif(sys.platform!='linux',reason='Linux protected numerical workers')
def test_real_round26_worker_isolation_and_shared_baseline_layout(tmp_path,monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','CI_FAKE_NOT_A_KEY')
    pool=tmp_path/'arms/meta/repeat';cfg,body,base=pool_fixture(pool,real=True)
    malicious='def fit_predict(data,budget):\n    import os\n    assert not os.environ.get("DEEPSEEK_API_KEY")\n    return data["query_support"]\n'
    result=runtime.assess(pool,malicious)
    assert result['status']=='valid',result.get('error')
    assert runtime.assess(pool,malicious)==result
    baseline_pool=tmp_path/'baselines/repeat';baseline=read(pool/'baseline_selection.json')
    prepare.pool_plan(baseline_pool,tmp_path/'baseline_budget','repeat','fixed',cfg,read(pool/'folds.json'),baseline,[],REPO,sys.executable,1.)
    result=runtime.assess(baseline_pool,body)
    assert result['status']=='valid',result.get('error')
