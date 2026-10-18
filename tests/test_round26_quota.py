"""Regression for the real nine-fold pilot quota failure and paid-artifact recovery."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import pytest
from test_round26 import pool_fixture,controller,FakeGateway,payload
from vcell.round26 import runtime,critic
from vcell.round26.prepare import pool_plan
from vcell.round26.recovery import recover_pilot
from vcell.model_agent.artifacts import read
from vcell.model_agent.contracts import digest
from vcell.utils import write_json,file_sha256
from vcell.evolution.journal import events


def test_common_seed_is_reused_only_for_identical_source_contract(tmp_path):
    pool=tmp_path/'arms/meta/repeat';cfg,body,base=pool_fixture(pool)
    assert runtime.assess(pool,body,execute_fn=lambda *args:pytest.fail('Duplicated shared source fit'))==base
    assert events(pool/'cache_reuse.jsonl')[-1]['kind']=='shared_source_cache_hit'
    plan=read(pool/'run_plan.json');plan['budget']['steps']+=1;write_json(pool/'run_plan.json',plan)
    with pytest.raises(ValueError,match='Common source plan'):runtime.assess(pool,body)


def test_confirmation_never_uses_a_source_seed_cache(tmp_path):
    pool=tmp_path/'arms/meta/repeat';_,body,_=pool_fixture(pool)
    with pytest.raises(ValueError,match='Freeze'):runtime.assess(pool,body,phase='confirmation')


def test_nine_folds_all_controls_fit_dedicated_pilot_and_old_policy_exhausts(tmp_path):
    pool=tmp_path/'pilot/meta/repeat';cfg,_,_=pool_fixture(pool)
    original_fold=read(pool/'folds.json')['search'][0]
    search=[{**original_fold,'name':f'fold_{i}'} for i in range(9)]
    write_json(pool/'folds.json',{'search':search,'confirmation':[original_fold]})
    write_json(pool/'common_sources.json',[])
    selector=read(pool/'baseline_selection.json');selector['search']={f['name']:next(iter(selector['search'].values())) for f in search}
    write_json(pool/'baseline_selection.json',selector)
    plan=read(pool/'run_plan.json');plan['budget'].update(seconds=120,evaluation_seconds_max=1800)
    plan['config'].update(worker_seconds=120,search_fraction=.8,maximum_task_search_share=.7)
    write_json(pool/'run_plan.json',plan)
    calls=[]
    def execute(code,data,dest,budget,python):
        root=Path(budget['root']);path=root/'evaluation_budget.json'
        value=read(path) if path.exists() else {'workers':{}}
        used=sum(r['charged_seconds'] for r in value['workers'].values())
        assert used+budget['seconds']<=budget['evaluation_seconds_max']
        calls.append(str(dest));value['workers'][str(dest)+'#1']={'charged_seconds':120.,'status':'complete'}
        write_json(path,value)
        return data['query_support']*.9
    body='def fit_predict(data,budget):\n    return data["query_support"]*.9\n'
    old=runtime.assess(pool,body,execute_fn=execute)
    assert old['status']=='invalid' and old['error']=='Task search compute quota exhausted' and len(calls)==8
    # New immutable pool/ledger; retain the exact old failed result.
    fixed=tmp_path/'fixed/pilot/meta/repeat';fixed_cfg={**cfg,'worker_seconds':120,'search_fraction':1.,'maximum_task_search_share':1.}
    pool_plan(fixed,tmp_path/'fixed/pilot/meta','repeat','pilot',fixed_cfg,read(pool/'folds.json'),selector,[],
        Path(plan['repo']),plan['python'],2.25)
    calls.clear();result=runtime.assess(fixed,body,execute_fn=execute)
    assert result['status']=='valid',result.get('error')
    assert len(calls)==65  # Nine source folds x seven controls + two contract workers.
    assert read(Path(old['directory'])/'assessment.json')['status']=='invalid'


def test_pilot_rejects_insufficient_allowance_before_any_api_call(tmp_path,monkeypatch):
    m=controller(monkeypatch);root=tmp_path/'run';pool=root/'arms/meta/repeat';cfg,_,_=pool_fixture(pool)
    cfg['pilot_hours']=.001
    shared={'repeat':{'folds':read(pool/'folds.json'),'baseline':read(pool/'baseline_selection.json'),'common':read(pool/'common_sources.json')}}
    monkeypatch.setattr(m,'gateway_for',lambda *args:pytest.fail('Paid gateway opened before budget check'))
    with pytest.raises(ValueError,match='before any API call'):m.live_smoke(root,cfg,shared,'unused')


def failed_source(tmp_path):
    source=tmp_path/'old';pool=source/'pilot/meta/repeat';cfg,_,base=pool_fixture(pool)
    old_cfg={**cfg,'pilot_hours':.5}
    write_json(source/'campaign_plan.json',{'parent_sha256':'frozen-parent','config':old_cfg})
    info=critic.evidence(pool);g=FakeGateway(pool.parent/'api_ledger.json')
    first=critic.decide(pool,g,'pilot_initial',cfg['critic_max_tokens'])
    body='# QUOTA_CHILD\ndef fit_predict(data,budget):\n    return data["query_support"]*.9\n'
    record=runtime.assess(pool,body,execute_fn=lambda *args:(_ for _ in ()).throw(RuntimeError('Task search compute quota exhausted')))
    assert record['status']=='invalid'
    write_json(pool.parent/'turns/000/plan.json',{'action':first['action'],'prediction_reference_sha256':base['module_sha256']})
    write_json(pool.parent/'turns/000/done.json',{'latest':record['module_sha256'],'attribution':'critic'})
    ledger=[{'attempt':i+1,'role':role,'status':'complete','http_status':200,'finish_reason':'stop','charged_output_tokens':100}
        for i,role in enumerate(('critic','mutation'))]
    write_json(pool.parent/'api_ledger.json',ledger)
    db=pool/'evolution/programs.sqlite';db.parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(db)
    con.execute('CREATE TABLE programs (id TEXT,parent_id TEXT,island_idx INT,code TEXT,correct INT)')
    con.execute('INSERT INTO programs VALUES (?,?,?,?,?)',('parent',None,0,Path(base['module']).read_text(),1))
    con.execute('INSERT INTO programs VALUES (?,?,?,?,?)',('child','parent',0,body,0));con.commit();con.close()
    shared={'folds':read(pool/'folds.json'),'baseline':read(pool/'baseline_selection.json'),'common':read(pool/'common_sources.json')}
    return source,cfg,shared,record


def test_resource_recovery_preserves_paid_decision_code_native_link_and_quota(tmp_path,monkeypatch):
    m=controller(monkeypatch);source,cfg,shared,old_record=failed_source(tmp_path)
    root=tmp_path/'new';write_json(root/'campaign_plan.json',{'parent_sha256':'frozen-parent'})
    before={p.relative_to(source).as_posix():file_sha256(p) for p in source.rglob('*') if p.is_file()}
    result=recover_pilot(root,source,cfg,shared)
    assert result['snapshot']['imported_paid_requests']==2 and result['snapshot']['native_parent_id']=='parent'
    assert read(root/'pilot/meta/api_ledger.json')==read(source/'pilot/meta/api_ledger.json')
    assert result['module']==Path(old_record['module']).read_text()
    assert before=={p.relative_to(source).as_posix():file_sha256(p) for p in source.rglob('*') if p.is_file()}
    def gateway(ar,*args):return FakeGateway(ar/'api_ledger.json')
    monkeypatch.setattr(m,'gateway_for',gateway)
    original=runtime.assess
    monkeypatch.setattr(m,'assess',lambda *args,**kwargs:original(*args,**kwargs,execute_fn=lambda code,d,*rest:d['query_support']*.9))
    monkeypatch.setattr(m,'generation',lambda *args:pytest.fail('Regenerated a paid candidate'))
    done=m.live_smoke(root,cfg,{'repeat':shared},'unused',recover_from=source)
    assert done['accepted_decisions']==2 and done['api_requests']==3
    assert done['recovery']['module_sha256']==old_record['module_sha256']
    assert read(source/'pilot/meta/api_ledger.json')[-1]['role']=='mutation'
    # Exact resume validates the recovered result, without another provider call.
    assert m.live_smoke(root,cfg,{'repeat':shared},'unused',recover_from=source)==done


@pytest.mark.parametrize('change',['folds','provider','code','failure'])
def test_recovery_rejects_changed_scope_ambiguous_requests_and_nonquota_failures(tmp_path,change):
    source,cfg,shared,record=failed_source(tmp_path);root=tmp_path/'new'
    write_json(root/'campaign_plan.json',{'parent_sha256':'frozen-parent'})
    if change=='folds':shared=copy.deepcopy(shared);shared['folds']['search'][0]['input_sha256']='changed'
    if change=='provider':
        path=source/'pilot/meta/api_ledger.json';rows=read(path);rows[-1]['status']='inflight';write_json(path,rows)
    if change=='code':Path(record['module']).write_text('changed')
    if change=='failure':
        record['error']='Actual shape mismatch';record.pop('sha256');record['sha256']=digest(record)
        write_json(Path(record['directory'])/'assessment.json',record)
    with pytest.raises(ValueError):recover_pilot(root,source,cfg,shared)
