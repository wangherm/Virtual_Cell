"""Virtual Cell domain, durable hypotheses, balanced scheduling and recovery contracts."""
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from test_round25_self_correction import make_pool,toy_execute
from vcell.round27 import runtime,critic,memory,domain,manager,tools
from vcell.model_agent.artifacts import read,verify
from vcell.utils import write_json,file_sha256

REPO=Path(__file__).resolve().parents[1]


def pool_fixture(pool,task='repeat',arm='meta'):
    _,body=make_pool(pool,arm,task)
    cfg=read(REPO/'configs/round27.json');cfg.update(device='cpu',panel_genes=6,steps=5,worker_seconds=20)
    plan=read(pool/'run_plan.json');plan.update(framework_sha256=runtime.fingerprint(REPO),config=cfg)
    write_json(pool/'run_plan.json',plan)
    base=runtime.assess(pool,body,execute_fn=toy_execute)
    assert base['status']=='valid',base.get('error')
    selection={**read(pool/'baseline_selection.json'),'method':'support','module_sha256':base['module_sha256'],
        'search_record':base,'confirmation_record':base}
    write_json(pool/'baseline_selection.json',selection)
    base=runtime.assess(pool,body,execute_fn=toy_execute)
    write_json(pool/'common_sources.json',[{'record':str(Path(base['directory'])/'assessment.json'),
        'sha256':file_sha256(Path(base['directory'])/'assessment.json')}])
    return cfg,body,base


def controller(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    spec=importlib.util.spec_from_file_location('round27_controller',REPO/'scripts/run_round27.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Gateway:
    def __init__(self,path):
        self.path=Path(path);self.records=read(path) if self.path.exists() else []
        self.model='offline';self.requests_max=30;self.output_tokens_max=900000;self.scope='';self.role='';self.responses=[]
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def forward(self,body):
        self.records.append({'status':'complete','role':self.role,'charged_output_tokens':1})
        write_json(self.path,self.records)
        text=self.responses.pop(0) if self.responses else json.loads(body['messages'][0]['content'].split('SOURCE evidence. ',1)[1])['example']
        return 200,{'choices':[{'message':{'content':json.dumps(text)},'finish_reason':'stop'}]}


def failed_hypothesis(root,pool,record,task='repeat',ticket='meta/first'):
    info=critic.evidence(pool);action=critic.parse_action(critic.contract(info)['example']['action'],info)
    result={'children':[{'module_sha256':record['module_sha256'],'status':'valid'}],'latest':record['module_sha256'],
        'prediction':{'status':'not_supported','metric':'mse','before':.01,'after':.011},'attribution':'critic'}
    memory.outcome(root,'meta',task,ticket,action,result)
    return action


def test_balanced_task_goal_and_explicit_incomplete_queue(tmp_path):
    cfg=read(REPO/'configs/round27.json')
    assert manager.choose_task(cfg,{'repeat':0,'transfer':4},{'repeat':1000,'transfer':0})=='repeat'
    assert manager.choose_task(cfg,{'repeat':5,'transfer':5},{'repeat':0,'transfer':0}) is None
    q=manager.task_queue(tmp_path,cfg,'meta',{'repeat':0,'transfer':10},'budget cap')
    assert not read(tmp_path/'tasks/meta.json')['coverage_complete']
    assert q[0]['status']=='blocked' and q[1]['status']=='complete'


def test_seed_cannot_be_stopped_for_matching_itself(tmp_path):
    pool=tmp_path/'arms/meta/repeat';_,_,_=pool_fixture(pool);info=critic.evidence(pool)
    a=critic.contract(info)['example']['action'];a['action']='stop_unproductive_branch'
    with pytest.raises(ValueError,match='actual new exploration'):critic.parse_action(a,info)
    info['exploration']['distinct_new']=5
    with pytest.raises(ValueError,match='matching itself'):critic.parse_action(a,info)


def test_pending_failure_survives_many_routine_events_and_requires_specific_revision(tmp_path):
    pool=tmp_path/'arms/meta/repeat';_,_,base=pool_fixture(pool);failed_hypothesis(tmp_path,pool,base)
    for i in range(15):memory.outcome(tmp_path,'meta','repeat','meta/native'+str(i),None,{'children':[]})
    info=critic.evidence(pool)
    assert info['required_revision_id']=='meta/first'
    envelope=critic.contract(info)['example']
    envelope['revision']['predecessor_experiment_id']='not-the-failed-experiment'
    with pytest.raises(ValueError,match='exact pending'):critic.parse_decision(envelope,info,None,None,{})
    envelope['revision']['predecessor_experiment_id']='meta/first'
    action,link=critic.parse_decision(envelope,info,None,None,{})
    memory.outcome(tmp_path,'meta','repeat','meta/second',action,{'children':[{'status':'valid'}],
        'prediction':{'status':'supported','metric':'mse','after':.009}},link)
    assert not memory.experience(tmp_path,'meta','repeat')['pending_revisions']
    assert manager.revision_summary(tmp_path)[0]['hypothesis_supported']


def test_memory_idempotent_resume_and_changed_commit_rejected(tmp_path):
    memory.remember(tmp_path,'one','experiment',arm='meta',task='repeat',value=1)
    memory.remember(tmp_path,'one','experiment',arm='meta',task='repeat',value=1)
    assert len(memory.events(tmp_path/'memory/events.jsonl'))==1
    with pytest.raises(ValueError,match='memory changed'):memory.remember(tmp_path,'one','experiment',arm='meta',task='repeat',value=2)


def test_repeated_probe_is_rejected_without_new_source_evidence(tmp_path):
    pool=tmp_path/'arms/meta/repeat';_,_,base=pool_fixture(pool);action=failed_hypothesis(tmp_path,pool,base)
    action=critic.ResearchAction(**{**asdict(action),'action':'run_source_probe','requested_source_diagnostic':'context_losses'})
    memory.outcome(tmp_path,'meta','repeat','meta/probe',action,{'probe':{'kind':'context_losses',
        'summary':[{'module_sha256':base['module_sha256']}]},'children':[]})
    with pytest.raises(ValueError,match='diagnostic already ran'):critic.parse_action(asdict(action),critic.evidence(pool))


def test_critic_schema_correction_and_cached_resume_preserve_revision(tmp_path):
    pool=tmp_path/'arms/meta/repeat';cfg,_,base=pool_fixture(pool);failed_hypothesis(tmp_path,pool,base)
    info=critic.evidence(pool);good=critic.contract(info)['example'];bad=copy.deepcopy(good);bad['revision']=None
    g=Gateway(pool.parent/'api_ledger.json');g.responses=[bad,good]
    value=critic.decide(pool,g,'review',cfg['critic_max_tokens'])
    assert value['attribution']=='critic' and value['revision']['predecessor_experiment_id']=='meta/first'
    assert [r['role'] for r in g.records]==['critic','critic_schema_correction']
    assert critic.decide(pool,g,'review',cfg['critic_max_tokens'])==value and len(g.records)==2


@pytest.mark.parametrize('kind',['reference_coverage','context_losses','target_losses','strong_response'])
def test_source_diagnostic_is_executed_frozen_and_checked(tmp_path,kind):
    pool=tmp_path/'arms/meta/repeat';_,_,base=pool_fixture(pool)
    path=tmp_path/'probe.json';value=tools.source_probe(pool,kind,path)
    assert value['scope']=='source_search_only' and value['methods'][0]['values']
    assert tools.source_probe(pool,kind,path)==value
    csv=next(Path(p) for p in value['evidence_files'] if Path(p).name=='scores.csv');csv.write_text(csv.read_text()+'\n')
    with pytest.raises(ValueError,match='evidence changed'):tools.source_probe(pool,kind,path)


def test_domain_catalog_tracks_actual_gene_order_and_unknown_metadata(tmp_path):
    pool=tmp_path/'arms/meta/repeat';cfg,_,_=pool_fixture(pool)
    catalog=domain.build(tmp_path,{'repeat':{'folds':read(pool/'folds.json')}},cfg)
    row=catalog['source_catalog'][0]
    assert row['genes']==6 and row['gene_order_sha256']
    assert row['auxiliary_metadata']['time']==['X']
    assert 'unknown' in row['metadata_note']
    assert 'Rep2' in catalog['tasks']['repeat']['question']
    assert catalog['evaluation']['sealed_external_available'] is False
    from vcell.round27.knowledge import lookup
    card=lookup(tmp_path,['JAK1'])['genes'][0]
    assert card['annotation_available'] and card['terms']
    assert all(r['relation']=='shared_pathway_annotation' for r in card['related_genes'])
    (tmp_path/'gene_knowledge.json').write_text('{}')
    with pytest.raises(ValueError,match='knowledge changed'):lookup(tmp_path,['JAK1'])


def test_long_term_export_import_is_source_only(tmp_path):
    root=tmp_path/'runs/old';pool=root/'arms/meta/repeat';cfg,_,base=pool_fixture(pool)
    cfg['arms']=['meta'];cfg['tasks']=['repeat'];failed_hypothesis(root,pool,base)
    memory.publish(root,tmp_path,cfg)
    value=memory.import_history(tmp_path/'runs/new',root)
    assert value['source_lessons'][0]['prediction']['status']=='not_supported'
    assert 'confirmation' not in json.dumps(value['source_lessons'])
    old=read(root/'memory_library.json');old['confirmation_included']=True;write_json(root/'memory_library.json',old)
    with pytest.raises(ValueError,match='source-only'):memory.import_history(tmp_path/'runs/other',root)


def test_pause_control_and_heartbeat_are_outside_completed_run_updates(tmp_path):
    work=tmp_path;root=work/'runs/trial';root.mkdir(parents=True)
    h=manager.Heartbeat(root,.01).__enter__();h.__exit__();original=(root/'heartbeat.json').read_bytes();h.__exit__()
    assert (root/'heartbeat.json').read_bytes()==original
    write_json(manager.control_path(work,'trial'),{'pause_requested':True})
    with pytest.raises(manager.PauseRequested):manager.checkpoint(work,'trial')
    write_json(manager.control_path(work,'trial'),{'pause_requested':False});manager.checkpoint(work,'trial')
    assert (root/'heartbeat.json').read_bytes()==original
    with pytest.raises(ValueError):manager.control_path(work,'../outside')


def test_native_module_has_no_torch_dependency():
    result=subprocess.run([sys.executable,'-c',f'import sys;sys.path.insert(0,{str(REPO/"src")!r});import vcell.round27.native;assert "torch" not in sys.modules'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_native_job_timeout_contains_all_nine_fold_workers(tmp_path):
    from vcell.round27.native import evaluation_walltime
    write_json(tmp_path/'run_plan.json',{'config':{'counterfactual_variants':list(range(7)),'worker_seconds':120}})
    write_json(tmp_path/'folds.json',{'search':[{}]*9})
    assert evaluation_walltime(tmp_path)=='02:12:00'


def test_balanced_search_executes_failed_hypothesis_revision_and_exact_resume(tmp_path,monkeypatch):
    m=controller(monkeypatch);root=tmp_path/'runs/campaign'
    cfg=None
    for task in ('repeat','transfer'):cfg,_,_=pool_fixture(root/'arms/meta'/task,task)
    cfg.update(task_targets={'repeat':2,'transfer':2},valid_target_per_arm=4,minimum_branch_exploration=1)
    gateways=[]
    def gateway(path,*args):g=Gateway(path/'api_ledger.json');gateways.append(g);return g
    monkeypatch.setattr(m,'gateway_for',gateway);calls=[]
    def generate(root,pool,folder,action,attribution,banned,turn,gateway,relay,shinka):
        calls.append(read(pool/'run_plan.json')['task']);gateway.role='mutation';gateway.records.append({'role':'mutation','status':'complete'})
        write_json(gateway.path,gateway.records)
        body=f'# CONTROLLED_{turn}\ndef fit_predict(data,budget):\n    return data["query_support"]*.9\n'
        before=len(runtime.novel(pool));r=runtime.assess(pool,body,execute_fn=lambda code,d,*args:d['query_support']*(.9-.01*turn))
        return {'latest':r['module_sha256'],'children':[{'status':r['status'],'module_sha256':r['module_sha256']}],
            'prediction':{'status':'not_supported','metric':'mse','after':r['metrics']['public']['mse']},
            'attribution':attribution,'before_novel':before}
    monkeypatch.setattr(m,'generation',generate)
    result=m.search_arm(root,'meta',cfg,'offline')
    assert result['coverage_complete'] and result['distinct_valid']=={'repeat':2,'transfer':2}
    assert calls[:2]==['repeat','transfer']
    chains=manager.revision_summary(root)
    assert {r['task'] for r in chains if r['measured']}=={'repeat','transfer'}
    count=len(calls);assert m.search_arm(root,'meta',cfg,'offline')==result and len(calls)==count


def test_pilot_budget_checked_before_provider(tmp_path,monkeypatch):
    m=controller(monkeypatch);pool=tmp_path/'arms/meta/repeat';cfg,_,_=pool_fixture(pool)
    cfg['pilot_hours']=.001
    shared={'repeat':{'folds':read(pool/'folds.json')}}
    monkeypatch.setattr(m,'gateway_for',lambda *args:pytest.fail('Paid provider opened too early'))
    with pytest.raises(ValueError,match='before any API'):m.live_smoke(tmp_path,cfg,shared,'offline')


def test_default_config_and_changed_task_targets(tmp_path,monkeypatch):
    m=controller(monkeypatch);cfg=read(REPO/'configs/round27.json');m.cfg_check(cfg)
    cfg['task_targets']['repeat']=0
    with pytest.raises(ValueError):m.cfg_check(cfg)


def test_frozen_prediction_tool_uses_exact_source_blocks_and_rejects_source_changes(tmp_path,monkeypatch):
    from vcell.round27 import predict
    root=tmp_path/'runs/campaign';pool=root/'arms/meta/repeat';cfg,_,base=pool_fixture(pool)
    fold=read(pool/'folds.json')['search'][0]
    registry={'models':[{'task':'repeat','module_sha256':base['module_sha256'],'module':base['module'],
        'template_input':fold['input'],'template_sha256':fold['input_sha256']}]}
    write_json(root/'model_registry.json',registry)
    write_json(root/'COMPLETE.json',{'files':{'model_registry.json':file_sha256(root/'model_registry.json')}})
    write_json(root/'campaign_plan.json',{'repo':str(REPO),'framework_sha256':runtime.fingerprint(REPO),'config':cfg})
    calls=[]
    def execute(code,d,dest,budget,python):
        calls.append(d);dest.mkdir(parents=True);assert 'gene_symbols' in d
        return d['query_support']*.9
    monkeypatch.setattr(predict,'execute',execute)
    result=predict.predict(root,tmp_path,base['module_sha256'],'repeat',fold['input'],'prediction1')
    assert result.exists() and len(calls)==1
    assert read(result.parent/'provenance.json')['evaluation'] is False
    with np.load(fold['input'],allow_pickle=False) as f:d={k:f[k] for k in f.files}
    d['train_y']=d['train_y']+1;np.savez_compressed(tmp_path/'bad.npz',**d)
    with pytest.raises(ValueError,match='source block'):predict.predict(root,tmp_path,base['module_sha256'],'repeat',tmp_path/'bad.npz','prediction2')


def test_full_controller_freezes_before_confirmation_exports_memory_and_resumes(tmp_path,monkeypatch):
    m=controller(monkeypatch);root=tmp_path/'runs/campaign';parent=tmp_path/'runs/round26';parent.mkdir(parents=True)
    write_json(parent/'campaign_plan.json',{'parent':str(tmp_path/'round25')});write_json(parent/'COMPLETE.json',{'files':{}})
    cfg=read(REPO/'configs/round27.json');cfg.update(device='cpu',panel_genes=6,steps=5,worker_seconds=20,
        task_targets={'repeat':1,'transfer':1},valid_target_per_arm=2,minimum_branch_exploration=1,disk_reserve_gib=.001)
    config=tmp_path/'config.json';write_json(config,cfg);shared={}
    for arm in cfg['arms']:
        for task in cfg['tasks']:
            pool=root/'arms'/arm/task;pool_fixture(pool,task,arm)
            shared[task]={'folds':read(pool/'folds.json'),'baseline':read(pool/'baseline_selection.json'),'common':read(pool/'common_sources.json')}
    monkeypatch.setattr(m,'prepare',lambda *args:shared);monkeypatch.setattr(m,'preflight',lambda *args:None)
    monkeypatch.setattr(m.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout='{}'))
    monkeypatch.setattr(m,'export_native_archives',lambda *args:None)
    def live(*args):write_json(root/'LIVE_SMOKE_COMPLETE.json',{'accepted_decisions':2,'api_requests':0})
    monkeypatch.setattr(m,'live_smoke',live);monkeypatch.setattr(m,'gateway_for',lambda path,*args:Gateway(path/'api_ledger.json'))
    def generation(root,pool,folder,action,attribution,banned,turn,gateway,relay,shinka):
        before=len(runtime.novel(pool));body=f'# CANDIDATE_{turn}\ndef fit_predict(data,budget):\n    return data["query_support"]*.9\n'
        r=runtime.assess(pool,body,execute_fn=lambda code,d,*args:d['query_support']*.9)
        return {'latest':r['module_sha256'],'children':[{'status':'valid','module_sha256':r['module_sha256']}],
            'prediction':{'status':'not_supported','metric':'mse','after':r['metrics']['public']['mse']} if action else None,
            'attribution':attribution,'before_novel':before}
    monkeypatch.setattr(m,'generation',generation)
    from vcell.round27 import report
    original=runtime.assess
    def measured(pool,*args,**kwargs):
        if kwargs.get('phase')=='confirmation':assert (root/'search_frozen.json').exists()
        return original(pool,*args,**kwargs,execute_fn=lambda code,d,*rest:d['query_support']*.9)
    monkeypatch.setattr(report,'assess',measured)
    a=SimpleNamespace(work_dir=str(tmp_path),name='campaign',parent_run=str(parent),config=str(config),device='cpu',
        resume=True,mode='all',shinka_python=sys.executable)
    assert m.launch(a)==root
    assert read(root/'summary.json')['task_coverage_complete']==dict.fromkeys(cfg['arms'],True)
    assert (root/'memory_library.json').exists() and (root/'model_registry.json').exists()
    assert (tmp_path/'agent_memory/campaign/source_library.json').exists()
    verify(root);heartbeat=(root/'heartbeat.json').read_bytes();m.launch(a)
    assert (root/'heartbeat.json').read_bytes()==heartbeat


@pytest.mark.skipif(sys.platform!='linux',reason='Real protected worker requires Linux Landlock/seccomp')
def test_actual_kernel_worker_keeps_credentials_outside_candidate(tmp_path,monkeypatch):
    pool=tmp_path/'arms/meta/repeat';_,_,_=pool_fixture(pool)
    monkeypatch.setenv('DEEPSEEK_API_KEY','never-in-a-candidate')
    body='def fit_predict(data,budget):\n    import os\n    assert "DEEPSEEK_API_KEY" not in os.environ\n    return data["query_support"]*.9\n'
    r=runtime.assess(pool,body)
    assert r['status']=='valid',r.get('error')
