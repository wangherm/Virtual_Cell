"""Real tiny refits and bounded head search; no provider or biological performance claim."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from vcell.model_agent.artifacts import read, seal, verify
from vcell.model_agent.candidate_code import BASE_HEAD, check_head, validate_code
from vcell.model_agent.contracts import Query, digest
from vcell.model_agent.nested import DEVELOPMENT, cases, holdout_targets, lines, partition
from vcell.model_agent.research import Proposal, ResearchPlanner, parse_proposal, proposal_observation
from vcell.model_agent.routing import OneShotPlanner
from vcell.model_agent.runtime import execute
from vcell.utils import write_json
REPO=Path(__file__).resolve().parents[1]


def proposal():return Proposal('propose_trial',None,'Test a bounded bilinear interaction',BASE_HEAD,16,.0005,.01,'mse')


def test_bounded_language_rejects_io_introspection_recursion_and_unbounded_layers():
    assert check_head(BASE_HEAD)['gradient_finite']
    for text in ['import os\n'+BASE_HEAD,BASE_HEAD.replace('t + b + t * b','self.forward(target, background)'),
        BASE_HEAD.replace('32','1024'),BASE_HEAD.replace('torch.tanh(self.target(target))','torch.load(target)'),
        BASE_HEAD.replace('t + b + t * b','self.__dict__'),BASE_HEAD.replace('t + b + t * b','torch.cat([t,b,t],dim=1)')]:
        with pytest.raises((ValueError,SyntaxError)):validate_code(text)
    # Valid syntax but unusable output/gradient is an admission failure.
    with pytest.raises(ValueError,match='shape'):check_head(BASE_HEAD.replace('return self.output(t + b + t * b)','return t + b'))
    p=asdict(proposal());parse_proposal(json.dumps(p))
    for change in ({'rank':64},{'learning_rate':1.},{'expert_id':'shell'},{'shell':'ls'}):
        with pytest.raises(ValueError):parse_proposal(json.dumps({**p,**change}))


def test_research_provider_cache_and_source_observation_redaction(tmp_path,monkeypatch):
    from test_deepseek_agent import Client, reply
    monkeypatch.setenv('DEEPSEEK_API_KEY','UNIT_SECRET')
    client=Client([reply(content=json.dumps(asdict(proposal())))])
    summary={'fixed_source_mse':{'ridge':.1},'best_fixed':'ridge','oracle_relative_gap':.05,'winner_counts':{'ridge':3},
        'contexts':['B','C'],'distinct_targets':5,'note':'source only','outer_truth':'DO_NOT_SEND','oracle_outer':-1}
    obs=proposal_observation(summary,[])
    planner=ResearchPlanner(tmp_path/'api',client=client,max_api_calls=2)
    assert planner.decide(obs)==proposal()
    assert planner.decide(obs)==proposal() and len(client.calls)==1
    body=json.dumps(client.calls[0][1]['json']);assert 'DO_NOT_SEND' not in body and 'oracle_outer' not in body and 'oracle_relative_gap' not in body
    assert planner.summary()['attempted_requests']==1
    assert 'UNIT_SECRET' not in ''.join(p.read_text() for p in (tmp_path/'api').rglob('*.json'))


def test_oneshot_is_exactly_one_api_request_then_controller_finish(tmp_path,monkeypatch):
    from test_deepseek_agent import Client, reply
    from vcell.model_agent.demo import make_demo
    monkeypatch.setenv('DEEPSEEK_API_KEY','UNIT_SECRET');base=make_demo(tmp_path/'demo')
    client=Client([reply()]);planner=OneShotPlanner(tmp_path/'api',client=client,max_api_calls=1)
    result=execute(base/'registry.json',base/'query.json',tmp_path/'run',planner=planner,planner_identity=planner.identity)
    assert result['selected_expert']=='fast_linear' and len(client.calls)==1 and result['provider_summary']['attempted_requests']==1


def test_formal_outer_rejects_partial_numerical_profiles(tmp_path):
    from vcell.model_agent.demo import make_demo
    base=make_demo(tmp_path/'demo');profile=read(base/'profiles/profile.json')
    profile['models'].pop('response_linear');write_json(base/'profiles/profile.json',profile);seal(base/'profiles')
    with pytest.raises(ValueError,match='lacks source OOF'):
        execute(base/'registry.json',base/'query.json',tmp_path/'formal',profile_path=base/'profiles',formal_outer=True)


def test_reference_conditions_and_whole_line_diagnostics(tmp_path):
    from vcell.model_agent.demo import make_demo
    from vcell.model_agent.artifacts import arrays
    from vcell.model_agent.contracts import load_query
    from vcell.model_agent.experts import Expert
    base=make_demo(tmp_path/'demo');folder=base/'fast_linear';card=read(folder/'expert.json')
    card['kind']='reference';card['provenance']['fit_cell_lines']=['synthetic_outer'];write_json(folder/'expert.json',card)
    stats=arrays(folder/'input.npz');stats.update(reference_targets=np.array(['DEMO','DEMO']),
        reference_contexts=np.array(['SOURCE_A','SOURCE_B']),reference_cell_lines=np.array(['SOURCE_A','SOURCE_B']),
        reference_study=np.array(['study','other']),reference_stimulus=np.array(['unstimulated','unstimulated']),reference_time=np.array(['24h','24h']),
        reference_batch=np.array(['batch','batch']),reference_modality=np.array(['CRISPRi','CRISPRi']),reference_delta=np.ones((2,3),np.float32))
    np.savez_compressed(folder/'input.npz',**stats);seal(folder);expert=Expert(folder)
    value=asdict(load_query(base/'query.json'));value.update(study='study',time='24h',batch='batch',stimulus='unstimulated')
    q=Query(**value)
    assert expert.donors(q).tolist()==[True,False]
    assert expert.diagnostics(q)['query_cell_line_in_fit']  # lowercase provenance does not evade checks
    for field,change in [('study','other_study'),('time','48h'),('batch','unknown'),('stimulus','IFNG')]:
        assert not expert.compatible(Query(**{**value,field:change}))[0]
    with pytest.raises(ValueError,match='Reserved'):Query(**{**value,'cell_line':'HT29','context':'misnamed_state'})


def test_cpu_admission_subprocess_and_training_dispatch(tmp_path,monkeypatch):
    sys.path.insert(0,str(REPO/'scripts'));import run_round17_oof as runner
    from test_round15_p4 import p4_worker_setup
    cfg=p4_worker_setup(tmp_path);source=tmp_path/'head.py';source.write_text(BASE_HEAD)
    monkeypatch.setenv('DEEPSEEK_API_KEY','NEVER_PASS_TO_WORKERS')
    result=runner.admit(source,16,tmp_path);assert result['gradient_finite']
    # Exercise the same real subprocess handler used by AutoDL, not a mocked trainer.
    root=tmp_path/'dispatch';root.mkdir()
    bc=read(Path(cfg['encoder']).parent/'manifest.json')['config'];bc.update(output_dir=str(root/'background'))
    runner.dispatch({'bg':bc},root,1,'test_background','background')
    verify(root/'background');assert 'NEVER_PASS_TO_WORKERS' not in (root/'logs/bg.log').read_text()


def fixture(tmp_path):
    from test_round15_p4 import p4_worker_setup
    from vcell.data import load_prepared, save_prepared
    cfg=p4_worker_setup(tmp_path);old=load_prepared(cfg['data_dir']);n=len(old['meta']);m=old['meta'].copy()
    # The same K562 line appears in both old unstimulated and added IFNG data.
    for field,value in [('cell_line','unknown'),('stimulus','unstimulated'),('time','24h'),('modality','CRISPRi')]:m[field]=value
    m.cell_line=m.context
    extra=[]
    for context in (*DEVELOPMENT,'K562_IFNG'):
        for t in range(12):
            for batch in ('Rep1','Rep2'):
                extra.append({'row_id':f'{context}:P{t}:{batch}','context':context,'cell_line':context.removesuffix('_IFNG'),
                    'perturbation':f'P{t}','batch':batch,'split':'train','dataset':'GSE_TEST','stimulus':'IFNG','time':'24h','modality':'CRISPRi',
                    'n_cells':20,'n_controls':40})
    meta=pd.concat([m,pd.DataFrame(extra)],ignore_index=True);rng=np.random.default_rng(123)
    path=tmp_path/'expanded';save_prepared(path,np.r_[old['baseline'],rng.random((len(extra),8)).astype(np.float32)],
        np.r_[old['delta'],rng.normal(0,.1,(len(extra),8)).astype(np.float32)],
        np.array([list(old['perturbations']).index(t) for t in meta.perturbation]),old['genes'],old['perturbations'],meta,
        {**old['audit'],'historical_rows':n})
    expanded=load_prepared(path);parent=tmp_path/'parent'
    for variant in ('reference','expanded'):
        dest=parent/'models'/f'context_{variant}_seed17';dest.mkdir(parents=True)
        c={**cfg,'variant':variant,'data_dir':str(path) if variant=='expanded' else cfg['data_dir'],'protocol':'context'}
        write_json(dest/'manifest.json',{'config':c});seal(dest)
    seal(parent);return cfg,expanded,parent,n


def test_whole_line_across_studies_and_unseen_target_exclusions(tmp_path):
    cfg,data,parent,n=fixture(tmp_path);m=data['meta'];held=holdout_targets(m);q=cases(m,'A549_IFNG',held,6)
    before=[]
    for variant in ('reference','expanded'):
        part=partition(data,n,['A549','K562'],q,held,variant)
        assert not set(lines(m)[part['fit']+part['calibration']])&{'A549','K562'}
        assert not set(m.iloc[part['fit']+part['calibration']].perturbation)&set(held)
        before.append(m.iloc[part['calibration']].row_id.tolist())
    assert before[0]==before[1]
    poisoned=copy.deepcopy(data);poisoned['delta'][q]+=100000
    assert partition(data,n,['A549','K562'],q,held,'expanded')==partition(poisoned,n,['A549','K562'],q,held,'expanded')


def test_actual_cpu_nested_refits_profiles_search_replay_and_resume(tmp_path,monkeypatch):
    sys.path.insert(0,str(REPO/'scripts'));import run_round17_oof as runner
    from vcell.model_agent.candidate_code import check_head
    from vcell.model_agent.experts import Expert, Registry
    cfg,data,parent,n=fixture(tmp_path)
    # Direct dispatch tests real trainer functions; deployment uses subprocess queues.
    def dispatch(specs,root,parallel,label,stage):
        for cfg in specs.values():
            if stage=='background':runner.pretrain(cfg,True)
            elif stage=='candidate':runner.candidate_worker(cfg,True)
            else:runner.worker(cfg,True)
    class Planner:
        def __init__(self,*args,**kwargs):pass
        def decide(self,obs):return proposal()
        def summary(self):return {'mock_provider':True,'attempted_requests':1}
    monkeypatch.setattr(runner,'dispatch',dispatch);monkeypatch.setattr(runner,'ResearchPlanner',Planner)
    monkeypatch.setattr(runner,'admit',lambda code,rank,dest:write_json(dest/'admission.json',check_head(code.read_text(),rank=rank)))
    args=['--work-dir',str(tmp_path),'--p4-run',str(parent),'--name','nested','--device','cpu','--max-steps','2',
        '--pretrain-steps','2','--save-every','1','--batch-size','8','--targets-per-context','4','--auto-improve','--trials','1']
    root=runner.main(args);verify(root)
    inputs=read(root/'inputs.json');assert inputs['historical_rows']==n
    assert read(root/'search_and_routing_sealed.json')['outer_scores_used_for_search'] is False
    assert (root/'comparison.csv').exists();scores=pd.read_csv(root/'comparison.csv')
    assert set(scores.policy)=={'fixed','rule','risk','workflow','softmax_gate'} and scores.mse.notna().all()
    profile=read(root/'evidence/A549_IFNG_seed17/base/profiles/profile.json')
    assert len(profile['models'])==6 and len(profile['source_contexts'])==2
    research=read(root/'research/A549_IFNG_seed17/selection.json');assert research['trials'][0]['status']=='evaluated'
    outerfold=inputs['folds'][0]['folds'][0];innerfold=inputs['folds'][0]['folds'][1]
    registry=Registry(root/'pools'/outerfold['name']/'registry.json')
    for expert in registry.experts.values():assert 'A549' not in expert.provenance['fit_cell_lines']+expert.provenance['calibration_cell_lines']
    # Candidate inference matches the same shared deployment interface.
    cp=root/'research/A549_IFNG_seed17/trial_01/bundles'/innerfold['name'];candidate=Expert(cp)
    row=innerfold['query_rows'][0];qpath=runner.query_path(data,row,root,innerfold['configs']['expanded'])
    from vcell.model_agent.contracts import load_query
    pred=candidate.predict(load_query(qpath));assert pred.shape==(8,) and np.isfinite(pred).all()
    # Exercise source-selected final-head activation even if this tiny trial loses.
    from types import SimpleNamespace
    selection={'selected_candidate':{'proposal':asdict(proposal()),'head_code':str(root/'research/A549_IFNG_seed17/trial_01/head.py'),
        'trial':1,'expert_id':'agent_head_01'}}
    extra_root=tmp_path/'activate';extra_root.mkdir()
    pool=runner.activate(inputs['folds'][0],selection,data,extra_root,{outerfold['name']:registry.path},SimpleNamespace(parallel_jobs=1))
    deployed=Registry(pool).experts['agent_head_01']
    assert deployed.card['recipe_fingerprint']==candidate.card['recipe_fingerprint']
    outerq=runner.query_path(data,outerfold['query_rows'][0],root,outerfold['configs']['expanded'])
    assert np.isfinite(deployed.predict(load_query(outerq))).all()
    # Candidate checkpoint interruption resumes exactly, including dropout RNG.
    import vcell.model_agent.research as research
    base_cfg=read(cp.parents[1]/'workers'/innerfold['name']/'manifest.json')['config']
    original=research.atomic_torch_save
    def stop(value,path):
        original(value,path)
        if Path(path).name=='last.pt' and value['step']==1:raise InterruptedError('controlled candidate interruption')
    cc={**base_cfg,'output_dir':str(tmp_path/'interrupted_candidate')}
    monkeypatch.setattr(research,'atomic_torch_save',stop)
    with pytest.raises(InterruptedError):research.candidate_worker(cc)
    monkeypatch.setattr(research,'atomic_torch_save',original);research.candidate_worker(cc,True)
    left=torch.load(Path(base_cfg['output_dir'])/'model.pt',weights_only=True)['state']
    right=torch.load(Path(cc['output_dir'])/'model.pt',weights_only=True)['state']
    for k in left:torch.testing.assert_close(left[k],right[k],rtol=0,atol=0)
    # Completed resume does not refit or call the provider.
    marker=(root/'COMPLETE.json').read_bytes();runner.main(args+['--resume']);assert (root/'COMPLETE.json').read_bytes()==marker
    # Formal outer cannot reuse a query that the exported experts fitted.
    fitrow=outerfold['configs']['expanded']['partition']['fit'][0]
    fitq=runner.query_path(data,fitrow,root,outerfold['configs']['expanded'])
    with pytest.raises(ValueError,match='excluded'):
        execute(registry.path,fitq,tmp_path/'bad',profile_path=root/'evidence/A549_IFNG_seed17/base/profiles',formal_outer=True)
