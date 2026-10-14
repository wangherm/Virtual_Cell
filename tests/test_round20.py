"""Leakage, deterministic resume, bounded code and tiny offline execution tests."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from vcell.model_agent.artifacts import arrays,read,seal,verify
from vcell.model_agent.candidate_code import BASE_HEAD
from vcell.round20.signal import fit_signal,exposure_weights,shuffle_target_prompts,promotion,summarize
from vcell.round20.reference import references,worker
from vcell.round19.predictor import DeepSeekPredictor,train_worker
from vcell.round19.agents import DEFAULT_RECIPE
from vcell.utils import write_json
from test_round19 import tiny_model,synthetic_worker_cache
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'scripts'))
import run_round20 as runner


def measured():
    meta=pd.DataFrame([{'row_id':f'{c}:{t}:{r}','context':c+'_IFNG','cell_line':c,'perturbation':t,
        'dataset':'study','stimulus':'IFNG','time':'24h','modality':'CRISPRi','batch':r,'replicate':r,'split':'train'}
        for c in ('A','B','C') for t in ('T0','T1','T2','T3') for r in ('Rep1','Rep2')])
    effects=np.array([[.2,-.3,.4,.1],[-.4,.5,-.1,.3],[.3,-.2,-.1,.6],[.5,.1,-.2,-.3]],np.float32)
    delta=np.tile(np.repeat(effects,2,axis=0),(3,1));return {'meta':meta,'delta':delta,'genes':np.array(['G0','G1','G2','G3'])}


def test_signal_uses_only_fit_and_preserves_outer_poison():
    data=measured();fit=np.arange(16)
    a,p,g,mask=fit_signal(data,fit);assert a['signal_gate'] and len(p)==8 and mask.all()
    poison=copy.deepcopy(data);poison['delta'][16:]+=10000
    b,pp,gg,mm=fit_signal(poison,fit);assert a==b
    pd.testing.assert_frame_equal(g,gg);np.testing.assert_array_equal(mask,mm)
    poison['delta'][8:16:2]*=-1
    assert not fit_signal(poison,fit)[0]['signal_gate']


def test_reference_conditions_whole_line_and_global_unknown_labels():
    data=measured();m=data['meta'];fit=np.flatnonzero(~m.cell_line.eq('C') & ~m.perturbation.eq('T3'))
    control=np.arange(len(m)*2).reshape(len(m),2).astype(np.float32)
    ref,_,av,audit=references(m,fit,data['delta'][fit],control,['T3'])
    assert av[16] and not av[22]  # C:T0 has source support; C:T3 is unknown everywhere.
    np.testing.assert_array_equal(ref[16],data['delta'][16])
    assert all(r['query_cell_line'] not in r['donor_cell_lines'] for r in audit)
    assert all(m.iloc[x].batch==r['conditions']['batch'] for r in audit for x in r['donor_rows'])
    poisoned=copy.deepcopy(data);poisoned['delta'][16:]+=1000
    r=references(m,fit,poisoned['delta'][fit],control,['T3'])[0];np.testing.assert_array_equal(r,ref)
    for mode in ('wrong','shuffled','none'):
        _,_,available,_=references(m,fit,data['delta'][fit],control,['T3'],mode)
        assert not available[m.perturbation.eq('T3')].any()
    with pytest.raises(ValueError,match='Globally unknown'):references(m,np.arange(16),data['delta'][:16],control,['T3'])
    altered=m.copy();altered.loc[16,'time']='48h'
    assert not references(altered,fit,data['delta'][fit],control,['T3'])[2][16]


def test_exposure_and_prompt_ablation():
    m=measured()['meta'];old=m.iloc[:2].copy();old.stimulus='unstimulated';mix=pd.concat([old,m],ignore_index=True)
    w=exposure_weights(mix);assert w[:2].sum()==pytest.approx(.5) and w[2:].sum()==pytest.approx(.5)
    texts=[f'Predict Human CRISPRi target T{i}; cell C{i}; stimulus IFNG.' for i in range(4)]
    shuffled=shuffle_target_prompts(texts,np.arange(4),17)
    assert texts!=shuffled
    assert all(t.split('; cell ')[1]==s.split('; cell ')[1] for t,s in zip(texts,shuffled))
    bad={'mse':1.,'zero_mse':1.,'strong_mse':None,'specificity_gap':None,'amplitude_ratio':0.}
    assert not promotion(bad,bad)
    good={'mse':.8,'strong_mse':.7,'specificity_gap':.1,'amplitude_ratio':.2}
    assert promotion(good,{**good,'mse':1.,'strong_mse':.8})
    baseline={**good,'mse':1.,'strong_mse':.8}
    assert not runner.recovery_gate({'wrong':good,'shuffled':good,'direct_shuffled_target':good},baseline)
    assert runner.recovery_gate({'matched':good},baseline)


def test_random_backbone_and_bounded_train_scale(tmp_path):
    spec=tiny_model(tmp_path/'m');recipe={**DEFAULT_RECIPE,'teachers':[],'structure':'direct','kd_weight':0.}
    torch.manual_seed(17);pre=DeepSeekPredictor(spec,4,5,6,3,recipe)
    torch.manual_seed(17);random=DeepSeekPredictor({**spec,'random_frozen':True,'correction_scale':'learned'},4,5,6,3,recipe)
    assert not torch.equal(pre.backbone.get_input_embeddings().weight,random.backbone.get_input_embeddings().weight)
    assert torch.equal(pre.control.weight,random.control.weight) and torch.equal(pre.output[-1].weight,random.output[-1].weight)
    assert all(not p.requires_grad for p in random.backbone.parameters())
    args=[torch.randn(2,4),torch.randn(2,5),torch.randn(2,3,6),torch.ones(2,3,dtype=torch.long),torch.ones(2,3,dtype=torch.long)]
    random(*args).square().mean().backward();assert random.scale_logit.grad is not None
    assert float(random.amplitude_scale().detach())==pytest.approx(.05)
    with torch.no_grad():random.scale_logit.fill_(100)
    assert 0<float(random.amplitude_scale().detach())<=1
    cache=synthetic_worker_cache(tmp_path/'cache');cfg={'cache':str(cache),'recipe':recipe,'model':{**spec,'random_frozen':True},
        'output_dir':str(tmp_path/'out'),'device':'cpu','seed':17,'max_steps':1,'batch_size':4,'save_every':1}
    train_worker(cfg);assert not read(tmp_path/'out/model_info.json')['pretrained_language_backbone']
    assert 'fit_prediction_to_truth_norm' in read(tmp_path/'out/history.json')[0]


def reference_cache(root,data):
    root.mkdir();m=data['meta'];fit=np.flatnonzero(~m.cell_line.eq(m.cell_line.iloc[16])&~m.perturbation.eq('T3'));cal=np.array([6,7]);query=np.arange(16,24)
    # Global T3 calibration rows are withheld too; source calibration from legal targets.
    cal=np.array([4,5]);fit=fit[~np.isin(fit,cal)]
    rng=np.random.default_rng(7)
    np.savez_compressed(root/'features.npz',control=rng.normal(size=(len(m),3)).astype(np.float32),
        target=rng.normal(size=(len(m),2)).astype(np.float32),basis=np.eye(4,dtype=np.float32),response_scale=np.float32(1),
        genes=data['genes'],query=query,row_ids=m.row_id.to_numpy(dtype='U'))
    np.savez_compressed(root/'source_labels.npz',fit=fit,calibration=cal,fit_delta=data['delta'][fit],calibration_delta=data['delta'][cal]);seal(root)
    return root


def test_reference_fixed_step_exact_resume_and_prediction_bound(tmp_path,monkeypatch):
    import vcell.round20.reference as module
    data=measured();meta=tmp_path/'metadata.csv';data['meta'].to_csv(meta,index=False);cache=reference_cache(tmp_path/'c',data)
    cfg={'cache':str(cache),'metadata':str(meta),'held_targets':['T3'],'mode':'matched','alpha':.2,'radius':.2,
        'learning_rate':.001,'weight_decay':.01,'code':BASE_HEAD,'device':'cpu','seed':17,'max_steps':3,'batch_size':4,'save_every':1,'output_dir':str(tmp_path/'full')}
    worker(cfg);p=arrays(tmp_path/'full/prediction.npz');lineage=read(tmp_path/'full/lineage.json')
    assert (p['prediction'][data['meta'].iloc[16:].perturbation.eq('T3')]==0).all()
    assert lineage['checkpoint_selection'].startswith('fixed final step')
    other={**cfg,'output_dir':str(tmp_path/'resume')};save=module.atomic_torch_save
    def stop(value,path):
        save(value,path)
        if Path(path).name=='last.pt' and value['step']==1:raise InterruptedError('fixture')
    monkeypatch.setattr(module,'atomic_torch_save',stop)
    with pytest.raises(InterruptedError):worker(other)
    monkeypatch.setattr(module,'atomic_torch_save',save);worker(other,True)
    np.testing.assert_array_equal(p['prediction'],arrays(tmp_path/'resume/prediction.npz')['prediction'])
    with pytest.raises(ValueError,match='Frozen'):worker({**other,'radius':.3},True)
    # No reference or correction output can exceed its declared L2 correction radius.
    direct=references(data['meta'],arrays(cache/'source_labels.npz')['fit'],arrays(cache/'source_labels.npz')['fit_delta'],arrays(cache/'features.npz')['control'],['T3'])[0][16:]
    assert np.linalg.norm(p['prediction']-.2*direct,axis=1).max()<=lineage['correction_L2_bound_normalized']+1e-6


def test_alpha_selection_never_reads_outer():
    f=pd.DataFrame({'context':['A']*6,'target':['T0','T1']*3,'method':['ridge']*6,'alpha':np.repeat([0.,.2,1.],2),
        'mse':[1.,1.,.8,.8,2.,2.]})
    chosen,selected=runner.choose_alphas(f);assert chosen=={'ridge':.2} and len(selected)==2


def test_full_gated_pipeline_freezes_before_outer_and_exact_resume(tmp_path,monkeypatch):
    from test_agent_nested import fixture
    from vcell.background_training import pretrain
    from vcell.round15_p4 import worker as p4worker
    import run_round18
    _,data,parent,_=fixture(tmp_path)
    src=run_round18.main(['--work-dir',str(tmp_path),'--p4-run',str(parent),'--name','src','--outer-contexts','BXPC3_IFNG',
        '--seeds','17','--device','cpu','--max-steps','1','--pretrain-steps','1','--batch-size','4','--save-every','1',
        '--profile-targets','0','--search-arms','fixed','--cycles','1','--trials','1','--prepare-only'])
    from types import SimpleNamespace
    args=SimpleNamespace(**read(src/'plan.json'));_,plans,bs,ws=run_round18.core.prepared_inputs(args,parent,src,False)
    for cfg in bs.values():pretrain(cfg,True)
    for cfg in ws.values():p4worker(cfg,True)
    seal(src)
    old=tmp_path/'runs/old';write_json(old/'plan.json',{'source_run':str(src)});write_json(old/'models.json',{'deepseek':tiny_model(tmp_path/'m'),'mlp':{'identity':{'fixture':'mock'}}});seal(old)
    def core_dispatch(specs,root,parallel,label,stage):
        for cfg in specs.values():(pretrain if stage=='background' else p4worker)(cfg,True)
    monkeypatch.setattr(runner.core,'dispatch',core_dispatch)
    checks=[]
    def run_jobs(specs,root,parallel):
        for cfg in specs.values():
            if '_outer_' in cfg['output_dir']:assert (root/'search_frozen.json').exists()
            checks.append(cfg['output_dir']);(train_worker if cfg['kind']=='direct' else worker)(cfg,True)
    monkeypatch.setattr(runner,'dispatch',run_jobs)
    monkeypatch.setattr(runner,'verify_models',lambda models:None)  # Explicit tiny fixture, no research snapshot claim.
    def never(*args,**kwargs):raise AssertionError('Agent must remain gated when source replicate evidence is unavailable')
    monkeypatch.setattr(runner,'SearchAgent',never)
    command=['--work-dir',str(tmp_path),'--round19-run',str(old),'--name','r20','--device','cpu','--max-steps','1','--teacher-steps','1','--save-every','1','--batch-size','4']
    root=runner.main(command);verify(root)
    summary=read(root/'summary.json');assert summary['attempted_candidate_slots']==0 and all(not g['agent_search'] for g in summary['gates'].values())
    assert not read(root/'search_frozen.json')['outer_used']
    comparison=pd.read_csv(root/'comparison.csv');assert set(comparison.method)>={'direct_mlp','matched','zero','reference_only'}
    marker=(root/'COMPLETE.json').read_bytes();count=len(checks);runner.main(command+['--resume'])
    assert marker==(root/'COMPLETE.json').read_bytes() and len(checks)==count
    assert (root/'round20_review_light.tar.gz').exists()


def test_agent_bounded_proposals_and_paid_truncation(tmp_path,monkeypatch):
    from test_deepseek_agent import Client,reply
    from vcell.round20.agents import SearchAgent
    from vcell.model_agent.research import Proposal
    import round19_recovery
    monkeypatch.setenv('DEEPSEEK_API_KEY','NO_SAVE')
    p=Proposal('propose_trial',None,'Try a small interaction',BASE_HEAD,32,.0005,.01,'mse')
    from dataclasses import asdict
    client=Client([reply(content=json.dumps(asdict(p))),reply(finish='length')]);agent=SearchAgent(tmp_path/'api',client=client,max_api_calls=2)
    decision,failure=round19_recovery.call(agent,{'iteration':1,'outer_results_available':False},'training');assert decision==p and failure is None
    decision,failure=round19_recovery.call(agent,{'iteration':2,'outer_results_available':False},'training');assert decision is None and agent.summary()['attempted_requests']==2
    assert 'NO_SAVE' not in ''.join(p.read_text() for p in (tmp_path/'api').glob('*.json'))


def test_autonomous_head_search_feedback_resume_and_isolated_admission(tmp_path,monkeypatch):
    from test_deepseek_agent import Client,reply
    from dataclasses import asdict
    from types import SimpleNamespace
    from vcell.round20.agents import SearchAgent,ShortReviewer
    from vcell.model_agent.research import Proposal
    data=measured();meta=data['meta'];names={'A':'HAP1','B':'BXPC3','C':'MCF7'}
    meta['cell_line']=meta.cell_line.map(names);meta['context']=meta.cell_line+'_IFNG'
    metadata=tmp_path/'meta.csv';meta.to_csv(metadata,index=False);cache=reference_cache(tmp_path/'c',data)
    # Reuse one real source fold in this software fixture, not a biological OOF claim.
    caches={'f':cache};masks={'f':np.ones(4,bool)}
    monkeypatch.setenv('DEEPSEEK_API_KEY','NOT_IN_WORKER')
    clients={};reviews={}
    def planner(folder,*args,**kwargs):
        arm=Path(folder).parent.name
        decisions=[Proposal('propose_trial',None,'Independent bounded head '+str(i),BASE_HEAD.replace('32',str(64+64*i)),32,.0005,.01,'mse') for i in range(2)]
        client=Client([reply(content=json.dumps(asdict(p))) for p in decisions]);clients[arm]=client
        return SearchAgent(folder,*args,**kwargs,client=client)
    def reviewer(folder,*args,**kwargs):
        client=Client([reply(content=json.dumps({'tool':'review','expert_id':None,'reason':'Source result checked','next_hypothesis':'Try a wider head, keep retrieval frozen'}))]*2)
        reviews['dual']=client;return ShortReviewer(folder,*args,**kwargs,client=client)
    monkeypatch.setattr(runner,'SearchAgent',planner);monkeypatch.setattr(runner,'ShortReviewer',reviewer)
    fits=[]
    def run_jobs(specs,*args):
        for cfg in specs.values():fits.append(cfg['output_dir']);worker(cfg,True)
    monkeypatch.setattr(runner,'dispatch',run_jobs)
    a=SimpleNamespace(seed=17,device='cpu',max_steps=2,batch_size=4,save_every=1,parallel_jobs=1,agent_model='deepseek-v4-pro')
    baseline={'mse':10.,'strong_mse':10.,'specificity_gap':-10.,'amplitude_ratio':0.}
    records,selected,ledger=runner.search(a,{'name':'dev','outer_context':'A549_IFNG'},data,caches,tmp_path,masks,{},metadata,['T3'],.2,baseline,2)
    assert len(records)==8 and len(clients['single'].calls)==len(clients['dual'].calls)==len(reviews['dual'].calls)==2
    assert all(not r['outer_used'] for r in records)
    assert all(r['status']=='evaluated' for r in records if r['arm'] in ('single','dual'))
    body=json.loads(clients['dual'].calls[1][1]['json']['messages'][1]['content'])
    assert body['previous_source_trials'][0]['review']['next_hypothesis'].startswith('Try a wider')
    assert body['outer_results_available'] is False
    assert (tmp_path/'search/dev/dual/trial_01/f_admission.json').exists()
    assert all(s['attempted_requests']==2 for s in ledger)
    # Provider actions and completed trials are replayed without new paid calls.
    runner.search(a,{'name':'dev','outer_context':'A549_IFNG'},data,caches,tmp_path,masks,{},metadata,['T3'],.2,baseline,2)
    assert all(len(c.calls)==0 for c in clients.values()) and len(reviews['dual'].calls)==0


def test_snapshot_verification_and_legacy_running_model_rejection(tmp_path,monkeypatch):
    lock=read(REPO/'configs/deepseek_predictor.lock.json');identity={k:lock[k] for k in ('model_id','revision','files')}
    models={'deepseek':{'identity':identity,'model_dir':str(tmp_path)}}
    monkeypatch.setattr(runner.previous,'matches',lambda *args:True);runner.verify_models(models)
    monkeypatch.setattr(runner.previous,'matches',lambda *args:False)
    with pytest.raises(ValueError,match='snapshot changed'):runner.verify_models(models)
    import round19_recovery
    old={'code':next(iter(round19_recovery.LEGACY_CODES))};write_json(tmp_path/'plan.json',old)
    monkeypatch.setattr(round19_recovery,'namespace_identity',lambda:'MODIFIED_MODEL_NAMESPACE')
    with pytest.raises(ValueError,match='unchanged model'):round19_recovery.freeze_plan(tmp_path,{'code':'new'},True,True)


def test_status_displays_nested_queue_states(tmp_path):
    import subprocess
    root=tmp_path/'runs/status';write_json(root/'progress.json',{'jobs':{'one':{'state':'complete'},'two':{'state':'running'}}})
    write_json(root/'stage.json',{'stage':'source experts','test_evaluated':False})
    result=subprocess.run([sys.executable,str(REPO/'scripts/round20_status.py'),'--work-dir',str(tmp_path),'--name','status'],capture_output=True,text=True)
    assert result.returncode==0 and "'complete': 1" in result.stdout and 'not whole experiment' in result.stdout
