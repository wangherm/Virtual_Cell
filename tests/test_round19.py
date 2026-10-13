"""Offline tiny RANDOM transformer software tests; never evidence of biological accuracy."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from vcell.model_agent.artifacts import read,verify,seal,arrays
from vcell.model_agent.contracts import digest
from vcell.round19.agents import DEFAULT_RECIPE,TrainingAgent,IterationAgent,guarded_call,validate_recipe
from vcell.round19.predictor import DeepSeekPredictor,train_worker
from vcell.round19.cache import crossfit_responses
from vcell.utils import write_json
from test_deepseek_agent import Client,reply
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'scripts'))
import run_round19 as runner


def tiny_model(folder):
    from transformers import Qwen2Config,Qwen2Model,PreTrainedTokenizerFast
    from tokenizers import Tokenizer,models,pre_tokenizers
    torch.set_num_threads(1);torch.manual_seed(1)
    Qwen2Model(Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=48,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)).save_pretrained(folder)
    t=Tokenizer(models.WordLevel({'[PAD]':0,'[UNK]':1,'Predict':2,'Human':3,'target':4},unk_token='[UNK]'))
    t.pre_tokenizer=pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=t,pad_token='[PAD]',unk_token='[UNK]').save_pretrained(folder)
    return {'model_dir':str(folder),'dtype':'float32','identity':{'fixture':'TINY_RANDOM_NOT_RESEARCH_PRETRAINING'}}


def test_agent_roles_strict_recipes_and_paid_rejections(tmp_path,monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','SECRET_NEVER_SAVE')
    training={'tool':'train','expert_id':None,'reason':'one testable change','recipe':DEFAULT_RECIPE}
    review={'tool':'review','expert_id':None,'reason':'Source evidence only','next_hypothesis':'Try LoRA with unchanged folds'}
    a=TrainingAgent(tmp_path/'a',client=Client([reply(content=json.dumps(training))]),max_api_calls=100)
    b=IterationAgent(tmp_path/'b',client=Client([reply(content=json.dumps(review))]),max_api_calls=100)
    obs={'iteration':1,'outer_results_available':False}
    assert guarded_call(a,obs)[0].recipe==DEFAULT_RECIPE
    assert guarded_call(b,obs)[0].next_hypothesis==review['next_hypothesis']
    assert a.config['prompt_sha256']!=b.config['prompt_sha256']
    assert guarded_call(a,obs)[0].recipe==DEFAULT_RECIPE and a.summary()['attempted_requests']==1
    malformed={**training,'recipe':{**DEFAULT_RECIPE,'learning_rate':1.}}
    bad=TrainingAgent(tmp_path/'bad',client=Client([reply(content=json.dumps(malformed))]),max_api_calls=100)
    decision,failure=guarded_call(bad,obs);assert decision is None and failure['validation_rule']=='Recipe exceeds fixed numerical bounds'
    assert guarded_call(bad,obs)[0] is None and bad.summary()['attempted_requests']==1
    text=''.join(p.read_text() for p in tmp_path.rglob('*.json'))
    assert 'SECRET_NEVER_SAVE' not in text and 'PRIVATE_REASONING' not in text
    for changed in ({'code':'import os'},{'rank':64},{'teachers':['state_st']},{'width':True},{'use_string':'yes'},{'learning_rate':float('nan')}):
        with pytest.raises(ValueError):validate_recipe({**DEFAULT_RECIPE,**changed})


def test_real_transformer_gradients_frozen_and_lora(tmp_path):
    spec=tiny_model(tmp_path/'model')
    for adaptation in ('frozen','lora'):
        model=DeepSeekPredictor(spec,4,5,6,3,{**DEFAULT_RECIPE,'adaptation':adaptation,'rank':16})
        inputs=[torch.randn(2,4),torch.randn(2,5),torch.randn(2,3,6),torch.ones(2,3,dtype=torch.long),torch.ones(2,3,dtype=torch.long)]
        prediction=model(*inputs);assert prediction.shape==(2,3)
        prediction.square().mean().backward()
        assert model.control.weight.grad.abs().sum()>0
        assert all(p.grad is None for p in model.backbone.parameters() if not p.requires_grad)
        if adaptation=='lora':assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'lora_B' in n)
        state=model.trainable_state();model.restore(state)
        assert all('base_layer' not in name for name in state)


def synthetic_worker_cache(root):
    rng=np.random.default_rng(19);root.mkdir(parents=True)
    values={'control':rng.normal(size=(12,4)).astype(np.float32),'target':rng.normal(size=(12,5)).astype(np.float32),
        'teachers':rng.normal(size=(12,3,6)).astype(np.float32),'basis':np.eye(6,dtype=np.float32),
        'response_scale':np.float32(1.),'query':np.array([10,11]),'context_codes':np.repeat([0,1,2],4),
        'target_codes':np.tile([0,1,2,3],3),'row_ids':np.array(['row'+str(i) for i in range(12)]),
        'genes':np.array(['G'+str(i) for i in range(6)]),'prompt_texts':np.array(['Predict Human target G']*12),
        'fit_weights':np.ones(8)/8,'calibration_weights':np.ones(2)/2}
    np.savez_compressed(root/'features.npz',**values)
    np.savez_compressed(root/'source_labels.npz',fit=np.arange(8),calibration=np.array([8,9]),
        fit_delta=rng.normal(size=(8,6)).astype(np.float32),calibration_delta=rng.normal(size=(2,6)).astype(np.float32))
    seal(root);return root


def test_worker_exact_step_resume_and_no_held_labels(tmp_path,monkeypatch):
    import vcell.round19.predictor as trainer
    spec=tiny_model(tmp_path/'model');cache=synthetic_worker_cache(tmp_path/'cache')
    cfg={'cache':str(cache),'recipe':{**DEFAULT_RECIPE,'adaptation':'lora','specificity_weight':.1},'model':spec,
        'output_dir':str(tmp_path/'full'),'device':'cpu','seed':17,'max_steps':3,'batch_size':4,'save_every':1}
    train_worker(cfg);actual=arrays(Path(cfg['output_dir'])/'prediction.npz')['prediction']
    interrupted={**cfg,'output_dir':str(tmp_path/'interrupted')};original=trainer.atomic_torch_save
    def stop(value,path):
        original(value,path)
        if Path(path).name=='last.pt' and value['step']==1:raise InterruptedError('fixture interruption')
    monkeypatch.setattr(trainer,'atomic_torch_save',stop)
    with pytest.raises(InterruptedError):train_worker(interrupted)
    monkeypatch.setattr(trainer,'atomic_torch_save',original);train_worker(interrupted,True)
    resumed=arrays(Path(interrupted['output_dir'])/'prediction.npz')['prediction']
    np.testing.assert_array_equal(actual,resumed)
    marker=(Path(interrupted['output_dir'])/'COMPLETE.json').read_bytes();train_worker(interrupted,True)
    assert marker==(Path(interrupted['output_dir'])/'COMPLETE.json').read_bytes()
    with pytest.raises(ValueError,match='Frozen'):train_worker({**interrupted,'max_steps':4},True)
    assert set(arrays(cache/'source_labels.npz'))=={'fit','calibration','fit_delta','calibration_delta'}
    assert not any(k.endswith('delta') for k in arrays(cache/'features.npz'))
    env=runner.worker_environment();assert not any(k for k in env if k!='TOKENIZERS_PARALLELISM' and any(x in k.upper() for x in ('KEY','SECRET','PASSWORD','TOKEN','CREDENTIAL')))


def test_response_teacher_oof_excludes_query_line_and_truth(tmp_path):
    torch.set_num_threads(1);rng=np.random.default_rng(3)
    meta=pd.DataFrame({'context':np.repeat(['A','B','C'],4),'cell_line':np.repeat(['A','B','C'],4),
        'perturbation':np.tile(['G0','G1','G2','G3'],3),'row_id':['row'+str(i) for i in range(12)]})
    data={'meta':meta,'genes':np.array(['g'+str(i) for i in range(5)]),'perturbations':np.array(['G0','G1','G2','G3']),
        'pert_idx':np.tile(np.arange(4),3),'delta':rng.normal(size=(12,5)).astype(np.float32)}
    control=rng.normal(size=(12,8)).astype(np.float32);assets={'relations':rng.random((4,3,5)).astype(np.float32)}
    p,audit=crossfit_responses(data,control,assets,np.arange(12),2,17)
    poison=copy.deepcopy(data);poison['delta'][:4]+=1000
    other,_=crossfit_responses(poison,control,assets,np.arange(12),2,17)
    np.testing.assert_allclose(p[:4],other[:4],rtol=0,atol=0)
    assert all(r['held_training_cell_line'] not in r['response_fit_cell_lines'] and not set(r['response_fit_rows'])&set(r['predicted_rows']) for r in audit)


def test_full_dual_pipeline_freezes_then_evaluates(tmp_path,monkeypatch):
    """Real tiny source experts, real transformer updates, mocked paid decisions."""
    from test_agent_nested import fixture
    import run_round18 as source
    from vcell.background_training import pretrain
    from vcell.round15_p4 import worker
    _,data,parent,_=fixture(tmp_path)
    src=source.main(['--work-dir',str(tmp_path),'--p4-run',str(parent),'--name','source',
        '--outer-contexts','BXPC3_IFNG','--seeds','17','--device','cpu','--max-steps','1',
        '--pretrain-steps','1','--batch-size','4','--save-every','1','--targets-per-context','3','--profile-targets','3',
        '--search-arms','fixed','--trials','1','--cycles','1','--prepare-only'])
    args=SimpleNamespace(**read(src/'plan.json'));_,plans,bs,ws=source.core.prepared_inputs(args,parent,src,freeze_inputs=False)
    for cfg in bs.values():pretrain(cfg,True)
    for cfg in ws.values():worker(cfg,True)
    for pl in plans:
        for fold in pl['folds']:source.zero_pool(source.core.pool_for(fold,src))
    seal(src)
    spec=tiny_model(tmp_path/'deepseek_new_model')
    monkeypatch.setattr(runner,'model_specs',lambda *args:{'deepseek':spec,'mlp':spec})
    # Run actual training directly for speed; production dispatch has stripped env.
    fits=[]
    def dispatch(specs,*args):
        for cfg in specs.values():fits.append(cfg['output_dir']);train_worker(cfg,True)
    monkeypatch.setattr(runner,'dispatch',dispatch);monkeypatch.setenv('DEEPSEEK_API_KEY','SECRET')
    recipe={**DEFAULT_RECIPE,'width':32,'learning_rate':.0003}
    tclient=Client([reply(content=json.dumps({'tool':'train','expert_id':None,'reason':'change width','recipe':recipe}))])
    bclient=Client([reply(content=json.dumps({'tool':'review','expert_id':None,'reason':'Source checked','next_hypothesis':'Next use LoRA'}))])
    monkeypatch.setattr(runner,'TrainingAgent',lambda *args,**kwargs:TrainingAgent(*args,**kwargs,client=tclient))
    monkeypatch.setattr(runner,'IterationAgent',lambda *args,**kwargs:IterationAgent(*args,**kwargs,client=bclient))
    command=['--work-dir',str(tmp_path),'--source-run',str(src),'--name','r19','--device','cpu',
        '--iterations','1','--max-steps','2','--teacher-steps','1','--batch-size','4','--save-every','1']
    root=runner.main(command);verify(root)
    assert read(root/'iteration_progress.json')['completed_rounds']==1
    assert read(root/'search_frozen.json')['outer_used'] is False
    assert read(root/'summary.json')['student_weights_updated'] and not read(root/'summary.json')['api_model_weights_updated']
    scores=pd.read_csv(root/'comparison.csv');assert set(scores.method)>={'zero','string_ridge','deepseek_fixed_fusion','mlp_same_inputs','dual_agent_selected'}
    # The fast teacher cache must really execute the existing P4 bundles.
    from vcell.model_agent.experts import Registry
    from vcell.model_agent.contracts import Query
    for fold in plans[0]['folds']:
        cache=root/'caches'/fold['name'];features=arrays(cache/'features.npz');row=int(features['query'][0])
        meta=data['meta'].iloc[row]
        q=Query(str(meta.row_id),str(meta.context),str(meta.perturbation),str(meta.stimulus),
            tuple(data['genes']),tuple(data['baseline'][row]),source_partition='train')
        pool=Registry(src/'pools'/fold['name']/'registry_with_zero.json')
        predictions=arrays(cache/'teacher_query_predictions.npz')['prediction'][0]
        for j,name in enumerate(runner.TEACHERS):
            np.testing.assert_allclose(predictions[j],pool.experts['expanded_'+name].predict(q),rtol=1e-5,atol=1e-7)
    for path in root.glob('caches/*/lineage.json'):
        lineage=read(path);assert lineage['external_native_teachers']['state_st']=='NOT_CONNECTED'
    body=''.join(json.dumps(c[1]['json']) for client in (tclient,bclient) for c in client.calls)
    assert 'outer_per_query' not in body and 'BXPC3_IFNG' not in body and 'source_mse' in body
    marker=(root/'COMPLETE.json').read_bytes();fit_count=len(fits);runner.main(command+['--resume'])
    assert marker==(root/'COMPLETE.json').read_bytes() and len(fits)==fit_count and len(tclient.calls)==len(bclient.calls)==1
    with pytest.raises(ValueError,match='Frozen'):runner.main([*command[:], '--iterations','2','--resume'])


def test_controller_actually_completes_100_rounds_and_200_calls(tmp_path,monkeypatch):
    """100 real controller rounds with mocked API/training, not a research result."""
    source=tmp_path/'source';source.mkdir()
    folds=[{'name':'fold'+str(i),'configs':{'expanded':{}}} for i in range(4)]
    mock_data=tmp_path/'mock_data';mock_data.mkdir()
    (mock_data/'dataset.npz').write_bytes(b'MOCK_DATASET');(mock_data/'metadata.csv').write_text('MOCK_METADATA')
    write_json(source/'inputs.json',{'data_dir':str(mock_data),'data_fingerprint':'mock',
        'folds':[{'outer_context':'BXPC3_IFNG','seed':17,'folds':folds}]});seal(source)
    monkeypatch.setattr(runner,'load_prepared',lambda _: {'audit':{'fingerprint':'mock'}})
    spec={'model_dir':'mock','dtype':'float32','identity':{'fixture':'MOCK'}}
    monkeypatch.setattr(runner,'model_specs',lambda *args:{'deepseek':spec,'mlp':spec})
    def cache(data,fold,pool,path,*args):return synthetic_worker_cache(path)
    monkeypatch.setattr(runner,'build_cache',cache)
    frame=pd.DataFrame({'row_id':['source_q'],'context':['SOURCE_A'],'target':['G0'],'method':['zero'],
        'mse':[1.],'zero_mse':[1.],'prediction_norm':[0.],'truth_norm':[1.],'cosine':[0.],'specificity_gap':[0.]})
    monkeypatch.setattr(runner,'source_baselines',lambda *args:({'zero':1.},frame.copy()))
    def scores(data,caches,specs,method):
        f=frame.copy();f['method']=method;f['mse']=.9
        return {'source_mse':.9,'zero_mse':1.,'contexts':{'SOURCE_A':{'mse':.9}},'outer_used':False},f
    monkeypatch.setattr(runner,'source_scores',scores)
    fits=[]
    def dispatch(specs,*args):
        for cfg in specs.values():
            path=Path(cfg['output_dir']);write_json(path/'manifest.json',{'config':cfg});seal(path);fits.append(str(path))
    monkeypatch.setattr(runner,'dispatch',dispatch);monkeypatch.setenv('DEEPSEEK_API_KEY','SECRET')
    responses=[]
    for i in range(1,101):
        recipe={**DEFAULT_RECIPE,'learning_rate':1e-5+i*5e-6}
        responses.append(reply(content=json.dumps({'tool':'train','expert_id':None,'reason':'Bounded unique recipe '+str(i),'recipe':recipe})))
    a=Client(responses);b=Client([reply(content=json.dumps({'tool':'review','expert_id':None,'reason':'Source review','next_hypothesis':'Modify learning rate within budget'})) for _ in range(100)])
    monkeypatch.setattr(runner,'TrainingAgent',lambda *args,**kwargs:TrainingAgent(*args,**kwargs,client=a))
    monkeypatch.setattr(runner,'IterationAgent',lambda *args,**kwargs:IterationAgent(*args,**kwargs,client=b))
    command=['--work-dir',str(tmp_path),'--source-run',str(source),'--name','hundred','--iterations','100','--device','cpu']
    root=runner.main(command);verify(root);progress=read(root/'iteration_progress.json')
    assert progress['completed_rounds']==progress['evaluated_candidates']==100
    assert len(a.calls)==len(b.calls)==100 and progress['training_api']['attempted_requests']==100
    assert len([p for p in fits if '/trial_' in p.replace('\\','/')])==300
    marker=(root/'COMPLETE.json').read_bytes();runner.main(command+['--resume'])
    assert (root/'COMPLETE.json').read_bytes()==marker and len(a.calls)==len(b.calls)==100
