"""Offline software tests. Tiny random Qwen and knowledge fixtures are NOT biology."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip('transformers')
pytest.importorskip('peft')
from test_qwen import setup
from test_round7 import expand_batches
from test_specialization import challenge_fixture
from vcell.data import load_prepared
from vcell.knowledge import load_knowledge, resolve_cards, build_knowledge
from vcell.round8 import ARMS, KnowledgeResponse, partitions, worker, load_checkpoint, predict, report
from vcell.train import tensors
from vcell.utils import write_json


def knowledge_fixture(data, path):
    rng=np.random.default_rng(8)
    n,g=len(data['perturbations']),len(data['genes'])
    modules=np.zeros((2,g),np.float32);modules[0,:g//2]=2/g;modules[1,g//2:]=2/g
    np.savez_compressed(path, genes=data['genes'],perturbations=data['perturbations'],
        semantic=rng.normal(size=(n,8)).astype(np.float32),gene_semantic=rng.normal(size=(g,8)).astype(np.float32),
        relations=rng.uniform(size=(n,3,g)).astype(np.float32)/g, modules=modules,module_names=np.array(['TEST_A','TEST_B']))
    return load_knowledge(path,data)


def config_fixture(cfg,data,path,output,arm='qwen_modules'):
    train=data['splits']['train'].tolist()
    return dict(data_dir=cfg['data_dir'],knowledge=str(path),partition={'fit':train[2:],'calibration':train[:2],
        'outer':data['splits']['val'].tolist()},protocol='context',arm=arm,seed=17,model=cfg['model'],device='cpu',
        rank=3,learning_rate=.0002,module_weight=.1,kd_weight=.5,max_steps=2,save_every=1,batch_size=2,
        gradient_accumulation=1,output_dir=str(output),teacher_cache=None)


def test_identifier_mapping_refuses_ambiguity():
    cards=resolve_cards(['A','B','C'],[{'query':'A','_id':'1','taxid':9606,'symbol':'A','summary':'ok'},
        {'query':'B','_id':'2','taxid':9606},{'query':'B','_id':'3','taxid':9606}])
    assert cards['A']['summary']=='ok'
    assert cards['B']['status']=='ambiguous' and cards['C']['status']=='missing'


def test_unseen_targets_and_double_splits_seal_test():
    meta=pd.DataFrame([{'context':c,'perturbation':f'P{p}','batch':f'b{b}','row_id':f'{c}_{p}_{b}'}
        for c in ('K562','RPE1','HepG2','Jurkat') for p in range(20) for b in range(5)])
    data={'meta':meta,'splits':{'train':np.flatnonzero(meta.context.isin(['K562','RPE1'])),
        'val':np.flatnonzero(meta.context=='HepG2'),'test':np.flatnonzero(meta.context=='Jurkat')}}
    for protocol in ('context','target','double'):
        parts=partitions(data,protocol)
        assert not set(sum(parts.values(),[])) & set(data['splits']['test'])
        if protocol != 'context':
            target_sets=[set(meta.iloc[parts[k]].perturbation) for k in ('fit','calibration','outer')]
            assert all(not target_sets[i]&target_sets[j] for i in range(3) for j in range(i))
        if protocol in ('context','double'):
            assert set(meta.iloc[parts['outer']].context)=={'HepG2'}


def test_all_branches_gradient_and_checkpoint_roundtrip(setup,tmp_path):
    cfg,data=setup
    path=tmp_path/'knowledge.npz';knowledge_fixture(data,path)
    specs={}
    for arm in ARMS:
        current=config_fixture(cfg,data,path,tmp_path/arm,arm)
        if arm=='qwen_kd':
            cache=tmp_path/'teacher.npz';gate=np.zeros(len(data['meta']),np.float32)
            gate[current['partition']['fit']]=.3
            np.savez_compressed(cache,delta=np.zeros_like(data['delta']),gate=gate,fit=current['partition']['fit'],
                row_ids=data['meta'].row_id.to_numpy(dtype='U'),genes=data['genes'])
            current['teacher_cache']=str(cache)
        worker(current)
        worker(current,resume=True)
        specs[arm]=current
        model,norm=load_checkpoint(Path(current['output_dir'])/'endpoint.pt',data)
        values,_=predict(model,tensors(data,norm),data['splits']['val'],'cpu',2)
        with np.load(Path(current['output_dir'])/'predictions.npz') as f:
            np.testing.assert_allclose(values*norm['scale'],f['outer'],atol=1e-6)
    result=report(data,specs,tmp_path)
    assert len(result)==5*len(ARMS)
    assert (tmp_path/'per_target.csv').exists()


def test_exact_resume_after_interrupt_and_mutation_guard(setup,tmp_path,monkeypatch):
    import vcell.round8 as engine
    cfg,data=setup;path=tmp_path/'knowledge.npz';knowledge_fixture(data,path)
    current=config_fixture(cfg,data,path,tmp_path/'interrupted')
    reference={**current,'output_dir':str(tmp_path/'reference')}
    worker(reference)
    save=engine.atomic_torch_save
    def interrupt(obj,path):
        save(obj,path)
        if Path(path).name=='last.pt' and obj['step']==1:
            raise InterruptedError('simulated interruption')
    monkeypatch.setattr(engine,'atomic_torch_save',interrupt)
    with pytest.raises(InterruptedError):worker(current)
    monkeypatch.setattr(engine,'atomic_torch_save',save)
    worker(current,resume=True)
    a=torch.load(tmp_path/'reference/endpoint.pt',weights_only=True)
    b=torch.load(tmp_path/'interrupted/endpoint.pt',weights_only=True)
    for key in a['state']:
        torch.testing.assert_close(a['state'][key],b['state'][key],rtol=0,atol=0)
    with pytest.raises(ValueError,match='identical'):
        worker({**current,'max_steps':3},resume=True)
    (tmp_path/'interrupted/predictions.npz').write_bytes(b'broken')
    with pytest.raises(ValueError,match='artifact changed'):worker(current,resume=True)


def test_outer_labels_cannot_change_fit_or_calibration(setup,tmp_path):
    from vcell.data import save_prepared
    cfg,data=setup;path=tmp_path/'knowledge.npz';knowledge_fixture(data,path)
    current=config_fixture(cfg,data,path,tmp_path/'original')
    worker(current)
    changed=copy.deepcopy(data);changed['delta'][data['splits']['val']]+=1000
    changed['delta'][data['splits']['test']]-=1000
    new=tmp_path/'changed_data'
    save_prepared(new,changed['baseline'],changed['delta'],changed['pert_idx'],changed['genes'],
                  changed['perturbations'],changed['meta'],changed['audit'])
    worker({**current,'data_dir':str(new),'output_dir':str(tmp_path/'changed')})
    a=torch.load(tmp_path/'original/endpoint.pt',weights_only=True)
    b=torch.load(tmp_path/'changed/endpoint.pt',weights_only=True)
    for key in a['state']:torch.testing.assert_close(a['state'][key],b['state'][key],rtol=0,atol=0)
    ja=json.loads((tmp_path/'original/calibration.json').read_text())
    jb=json.loads((tmp_path/'changed/calibration.json').read_text())
    assert ja['mix']==jb['mix'] and ja['alpha']==jb['alpha']


def test_launcher_jobs_resume_and_review_package(setup,tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=importlib.import_module('run_round8')
    cfg,data=setup;data=expand_batches(data,tmp_path/'batches')
    challenge_fixture(data,tmp_path/'challenge')
    from vcell.expansion import prepare_expansion
    expanded=load_prepared(prepare_expansion(data,load_prepared(tmp_path/'challenge'),tmp_path/'panel')['plus_h1'])
    knowledge_fixture(expanded,tmp_path/'knowledge.npz')
    write_json(tmp_path/'student.json',cfg)
    work=tmp_path/'work';work.mkdir()
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    args=['--work-dir',str(work),'--data',str(tmp_path/'batches'),'--challenge-data',str(tmp_path/'challenge'),
        '--knowledge-npz',str(tmp_path/'knowledge.npz'),'--model-dir',cfg['model']['model_id'],
        '--student-config',str(tmp_path/'student.json'),'--device','cpu','--max-steps','1',
        '--batch-size','2','--protocols','context','--seeds','17','--arms','mlp_semantic','qwen_modules']
    root=runner.main(args)
    assert json.loads((root/'COMPLETE.json').read_text())['students']==2
    runner.main(args+['--resume'])
    import tarfile
    with tarfile.open(root/'round8_review.tar.gz') as archive:
        assert 'splits/context/evaluation.npz' in archive.getnames()
        assert 'comparison.csv' in archive.getnames()
    with pytest.raises(ValueError,match='identical'):
        runner.main(args+['--resume','--max-steps','2'])
    with runner.run_lock(work/'runs'/'.round8_01.lock'):
        with pytest.raises(RuntimeError,match='already active'):
            runner.main(args+['--resume'])


def test_knowledge_builder_snapshot_and_mutation(setup,tmp_path,monkeypatch):
    import vcell.knowledge as module
    from vcell.utils import file_sha256
    cfg,data=setup
    # Artificial public-service replies for offline software coverage only.
    names=sorted(set(data['genes'])|set(data['perturbations']))
    def request(folder,url,payload):
        Path(folder).mkdir(parents=True,exist_ok=True)
        if 'mygene' in url:
            return [{'query':g,'_id':str(i),'taxid':9606,'symbol':g,'name':g,'summary':'TEST ONLY function text.'}
                    for i,g in enumerate(names)]
        return [{'preferredName_A':data['perturbations'][0],'preferredName_B':data['genes'][0],'score':.9}]
    monkeypatch.setattr(module,'cached_request',request)
    assets=tmp_path/'assets';assets.mkdir()
    path=assets/'Reactome_TEST.gmt';path.write_text('TEST_MODULE\tdescription\t'+'\t'.join(names))
    write_json(assets/'sources.json',{'files':[{'library':'Reactome_TEST','sha256':file_sha256(path)}]})
    output=tmp_path/'knowledge'
    result=build_knowledge(data,output,assets,cfg['model'],device='cpu',batch_size=4)
    values=load_knowledge(result,data)
    assert values['semantic'].shape==(len(data['perturbations']),32)
    assert values['relations'][:,2].sum()>0
    build_knowledge(data,output,assets,cfg['model'],device='cpu')
    (output/'cards.json').write_text('{}')
    with pytest.raises(ValueError,match='content changed'):
        build_knowledge(data,output,assets,cfg['model'],device='cpu')


def test_teacher_refit_uses_fit_labels_only_and_masks_missing(setup,tmp_path):
    from vcell.knowledge_teachers import prepare_kd
    from vcell.utils import file_sha256
    cfg,data=setup
    parts={'fit':data['splits']['train'].tolist(),'calibration':[],'outer':[]}
    file=tmp_path/'features.npz'
    np.savez_compressed(file,features=data['baseline'],data_fingerprint=data['audit']['fingerprint'],
        row_ids=data['meta'].row_id.to_numpy(dtype='U'))
    write_json(file.with_suffix('.json'),{'feature_file_sha256':file_sha256(file),
        'artifact_kind':'frozen_control_features_NOT_predictions','source':'TEST_ONLY'})
    a=prepare_kd(data,data,parts,{'TEST_ONLY':str(file)},tmp_path/'a',epochs=1)
    other=copy.deepcopy(data)
    other['delta'][other['splits']['val']]+=999
    other['delta'][other['splits']['test']]-=999
    b=prepare_kd(other,other,parts,{'TEST_ONLY':str(file)},tmp_path/'b',epochs=1)
    with np.load(a) as x,np.load(b) as y:
        np.testing.assert_array_equal(x['delta'],y['delta'])
        np.testing.assert_array_equal(x['gate'],y['gate'])
        assert not x['gate'][data['splits']['val']].any()
    audit=json.loads((tmp_path/'a/audit.json').read_text())
    for fold in audit['folds']:
        assert not set(fold['fit'])&set(fold['held'])
        assert set(fold['fit'])<=set(parts['fit'])
