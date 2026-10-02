"""Tiny artificial backgrounds/Qwen exercise software, not biological performance."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from test_qwen import setup
from test_round8 import knowledge_fixture, config_fixture
from test_round7 import expand_batches
from vcell.background_training import prepare_background, pretrain, masked_loss, matched_controls
from vcell.round9 import BackgroundResponse, module_matrix, module_loss, arm_matrix, diagnostics, report, fit_probe
from vcell.round8 import worker, load_checkpoint, predict
from vcell.train import tensors
from vcell.utils import write_json, file_sha256


def background_fixture(root,genes):
    root.mkdir();(root/'panel.txt').write_text('\n'.join(genes)+'\n')
    summary={};rng=np.random.default_rng(13)
    for key,splits in [('ts_blood',['train','val']),('hlca_core',['external_candidate'])]:
        folder=root/key;folder.mkdir();shards=[]
        for split in splits:
            n=8;mask=np.ones(len(genes),bool);mask[-1]=False
            x=rng.random((n,len(genes))).astype(np.float32);x[:,-1]=0
            obs=pd.DataFrame({'split':split,'donor_id':split,'donor_group':('tabula:' if key.startswith('ts_') else key+':')+split,
                'tissue':'blood','cell_type':'T cell','assay':'10x','dataset':key},index=[f'{split}_{i}' for i in range(n)])
            a=ad.AnnData(sparse.csr_matrix(x),obs=obs,var=pd.DataFrame({'available':mask},index=genes))
            path=folder/(split+'.h5ad');a.write_h5ad(path)
            shards.append({'file':path.name,'sha256':file_sha256(path),'cells':n,'split':split})
        member=folder/'membership.csv.gz';member.write_bytes(b'fixture-only')
        write_json(folder/'COMPLETE.json',{'shards':shards,'membership_sha256':file_sha256(member)})
        summary[key]={'selected_cells':sum(s['cells'] for s in shards)}
    write_json(root/'report.json',{'stage':'complete','datasets':summary})
    write_json(root/'COMPLETE.json',{'stage':'complete'});write_json(root/'run_config.json',{'fixture':True})
    return root


def fixtures(cfg,data,tmp_path,family='mlp',background='b2'):
    k=tmp_path/'knowledge.npz';knowledge_fixture(data,k)
    source=background_fixture(tmp_path/'source',data['genes'])
    cache=prepare_background(source,tmp_path/'cache',data['genes'])
    student=config_fixture(cfg,data,k,tmp_path/'student',family+'_b2_aux_true')
    student.update(experiment='round9',family=family,background=background,module_mode='aux_true',module_permutation_seed=818,
                   encoder=str(tmp_path/'pretrain/encoder.pt'),diagnostic_every=1)
    bg=dict(stage='background',data_dir=cfg['data_dir'],partition=student['partition'],background_cache=str(cache),
            background=background,family=family,model=cfg['model'],seed=17,device='cpu',pretrain_steps=2,
            pretrain_batch=4,save_every=1,output_dir=str(tmp_path/'pretrain'))
    return bg,student


def test_masks_random_modules_projection_and_ridge():
    pred=torch.tensor([[2.,500.]],requires_grad=True)
    loss=masked_loss(pred,torch.zeros_like(pred),torch.tensor([[True,False]]));loss.backward()
    assert loss==4 and pred.grad[0,1]==0
    m=np.array([[.5,.5,0,0],[0,.5,.5,0]],np.float32);r=module_matrix(m,'aux_random')
    np.testing.assert_allclose(m@m.T,r@r.T)
    assert np.array_equal((m!=0).sum(1),(r!=0).sum(1))
    p=torch.ones((2,4),requires_grad=True);head=torch.zeros((2,2),requires_grad=True)
    module_loss(p,head,torch.zeros_like(p),torch.tensor(m),'projection_true').backward()
    assert p.grad.abs().sum()>0 and head.grad is None
    rng=np.random.default_rng(1);x=np.repeat(rng.normal(size=(4,7)),3,axis=0);y=rng.normal(size=(12,2));w=rng.uniform(.1,2,12)
    coef,intercept=fit_probe(x,y,w,1.)
    xm=np.average(x,axis=0,weights=w);ym=np.average(y,axis=0,weights=w)
    expected=np.linalg.solve((x-xm).T@(w[:,None]*(x-xm))+np.eye(7),(x-xm).T@(w[:,None]*(y-ym)))
    np.testing.assert_allclose(coef,expected,atol=1e-10);np.testing.assert_allclose(intercept,ym-xm@expected)
    assert len(arm_matrix())==22


@pytest.mark.parametrize('family',['mlp','qwen'])
def test_full_transfer_all_losses_reload_and_reports(setup,tmp_path,family):
    cfg,data=setup;bg,student=fixtures(cfg,data,tmp_path,family)
    pretrain(bg);pretrain(bg,True)
    specs={}
    for mode in ('none','aux_true','aux_random','projection_true','projection_random'):
        current={**student,'module_mode':mode,'arm':family+'_b2_'+mode,'output_dir':str(tmp_path/mode)}
        worker(current);worker(current,True);specs[current['arm']]=current
        model,norm=load_checkpoint(Path(current['output_dir'])/'endpoint.pt',data)
        values,_=predict(model,tensors(data,norm),data['splits']['val'],'cpu',2)
        with np.load(Path(current['output_dir'])/'predictions.npz') as f:
            np.testing.assert_allclose(values*norm['scale'],f['outer'],atol=1e-6)
    assets=knowledge_fixture(data,tmp_path/'diagnostic_knowledge.npz')
    diagnostics(data,assets,student['partition'],tmp_path/'diagnostics')
    report(data,specs,tmp_path)
    assert (tmp_path/'target_sensitivity.csv').exists()
    assert not json.loads((tmp_path/'evaluation_scope.json').read_text())['test_evaluated']
    oracle=pd.read_csv(tmp_path/'diagnostics/basis_oracle.csv')
    assert np.isfinite(oracle.oracle_projection_mse).all()


def test_pretraining_exact_resume_and_heldout_invariance(setup,tmp_path,monkeypatch):
    import vcell.background_training as engine
    from vcell.data import save_prepared
    cfg,data=setup;bg,student=fixtures(cfg,data,tmp_path)
    reference={**bg,'output_dir':str(tmp_path/'reference')};pretrain(reference)
    save=engine.atomic_torch_save
    def interrupt(obj,path):
        save(obj,path)
        if Path(path).name=='last.pt' and obj['step']==1:raise InterruptedError('test interruption')
    monkeypatch.setattr(engine,'atomic_torch_save',interrupt)
    with pytest.raises(InterruptedError):pretrain(bg)
    monkeypatch.setattr(engine,'atomic_torch_save',save);pretrain(bg,True)
    a=torch.load(tmp_path/'reference/encoder.pt',weights_only=True);b=torch.load(tmp_path/'pretrain/encoder.pt',weights_only=True)
    for k in a['state']:torch.testing.assert_close(a['state'][k],b['state'][k],rtol=0,atol=0)
    changed=copy.deepcopy(data);sealed=np.concatenate([data['splits']['val'],data['splits']['test']])
    changed['delta'][sealed]+=1000;changed['baseline'][sealed]+=1000
    path=tmp_path/'changed_data'
    save_prepared(path,changed['baseline'],changed['delta'],changed['pert_idx'],changed['genes'],changed['perturbations'],changed['meta'],changed['audit'])
    pretrain({**bg,'data_dir':str(path),'output_dir':str(tmp_path/'changed_run')})
    c=torch.load(tmp_path/'changed_run/encoder.pt',weights_only=True)
    for k in b['state']:torch.testing.assert_close(b['state'][k],c['state'][k],rtol=0,atol=0)
    assert len(matched_controls(data,student['partition']))<=len(student['partition']['fit'])
    with pytest.raises(ValueError,match='identical'):pretrain({**bg,'pretrain_steps':3},True)


def test_source_and_donor_leakage_guards(tmp_path):
    genes=['a','b','c'];source=background_fixture(tmp_path/'source',genes)
    prepare_background(source,tmp_path/'cache',genes)
    with pytest.raises(ValueError,match='panel'):prepare_background(source,tmp_path/'other',genes[::-1])
    val=source/'ts_blood/val.h5ad';a=ad.read_h5ad(val);a.obs['donor_group']='tabula:train';a.write_h5ad(val)
    done=json.loads((source/'ts_blood/COMPLETE.json').read_text())
    with pytest.raises(ValueError,match='checksum'):prepare_background(source,tmp_path/'other',genes)
    done['shards'][1]['sha256']=file_sha256(val);write_json(source/'ts_blood/COMPLETE.json',done)
    with pytest.raises(ValueError,match='donor leakage'):prepare_background(source,tmp_path/'other',genes)


def test_b1_does_not_fit_public_values_and_student_heldout_isolation(setup,tmp_path):
    from vcell.data import save_prepared
    cfg,data=setup;bg,student=fixtures(cfg,data,tmp_path,background='b1')
    pretrain(bg);worker(student)
    # Changing public training values changes diagnostics, not B1 fitted weights.
    public=np.load(Path(bg['background_cache'])/'train.npy');public[:,:-1]+=10
    np.save(Path(bg['background_cache'])/'train.npy',public)
    altered={**bg,'output_dir':str(tmp_path/'public_changed')};pretrain(altered)
    a=torch.load(tmp_path/'pretrain/encoder.pt',weights_only=True)
    b=torch.load(tmp_path/'public_changed/encoder.pt',weights_only=True)
    for k in a['state']:torch.testing.assert_close(a['state'][k],b['state'][k],rtol=0,atol=0)
    changed=copy.deepcopy(data)
    held=np.concatenate([data['splits']['val'],data['splits']['test']])
    changed['delta'][held]+=100
    path=tmp_path/'changed'
    save_prepared(path,changed['baseline'],changed['delta'],changed['pert_idx'],changed['genes'],changed['perturbations'],changed['meta'],changed['audit'])
    worker({**student,'data_dir':str(path),'output_dir':str(tmp_path/'changed_student')})
    a=torch.load(tmp_path/'student/endpoint.pt',weights_only=True)
    b=torch.load(tmp_path/'changed_student/endpoint.pt',weights_only=True)
    for k in a['state']:torch.testing.assert_close(a['state'][k],b['state'][k],rtol=0,atol=0)
    ca=json.loads((tmp_path/'student/calibration.json').read_text());cb=json.loads((tmp_path/'changed_student/calibration.json').read_text())
    assert ca['mix']==cb['mix'] and ca['alpha']==cb['alpha']


def test_launcher_end_to_end_and_resume(setup,tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=importlib.import_module('run_round9');cfg,data=setup
    data=expand_batches(data,tmp_path/'batches')
    knowledge_fixture(data,tmp_path/'knowledge.npz');source=background_fixture(tmp_path/'source',data['genes'])
    write_json(tmp_path/'student.json',cfg);work=tmp_path/'work';work.mkdir()
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda _:SimpleNamespace(free=1000*1024**3))
    args=['--work-dir',str(work),'--data',str(tmp_path/'batches'),'--background-dir',str(source),
        '--knowledge-npz',str(tmp_path/'knowledge.npz'),'--model-dir',cfg['model']['model_id'],
        '--student-config',str(tmp_path/'student.json'),'--device','cpu','--max-steps','1','--pretrain-steps','1',
        '--pretrain-batch','4','--batch-size','2','--save-every','1','--protocols','context','--seeds','17',
        '--arms','mlp_b0_none','qwen_b2_projection_true']
    root=runner.main(args)
    assert json.loads((root/'COMPLETE.json').read_text())['students']==2
    runner.main(args+['--resume'])
    import tarfile
    with tarfile.open(root/'round9_review_light.tar.gz') as t:
        assert 'comparison.csv' in t.getnames()
        assert not any(n.endswith(('.pt','.npz','.npy','.h5ad')) for n in t.getnames())
    with pytest.raises(ValueError,match='identical'):runner.main(args+['--resume','--max-steps','2'])
