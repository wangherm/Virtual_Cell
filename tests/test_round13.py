"""Round13 numerical graph checks, causal split boundaries and exact recovery."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest
import requests
from scipy import sparse
import torch

from test_round12 import worker_setup
from vcell.round12 import worker as round12_worker, save_complete
from vcell.round13 import (ARMS, FactorizedResponse, episode_mask, identity_indices,
                          fit_factor, retrieval, representation, worker, report)
from vcell.string_diffusion import diffuse, map_targets, read_graph, download_file
from vcell.round11 import verify_files
from vcell.utils import write_json, file_sha256


def test_diffusion_matches_dense_ppr_and_mapping():
    graph=sparse.csr_matrix([[0.,2,0,0],[2,0,1,0],[0,1,0,0],[0,0,0,0]])
    values,random,audit=diffuse(graph,dim=3,tol=1e-9)
    degree=np.asarray(graph.sum(1)).ravel();t=graph.toarray()/np.maximum(degree[:,None],1e-30)
    expected=np.linalg.solve(np.eye(4)-.85*t,.15*random)
    np.testing.assert_allclose(values,expected,atol=1e-7);assert not values[3].any()
    info=pd.DataFrame({'preferred_name':['A','B','B','D']})
    coverage=pd.DataFrame({'target':['A','alias','D','missing'],'canonical_symbol':['A','B','D',None]})
    mapped,_,audit=map_targets(coverage.target.tolist(),coverage,info,graph,values,random)
    np.testing.assert_allclose(mapped[1],values[[1,2]].mean(0));assert not mapped[2:].any()
    assert audit.available.tolist()==[True,True,False,False]
    with pytest.raises(ValueError,match='converge'):diffuse(graph,max_iter=1,tol=1e-20)


def test_directed_pairs_not_double_counted(tmp_path):
    info=pd.DataFrame({'#string_protein_id':['9606.a','9606.b','9606.c'],'preferred_name':['A','B','C']})
    info.to_csv(tmp_path/'info.gz',sep='\t',index=False,compression='gzip')
    links=pd.DataFrame({'protein1':['9606.a','9606.b','9606.b'],'protein2':['9606.b','9606.a','9606.c'],'combined_score':[800,800,600]})
    links.to_csv(tmp_path/'links.gz',sep=' ',index=False,compression='gzip')
    _,graph=read_graph(tmp_path/'info.gz',tmp_path/'links.gz')
    assert graph.nnz==2;assert graph[0,1]==pytest.approx(.8)


def test_download_resumes_partial_and_rejects_bad_range(tmp_path,monkeypatch):
    import vcell.string_diffusion as engine
    content=b'abcdef';record={'name':'sample.gz','url':'https://example.invalid/file','bytes':6,'sha256':hashlib.sha256(content).hexdigest()}
    calls=[]
    class Response:
        def __init__(self,offset):self.status_code=206 if offset else 200;self.headers={'Content-Range':f'bytes {offset}-5/6'};self.offset=offset
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,size):
            if len(calls)==1:
                yield content[:3];raise requests.ConnectionError('interrupted')
            yield content[self.offset:]
    def get(url,headers,**kwargs):
        calls.append(headers);return Response(3 if 'Range' in headers else 0)
    monkeypatch.setattr(engine.requests,'get',get);monkeypatch.setattr(engine.time,'sleep',lambda _:None)
    p=download_file(record,tmp_path);assert p.read_bytes()==content;assert calls[1]['Range']=='bytes=3-'
    assert download_file(record,tmp_path)==p
    p.write_bytes(b'bad')
    with pytest.raises(ValueError,match='checksum'):download_file(record,tmp_path)


def test_target_mask_unseen_and_gradients():
    torch.manual_seed(17);model=FactorizedResponse(4,3,5,2,True,True)
    with torch.no_grad():
        model.target_out.weight.fill_(.1);model.cross_out.weight.fill_(.2);model.ids.weight[1:].fill_(.5)
    target=torch.randn(6,4);bg=torch.randn(6,3);ids=torch.tensor([0,1,1,2,2,3])
    mask=torch.tensor(episode_mask(5,17,10,.5));assert not mask[0]
    assert torch.equal(mask[ids][1],mask[ids][2]);assert torch.equal(mask[ids][3],mask[ids][4])
    masked=model(target,bg,ids,torch.zeros(5,dtype=torch.bool))
    no_ids=model(target,bg,torch.zeros_like(ids))
    torch.testing.assert_close(masked,no_ids)
    model(target,bg,ids,mask).square().mean().backward()
    assert model.ids.weight.grad[0].abs().sum()==0
    assert model.cross_out.weight.grad.abs().sum()>0


def test_factor_resume_and_outer_label_invariance(tmp_path,monkeypatch):
    import vcell.round13 as engine
    rng=np.random.default_rng(2);n=40
    target=rng.normal(size=(n,5)).astype(np.float32);bg=rng.normal(size=(n,3)).astype(np.float32)
    ids=np.arange(n)%5;y=rng.normal(size=(n,2)).astype(np.float32);fit=np.arange(20);cal=np.arange(20,30)
    cfg={'seed':17,'device':'cpu','max_steps':6,'batch_size':8,'save_every':2,
         'fit_weights':(np.ones(20)/20).tolist(),'cal_weights':(np.ones(10)/10).tolist()}
    for name in ('full','interrupted','poison'):(tmp_path/name).mkdir()
    arm='diffusion_factor_id_dropout'
    expected=fit_factor(target,bg,ids,y,fit,cal,cfg,arm,tmp_path/'full','stamp')
    real=engine.atomic_torch_save
    def interrupt(value,path):
        real(value,path)
        if Path(path).name=='last.pt' and value['step']==2:raise InterruptedError('test interruption')
    monkeypatch.setattr(engine,'atomic_torch_save',interrupt)
    with pytest.raises(InterruptedError):fit_factor(target,bg,ids,y,fit,cal,cfg,arm,tmp_path/'interrupted','stamp')
    monkeypatch.setattr(engine,'atomic_torch_save',real)
    actual=fit_factor(target,bg,ids,y,fit,cal,cfg,arm,tmp_path/'interrupted','stamp');np.testing.assert_array_equal(expected,actual)
    saved=torch.load(tmp_path/'full/model.pt',weights_only=True)
    restored=FactorizedResponse(saved['target_dim'],saved['background_dim'],saved['n_ids'],saved['rank'],saved['interaction'],saved['use_id'])
    restored.load_state_dict(saved['state'])
    with torch.no_grad():reloaded=restored(torch.tensor(target),torch.tensor(bg),torch.tensor(ids)).numpy()
    np.testing.assert_array_equal(expected,reloaded)
    y[30:]=10000
    poisoned=fit_factor(target,bg,ids,y,fit,cal,cfg,arm,tmp_path/'poison','stamp');np.testing.assert_array_equal(expected,poisoned)


def test_tie_aware_retrieval_top5():
    meta=pd.DataFrame({'context':['A']*10,'perturbation':[f'P{i}' for i in range(10)]})
    records=retrieval(meta,np.eye(10),np.zeros((10,10)),'zero')
    assert all(r['midrank']==5.5 and r['top1_credit']==.1 and r['top5_credit']==.5 for r in records)
    true=retrieval(meta,np.eye(10),np.eye(10),'oracle')
    assert all(r['midrank']==1 and r['top1_credit']==1 and r['top5_credit']==1 for r in true)


def parent_fixture(tmp_path):
    data,pc,_,_=worker_setup(tmp_path)
    pc.update(variant='expanded',representations=['none','id','string'],predictors=['ridge'])
    round12_worker(pc)
    diffusion=tmp_path/'diffusion';diffusion.mkdir()
    rng=np.random.default_rng(81);n=len(data['perturbations'])
    np.savez_compressed(diffusion/'embeddings.npz',targets=data['perturbations'],diffusion=rng.normal(size=(n,10)).astype(np.float32),
                        undiffused_random=rng.normal(size=(n,10)).astype(np.float32),available=np.ones(n,bool))
    write_json(diffusion/'audit.json',{'fixture':True});write_json(diffusion/'plan.json',{'fixture':True})
    pd.DataFrame({'target':data['perturbations'],'available':True}).to_csv(diffusion/'mapping.csv',index=False)
    save_complete(diffusion)
    cfg={'parent_worker':pc['output_dir'],'diffusion':str(diffusion/'embeddings.npz'),'seed':17,'protocol':'target',
         'device':'cpu','arms':list(ARMS),'max_steps':2,'batch_size':8,'save_every':1,'output_dir':str(tmp_path/'round13')}
    return data,pc,cfg


def test_worker_all_arms_reports_and_cache_guard(tmp_path):
    data,pc,cfg=parent_fixture(tmp_path);worker(cfg);verify_files(Path(cfg['output_dir']));worker(cfg,True)
    ids,_=identity_indices(data,pc['partition']['fit']);assert not ids[pc['partition']['outer']].any()
    reports=tmp_path/'reports';reports.mkdir();report({'one':cfg},reports)
    f=pd.read_csv(reports/'comparison_mean.csv');assert len(f)==len(ARMS)+5
    assert np.isfinite(f.mse_mean).all();assert set(pd.read_csv(reports/'oracle_sweep.csv').method)=={'oracle_rank32','oracle_rank64'}
    # Reused baseline predictions are bit-identical to Round12's published scores.
    r12=pd.read_csv(Path(pc['output_dir'])/'per_target.csv');r13=pd.read_csv(reports/'per_target.csv')
    np.testing.assert_allclose(r12[r12.method=='string_ridge'].mse,r13[r13.method=='r12_string_ridge'].mse)
    (Path(cfg['output_dir'])/'diffusion_factor/model.pt').write_bytes(b'bad')
    with pytest.raises(ValueError,match='checksum'):worker(cfg,True)


def test_launcher_uses_only_existing_backgrounds(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import run_round13 as launcher
    data,pc,cfg=parent_fixture(tmp_path)
    previous=tmp_path/'parent';parentworker=previous/'benchmarks/target_expanded_seed17';parentworker.parent.mkdir(parents=True)
    # Copy a completed fixture, retaining explicit historical dependency paths.
    import shutil
    shutil.copytree(pc['output_dir'],parentworker)
    write_json(previous/'plan.json',{'experiment':'round12','seeds':[17],'protocols':['target']})
    pd.DataFrame({'target':data['perturbations'],'canonical_symbol':data['perturbations']}).to_csv(previous/'annotation_coverage.csv',index=False)
    save_complete(previous)
    monkeypatch.setattr(launcher,'prepare',lambda *args:Path(cfg['diffusion']))
    def inline(jobs,root,*args,**kwargs):
        results={}
        for name,job in jobs.items():
            command=job['cmd'];current=json.loads(Path(command[command.index('--worker-config')+1]).read_text())
            worker(current,'--resume' in command);results[name]=0
        return results
    monkeypatch.setattr(launcher,'run_jobs',inline)
    work=tmp_path/'work';work.mkdir()
    args=['--work-dir',str(work),'--round12-run',str(previous),'--diffusion-dir',str(Path(cfg['diffusion']).parent),
          '--seeds','17','--protocols','target','--device','cpu','--arms','diffusion_interaction_ridge','diffusion_factor_id_dropout',
          '--max-steps','2','--save-every','1']
    root=launcher.main(args);verify_files(root);launcher.main(args+['--resume'])
    budget=json.loads((root/'budget.json').read_text());assert budget['background_jobs']==0
    with tarfile.open(root/'round13_review_light.tar.gz') as t:
        assert 'comparison_mean.csv' in t.getnames();assert not any(n.endswith(('.pt','.npz','.npy')) for n in t.getnames())
