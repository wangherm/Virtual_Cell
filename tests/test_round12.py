"""Round12 split boundaries, raw expansion, label poisoning and exact resumption."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from test_round9 import background_fixture
from vcell.background_training import pretrain, prepare_background
from vcell.data import save_prepared, load_prepared
from vcell.round12 import (make_partition, assert_extension, target_features, projection, fit_ridge,
                          specificity, worker, mlp_fit, report, half_diagnostic)
from vcell.round12_data import expand, aggregate_halves
from vcell.round11 import verify_files
from vcell.utils import write_json, file_sha256


def dataset(path):
    rng=np.random.default_rng(41);genes=np.array([f'G{i}' for i in range(8)])
    rows=[]
    for context,split,targets in [('K562','train',range(30)),('RPE1','train',range(30)),
                                  ('H1_hESC','train',range(20,40)),('HepG2','val',range(30)),('Jurkat','test',range(30))]:
        for p in targets:
            for b in range(2):
                rows.append(dict(row_id=f'{context}:{p}:{b}',context=context,perturbation=f'P{p}',batch=str(b),
                                 split=split,dataset=context,n_cells=20,n_controls=40))
    meta=pd.DataFrame(rows);vocab=sorted(set(meta.perturbation));n=len(meta)
    baseline=rng.random((n,8)).astype(np.float32);delta=rng.normal(0,.1,(n,8)).astype(np.float32)
    save_prepared(path,baseline,delta,np.array([vocab.index(p) for p in meta.perturbation]),genes,vocab,meta,
                  {'normalization':'mean(log1p(10000*counts/full_library))','target_sum':10000})
    return load_prepared(path)


def raw_sources(work,genes):
    entries=[];sources=[];rng=np.random.default_rng(12)
    for context in ('K562','RPE1'):
        targets=['control']*40+[f'P{i}' for i in range(45) for _ in range(20)]
        obs=pd.DataFrame({'gene':targets,'batch':'a'},index=[f'cell{i}' for i in range(len(targets))])
        x=rng.poisson(4,size=(len(obs),len(genes))).astype(np.float32)
        path=work/'raw'/(context+'.h5ad');path.parent.mkdir(parents=True,exist_ok=True)
        ad.AnnData(x,obs=obs,var=pd.DataFrame(index=genes)).write_h5ad(path)
        entries.append({'id':context,'context':context,'path':str(path),'perturbation_key':'gene',
                        'batch_key':'batch','control_values':['control']})
        sources.append({'id':context,'path':str(path)})
    write_json(work/'prepared/real_min10_01/data_audit.json',{'sources':sources,'manifest':{
        'datasets':entries,'chunk_size':100,'min_cells':10,'min_control_cells':30,'target_sum':10000}})
    return entries


def assets(data,path):
    rng=np.random.default_rng(72);n=len(data['perturbations']);g=len(data['genes'])
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,genes=data['genes'],perturbations=data['perturbations'],
        semantic=rng.normal(size=(n,12)).astype(np.float32),gene_semantic=rng.normal(size=(g,12)).astype(np.float32),
        relations=rng.random((n,3,g)).astype(np.float32),modules=np.ones((2,g),np.float32)/g,module_names=np.array(['a','b']))
    write_json(path.parent/'COMPLETE.json',{'files':{path.name:file_sha256(path)}})
    with np.load(path) as f:return {k:f[k] for k in f.files}


def test_expansion_preserves_old_rows_and_split_boundaries(tmp_path):
    old=dataset(tmp_path/'old');raw_sources(tmp_path,old['genes'])
    new=expand(tmp_path,old,tmp_path/'expanded');assert_extension(old,new)
    again=expand(tmp_path,old,tmp_path/'expanded')
    assert again['audit']['fingerprint']==new['audit']['fingerprint']
    assert len(new['meta'])>len(old['meta'])
    for protocol in ('target','context','double'):
        a=make_partition(old,old,protocol);b=make_partition(old,new,protocol)
        assert old['meta'].iloc[a['outer']].row_id.tolist()==new['meta'].iloc[b['outer']].row_id.tolist()
        fit=set(new['meta'].iloc[b['fit']].perturbation);cal=set(new['meta'].iloc[b['calibration']].perturbation)
        outer=set(new['meta'].iloc[b['outer']].perturbation)
        assert not fit&cal
        if protocol!='context':assert not fit&outer
        else:assert outer<=fit
        assert 'Jurkat' not in set(new['meta'].iloc[sum(b.values(),[])].context)
    half_diagnostic(tmp_path/'expanded',tmp_path/'diag')
    assert (tmp_path/'diag/half_specificity.csv').exists()
    corrupted=copy.deepcopy(new);corrupted['delta'][0,0]+=1
    with pytest.raises(ValueError,match='Historical'):assert_extension(old,corrupted)


def test_raw_strata_and_noninteger_rejected(tmp_path):
    old=dataset(tmp_path/'old');entry=raw_sources(tmp_path,old['genes'])[0]
    a=ad.read_h5ad(entry['path']);a.obs['timepoint']=['a','b']*(len(a)//2);a.write_h5ad(entry['path'])
    with pytest.raises(ValueError,match='varies'):aggregate_halves(entry,old['genes'],{'chunk_size':100})
    del a.obs['timepoint'];a.X[0,0]=.25;a.write_h5ad(entry['path'])
    with pytest.raises(ValueError,match='Non-integer'):aggregate_halves(entry,old['genes'],{'chunk_size':100})


def test_fit_only_projection_unknown_id_and_shuffle(tmp_path):
    data=dataset(tmp_path/'data');k=assets(data,tmp_path/'knowledge/knowledge.npz')
    fit=np.array(make_partition(data,data,'target')['fit'])
    seen=np.unique(data['pert_idx'][fit]);unknown=np.setdiff1d(np.arange(len(data['perturbations'])),seen)
    f,_,_=target_features(data,k,fit,'id',17);assert (f[unknown]==0).all()
    values,t=projection(k['semantic'],seen,4)
    changed=k['semantic'].copy();changed[unknown]+=10000
    values2,t2=projection(changed,seen,4)
    for key in t:np.testing.assert_allclose(t[key],t2[key])
    np.testing.assert_allclose(values[seen],values2[seen])
    original,_,_=target_features(data,k,fit,'string',17)
    shuffled,tr,_=target_features(data,k,fit,'shuffled_string_1',17)
    assert not np.allclose(original,shuffled);assert sorted(tr['permutation'])==list(range(len(f)))


def test_ridge_solution_and_tie_aware_ranking():
    rng=np.random.default_rng(4);x=rng.normal(size=(40,4));y=x@rng.normal(size=(4,3))+2
    coef,bias=fit_ridge(x,y,np.ones(40),1e-8);np.testing.assert_allclose(x@coef+bias,y,atol=1e-6)
    m=pd.DataFrame({'context':['A']*4,'perturbation':['a','b','c','d']})
    results=specificity(m,np.eye(4),np.zeros((4,4)),'zero')
    assert all(r['midrank']==2.5 and r['top1_credit']==.25 for r in results)


def test_exact_mlp_resume(tmp_path,monkeypatch):
    import vcell.round12 as engine
    rng=np.random.default_rng(2);x=rng.normal(size=(30,6)).astype(np.float32);y=rng.normal(size=(30,3)).astype(np.float32)
    fit=np.arange(20);w=np.ones(20)/20;cfg={'seed':17,'device':'cpu','max_steps':4,'batch_size':8,'save_every':1}
    full=tmp_path/'full';part=tmp_path/'part';full.mkdir();part.mkdir()
    expected=mlp_fit(x,y,fit,w,cfg,full,'stamp')
    original=engine.atomic_torch_save
    def interrupt(value,path):
        original(value,path)
        if Path(path).name=='last.pt' and value['step']==2:raise InterruptedError('test interruption')
    monkeypatch.setattr(engine,'atomic_torch_save',interrupt)
    with pytest.raises(InterruptedError):mlp_fit(x,y,fit,w,cfg,part,'stamp')
    monkeypatch.setattr(engine,'atomic_torch_save',original)
    actual=mlp_fit(x,y,fit,w,cfg,part,'stamp');np.testing.assert_array_equal(actual,expected)


def worker_setup(tmp_path):
    data=dataset(tmp_path/'data');knowledge=tmp_path/'knowledge/knowledge.npz';assets(data,knowledge)
    model=tmp_path/'model';model.mkdir();write_json(model/'config.json',{'hidden_size':8})
    spec={'model_id':str(model),'control_tokens':2}
    source=background_fixture(tmp_path/'public',data['genes']);cache=prepare_background(source,tmp_path/'cache',data['genes'])
    part=make_partition(data,data,'target')
    bg={'stage':'background','data_dir':str(tmp_path/'data'),'partition':part,'background_cache':str(cache),
        'background':'b2','family':'qwen','model':spec,'seed':17,'device':'cpu','pretrain_steps':2,
        'pretrain_batch':4,'save_every':1,'output_dir':str(tmp_path/'bg')}
    pretrain(bg)
    cfg={'data_dir':str(tmp_path/'data'),'knowledge':str(knowledge),'encoder':str(tmp_path/'bg/encoder.pt'),
         'partition':part,'model':spec,'seed':17,'device':'cpu','rank':4,'background_dim':2,'target_dim':3,
         'representations':['none','id','random','text','string','shuffled_string_1'],
         'predictors':['ridge','mlp'],'ridge_alphas':[.01,.1],'max_steps':2,'batch_size':8,'save_every':1,
         'protocol':'target','variant':'original','output_dir':str(tmp_path/'heads')}
    return data,cfg,bg,source


def test_worker_all_heads_resume_and_report(tmp_path):
    data,cfg,_,_=worker_setup(tmp_path)
    worker(cfg);verify_files(Path(cfg['output_dir']));worker(cfg,True)
    report({'job':cfg},tmp_path)
    scores=pd.read_csv(tmp_path/'comparison.csv');assert len(scores)==15
    assert np.isfinite(scores.mse).all()
    with np.load(Path(cfg['output_dir'])/'text_ridge/model.npz') as f:before=f['coef'].copy()
    # Outer labels may affect reported scores/oracle, never fitted feature transforms,
    # response basis, ridge model or ridge hyperparameter selection.
    data['delta'][cfg['partition']['outer']]+=100
    save_prepared(tmp_path/'poison',data['baseline'],data['delta'],data['pert_idx'],data['genes'],
                  data['perturbations'],data['meta'],data['audit'].copy())
    # Refit background under the new data fingerprint to enforce cache provenance.
    altered={**cfg,'data_dir':str(tmp_path/'poison'),'output_dir':str(tmp_path/'poison_heads'),
             'encoder':str(tmp_path/'poison_bg/encoder.pt')}
    bg=json.loads((tmp_path/'bg/manifest.json').read_text())['config']
    pretrain({**bg,'data_dir':str(tmp_path/'poison'),'output_dir':str(tmp_path/'poison_bg')})
    worker(altered)
    with np.load(Path(altered['output_dir'])/'text_ridge/model.npz') as f:np.testing.assert_allclose(before,f['coef'],atol=1e-10)
    (Path(cfg['output_dir'])/'text_ridge/per_target.csv').write_text('corrupt')
    with pytest.raises(ValueError,match='checksum'):worker(cfg,True)


def test_launcher_whole_flow_and_light_package(tmp_path,monkeypatch):
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'scripts'))
    import run_round12 as launcher
    data,cfg,bg,source=worker_setup(tmp_path)
    work=tmp_path/'work';work.mkdir();raw_sources(work,data['genes'])
    expanded_path=tmp_path/'expanded';new=expand(work,data,expanded_path)
    k=tmp_path/'expanded_knowledge/knowledge.npz';assets(new,k)
    previous=work/'runs/round10_01';previous.mkdir(parents=True)
    write_json(previous/'plan.json',{'experiment':'round10','data_path':cfg['data_dir'],'data':data['audit']['fingerprint'],
                'model':cfg['model'],'background_path':str(source),'pretrain_batch':4})
    write_json(previous/'COMPLETE.json',{'complete':True});write_json(previous/'budget.json',{'background_optimizer_steps':2})
    def inline(jobs,root,*args,**kwargs):
        result={}
        for name,value in jobs.items():
            command=value['cmd'];conf=json.loads(Path(command[command.index('--worker-config')+1]).read_text())
            if conf.get('stage')=='background':pretrain(conf,'--resume' in command)
            else:worker(conf,'--resume' in command)
            result[name]=0
        return result
    monkeypatch.setattr(launcher,'run_jobs',inline)
    args=['--work-dir',str(work),'--expanded-data',str(expanded_path),'--knowledge-npz',str(k),
          '--device','cpu','--seeds','17','--protocols','target','context','double','--representations','none','string',
          '--predictors','ridge','--max-steps','2','--save-every','1']
    root=launcher.main(args);verify_files(root)
    launcher.main(args+['--resume'])
    assert (root/'COMPLETE.json').exists();assert (root/'report.html').exists()
    with tarfile.open(root/'round12_review_light.tar.gz') as f:
        names=f.getnames();assert 'comparison_mean.csv' in names
        assert not any(n.endswith(('.pt','.npz','.npy')) for n in names)
    paired=pd.read_csv(root/'paired_comparisons.csv');assert 'expansion' in set(paired.contrast)
    assert not json.loads((root/'evaluation_scope.json').read_text())['jurkat_scored']
