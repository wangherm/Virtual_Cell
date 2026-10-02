"""Offline tests for granularity controls, new holdouts and resumed warmup."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from test_qwen import setup
from test_round9 import background_fixture, fixtures
from test_round8 import knowledge_fixture
from test_round7 import expand_batches
from vcell.background_training import prepare_background, pretrain
from vcell.round8 import worker, load_checkpoint, predict
from vcell.round10 import (prepare_aggregates, load_aggregates, make_partition, partition_audit,
                          warmup_weight, paired_summary, arm_matrix, arm_allowed, report)
from vcell.train import tensors
from vcell.utils import write_json


def test_aggregate_is_mean_log_and_strictly_within_group(tmp_path):
    source=background_fixture(tmp_path/'source',['a','b','c'])
    cache=prepare_background(source,tmp_path/'cache',['a','b','c'])
    meta=pd.read_csv(cache/'train_metadata.csv',keep_default_na=False)
    meta.loc[:3,'donor_group']='tabula:A';meta.loc[4:,'donor_group']='tabula:B'
    meta['sample_id']=['sample1']*2+['sample2']*2+['sample3']*4
    meta.to_csv(cache/'train_metadata.csv',index=False)
    folder=prepare_aggregates(cache,tmp_path/'aggregate');values,index=load_aggregates(folder,cache)
    x=np.load(cache/'train.npy')
    assert len(values)==3 and len(index)==len(x)
    for g in range(len(values)):
        rows=np.flatnonzero(index==g)
        np.testing.assert_allclose(values[g],x[rows].mean(0),rtol=1e-6)
        assert meta.iloc[rows].donor_group.nunique()==1 and meta.iloc[rows].sample_id.nunique()==1
    assert np.all(values[:,-1]==0)
    audit=json.loads((folder/'audit.json').read_text());assert audit['input_cells']==8 and audit['groups']==3
    prepare_aggregates(cache,folder)
    with (folder/'means.npy').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):load_aggregates(folder,cache)
    meta.loc[0,'split']='val';meta.to_csv(cache/'train_metadata.csv',index=False)
    with pytest.raises(ValueError,match='train cells'):prepare_aggregates(cache,tmp_path/'bad')


def test_new_holdouts_no_held_context_in_fit_or_calibration(tmp_path):
    meta=pd.DataFrame([{'context':c,'perturbation':f'P{p}','batch':f'b{b}','row_id':f'{c}_{p}_{b}'}
        for c in ('K562','RPE1','H1_hESC','HepG2','Jurkat') for p in range(20) for b in range(5)])
    data={'meta':meta,'splits':{'train':np.flatnonzero(meta.context.isin(['K562','RPE1','H1_hESC'])),
        'val':np.flatnonzero(meta.context=='HepG2'),'test':np.flatnonzero(meta.context=='Jurkat')}}
    for protocol,held in [('context_rpe1','RPE1'),('context_h1','H1_hESC')]:
        parts=make_partition(data,protocol)
        assert set(meta.iloc[parts['outer']].context)=={held}
        assert held not in set(meta.iloc[parts['fit']+parts['calibration']].context)
        assert not set(sum(parts.values(),[]))&set(data['splits']['test'])
        assert not set(sum(parts.values(),[]))&set(data['splits']['val'])
        partition_audit(data,parts,tmp_path/protocol)
        assert (tmp_path/protocol/'coverage.csv').exists()
    assert sum(arm_allowed(p,m) for p in ('context','target','context_rpe1','context_h1') for _,_,m in arm_matrix())*3==54


def test_primary_metric_not_flat_target_mean_and_warmup():
    # Unequal context coverage: primary improvement is (2 + 10)/2, not 14/3.
    units=pd.DataFrame({'context':['A','A','B']*3,'target':['t1','t2','t1']*3,
                        'seed':np.repeat([17,29,43],3),'gain':[1.,3.,10.]*3})
    summary=paired_summary(units,n_boot=25)
    assert summary['primary_mse_improvement']==6 and summary['bootstrap_replicates']==25
    assert summary['largest_contribution_target']=='t1'
    assert summary['primary_gain_without_largest'] is None  # Removing it empties context B.
    assert [warmup_weight(s,500,.1) for s in (0,250,500,1000)]==[0,.05,.1,.1]


def test_b3_transfer_and_exact_warmup_resume(setup,tmp_path,monkeypatch):
    import vcell.round8 as engine
    cfg,data=setup;bg,student=fixtures(cfg,data,tmp_path,family='qwen',background='b3')
    aggregate=prepare_aggregates(bg['background_cache'],tmp_path/'aggregates');bg['aggregate_dir']=str(aggregate)
    pretrain(bg);pretrain(bg,True)
    student.update(experiment='round10',module_warmup_steps=1,arm='qwen_b3_aux_true_warmup')
    reference={**student,'output_dir':str(tmp_path/'reference')};worker(reference)
    save=engine.atomic_torch_save
    def interrupt(obj,path):
        save(obj,path)
        if Path(path).name=='last.pt' and obj['step']==1:raise InterruptedError('test interruption')
    monkeypatch.setattr(engine,'atomic_torch_save',interrupt)
    with pytest.raises(InterruptedError):worker(student)
    monkeypatch.setattr(engine,'atomic_torch_save',save);worker(student,True)
    a=torch.load(tmp_path/'reference/endpoint.pt',weights_only=True);b=torch.load(tmp_path/'student/endpoint.pt',weights_only=True)
    for k in a['state']:torch.testing.assert_close(a['state'][k],b['state'][k],rtol=0,atol=0)
    history=pd.read_csv(tmp_path/'student/history.csv');assert history.module_weight.tolist()==[0,.1]
    model,norm=load_checkpoint(tmp_path/'student/endpoint.pt',data)
    values,_=predict(model,tensors(data,norm),data['splits']['val'],'cpu',2)
    with np.load(tmp_path/'student/predictions.npz') as f:np.testing.assert_allclose(values*norm['scale'],f['outer'],atol=1e-6)


def test_round10_launcher_and_light_review(setup,tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=importlib.import_module('run_round9');cfg,data=setup
    data=expand_batches(data,tmp_path/'batches')
    knowledge_fixture(data,tmp_path/'knowledge.npz');source=background_fixture(tmp_path/'source',data['genes'])
    write_json(tmp_path/'student.json',cfg);work=tmp_path/'work';work.mkdir()
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda _:SimpleNamespace(free=1000*1024**3))
    args=['--work-dir',str(work),'--data',str(tmp_path/'batches'),'--background-dir',str(source),
        '--knowledge-npz',str(tmp_path/'knowledge.npz'),'--model-dir',cfg['model']['model_id'],
        '--student-config',str(tmp_path/'student.json'),'--device','cpu','--max-steps','2','--pretrain-steps','1',
        '--pretrain-batch','4','--batch-size','2','--save-every','1','--protocols','context','--seeds','17',
        '--arms','qwen_b2_none','qwen_b3_none','qwen_b2_aux_true_warmup','qwen_b2_aux_random_warmup',
        '--module-warmup-steps','1']
    root=runner.main(args,experiment='round10')
    assert root.name=='round10_01'
    assert json.loads((root/'COMPLETE.json').read_text())['students']==4
    assert not (root/'diagnostics').exists()  # Do not rerun Round9 probes/oracles.
    runner.main(args+['--resume'],experiment='round10')
    pairs=pd.read_csv(root/'paired_diagnostics.csv');assert len(pairs)>0
    assert (root/'target_coverage_scores.csv').exists()
    import tarfile
    with tarfile.open(root/'round10_review_light.tar.gz') as t:
        assert 'background_aggregates/audit.json' in t.getnames()
        assert not any(n.endswith(('.pt','.npz','.npy','.h5ad')) for n in t.getnames())
    with pytest.raises(ValueError,match='identical'):
        runner.main(args+['--resume','--module-warmup-steps','2'],experiment='round10')
