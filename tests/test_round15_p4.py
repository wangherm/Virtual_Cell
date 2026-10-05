"""Real CPU training, canonical holdouts, deterministic resume and report integrity."""
import copy
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest
import torch

from test_round12 import assets, dataset, worker_setup
from test_round15_adapter import fake_aggregation
from test_round9 import background_fixture
from vcell.data import load_prepared
from vcell.round12 import make_partition
from vcell.round15 import reconstruct, seal
from vcell.round15_adapter import POLICY, prepare_extension
from vcell.round15_p4 import (assert_mixscale_extension, harmonize, load_relations,
                             prepare_relations, target_aliases, worker)
from vcell.utils import write_json

REPO=Path(__file__).resolve().parents[1]


def annotated_parent(data, folder):
    assets(data,folder/'knowledge.npz')
    cards={str(label):{'status':'mapped','symbol':str(label),'entrezgene':100+i}
           for i,label in enumerate(data['perturbations'])}
    cards.update({str(g):{'status':'mapped','symbol':str(g),'entrezgene':10000+i}
                  for i,g in enumerate(data['genes'])})
    write_json(folder/'cards.json',cards);seal(folder)
    return cards


def adapter_fixture(tmp_path):
    old=dataset(tmp_path/'old');adapter=tmp_path/'adapter'
    fake_aggregation(old,tmp_path/'agg')
    prepare_extension(old,old,tmp_path/'agg',adapter/'prepared',POLICY)
    write_json(adapter/'training_handoff.json',{'reference_data':str(tmp_path/'old'),
        'original_data':str(tmp_path/'old'),'data_path':str(adapter/'prepared'),
        'training_data_ready':True,'test_evaluated':False})
    write_json(adapter/'ADAPTER_STATUS.json',{'state':'ADAPTER_READY'})
    seal(adapter)
    return old,load_prepared(adapter/'prepared'),adapter


def fake_requests(folder,url,payload):
    if 'mygene' in url:
        return [{'query':label,'_id':'99999','taxid':9606,'symbol':label,'entrezgene':99999}
                for label in payload['q'].split(',')]
    return [{'preferredName_A':'NEW','preferredName_B':'G0','score':.9}]


def test_incremental_relations_preserve_all_old_vectors_and_cache_resume(tmp_path,monkeypatch):
    import vcell.round15_p4 as engine
    old,expanded,_=adapter_fixture(tmp_path);parent=tmp_path/'parent'
    annotated_parent(old,parent);calls=[]
    def query(folder,url,payload):
        calls.append((url,payload));return fake_requests(folder,url,payload)
    monkeypatch.setattr(engine,'cached_request',query)
    out=tmp_path/'relations';assert prepare_relations(old,expanded,parent,out)=={}
    assert len(calls)==2 and calls[0][1]['q']=='NEW'
    a=load_relations(out/'relations.npz',old)
    with np.load(parent/'knowledge.npz') as f:np.testing.assert_array_equal(a['relations'],f['relations'])
    values=load_relations(out/'relations.npz',expanded)
    ix=list(expanded['perturbations']).index('NEW')
    np.testing.assert_array_equal(values['relations'][ix,2],np.r_[1.,np.zeros(7)])
    assert prepare_relations(old,expanded,parent,out)=={} and len(calls)==2
    # Numerical corruption is rejected on completed-cache reuse.
    (out/'relations.npz').write_bytes(b'bad')
    with pytest.raises(ValueError,match='checksum'):prepare_relations(old,expanded,parent,out)


def test_new_alias_joins_only_added_rows_and_cannot_evade_target_holdout(tmp_path):
    old,expanded,_=adapter_fixture(tmp_path);cards=annotated_parent(old,tmp_path/'parent')
    parts=make_partition(old,old,'target')
    held=str(old['meta'].iloc[parts['outer'][0]].perturbation)
    cards['NEW']={**cards[held],'symbol':'different_current_name'}
    aliases,audit=target_aliases(old,expanded,cards)
    assert aliases=={'NEW':held}
    dest=harmonize(old,expanded,aliases,tmp_path/'joined');joined=load_prepared(dest)
    assert_mixscale_extension(old,joined,allow_target_aliases=True)
    np.testing.assert_array_equal(joined['delta'],expanded['delta'])
    assert joined['meta'].perturbation.iloc[:len(old['meta'])].equals(old['meta'].perturbation)
    parts=make_partition(old,joined,'target')
    assert held not in set(joined['meta'].iloc[parts['fit']].perturbation)
    assert harmonize(old,expanded,aliases,tmp_path/'joined')==dest
    cards['P1']={**cards['P0']}
    with pytest.raises(ValueError,match='Historical target alias collision'):target_aliases(old,expanded,cards)


def test_missing_added_identity_and_modified_history_fail(tmp_path):
    old,expanded,_=adapter_fixture(tmp_path);cards=annotated_parent(old,tmp_path/'parent')
    cards['NEW']={'status':'ambiguous'}
    with pytest.raises(ValueError,match='missing/ambiguous'):target_aliases(old,expanded,cards)
    bad=copy.deepcopy(expanded);bad['delta'][0,0]+=1
    with pytest.raises(ValueError,match='Historical expression'):assert_mixscale_extension(old,bad)
    bad=copy.deepcopy(expanded);bad['meta'].loc[len(old['meta']),'context']='HT29_IFNG'
    with pytest.raises(ValueError,match='audited development'):assert_mixscale_extension(old,bad)


def p4_worker_setup(tmp_path):
    _,cfg,_,_=worker_setup(tmp_path)
    with np.load(cfg['knowledge']) as f:
        path=tmp_path/'relations.npz'
        np.savez_compressed(path,genes=f['genes'],perturbations=f['perturbations'],relations=f['relations'])
    cfg.update(relations=str(path),methods=['string_ridge','factor_rank16','factor_rank32'],
               output_dir=str(tmp_path/'worker'),max_steps=2,save_every=1,batch_size=8,
               ridge_alphas=[.01,.1],variant='reference')
    return cfg


def test_real_worker_exact_head_resume_and_completed_checksum(tmp_path,monkeypatch):
    import vcell.round15 as trainer
    cfg=p4_worker_setup(tmp_path)
    expected=worker(cfg)
    interrupted=copy.deepcopy(cfg);interrupted['output_dir']=str(tmp_path/'interrupted')
    save=trainer.atomic_torch_save
    def stop(value,path):
        save(value,path)
        if Path(path).name=='last.pt' and value['step']==1:raise InterruptedError('controlled interruption')
    monkeypatch.setattr(trainer,'atomic_torch_save',stop)
    with pytest.raises(InterruptedError):worker(interrupted)
    monkeypatch.setattr(trainer,'atomic_torch_save',save)
    actual=worker(interrupted,True)
    for method in cfg['methods']:
        np.testing.assert_array_equal(reconstruct(actual/method/'coefficients.npz'),reconstruct(expected/method/'coefficients.npz'))
    before=(actual/'COMPLETE.json').read_bytes();worker(interrupted,True)
    assert before==(actual/'COMPLETE.json').read_bytes()
    (actual/'factor_rank16/model.pt').write_bytes(b'corruption')
    with pytest.raises(ValueError,match='checksum'):worker(interrupted,True)


def test_outer_label_poisoning_does_not_change_fitted_heads(tmp_path):
    from vcell.background_training import pretrain
    from vcell.data import save_prepared
    cfg=p4_worker_setup(tmp_path);first=worker(cfg)
    data=load_prepared(cfg['data_dir']);changed=data['delta'].copy()
    changed[cfg['partition']['outer']]+=10000
    poison=tmp_path/'poison'
    save_prepared(poison,data['baseline'],changed,data['pert_idx'],data['genes'],data['perturbations'],data['meta'],dict(data['audit']))
    bg=json.loads((Path(cfg['encoder']).parent/'manifest.json').read_text())['config']
    bg.update(data_dir=str(poison),output_dir=str(tmp_path/'poison_bg'));pretrain(bg)
    cfg2={**cfg,'data_dir':str(poison),'encoder':str(tmp_path/'poison_bg/encoder.pt'),'output_dir':str(tmp_path/'poison_worker')}
    second=worker(cfg2)
    for method in cfg['methods']:
        np.testing.assert_array_equal(reconstruct(first/method/'coefficients.npz'),reconstruct(second/method/'coefficients.npz'))
        assert json.loads((first/method/'selection.json').read_text())==json.loads((second/method/'selection.json').read_text())


def test_full_launcher_cpu_training_matching_reports_and_resume(tmp_path,monkeypatch):
    sys.path.insert(0,str(REPO/'scripts'))
    import run_round15_p4 as launcher
    import vcell.round15_p4 as engine
    old,expanded,adapter=adapter_fixture(tmp_path)
    annotated_parent(old,tmp_path/'knowledge/round12')
    public=background_fixture(tmp_path/'public',old['genes'])
    model=tmp_path/'model';model.mkdir();write_json(model/'config.json',{'hidden_size':8})
    previous=tmp_path/'runs/round10_01'
    write_json(previous/'plan.json',{'experiment':'round10','data':old['audit']['fingerprint'],
        'model':{'model_id':str(model),'control_tokens':2},'background_path':str(public),'pretrain_batch':4})
    write_json(previous/'budget.json',{'background_optimizer_steps':2})
    write_json(previous/'COMPLETE.json',{'test_evaluated':False,'students':1})
    monkeypatch.setattr(engine,'cached_request',fake_requests)
    monkeypatch.setattr(launcher,'hardware',lambda _: {'fixture':True})
    interrupt_queue={'once':True}
    def dispatch(jobs,root,slots,resume,label):
        result={}
        for name,job in jobs.items():
            # Complete one head worker, then simulate a separate worker failure.
            # Backgrounds and the completed worker must be reused with checksum
            # validation on orchestration resume, not retrained or overwritten.
            if interrupt_queue['once'] and not name.startswith('background_') and result:
                interrupt_queue['once']=False;result[name]=1;return result
            launcher.main(['--worker-config',job['cmd'][job['cmd'].index('--worker-config')+1]]+
                          (['--resume'] if '--resume' in job['cmd'] else []))
            result[name]=0
        return result
    monkeypatch.setattr(launcher,'run_jobs',dispatch)
    args=['--work-dir',str(tmp_path),'--adapter-dir',str(adapter),'--device','cpu',
          '--protocols','target','--seeds','17','--max-steps','2','--save-every','1','--batch-size','8']
    with pytest.raises(RuntimeError,match='failed'):launcher.main(args)
    root=tmp_path/'runs/round15_p4_01'
    assert not (root/'COMPLETE.json').exists() and (root/'INCOMPLETE.json').exists()
    completed={path:path.read_bytes() for path in (root/'background_pretraining').rglob('COMPLETE.json')}
    completed.update({path:path.read_bytes() for path in (root/'models').rglob('COMPLETE.json')})
    root=launcher.main(args+['--resume'])
    assert all(path.read_bytes()==value for path,value in completed.items())
    assert not (root/'INCOMPLETE.json').exists()
    assert json.loads((root/'budget.json').read_text())['fitted_heads']==6
    summary=pd.read_csv(root/'comparison_mean.csv')
    assert set(summary.variant)=={'reference','expanded'}
    assert set(summary.method)==set(engine.METHODS)|{'zero','mean_transfer'}
    gain=pd.read_csv(root/'paired_data_gain.csv');assert gain.loc[gain.method.eq('zero'),'reference_minus_expanded_mse'].eq(0).all()
    with tarfile.open(root/'round15_p4_review_light.tar.gz') as archive:
        assert 'ARCHIVE_MANIFEST.json' in archive.getnames()
        assert not any(name.endswith(('.npz','.pt','.npy')) for name in archive.getnames())
        assert not any(name.startswith(('background_cache/','harmonized/')) for name in archive.getnames())
        import hashlib
        manifest=json.load(archive.extractfile('ARCHIVE_MANIFEST.json'))
        for name,identity in manifest['files'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==identity['sha256']
    before=(root/'COMPLETE.json').read_bytes();launcher.main(args+['--resume'])
    assert before==(root/'COMPLETE.json').read_bytes()
    with pytest.raises(ValueError,match='Frozen artifact changed'):launcher.main(args+['--resume','--max-steps','3'])
