"""Round15 gradient, leakage, reconstruction, recovery and execution accounting tests."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest
import torch

from test_round14 import fixture
from vcell.round12 import save_complete
from vcell.round14 import ResponseHead, worker as previous_worker
from vcell.round15 import train_head, infer, worker, report, reconstruct, verify_marker, paired_report
from vcell.round15_heads import ARMS, REUSE, VARIANTS, BLOCKED, ExperimentalHead, contrastive_loss
from vcell.round15_references import reference_bank, ridge_reference, target_folds
from vcell.utils import write_json


def synthetic():
    rng=np.random.default_rng(12);n=36;g=8
    meta=pd.DataFrame({'row_id':[f'r{i}' for i in range(n)],'context':np.repeat(['A','B','C'],12),'perturbation':np.tile(np.repeat(['P0','P1','P2','P3'],3),3)})
    return dict(meta=meta,delta=rng.normal(size=(n,g)).astype(np.float32),perturbations=np.array(['P0','P1','P2','P3']),
                pert_idx=np.tile(np.repeat(np.arange(4),3),3),baseline=rng.normal(size=(n,g)).astype(np.float32))


def test_standard_forward_and_amplitude_have_expected_gradients():
    t=torch.randn(8,5);b=torch.randn(8,3);ids=torch.arange(8)%4
    for name in REUSE:
        torch.manual_seed(9);old=ResponseHead(5,3,4,2,ARMS[name])
        torch.manual_seed(9);new=ExperimentalHead(5,3,4,2,ARMS[name],17)
        with torch.no_grad():
            old.target_out.weight.fill_(.14);new.load_state_dict(old.state_dict())
        torch.testing.assert_close(new(t,b,ids),old(t,b,ids),rtol=0,atol=0)
    model=ExperimentalHead(5,3,4,2,ARMS['amplitude_direction'],17)
    (model(t,b,ids)-torch.randn(8,2)).square().mean().backward()
    assert model.amplitude.bias.grad.abs().sum()>0
    assert model.target_out.weight.grad.abs().sum()>0
    assert len(ARMS)==24 and len(VARIANTS)==12 and len(BLOCKED)==4


def test_contrastive_partial_conditions_are_finite_and_do_not_cross_groups():
    pred=torch.randn(5,3,requires_grad=True);truth=torch.tensor([[1.,0,0],[0,1,0],[1,0,0],[0,0,0],[0,0,1]])
    groups=torch.tensor([0,0,0,0,1]);targets=torch.tensor([0,1,0,2,3])
    loss,count=contrastive_loss(pred,truth,groups,targets,.01);loss.backward()
    assert count==3 and torch.isfinite(pred.grad).all()
    assert not pred.grad[3:].any()
    empty,count=contrastive_loss(pred,truth,torch.arange(5),targets,.01)
    assert count==0 and empty.item()==0


@pytest.mark.parametrize('mode',['reference','neighbor','shared'])
def test_reference_bank_excludes_own_labels_and_outer_truth(mode):
    data=synthetic();fit=np.arange(24);profiles=np.ones((4,5))
    before,available,audit=reference_bank(data,fit,profiles,mode)
    for query in audit['queries']:
        rows=set(np.flatnonzero((data['meta'].context==query['context'])&(data['meta'].perturbation==query['target'])))
        for j in query['donor_units']:
            unit=audit['units'][j]
            assert set(unit['rows'])<=set(fit) and not rows.intersection(unit['rows'])
            if mode=='reference':assert unit['context']!=query['context']
            else:assert unit['target']!=query['target']
    data['delta'][24:]=1e8
    np.testing.assert_array_equal(before,reference_bank(data,fit,profiles,mode)[0])
    data['delta'][:3]+=1e6
    np.testing.assert_array_equal(before[:3],reference_bank(data,fit,profiles,mode)[0][:3])


def test_ridge_target_oof_excludes_held_target_and_outer_responses():
    data=synthetic();fit=np.arange(24);query=np.arange(24,36)
    rng=np.random.default_rng(3);assets={'relations':rng.normal(size=(4,3,5)).astype(np.float32)}
    bg=rng.normal(size=(36,2)).astype(np.float32)
    before,audit=ridge_reference(data,assets,bg,fit,query)
    for inner,held in target_folds(data,fit):
        assert not set(data['meta'].iloc[inner].perturbation)&set(data['meta'].iloc[held].perturbation)
    held=target_folds(data,fit)[0][1];data['delta'][held]+=1e5
    after,_=ridge_reference(data,assets,bg,fit,query)
    np.testing.assert_array_equal(before[held],after[held])
    data['delta'][query]+=1e8
    np.testing.assert_array_equal(after,ridge_reference(data,assets,bg,fit,query)[0])


def test_training_exact_resume_outer_invariance_and_checkpoint_reload(tmp_path,monkeypatch):
    import vcell.round15 as engine
    data=synthetic();rng=np.random.default_rng(22);n=36
    t=rng.normal(size=(n,5)).astype(np.float32);b=rng.normal(size=(n,3)).astype(np.float32);ids=data['pert_idx']
    basis=np.eye(8,dtype=np.float32)[:3];ref=np.zeros((n,4),np.float32);offset=np.zeros_like(data['delta'])
    fit=np.arange(24);cal=np.arange(24,30);cfg=dict(seed=17,device='cpu',max_steps=6,batch_size=8,save_every=2)
    for name in ('full','resume','poison'):(tmp_path/name).mkdir()
    def run(name):return train_head(t,b,ids,ref,offset,data,basis,1.,fit,cal,cfg,'local_factor_noid','mse_target_contrastive',tmp_path/name,'stamp')
    expected=run('full');prediction=infer(expected,t,b,ids,ref,'cpu');original=engine.atomic_torch_save
    def interrupt(value,path):
        original(value,path)
        if Path(path).name=='last.pt' and value['step']==2:raise InterruptedError('test')
    monkeypatch.setattr(engine,'atomic_torch_save',interrupt)
    with pytest.raises(InterruptedError):run('resume')
    monkeypatch.setattr(engine,'atomic_torch_save',original)
    np.testing.assert_array_equal(prediction,infer(run('resume'),t,b,ids,ref,'cpu'))
    data['delta'][30:]=1e8
    np.testing.assert_array_equal(prediction,infer(run('poison'),t,b,ids,ref,'cpu'))
    saved=torch.load(tmp_path/'full/model.pt',weights_only=True)
    restored=ExperimentalHead(saved['target_dim'],saved['background_dim'],saved['n_ids'],saved['rank'],saved['spec'],saved['seed'])
    restored.load_state_dict(saved['state'])
    np.testing.assert_array_equal(prediction,infer(restored,t,b,ids,ref,'cpu'))


def test_macro_contribution_uses_context_weights():
    rows=[]
    for ctx,target,gain in [('A','x',1.),('A','y',1.),('A','z',1.),('B','w',.6)]:
        for method,value in [('base',2.),('new',2.-gain)]:
            rows.append(dict(protocol='target',method=method,context=ctx,target=target,mse=value))
    result=paired_report(pd.DataFrame(rows),'mse',[('base','new')]).iloc[0]
    assert result.largest_gain_target=='w'  # .6/2 > 1/(3*2)
    assert result.gain==pytest.approx(.8)
    assert np.isnan(result.gain_without_largest_target)  # Removing w empties context B.


def test_all_available_heads_losses_reports_and_launcher_resume(tmp_path,monkeypatch):
    import vcell.round15 as engine
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import run_round15 as launcher
    data,pc,_,cfg=fixture(tmp_path);cfg['arms']=list(REUSE.values());previous_worker(cfg)
    previous=tmp_path/'parent14';parent=previous/'models/target_seed17';parent.parent.mkdir(parents=True)
    shutil.copytree(cfg['output_dir'],parent)
    write_json(previous/'plan.json',{'experiment':'round14','seeds':[17],'protocols':['target']});save_complete(previous)
    # This fixture's historical round was generated with today's source tree. Production
    # compares the historical hash of all pre-Round15 modules, never this test override.
    historical=json.loads((parent/'manifest.json').read_text())['source']
    monkeypatch.setattr(engine,'historical_source',lambda:historical)
    def inline(jobs,root,*args,**kwargs):
        result={}
        for name,job in jobs.items():
            command=job['cmd'];current=json.loads(Path(command[command.index('--worker-config')+1]).read_text())
            worker(current,'--resume' in command);result[name]=0
        return result
    monkeypatch.setattr(launcher,'run_jobs',inline)
    args=['--work-dir',str(tmp_path/'work'),'--round14-run',str(previous),'--seeds','17','--protocols','target',
          '--device','cpu','--max-steps','2','--save-every','1','--batch-size','8','--skip-data']
    root=launcher.main(args);verify_marker(root,'CORE_COMPLETE.json');launcher.main(args+['--resume'])
    summary=json.loads((root/'screen_summary.json').read_text())
    assert summary['counts']=={'COMPLETED':25,'REUSED':7,'BLOCKED':4}
    results=pd.read_csv(root/'comparison.csv');assert np.isfinite(results.mse).all()
    dest=root/'models/target_seed17';coef=dest/'ridge_residual/coefficients.npz'
    pred=reconstruct(coef);outer=pc['partition']['outer'];truth=data['delta'][outer]
    with pytest.raises(ValueError,match='gene alignment'):reconstruct(coef,genes=data['genes'][::-1])
    with pytest.raises(ValueError,match='row alignment'):reconstruct(coef,row_ids=np.array(['wrong']))
    meta=data['meta'].iloc[outer].reset_index(drop=True)
    scores=pd.read_csv(dest/'ridge_residual/per_target.csv')
    for row in scores.itertuples():
        ix=(meta.context==row.context)&(meta.perturbation==row.target)
        assert row.mse==pytest.approx(float(np.square(pred[ix]-truth[ix]).mean()),rel=1e-5)
    for name in BLOCKED:
        folder=dest/(name if name in ARMS else 'training_'+name)
        assert (folder/'BLOCKED.json').exists() and not (folder/'COMPLETE.json').exists()
    with tarfile.open(root/'round15_review_light.tar.gz') as archive:
        manifest=json.load(archive.extractfile('ARCHIVE_MANIFEST.json'))
        for name,entry in manifest['files'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==entry['sha256']
        assert not any(n.endswith(('.npz','.pt')) for n in archive.getnames())
    with pytest.raises(ValueError):launcher.main(args+['--resume','--max-steps','3'])
    (dest/'ridge_residual/model.pt').write_bytes(b'bad')
    with pytest.raises(ValueError,match='checksum'):launcher.main(args+['--resume'])
