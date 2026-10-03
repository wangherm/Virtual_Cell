"""Numerical, leakage-boundary and historical checkpoint reuse tests."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_qwen import setup
from test_round9 import fixtures
from vcell.background_training import pretrain
from vcell.round8 import worker
from vcell.round11 import (moments, mixture_error, fit_coefficients, support_plan, few_shot,
                          freeze, assert_oof_eligible, context_partition, audit_cache,
                          summarize, release_inference, query_target_specificity)
from vcell.perturbation_coverage import annotation_counts
from vcell.selection import row_weights, mse
from vcell.round7 import calibrated_mix
from vcell.utils import write_json


def stats_fixture(n=140):
    rng=np.random.default_rng(33)
    meta=pd.DataFrame({'row_id':[f'r{i}' for i in range(n*2)],'context':'H1_hESC',
                       'perturbation':np.repeat([f'P{i}' for i in range(n)],2),'seen_in_fit':False})
    p=rng.normal(size=(len(meta),7));y=.3*p;r=rng.normal(size=p.shape)
    return moments(meta,y,p,r),y,p,r


def test_moments_exact_mixture_and_same_calibration_family():
    frame,y,p,r=stats_fixture()
    for a,b in [(0,0),(1,0),(.2,.7)]:
        np.testing.assert_allclose(mixture_error(frame,a,b),((a*p+b*r-y)**2).mean(1),atol=1e-12)
    assert fit_coefficients(frame,True)[0]==pytest.approx(.3)
    expected=calibrated_mix([p,r,np.zeros_like(p)],y,row_weights(frame))
    np.testing.assert_allclose(fit_coefficients(frame),expected[:2],atol=1e-8)
    flipped=moments(frame,-p,p,r)
    assert fit_coefficients(flipped,True)[0]==0


def test_support_query_boundaries_and_query_poisoning(tmp_path):
    frame,y,p,r=stats_fixture();plan=support_plan(frame.perturbation)
    assert len(plan['query'])==100 and len(plan['support_pool'])==40
    assert not set(plan['query'])&set(plan['support_pool'])
    assert support_plan(reversed(frame.perturbation.tolist()))==plan
    before,coeff=few_shot(frame,plan,43)
    query=frame.perturbation.isin(plan['query'])
    poisoned=frame.copy();poisoned.loc[query,['c','t','h_ref']]=1000
    _,other=few_shot(poisoned,plan,43)
    assert coeff==other
    for row in coeff:
        assert row['alpha']==pytest.approx(.3 if row['support_targets'] else 1.)
        assert row['target_ids']==plan['draws'][row['draw']][:row['support_targets']]
    assert all(r['target'] in plan['query'] for r in before)
    specificity=query_target_specificity(frame,y,p,plan,43)
    assert all(s['true_target_midrank']==1 and s['own_target_pearson']==pytest.approx(1) for s in specificity)
    assert all(s['shuffled_prediction_target']!=s['target'] for s in specificity)
    scores=summarize(before,tmp_path,[('zero','adapted')])
    assert scores[ scores.method=='adapted'].targets.eq(100).all()
    freeze(tmp_path/'plan.json',plan);freeze(tmp_path/'plan.json',plan)
    with pytest.raises(ValueError,match='Frozen'):freeze(tmp_path/'plan.json',{'different':True})


def test_source_fold_boundaries_and_nested_leakage():
    meta=pd.DataFrame([{'context':c,'perturbation':f'P{p}','batch':f'b{b}','row_id':f'{c}_{p}_{b}'}
        for c in ('K562','RPE1','H1_hESC','HepG2','Jurkat') for p in range(20) for b in range(5)])
    data={'meta':meta,'splits':{'train':np.flatnonzero(meta.context.isin(['K562','RPE1','H1_hESC'])),
        'val':np.flatnonzero(meta.context=='HepG2'),'test':np.flatnonzero(meta.context=='Jurkat')}}
    caches=[]
    for c in ('K562','RPE1','H1_hESC'):
        parts=context_partition(data,c)
        assert c not in set(meta.iloc[parts['fit']+parts['calibration']].context)
        assert set(meta.iloc[parts['outer']].context)=={c}
        caches.append({'cfg':{},'meta':meta.iloc[parts['outer']],
                       'identity':{'fit_contexts':sorted(set(meta.iloc[parts['fit']].context))}})
    assert_oof_eligible(caches,'HepG2');assert_oof_eligible(caches,'Jurkat')
    with pytest.raises(ValueError,match='unseen source'):assert_oof_eligible(caches,'H1_hESC')
    with pytest.raises(ValueError,match='Missing'):assert_oof_eligible(caches[:2],'Jurkat')
    caches[0]['identity']['fit_contexts'].append('Jurkat')
    with pytest.raises(ValueError,match='contamination'):assert_oof_eligible(caches,'Jurkat')


def test_original_annotation_counts_do_not_mix_control_strata():
    obs=pd.DataFrame({'target':['control']*5+['P']*4+['Q']*3,'batch':['b']*12,
                      'timepoint':['24h']*9+['48h']*3})
    result=annotation_counts(obs,{'perturbation_key':'target','batch_key':'batch','control_values':['control']},3,3)
    assert result.set_index('target').loc['P','count_eligible']
    assert not result.set_index('target').loc['Q','count_eligible']
    assert result.set_index('target').loc['Q','n_controls']==0


def test_historical_cache_audit_and_release_gate(setup,tmp_path):
    cfg,data=setup
    bg,student=fixtures(cfg,data,tmp_path,'qwen','b2')
    student.update(arm='qwen_b2_none',module_mode='none',kd_weight=0.)
    pretrain(bg);worker(student)
    cache=audit_cache(data,tmp_path/'student',student['partition'])
    assert cache['identity']['checkpoint_sha256']
    tokenizer=Path(cfg['model']['model_id'])/'tokenizer.json';original=tokenizer.read_bytes()
    tokenizer.write_bytes(original+b' ')
    with pytest.raises(ValueError,match='backbone dependency'):audit_cache(data,tmp_path/'student',student['partition'])
    tokenizer.write_bytes(original)
    wrong=copy.deepcopy(student['partition']);wrong['fit']=wrong['fit'][::-1]
    with pytest.raises(ValueError,match='partition'):audit_cache(data,tmp_path/'student',wrong)
    path=tmp_path/'release.json';write_json(path,{'checkpoint_sha256':[], 'data_fingerprint':data['audit']['fingerprint']})
    with pytest.raises(ValueError,match='not frozen'):release_inference(data,cache,data['splits']['test'],path,'cpu')
    write_json(path,{'checkpoint_sha256':[cache['identity']['checkpoint_sha256']], 'data_fingerprint':data['audit']['fingerprint']})
    pred,ref,seen=release_inference(data,cache,data['splits']['test'],path,'cpu')
    assert pred.shape==data['delta'][data['splits']['test']].shape and np.isfinite(pred).all()
    with (tmp_path/'student/predictions.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):audit_cache(data,tmp_path/'student',student['partition'])


def test_complete_pipeline_reuses_folds_and_freezes_release(setup,tmp_path,monkeypatch):
    """Train real tiny endpoints, run all stages, then verify exact resume boundaries."""
    import importlib
    from types import SimpleNamespace
    from test_round8 import knowledge_fixture
    from test_round9 import background_fixture
    from vcell.background_training import prepare_background
    from vcell.data import save_prepared, load_prepared
    from vcell.round8 import partitions
    from vcell.train import source_fingerprint
    from vcell.utils import read_config
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=importlib.import_module('run_round11')
    cfg,_=setup;work=tmp_path/'work';old=work/'runs/round10_01';old.mkdir(parents=True)
    meta=pd.DataFrame([{'context':c,'perturbation':f'P{p}','batch':f'b{b}','row_id':f'{c}_{p}_{b}',
                        'split':'val' if c=='HepG2' else 'test' if c=='Jurkat' else 'train'}
        for c in ('K562','RPE1','H1_hESC','HepG2','Jurkat') for p in range(140) for b in range(3)])
    rng=np.random.default_rng(123);n=len(meta);genes=[f'G{i}' for i in range(6)];targets=[f'P{i}' for i in range(140)]
    prepared=work/'prepared/fixture'
    save_prepared(prepared,rng.uniform(0,1,(n,6)).astype('float32'),rng.normal(0,.1,(n,6)).astype('float32'),
                  np.array([targets.index(p) for p in meta.perturbation]),genes,targets,meta,
                  {'synthetic':True,'normalization':'mean(log1p(10000*counts/full_library))','target_sum':10000})
    data=load_prepared(prepared);knowledge=work/'knowledge.npz';knowledge_fixture(data,knowledge)
    source=background_fixture(work/'source',genes);cache=prepare_background(source,old/'background_cache',genes)
    plan={'experiment':'round10','source':source_fingerprint(),'seeds':[17,29,43],
          'data_path':str(prepared),'data':data['audit']['fingerprint']}
    write_json(old/'plan.json',plan)
    for seed in (17,29,43):
        for protocol,held,arms in [('context',None,('mlp_b0_none','qwen_b1_none','qwen_b2_none')),
                                  ('context_h1','H1_hESC',('qwen_b2_none',)),('context_rpe1','RPE1',('qwen_b2_none',))]:
            parts=partitions(data,'context') if held is None else context_partition(data,held)
            for arm in arms:
                family,bg,_=arm.split('_');name=f'{protocol}_{arm}_seed{seed}';encoder=None
                if bg!='b0':
                    dest=old/'background_pretraining'/f'{protocol}_{bg}_seed{seed}'
                    bc=dict(stage='background',data_dir=str(prepared),partition=parts,background_cache=str(cache),
                            background=bg,family=family,model=cfg['model'],seed=seed,device='cpu',pretrain_steps=1,
                            pretrain_batch=4,save_every=1,output_dir=str(dest))
                    pretrain(bc);encoder=str(dest/'encoder.pt')
                student=dict(experiment='round10',data_dir=str(prepared),knowledge=str(knowledge),partition=parts,
                    protocol=protocol,arm=arm,family=family,background=bg,module_mode='none',module_permutation_seed=818,
                    seed=seed,model=cfg['model'],device='cpu',rank=3,learning_rate=.0002,module_weight=.1,kd_weight=0.,
                    max_steps=1,save_every=1,batch_size=128,gradient_accumulation=1,output_dir=str(old/'students'/name),
                    teacher_cache=None,encoder=encoder,diagnostic_every=1)
                worker(student)
    write_json(old/'COMPLETE.json',{'students':15,'test_evaluated':False})
    # Keep the integration test fast while still running the real worker and pretrainer.
    # The scheduler's actual subprocess/resume behavior is covered by Round9/10 tests.
    executed=[]
    def synchronous(queue,root,*args,**kwargs):
        result={}
        for name,spec in queue.items():
            command=spec['cmd'];c=read_config(command[command.index('--worker-config')+1]);executed.append(name)
            (pretrain if c.get('stage')=='background' else worker)(c,resume='--resume' in command);result[name]=0
        return result
    monkeypatch.setattr(runner,'run_jobs',synchronous)
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda _:SimpleNamespace(free=1000*1024**3))
    args=['--work-dir',str(work),'--device','cpu']
    root=runner.main(args+['--development-only'])
    assert not (root/'TEST_OPENED.json').exists() and not (root/'COMPLETE.json').exists()
    release=json.loads((root/'release_manifest.json').read_text());assert len(release['checkpoint_sha256'])==9
    assert len(executed)==6 and all('k562' in name for name in executed)
    runner.main(args+['--resume'])
    assert json.loads((root/'COMPLETE.json').read_text())['test_evaluated']
    assert json.loads((root/'release_manifest.json').read_text())==release
    scores=pd.read_csv(root/'jurkat/comparison.csv');assert scores.seed.nunique()==3 and scores.method.nunique()==8
    original_count=len(executed);runner.main(args+['--resume']);assert len(executed)==original_count
    import tarfile
    with tarfile.open(root/'round11_review_light.tar.gz') as t:
        assert 'evidence/source_oof_seed17.csv' in t.getnames()
        assert not any(n.endswith(('.npz','.pt','.npy')) for n in t.getnames())
    prediction=next((root/'test_predictions').rglob('predictions.npz'))
    with prediction.open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):runner.main(args+['--resume'])
