import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from vcell.round24 import modules,runtime,prepare,stats
from vcell.round22 import data,evaluator
from vcell.model_agent.artifacts import read,seal,verify
from vcell.utils import write_json,file_sha256

REPO=Path(__file__).resolve().parents[1]


def fixture_root(tmp_path):
    rng=np.random.default_rng(24);n=15;q=3;g=6
    genes=np.array(['ENSG00000162772','ENSG00000162434','ENSG00000125347','G3','G4','G5'])
    targets=np.resize(np.array(['ATF3','JAK1','IRF1']),n)
    y=rng.normal(size=(n,g));truth=rng.normal(size=(q,g));support=y+rng.normal(scale=.2,size=(n,g));qs=truth+rng.normal(scale=.2,size=(q,g))
    fold=data.pack(tmp_path,'fold','repeat',np.arange(n),np.arange(n,n+q),y,truth,rng.normal(size=(n,4)),rng.normal(size=(q,4)),targets,['ATF3','JAK1','IRF1'],list('abc'),['A549_IFNG']*q,genes,
        support,qs,qs[::-1].copy(),rng.normal(size=(q,4)),np.ones((q,g),bool),np.ones(q,bool),np.ones(q,bool),np.zeros(q,bool),
        extra={'control':(rng.normal(size=(n,g)),rng.normal(size=(q,g))),'prior':(rng.normal(size=(n,g)),rng.normal(size=(q,g))),
            'source_context':(np.repeat('A549_IFNG',n),np.repeat('A549_IFNG',q))},wrong_targets=['JAK1','IRF1','ATF3'],
        shuffled=qs[[1,2,0]],shuffled_x=rng.normal(size=(q,4)),shuffled_targets=['IRF1','ATF3','JAK1'])
    d,_,_=evaluator.inputs(fold)
    aux={k:np.array(['historical','papalexi_thp1','adamson_k562']) if k=='dataset' else np.repeat('X',3) for k in ('target','dataset','context','stimulus','modality','batch','time')}
    values={**d,**{'aux_'+k:v for k,v in aux.items()},'aux_baseline':rng.normal(size=(3,g)),'aux_delta':rng.normal(size=(3,g))}
    np.savez_compressed(fold['input'],**values);fold['input_sha256']=file_sha256(fold['input'])
    baseline=tmp_path/'baseline.npz';np.savez_compressed(baseline,prediction=evaluator.fixed_predictions(values)['demeaned_support'])
    fold.update(baseline=str(baseline),baseline_sha256=file_sha256(baseline))
    write_json(tmp_path/'folds.json',{'tasks':{t:{p:[fold] for p in ('search','confirmation')} for t in ('repeat','transfer')}})
    cfg=read(REPO/'configs/round24.json');cfg.update(worker_seconds=30,steps=5)
    write_json(tmp_path/'run_plan.json',{'repo':str(REPO),'python':sys.executable,'framework_sha256':runtime.fingerprint(REPO),'config':cfg,
        'budget':{'root':str(tmp_path),'seed':17,'device':'cpu','steps':5,'seconds':30,'ram_bytes':2*1024**3,'evaluation_seconds_max':1000}})
    return fold


def test_entire_editable_module_is_composed_without_shell_byte_matching():
    module='import collections\n# EVOLVE-BLOCK-START\ndef fit_predict(data,budget):\n    return np.zeros((len(data["query_ids"]),len(data["genes"])))\n# EVOLVE-BLOCK-END\n'
    ast.parse(modules.compose(module))
    assert 'import collections' in modules.compose(module)
    with pytest.raises(ValueError,match='Exactly one'):modules.compose('def other():\n    return 1\n')
    with pytest.raises(ValueError,match='Exactly one'):modules.compose(module+'\ndef fit_predict(data,budget): return 2\n')


def test_frozen_winner_and_every_ablation_have_supported_anchors():
    code=(REPO/'experiments/round24_evolve/round23_winner.py').read_text()
    assert hashlib.sha256(code.encode()).hexdigest()==modules.WINNER
    variants=modules.methods(code)
    for body in variants.values():ast.parse(modules.compose(body))
    assert '_r24_align(data)' in variants['winner_repaired']
    assert '_r24_active' in variants['no_target_branches'] and "data['gene_symbols']" in variants['winner_repaired']
    with pytest.raises(ValueError,match='Unsupported winner'):modules.methods(code+'# change')


def test_data_ablation_holds_main_training_and_gene_panel_constant(tmp_path):
    fold=fixture_root(tmp_path);d,_,_=evaluator.inputs(fold)
    expected={'none':[],'historical':['historical'],'new':['papalexi_thp1','adamson_k562'],'all':['historical','papalexi_thp1','adamson_k562']}
    for arm,labels in expected.items():
        a=runtime.public_data(d,arm,['papalexi_thp1','adamson_k562'])
        assert a['aux_dataset'].tolist()==labels
        for key in ('train_y','calibration_y','query_support','genes'):np.testing.assert_array_equal(a[key],d[key])
        assert a['gene_symbols'][0]=='ATF3'


def test_identity_and_support_controls_change_separate_inputs(tmp_path):
    fold=fixture_root(tmp_path);d,g,_=evaluator.inputs(fold)
    support=runtime.counterfactual(d,g,'support_wrong');identity=runtime.counterfactual(d,g,'identity_wrong');both=runtime.counterfactual(d,g,'both_wrong')
    np.testing.assert_array_equal(support['query_targets'],d['query_targets']);np.testing.assert_array_equal(support['query_x'],d['query_x'])
    np.testing.assert_array_equal(identity['query_support'],d['query_support']);np.testing.assert_array_equal(identity['query_targets'],g['wrong_targets'])
    np.testing.assert_array_equal(both['query_support'],support['query_support']);np.testing.assert_array_equal(both['query_targets'],identity['query_targets'])
    for value in (support,identity,both):np.testing.assert_array_equal(value['query_control'],d['query_control'])


def test_overall_gain_is_not_target_response_when_strong_error_worsens():
    cfg=read(REPO/'configs/round24.json')
    base=pd.DataFrame({'row_id':['a','b'],'context':['A','A'],'target':['X','Y'],'mse':[1.,1.],'strong_mse':[2.,2.],
        'specificity_gap':[.2,.2],'prediction_norm':[1.,1.],'truth_norm':[2.,2.],'residual_norm':[1.,1.],'truth_residual_norm':[2.,2.],'common_support':[True,True]})
    frame=base.copy();frame['mse']=.95;frame['strong_mse']=2.1;frame['specificity_gap']=.1
    cf=pd.concat([base.assign(variant=v) for v in cfg['counterfactual_variants'][1:]])
    m=runtime.metrics(frame,base,cf,cfg)['public']
    assert m['overall_error_candidate'] and not m['target_response_candidate']
    frame['strong_mse']=1.9;frame['specificity_gap']=.21
    assert runtime.metrics(frame,base,cf,cfg)['public']['target_response_candidate']
    frame['prediction_norm']=.01
    assert runtime.metrics(frame,base,cf,cfg)['public']['target_response_candidate']  # no forced amplitude inflation


def test_assessment_cache_and_confirmation_gate(tmp_path):
    fixture_root(tmp_path);body=modules.simple_module();calls=[]
    def execute(code,d,dest,*args):calls.append(1);return d['query_support']
    r=runtime.assess(tmp_path,body,controls=False,execute_fn=execute)
    assert r['status']=='valid'
    assert runtime.assess(tmp_path,body,controls=False,execute_fn=lambda *args:pytest.fail('Cached refit'))==r
    with pytest.raises(ValueError,match='freeze'):runtime.assess(tmp_path,body,phase='confirmation',scope='evolved')
    with pytest.raises(ValueError,match='outer'):runtime.assess(tmp_path,body,phase='outer')
    assert len(calls)==1
    write_json(Path(r['directory'])/'assessment.json',{'corrupt':True})
    with pytest.raises((ValueError,KeyError)):runtime.assess(tmp_path,body,controls=False,execute_fn=execute)


def test_target_cluster_stats_preserve_context_target_pairs():
    b=pd.DataFrame([{'row_id':str(i),'context':c,'target':t,'mse':1.} for i,(c,t) in enumerate([('A','X'),('A','Y'),('B','X'),('B','Z')])]);a=b.copy();a['mse']=.9
    result=stats.cluster_gain(a,b,draws=100)
    assert result['gain']==pytest.approx(.1) and result['context_target_wins']==4 and result['distinct_targets']==3
    np.testing.assert_allclose(result['exploratory_interval_95'],[.1,.1])


def test_seed_and_finalist_selection_cannot_use_confirmation(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    spec=importlib.util.spec_from_file_location('round24controller',REPO/'scripts/run_round24.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    def record(name,phase,value):return {'name':name,'phase':phase,'task':'repeat','arm':'all','metrics':{'public':{'mse':value}}}
    rows=[record('short','search',1.001),record('long','search',1.),record('short','confirmation',100.),record('long','confirmation',0.)]
    chosen,best=m.select_seed(rows,{'short':'line','long':'\n'.join(['line']*100)})
    assert chosen['name']=='short' and best['name']=='long'
    for r in rows:
        if r['phase']=='confirmation':r['metrics']['public']['mse']=-1000
    assert m.select_seed(rows,{'short':'line','long':'\n'.join(['line']*100)})[0]['name']=='short'
    with pytest.raises(ValueError,match='Confirmation'):m.freeze_search(tmp_path,[{'identity':{'phase':'confirmation'}}],read(REPO/'configs/round24.json'),'test')


@pytest.mark.skipif(sys.platform!='linux',reason='Real Landlock/seccomp worker requires Linux')
def test_real_module_worker_replay_and_repaired_branch_audit(tmp_path):
    fixture_root(tmp_path);code=(REPO/'experiments/round24_evolve/round23_winner.py').read_text()
    old=runtime.assess(tmp_path,modules.winner_module(code),controls=False,audit=True)
    new=runtime.assess(tmp_path,modules.winner_module(code,repair=True),controls=False,audit=True)
    assert old['status']=='valid',old.get('error')
    assert new['status']=='valid',new.get('error')
    a=read(Path(old['directory'])/'branch_audit.json')[0];b=read(Path(new['directory'])/'branch_audit.json')[0]
    assert a['query_contexts']==['global'] and b['query_contexts']==['a549_ifng']
    assert b['target_output_index_matches']>a['target_output_index_matches']
    files=list((tmp_path/'input_cache').glob('*.npz'));assert files and any(p.stat().st_nlink>1 for p in files)


def test_prepare_requires_full_source_predictions_and_never_opens_outer(tmp_path):
    source=tmp_path/'source';source.mkdir();fold=fixture_root(source)
    code=(REPO/'experiments/round24_evolve/round23_winner.py').read_text();phase_specs={p:[fold] for p in ('search','confirmation')}
    for task in ('repeat','transfer'):
        pool=source/'pools'/f'{task}_expanded';pool.mkdir(parents=True)
        write_json(pool/'data_plan.json',{'gene_order':evaluator.inputs(fold)[0]['genes'].tolist(),'folds':{task:{**phase_specs,'outer':[{'must_not_read':True}]}}})
        write_json(pool/'baseline_selection.json',{task:'demeaned_support'})
    p=source/'pools/repeat_expanded/evaluations/search'/modules.WINNER;p.mkdir(parents=True);(p/'candidate.py').write_text(code)
    write_json(source/'confirmation_frozen.json',{'repeat_expanded':{'winner':{'code_sha256':modules.WINNER}}});seal(source)
    root=tmp_path/'destination';root.mkdir();cfg=read(REPO/'configs/round24.json');cfg['panel_genes']=6
    with pytest.raises(ValueError,match='full Round23'):prepare.prepare(source,root,cfg)
    for phase in ('search','confirmation'):
        p=source/'pools/repeat_expanded/evaluations'/phase/modules.WINNER/'full/workers/fold/correct/output';p.mkdir(parents=True);np.savez_compressed(p/'prediction.npz',prediction=np.zeros((3,6)))
    seal(source);prepare.prepare(source,root,cfg)
    assert set(read(root/'folds.json')['tasks']['repeat'])=={'search','confirmation'}
    assert not read(root/'folds.json')['outer_included']


def test_controller_resumes_after_search_freeze_without_reopening_paid_search(tmp_path,monkeypatch):
    from argparse import Namespace
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    spec=importlib.util.spec_from_file_location('round24resume',REPO/'scripts/run_round24.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    parent=tmp_path/'parent';parent.mkdir();write_json(parent/'COMPLETE.json',{'fixture':True})
    config=tmp_path/'config.json';write_json(config,read(REPO/'configs/round24.json'))
    monkeypatch.setattr(m,'prepare',lambda *args:'frozen source')
    monkeypatch.setattr(m,'preflight',lambda *args:None)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda *args:SimpleNamespace(free=10**12))
    monkeypatch.setattr(m,'diagnostics',lambda *args:[])
    charged=[]
    def search(root,*args):
        if (root/'search_frozen.json').exists():return read(root/'search_frozen.json')
        charged.append(1);r={'valid_new':0,'finalists':[]};write_json(root/'search_frozen.json',r);return r
    monkeypatch.setattr(m,'evolve',search)
    attempts=[]
    def random(root,*args):
        attempts.append(1)
        if len(attempts)==1:raise RuntimeError('simulated interruption after search freeze')
        return {'records':[],'matched':True}
    monkeypatch.setattr(m,'random_control',random)
    monkeypatch.setattr(m,'report',lambda root,*args:write_json(root/'summary.json',{'test_evaluated':False}))
    a=Namespace(work_dir=str(tmp_path),name='campaign',round23_run=str(parent),config=str(config),resume=False,device='cpu',mode='all',shinka_python='unused')
    with pytest.raises(RuntimeError,match='interruption'):m.launch(a)
    a.resume=True;root=m.launch(a);verify(root)
    assert charged==[1] and (root/'round24_review_light.tar.gz').exists()
    m.launch(a);assert charged==[1]
    changed=read(config);changed['seed']+=1;write_json(config,changed)
    with pytest.raises(ValueError):m.launch(a)


def test_feedback_excludes_confirmation_scores(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    spec=importlib.util.spec_from_file_location('round24memory',REPO/'scripts/run_round24.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for phase,secret in [('search','source allowed'),('confirmation','CONFIRMATION_SENTINEL')]:
        p=tmp_path/'assessments'/phase/'assessment.json';write_json(p,{'identity':{'phase':phase,'scope':'evolved'},'metrics':{'public':{'evidence':secret}},'module_sha256':'code'})
    m.source_memory(tmp_path)
    text=(tmp_path/'source_feedback.json').read_text();assert 'source allowed' in text and 'CONFIRMATION_SENTINEL' not in text


def feedback_entry(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    spec=importlib.util.spec_from_file_location('round24_feedback_entry',REPO/'scripts/round24_entry.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_nullable_journal_feedback_preserves_bounds_and_source_boundary(tmp_path,monkeypatch):
    from vcell.evolution.journal import append,events
    entry=feedback_entry(monkeypatch);journal=tmp_path/'journal.jsonl'
    append(journal,'assessment',phase='search',error=None,score=.25)
    append(journal,'assessment',phase='search',score=.5)
    append(journal,'assessment',phase='search',error='prefix'+('x'*1700))
    append(journal,'assessment',phase='confirmation',error='CONFIRMATION_SENTINEL',score=-999)
    append(journal,'assessment',phase='outer',error='OUTER_SENTINEL',score=-1000)
    raw=journal.read_bytes();result=entry.source_feedback(journal,limit=3)
    assert [r['error'] for r in result]==['','','x'*1600]
    assert [r['score'] for r in result[:2]]==[.25,.5]
    assert len(entry.source_feedback(journal,limit=1))==1
    assert 'SENTINEL' not in json.dumps(result)
    assert journal.read_bytes()==raw and len(events(journal))==5
    audit=read(tmp_path/'feedback_compatibility.json')
    assert audit['journal_sha256']==file_sha256(journal)
    assert not audit['journal_rewritten'] and not audit['numerical_fingerprint_overridden']


def test_feedback_fix_reuses_real_assessment_and_frozen_plan(tmp_path,monkeypatch):
    from vcell.evolution.journal import events
    fixture_root(tmp_path);entry=feedback_entry(monkeypatch)
    spec=importlib.util.spec_from_file_location('round24_null_resume',REPO/'scripts/run_round24.py')
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    result=runtime.assess(tmp_path,modules.simple_module(),controls=False,
        execute_fn=lambda code,d,*args:d['query_support'])
    assert result['status']=='valid' and events(tmp_path/'journal.jsonl')[-1]['error'] is None
    plan=(tmp_path/'run_plan.json').read_bytes();journal=(tmp_path/'journal.jsonl').read_bytes()
    assessment=Path(result['directory'])/'assessment.json';saved=assessment.read_bytes()
    before=runtime.fingerprint(REPO)
    entry.configure_controller(controller).source_memory(tmp_path)
    assert read(tmp_path/'source_feedback.json')['history'][-1]['error']==''
    assert runtime.assess(tmp_path,modules.simple_module(),controls=False,
        execute_fn=lambda *args:pytest.fail('Completed assessment was retrained'))==result
    assert runtime.fingerprint(REPO)==before
    assert (tmp_path/'run_plan.json').read_bytes()==plan
    assert (tmp_path/'journal.jsonl').read_bytes()==journal
    assert assessment.read_bytes()==saved


def test_feedback_fix_does_not_hide_corrupt_journal_or_invalid_error(tmp_path,monkeypatch):
    from vcell.evolution.journal import append
    entry=feedback_entry(monkeypatch);journal=tmp_path/'journal.jsonl'
    append(journal,'assessment',phase='search',error={'unexpected':'object'})
    with pytest.raises(TypeError,match='string or null'):entry.source_feedback(journal)
    journal.write_bytes(journal.read_bytes().replace(b'assessment',b'tampered!!'))
    with pytest.raises(ValueError,match='journal chain'):entry.source_feedback(journal)


def test_default_launcher_dispatches_compatibility_entry(monkeypatch):
    from types import SimpleNamespace
    entry=feedback_entry(monkeypatch);calls=[]
    controller=SimpleNamespace(feedback=None,main=lambda:calls.append(tuple(sys.argv)))
    monkeypatch.setitem(sys.modules,'run_round24',controller)
    monkeypatch.setattr(sys,'argv',['round24_entry.py','--name','existing','--resume'])
    entry.main()
    assert calls==[('round24_entry.py','--name','existing','--resume')]
    assert controller.feedback is entry.source_feedback
    launcher=(REPO/'scripts/run_round24_autodl.sh').read_text()
    assert 'scripts/round24_entry.py' in launcher
