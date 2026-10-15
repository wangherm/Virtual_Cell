import copy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from vcell.evolution import journal,catalog,adapter,grader,pools,seeds
from vcell.round22 import data,evaluator
from vcell.utils import write_json,file_sha256

REPO=Path(__file__).resolve().parents[1]


def controller(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    spec=importlib.util.spec_from_file_location('campaign23',REPO/'scripts/round23.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fixture_pool(tmp_path):
    rng=np.random.default_rng(21);n=14;q=3;g=6
    fold=data.pack(tmp_path,'fold','repeat',np.arange(n),np.arange(n,n+q),rng.normal(size=(n,g)),rng.normal(size=(q,g)),
        rng.normal(size=(n,4)),rng.normal(size=(q,4)),[f'T{i}' for i in range(n)],[f'Q{i}' for i in range(q)],list('abc'),['A549_IFNG']*q,np.array([f'G{i}' for i in range(g)]),
        rng.normal(size=(n,g)),rng.normal(size=(q,g)),rng.normal(size=(q,g)),rng.normal(size=(q,4)),np.ones((q,g),bool),np.ones(q,bool),np.ones(q,bool),np.zeros(q,bool),
        extra={'control':(rng.normal(size=(n,g)),rng.normal(size=(q,g))),'prior':(rng.normal(size=(n,g)),rng.normal(size=(q,g))),
        'source_context':(np.repeat('C',n),np.repeat('A549_IFNG',q))},wrong_targets=[f'T{i}' for i in range(q)])
    write_json(tmp_path/'run_plan.json',{'repo':str(REPO),'code_sha256':evaluator.namespace(REPO),'framework_sha256':journal.fingerprint(REPO),
        'python':sys.executable,'task':'repeat','arm':'core','budget':{'steps':10,'seed':17,'device':'cpu','seconds':120,'root':str(tmp_path)}})
    write_json(tmp_path/'data_plan.json',{'folds':{'repeat':{'search':[fold],'confirmation':[fold],'outer':[]}}})
    return fold


def test_journal_chain_detects_tampering_and_recovers_interrupted_tail(tmp_path):
    path=tmp_path/'events.jsonl';journal.append(path,'trial',phase='search',value=1)
    with path.open('ab') as f:f.write(b'{"partial":')
    assert len(journal.events(path))==1
    r=journal.append(path,'trial',phase='search',value=2)
    assert r['interrupted_append']['uncommitted_bytes']>0 and len(journal.events(path))==2
    path.write_text(path.read_text().replace('"value": 1','"value": 9'))
    with pytest.raises(ValueError,match='chain'):journal.events(path)


def test_auxiliary_excludes_whole_lines_test_global_and_confirmation_targets():
    m=pd.DataFrame({'context':['HAP1_other','A549_IFNG','C','C','C','C','C','C'],
        'cell_line':['HAP1','A549','X','X','X','X','Jurkat','X'],
        'perturbation':['A','A','GLOBAL','CONF','QUERY','OK','A','TEST'],'split':['train']*7+['test']})
    ok=catalog.legal_auxiliary(m,'HAP1_IFNG',['A549_IFNG'],['QUERY'],['CONF'],['GLOBAL'],'repeat','search')
    assert np.flatnonzero(ok).tolist()==[5]
    ok=catalog.legal_auxiliary(m,'HAP1_IFNG',['A549_IFNG'],[],['CONF'],['GLOBAL'],'transfer','confirmation')
    assert np.flatnonzero(ok).tolist()==[3,4,5]


def test_source_bundle_requires_gene_alignment_and_role(tmp_path):
    path=tmp_path/'bundle.npz';genes=np.array(['G1','G2']);values={k:np.array(['x']) for k in ('context','cell_line','target','dataset','stimulus','modality','batch','time')}
    np.savez_compressed(path,genes=genes,baseline=np.ones((1,2)),delta=np.zeros((1,2)),**values)
    e={'bundle':str(path),'sha256':file_sha256(path),'role':'source'}
    assert catalog.load_source_bundle(e,genes)['delta'].shape==(1,2)
    with pytest.raises(ValueError,match='role'):catalog.load_source_bundle({**e,'role':'sealed_external'},genes)
    with pytest.raises(ValueError,match='panel'):catalog.load_source_bundle(e,genes[::-1])
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):catalog.load_source_bundle(e,genes)


def test_h5ad_adapter_full_library_and_completed_artifact_immutable(tmp_path):
    import anndata as ad
    x=np.array([[1,2,7],[2,1,7],[4,2,4],[3,4,3]],dtype=np.uint16)
    a=ad.AnnData(x,obs=pd.DataFrame({'target':['NT','NT','JAK1','JAK1'],'context':['THP1']*4,'rep':['R1']*4},index=list('abcd')),var=pd.DataFrame(index=['G1','G2','OTHER']))
    path=tmp_path/'raw.h5ad';a.write_h5ad(path)
    cfg={'target_column':'target','control_values':['NT'],'context_column':'context','cell_line':'THP1','stratum_columns':['rep'],
        'stimulus':'IFNG','modality':'CRISPR-KO','time':'unknown','dataset':'fixture','target_namespace':'HGNC_symbol','min_cells':2,'min_controls':2}
    entry=adapter.adapt(path,cfg,['G2','G1'],tmp_path/'ready');d=catalog.load_source_bundle(entry,np.array(['G2','G1']))
    expected=np.log1p(10000*x.astype(float)/x.sum(1)[:,None])
    np.testing.assert_allclose(d['delta'][0],(expected[2:].mean(0)-expected[:2].mean(0))[[1,0]],atol=1e-6)
    assert adapter.adapt(path,cfg,['G2','G1'],tmp_path/'ready')==entry
    with pytest.raises(ValueError,match='artifact'):adapter.adapt(path,{**cfg,'stimulus':'other'},['G2','G1'],tmp_path/'ready')
    with pytest.raises(ValueError,match='Missing'):adapter.adapt(path,cfg,['MISSING'],tmp_path/'missing')
    a.X=x.astype(float)/3;a.write_h5ad(tmp_path/'logged.h5ad')
    with pytest.raises(ValueError,match='raw counts'):adapter.adapt(tmp_path/'logged.h5ad',cfg,['G1'],tmp_path/'badcounts')


def test_candidate_failures_are_cached_and_never_imported_in_grader(tmp_path):
    fixture_pool(tmp_path);calls=[]
    def bad(code,d,dest,budget,python):calls.append(1);raise RuntimeError('mat1 and mat2 dtype float/double')
    code=seeds.seed(REPO);r=grader.assess(code,tmp_path,execute_fn=bad)
    assert r['status']=='invalid' and r['category']=='dtype_or_linear_algebra'
    assert grader.assess(code,tmp_path,execute_fn=lambda *args:pytest.fail('Duplicate retrained'))['status']=='invalid'
    assert calls==[1] and journal.feedback(tmp_path/'events.jsonl')[0]['kind']=='invalid_candidate'


def test_trusted_seed_matches_strong_fixed_baseline_and_random_code_runs(tmp_path):
    fold=fixture_pool(tmp_path);d,_,_=evaluator.inputs(fold)
    context={};exec(seeds.seed(REPO),context)
    prediction=context['fit_predict'](d,{'steps':5,'seed':17,'device':'cpu'})
    np.testing.assert_allclose(prediction,evaluator.fixed_predictions(d)['demeaned_support'],rtol=1e-6,atol=1e-7)
    for name,wanted in evaluator.fixed_predictions(d).items():
        namespace={};exec(seeds.seed(REPO,name),namespace)
        np.testing.assert_allclose(namespace['fit_predict'](d,{'steps':5}),wanted,rtol=1e-6,atol=1e-7)
    d['task']=np.array('transfer');assert not context['fit_predict'](d,{ }).any()
    for n in (17,18):
        context={};exec(seeds.random_method(REPO,n),context)
        result=context['fit_predict'](d,{'steps':5,'device':'cpu'})
        assert result.shape==(len(d['query_ids']),len(d['genes'])) and np.isfinite(result).all()


def test_contract_then_full_ablations_and_frozen_stage_guards(tmp_path):
    fixture_pool(tmp_path);calls=[]
    def worker(code,d,dest,budget,python):
        assert 'truth' not in d;calls.append((len(d['query_ids']),budget['steps']));return d['query_support']
    code=seeds.seed(REPO);r=grader.assess(code,tmp_path,execute_fn=worker)
    assert r['status']=='valid' and calls[0]==calls[1]==(2,5) and len(calls)==5
    assert grader.assess(code,tmp_path,execute_fn=lambda *args:pytest.fail('Duplicate evaluation'))==r
    # Confirmation cannot run until the source pool has frozen.
    with pytest.raises(ValueError,match='search freeze'):grader.assess(code,tmp_path,'confirmation',worker)
    assert not (tmp_path/'evaluations/confirmation').exists()
    write_json(tmp_path/'baseline_selection.json',{'repeat':'zero'})
    with pytest.raises(ValueError,match='baseline identity'):grader.assess(code,tmp_path,execute_fn=worker)


def test_novel_count_excludes_seed_random_and_duplicate_predictions(tmp_path,monkeypatch):
    m=controller(monkeypatch);(tmp_path/'seed.py').write_text('initial')
    codeA='method A';codeB='method B';codeR='random'
    records=[]
    for i,code in enumerate(('initial',codeA,codeB,codeR)):
        import hashlib
        key=hashlib.sha256(code.encode()).hexdigest();r={'code_sha256':key,'behavior_sha256':'initial' if i==0 else 'same_new_behavior','metrics':{'combined_score':float(i)}}
        records.append(r)
        if code!=codeR:
            p=tmp_path/'evolution'/f'gen_{i}'/'main.py';p.parent.mkdir(parents=True);p.write_text(code)
    monkeypatch.setattr(m,'valid_candidates',lambda pool:records)
    assert len(m.novel(tmp_path))==1 and m.novel(tmp_path)[0]['code_sha256']==records[2]['code_sha256']


def test_framework_hash_covers_new_trusted_code(tmp_path,monkeypatch):
    fixture_pool(tmp_path);monkeypatch.setattr(grader,'fingerprint',lambda repo:'changed')
    with pytest.raises(ValueError,match='framework'):grader.assess(seeds.seed(REPO),tmp_path)


def test_database_exports_include_failed_proposals(tmp_path,monkeypatch):
    import sqlite3
    m=controller(monkeypatch);p=tmp_path/'evolution';p.mkdir();con=sqlite3.connect(p/'programs.sqlite')
    con.execute('CREATE TABLE attempt_log (generation INTEGER, status TEXT, details TEXT)');con.execute('INSERT INTO attempt_log VALUES (1,?,?)',('failed','shape error'));con.commit();con.close()
    m.export_database(tmp_path)
    assert json.loads((tmp_path/'attempt_log_export.json').read_text())[0]['status']=='failed'


def test_core_expanded_keep_identical_truth_and_exclude_query_labels(tmp_path,monkeypatch):
    base=tmp_path/'fixture';base.mkdir();fold=fixture_pool(base)
    plan={'folds':{t:{'search':[copy.deepcopy(fold)],'confirmation':[copy.deepcopy(fold)],'outer':[]} for t in ('repeat','transfer')},
        'outer_context':'HAP1_IFNG','confirmation_targets':['CONF'],'global_held_targets':['GLOBAL']}
    parent=tmp_path/'parent';parent.mkdir();write_json(parent/'lineage.json',{'data_dir':'fixture'})
    meta=pd.DataFrame({'row_id':list('abcdef'),'context':['X','X','HAP1_IFNG','A549_IFNG','X','X'],
        'dataset':['D']*6,'batch':[str(i) for i in range(6)],'perturbation':['OK','CONF','OK','OK','GLOBAL','Q0'],
        'split':['train']*6,'stimulus':['IFNG']*6,'modality':['CRISPRi']*6,'time':['24h']*6})
    genes=evaluator.inputs(fold)[0]['genes'];historical={'genes':genes,'baseline':np.ones((6,len(genes))),'delta':np.ones((6,len(genes))),'meta':meta}
    monkeypatch.setattr(pools,'prepare',lambda *args:copy.deepcopy(plan));monkeypatch.setattr(pools,'load_prepared',lambda p:historical)
    monkeypatch.setattr(pools,'canonical',lambda value:str(value));monkeypatch.setattr(pools,'target_aliases',lambda:{})
    root=tmp_path/'campaign';root.mkdir()
    result=pools.compile_pools(parent,root,['repeat','transfer'],['core','expanded'],{'repo':str(REPO),'seed':17,'outer_context':'HAP1_IFNG','worker':{}},sys.executable,journal.fingerprint(REPO),{'sources':[]})
    for name,pool in result.items():
        f=json.loads((pool/'data_plan.json').read_text())['folds'][name.split('_')[0]]['search'][0]
        d,g,m=evaluator.inputs(f)
        if name.endswith('core'):assert len(d['aux_delta'])==0
        elif name.startswith('repeat'):assert d['aux_target'].tolist()==['OK']
        else:assert d['aux_target'].tolist()==['OK','Q0']
        np.testing.assert_array_equal(g['truth'],evaluator.inputs(fold)[1]['truth'])
        sibling=result[name.replace('expanded','core')]
        same=json.loads((sibling/'data_plan.json').read_text())['folds'][name.split('_')[0]]['search'][0]
        assert f['grade_sha256']==same['grade_sha256']
    # A process restart can rebuild identical packs without altering any frozen inputs.
    pools.compile_pools(parent,root,['repeat','transfer'],['core','expanded'],{'repo':str(REPO),'seed':17,'outer_context':'HAP1_IFNG','worker':{}},sys.executable,journal.fingerprint(REPO),{'sources':[]})
    subset=tmp_path/'subset';subset.mkdir()
    small=pools.compile_pools(parent,subset,['repeat','transfer'],['core','expanded'],{'repo':str(REPO),'seed':17,'outer_context':'HAP1_IFNG','worker':{}},sys.executable,journal.fingerprint(REPO),{'sources':[],'output_panel':{'genes':genes[[3,1]].tolist()}})
    truth=[]
    for name in ('repeat_core','repeat_expanded'):
        f=json.loads((small[name]/'data_plan.json').read_text())['folds']['repeat']['search'][0]
        d,g,_=evaluator.inputs(f);assert d['genes'].tolist()==genes[[3,1]].tolist();truth.append(g['truth'])
    np.testing.assert_array_equal(truth[0],truth[1]);np.testing.assert_array_equal(truth[0],evaluator.inputs(fold)[1]['truth'][:,[3,1]])


def test_campaign_smoke_then_frozen_finalize_never_reopens_api(tmp_path,monkeypatch):
    from argparse import Namespace
    from vcell.model_agent.artifacts import verify
    m=controller(monkeypatch);parent=tmp_path/'parent';parent.mkdir();write_json(parent/'COMPLETE.json',{'fixture':True})
    cfg=json.loads((REPO/'configs/round23_campaign.json').read_text());cfg.update(valid_new_candidates_per_pool=1,max_cycles_per_pool=1)
    config=tmp_path/'config.json';write_json(config,cfg);cat=tmp_path/'catalog.json';write_json(cat,{'sources':[]})
    def compiled(parent,root,tasks,arms,budget,python,framework,catalog):
        out={}
        for task in tasks:
            for arm in arms:
                pool=root/'pools'/f'{task}_{arm}';pool.mkdir(parents=True,exist_ok=True)
                if not (pool/'data_plan.json').exists():
                    fixture_pool(pool);spec=json.loads((pool/'data_plan.json').read_text());spec['folds'][task]=copy.deepcopy(spec['folds']['repeat']);write_json(pool/'data_plan.json',spec)
                    plan=json.loads((pool/'run_plan.json').read_text());plan.update(task=task,arm=arm,budget=budget['worker']);write_json(pool/'run_plan.json',plan)
                out[pool.name]=pool
        return out
    monkeypatch.setattr(m,'compile_pools',compiled);monkeypatch.setattr(m,'preflight',lambda *args:None)
    def numerical(code,pool,phase='search'):
        return grader.assess(code,pool,phase,execute_fn=lambda code,d,*args:d['query_support'])
    monkeypatch.setattr(m,'assess',numerical);monkeypatch.setattr(m,'Gateway',lambda *args,**kwargs:pytest.fail('Frozen campaign reopened paid API'))
    a=Namespace(work_dir=str(tmp_path),name='campaign',config=str(config),catalog=str(cat),round21_run=str(parent),resume=False,device='cpu',mode='smoke',shinka_python=None)
    root=m.campaign(a);assert not (root/'COMPLETE.json').exists()
    selections={}
    for pool in (root/'pools').glob('*'):
        r={**numerical((pool/'seed.py').read_text(),pool),'arm':'fixed'}
        selections[pool.name]={'finalists':[r],'source_records':[r],'valid_count':0,'random_count':0}
    write_json(root/'search_frozen.json',{'pools':selections,'confirmation_used':False,'outer_used':False})
    monkeypatch.delenv('DEEPSEEK_API_KEY',raising=False);a.resume=True;a.mode='all'
    m.campaign(a);verify(root)
    summary=json.loads((root/'summary.json').read_text());assert not summary['sealed_external_evaluated']
    assert all(not x['confirmed_promotion'] and not x['goal_reached'] for x in summary['pools'].values())
    assert (root/'round23_review_light.tar.gz').exists()
    m.campaign(a);verify(root)


def test_fetch_checks_resume_checksum_and_never_deletes_bad_artifacts(tmp_path,monkeypatch):
    import hashlib
    from types import SimpleNamespace
    payload=b'public data';entry={'id':'source','url':'https://zenodo.org/records/1/files/a.h5ad','filename':'a.h5ad','md5':hashlib.md5(payload).hexdigest()}
    monkeypatch.setattr(catalog.shutil,'disk_usage',lambda path:SimpleNamespace(free=10**12))
    class Response:
        is_redirect=False;status_code=200;headers={'Content-Length':str(len(payload))}
        def raise_for_status(self):pass
        def iter_content(self,*args):yield payload
        def __enter__(self):return self
        def __exit__(self,*args):pass
    calls=[];monkeypatch.setattr(catalog.requests,'get',lambda *args,**kwargs:(calls.append(1) or Response()))
    p=catalog.fetch(entry,tmp_path);assert p.read_bytes()==payload and calls==[1]
    assert catalog.fetch(entry,tmp_path)==p and calls==[1]
    p.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):catalog.fetch(entry,tmp_path)
    assert p.read_bytes()==b'corrupt'
    with pytest.raises(ValueError,match='approved'):catalog.fetch({**entry,'url':'http://127.0.0.1/private'},tmp_path)


def test_data_agent_uses_only_source_metadata_and_freezes_common_panel(tmp_path,monkeypatch):
    import anndata as ad
    from vcell.evolution import data_agent
    gene_names=[f'G{i:04d}' for i in range(1100)]
    a=ad.AnnData(np.ones((40,1100),dtype=np.uint16),obs=pd.DataFrame({'perturbation':['NT']*20+['JAK1']*20,'replicate':['R1']*40},index=[f'c{i}' for i in range(40)]),var=pd.DataFrame(index=gene_names))
    raw=tmp_path/'source.h5ad';a.write_h5ad(raw);panel=tmp_path/'panel.npz';np.savez_compressed(panel,genes=np.array(gene_names+['ABSENT1','ABSENT2'],dtype='U'))
    calls=[];monkeypatch.setattr(data_agent,'fetch',lambda e,d:(calls.append(e['id']) or raw))
    proposal={'target_column':'perturbation','control_values':['NT'],'stratum_columns':['replicate'],'context_column':'__dataset_context__','gene_column':'index','layer':'X'}
    seen=[]
    def provider(body):
        seen.append(body);return 200,{'choices':[{'message':{'content':json.dumps(proposal)}}]}
    source={'id':'source','role':'source','cell_line':'THP1','modality':'CRISPR-KO','publication':'public source'}
    cat={'sources':[source,{'id':'SEALED','role':'sealed_external','publication':'never sent'}]}
    dest=data_agent.prepare_sources(tmp_path,cat,panel,tmp_path/'data-agent','not-a-real-key',request_fn=provider)
    ready=json.loads(dest.read_text());assert calls==['source'] and len(seen)==1
    assert 'SEALED' not in seen[0]['messages'][0]['content']
    assert ready['output_panel']['gene_count']==1100 and len(ready['sources'])==2
    b=catalog.load_source_bundle(ready['sources'][0],np.array(gene_names));assert b['delta'].shape==(1,1100)
    assert data_agent.prepare_sources(tmp_path,cat,panel,tmp_path/'data-agent','unused',request_fn=lambda _:pytest.fail('Completed adapter reopened API'))==dest
    inspection=catalog.inspect_h5ad(raw)
    with pytest.raises(ValueError,match='declared'):
        data_agent.parse_proposal(json.dumps({**proposal,'gene_map':{'evil':'invented'}}),inspection,source,tmp_path)


def test_hgnc_ids_and_previous_names_share_target_exclusion():
    from vcell.evolution.identity import canonical,target_aliases
    assert canonical('SEPT7')==canonical('SEPTIN7')=='SEPTIN7'
    assert canonical('ENSG00000122545.9')=='SEPTIN7'
    m=pd.DataFrame({'context':['X','X'],'perturbation':['SEPT7','ENSG00000122545.9'],'split':['train','train']})
    ok=catalog.legal_auxiliary(m,'HAP1_IFNG',[],[],[],['SEPTIN7'],'transfer','search',target_aliases())
    assert not ok.any()
