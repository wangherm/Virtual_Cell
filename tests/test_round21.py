"""Offline task boundaries, common support, independent controls and exact resume."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from vcell.model_agent.artifacts import arrays,read,seal,verify
from vcell.model_agent.nested import DEVELOPMENT,holdout_targets
from vcell.data import save_prepared,load_prepared
from vcell.round21 import signal,repeat,transfer,audit
from vcell.utils import write_json,file_sha256
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'scripts'))
import run_round21 as runner


def measured():
    rng=np.random.default_rng(21);responses=rng.normal(0,.3,(12,8)).astype(np.float32)
    m=pd.DataFrame([{'row_id':f'{c}:T{t}:{rep}','context':c,'cell_line':c.removesuffix('_IFNG'),'perturbation':f'T{t}',
        'dataset':'GSE_TEST','stimulus':'IFNG','time':'24h','modality':'CRISPRi','batch':rep,'replicate':rep,
        'split':'train','n_cells':30,'n_controls':40,'n_guides':2}
        for c in (*DEVELOPMENT,'K562_IFNG') for t in range(12) for rep in ('Rep1','Rep2')])
    values=np.tile(np.repeat(responses,2,axis=0),(5,1))+rng.normal(0,.005,(len(m),8))
    return {'meta':m,'delta':values.astype(np.float32),'baseline':rng.random((len(m),8)).astype(np.float32),
        'genes':np.array(['G'+str(i) for i in range(8)]),'perturbations':np.array(['T'+str(i) for i in range(12)])}


def test_target_specific_rules_do_not_average_away_local_effects():
    d=measured();rows=np.arange(24);table,mismatch,masks=signal.signature_table(d['meta'],rows,d['delta'][rows])
    assert len(table)==12 and len(mismatch)==12*11 and table.reliable.sum()>5
    assert any(mask.any() for mask in masks.values())
    # A common vector repeated for every target is reproducible, not target-specific.
    shared=np.tile(np.arange(8,dtype=np.float32)/10,(24,1));q,_,_=signal.signature_table(d['meta'],rows,shared)
    assert (q.own_pearson>.99).all() and not q.reliable.any()
    partial=np.arange(23);q,_,_=signal.signature_table(d['meta'],partial,d['delta'][partial])
    missing=q.loc[q.perturbation.eq('T11')];assert missing.pair_status.iloc[0]=='MISSING_Rep2' and not missing.reliable.iloc[0]


def test_references_exclude_query_line_and_unknown_even_in_mismatches():
    d=measured();m=d['meta'];held=['T11'];fit=np.flatnonzero(~m.cell_line.eq('A549')&~m.perturbation.isin(held))
    control=d['baseline'][:,:3];result,_,av,common,strong,table,lineage=signal.references(m,fit,d['delta'][fit],control,held)
    assert av[:22].all() and common[:22].all() and not av[m.perturbation.eq('T11')].any()
    assert strong[:22].any()
    for record in lineage:
        i=m.index[m.row_id.eq(record['row_id'])][0]
        for key in ('matched_rows','wrong_rows','shuffled_rows'):
            assert all(m.iloc[j].cell_line!=m.iloc[i].cell_line and m.iloc[j].perturbation not in held for j in record[key])
        assert all(m.iloc[j].perturbation!=m.iloc[i].perturbation for j in record['wrong_rows']+record['shuffled_rows'])
    poison=copy.deepcopy(d);poison['delta'][:24]+=10000
    again=signal.references(m,fit,poison['delta'][fit],control,held)[0]
    for key in result:np.testing.assert_array_equal(result[key],again[key])
    with pytest.raises(ValueError,match='Globally unknown'):signal.references(m,np.arange(len(m)),d['delta'],control,held)
    # Reliability thresholds stay in original response units after normalization.
    scaled=signal.references(m,fit,d['delta'][fit]/10,control,held,response_scale=10)
    np.testing.assert_array_equal(common,scaled[3]);np.testing.assert_array_equal(strong,scaled[4])


def test_repeat_pack_and_predictions_never_use_query_rep2(tmp_path):
    d=measured();rows=np.arange(24);plans=repeat.repeat_plan(d['meta'],rows,d['delta'][rows],d['baseline']);spec=plans[0]
    cache=repeat.pack(spec,d['baseline'],tmp_path/'c');cfg={'kind':'repeat','cache':str(cache),'output_dir':str(tmp_path/'out'),
        'seed':17,'device':'cpu','max_steps':3,'batch_size':4,'save_every':1}
    repeat.worker(cfg);p=arrays(tmp_path/'out/prediction.npz')
    assert set(arrays(cache/'data.npz'))=={'train_rep1','train_rep2','query_rep1','train_control_difference','query_control_difference','train_targets','query_targets'}
    poisoned=copy.deepcopy(spec)
    for q in poisoned['query']:q['rep2']+=1000
    c2=repeat.pack(poisoned,d['baseline'],tmp_path/'c2');repeat.worker({**cfg,'cache':str(c2),'output_dir':str(tmp_path/'out2')})
    p2=arrays(tmp_path/'out2/prediction.npz')
    for key in p:np.testing.assert_array_equal(p[key],p2[key])
    assert not read(tmp_path/'out/lineage.json')['query_rep2_loaded']


def test_repeat_exact_optimizer_resume(tmp_path,monkeypatch):
    d=measured();spec=repeat.repeat_plan(d['meta'],np.arange(24),d['delta'][:24],d['baseline'])[0]
    cache=repeat.pack(spec,d['baseline'],tmp_path/'c');cfg={'cache':str(cache),'output_dir':str(tmp_path/'full'),'seed':17,'device':'cpu','max_steps':3,'batch_size':4,'save_every':1}
    repeat.worker(cfg);save=repeat.atomic_torch_save
    def stop(s,path):
        save(s,path)
        if Path(path).name=='last.pt' and s['step']==1:raise InterruptedError('fixture')
    monkeypatch.setattr(repeat,'atomic_torch_save',stop)
    other={**cfg,'output_dir':str(tmp_path/'resume')}
    with pytest.raises(InterruptedError):repeat.worker(other)
    monkeypatch.setattr(repeat,'atomic_torch_save',save);repeat.worker(other,True)
    np.testing.assert_array_equal(arrays(tmp_path/'full/prediction.npz')['mlp_correction'],arrays(tmp_path/'resume/prediction.npz')['mlp_correction'])
    with pytest.raises(ValueError,match='Frozen'):repeat.worker({**other,'max_steps':4},True)


def parent_cache(path,d,fit,cal,query):
    path.mkdir(parents=True);rng=np.random.default_rng(11)
    np.savez_compressed(path/'features.npz',control=d['baseline'][:,:3],target=np.eye(12,dtype=np.float32)[[int(t[1:]) for t in d['meta'].perturbation]],
        basis=np.eye(8,dtype=np.float32),response_scale=np.float32(1),query=np.array(query),genes=d['genes'],row_ids=d['meta'].row_id.to_numpy(dtype='U'))
    np.savez_compressed(path/'source_labels.npz',fit=np.array(fit),calibration=np.array(cal),fit_delta=d['delta'][fit],calibration_delta=d['delta'][cal])
    np.savez_compressed(path/'teacher_query_predictions.npz',prediction=np.zeros((len(query),3,8),np.float32))
    write_json(path/'manifest.json',{'fixture':'software only; no research accuracy claim'});seal(path);return path


def test_transfer_worker_resume_and_real_source_only_masks(tmp_path,monkeypatch):
    d=measured();m=d['meta'];metadata=tmp_path/'meta.csv';m.to_csv(metadata,index=False)
    used=np.flatnonzero(~m.cell_line.eq('A549')&~m.perturbation.eq('T11'));cal=used[:2];fit=used[2:];query=np.arange(24)
    parent=parent_cache(tmp_path/'p',d,fit,cal,query);cache=transfer.cache(parent,metadata,['T11'],tmp_path/'c')
    cfg={'cache':str(cache),'output_dir':str(tmp_path/'full'),'seed':17,'device':'cpu','max_steps':3,'batch_size':4,'save_every':1,
        'rank':32,'code':runner.BASE_HEAD,'learning_rate':.0005,'weight_decay':.01,'loss':'mse','alpha':.7}
    transfer.worker(cfg);assert read(tmp_path/'full/training_status.json')['status']=='TRAINED'
    output=arrays(tmp_path/'full/prediction.npz');assert (output['prediction'][22:]==0).all()
    assert not any(key.endswith('delta') for key in arrays(cache/'features.npz'))
    save=transfer.atomic_torch_save
    def stop(s,path):
        save(s,path)
        if Path(path).name=='last.pt' and s['step']==1:raise InterruptedError('fixture')
    monkeypatch.setattr(transfer,'atomic_torch_save',stop);other={**cfg,'output_dir':str(tmp_path/'resume')}
    with pytest.raises(InterruptedError):transfer.worker(other)
    monkeypatch.setattr(transfer,'atomic_torch_save',save);transfer.worker(other,True)
    np.testing.assert_array_equal(output['prediction'],arrays(tmp_path/'resume/prediction.npz')['prediction'])
    with pytest.raises(ValueError,match='Frozen'):transfer.worker({**other,'alpha':1.},True)


def test_control_pool_detects_cross_repeat_reuse():
    cells=pd.DataFrame({'cell_id':['n1','n2','p1','p2'],'stratum_id':[1,2,1,2],'group_id':[1,2,3,4],
        'is_control':['1','1','0','0'],'Batch_info':['Rep1','Rep2','Rep1','Rep2']})
    groups=pd.DataFrame({'group_id':[3,4],'cell_type':['HAP1']*2,'Batch_info':['Rep1','Rep2'],'gene':['BATF2']*2})
    links=pd.DataFrame({'group_id':[3,4],'stratum_id':[1,2]});guides=pd.DataFrame({'group_id':[3,4],'guide_eligible':[True,True]})
    library=pd.DataFrame({'cell_id':['n1','n2','p1','p2'],'library_size':[10,20,30,40]})
    frame,verified=audit.control_audit(cells,groups,links,library,guides);assert verified and (frame.controls_shared_with_other_repeat==0).all()
    links.loc[1,'stratum_id']=1
    frame,verified=audit.control_audit(cells,groups,links,library,guides);assert not verified and frame.controls_shared_with_other_repeat.max()==1


def full_parent(tmp_path):
    d=measured();data_path=tmp_path/'data';extra=[]
    for name in ('val','test'):
        q=d['meta'].iloc[:2].copy();q['context']=name.upper();q['cell_line']=name.upper();q['split']=name;q['row_id']=[name+str(i) for i in range(2)];extra.append(q)
    d['meta']=pd.concat([d['meta'],*extra],ignore_index=True);d['baseline']=np.r_[d['baseline'],d['baseline'][:2],d['baseline'][:2]];d['delta']=np.r_[d['delta'],d['delta'][:2],d['delta'][:2]]
    m=d['meta'];save_prepared(data_path,d['baseline'],d['delta'],np.array([int(t[1:]) for t in m.perturbation]),d['genes'],d['perturbations'],m,
        {'normalization':'mean(log1p(10000*counts/full_library))','target_sum':10000})
    data=load_prepared(data_path);held=holdout_targets(m);parent=tmp_path/'runs/r20';plans=[];lineage={}
    for outer in ('BXPC3_IFNG','HAP1_IFNG'):
        folds=[]
        for j,context in enumerate([outer]+[c for c in DEVELOPMENT if c!=outer]):
            excluded={outer} if j==0 else {outer,context};allowed=np.flatnonzero(m.split.eq('train')&~m.context.isin(excluded)&~m.perturbation.isin(held));cal=allowed[:2];fit=allowed[2:];query=np.flatnonzero(m.context.eq(context))
            name=outer+'_seed17_'+('outer' if j==0 else 'inner'+str(j));cache=parent_cache(parent/'caches'/name,d,fit,cal,query)
            lineage[name]={'path':str(cache),'complete_sha256':file_sha256(cache/'COMPLETE.json')};folds.append({'name':name,'query_rows':query.tolist(),'configs':{'expanded':{'partition':{'fit':fit.tolist(),'calibration':cal.tolist()}}}})
        plans.append({'name':outer+'_seed17','outer_context':outer,'seed':17,'folds':folds})
    write_json(parent/'inputs.json',{'data_dir':str(data_path),'metadata_sha256':file_sha256(data_path/'metadata.csv'),
        'dataset_sha256':file_sha256(data_path/'dataset.npz'),'gene_order':d['genes'].tolist(),'held_targets':held,'folds':plans})
    write_json(parent/'cache_lineage.json',lineage);seal(parent);return parent,data


def test_complete_numeric_pipeline_and_freeze_resume(tmp_path,monkeypatch):
    parent,data=full_parent(tmp_path);calls=[]
    def direct(specs,root,*args):
        for cfg in specs.values():
            if '_outer_' in cfg['output_dir']:assert (root/'search_frozen.json').exists()
            calls.append(cfg['output_dir']);(repeat.worker if cfg['kind']=='repeat' else transfer.worker)(cfg,True)
    monkeypatch.setattr(runner,'dispatch',direct)
    def forbidden(*args,**kwargs):raise AssertionError('Unverified raw controls must never enable API search')
    monkeypatch.setattr(runner,'SearchAgent',forbidden)
    command=['--work-dir',str(tmp_path),'--round20-run',str(parent),'--name','r21','--device','cpu','--max-steps','1','--save-every','1','--batch-size','4','--skip-raw-audit']
    root=runner.main(command);verify(root);summary=read(root/'summary.json')
    assert summary['candidate_slots_attempted']==0 and not summary['prepared_data_modified']
    assert not read(root/'search_frozen.json')['outer_used']
    comparison=pd.read_csv(root/'comparison.csv');assert set(comparison.task)=={'supported_technical_repeat','source_cross_cell_transfer','held_development_cross_cell_transfer'}
    assert set(comparison['subset'])>={'all','common_reference_support','globally_unknown'}
    assert (root/'replicate_qc_by_target.csv').exists() and (root/'round21_review_light.tar.gz').exists()
    marker=(root/'COMPLETE.json').read_bytes();count=len(calls);runner.main(command+['--resume']);assert (root/'COMPLETE.json').read_bytes()==marker and len(calls)==count
    assert not read(root/'raw_audit/RAW_STATUS.json')['independent_repeat_controls_verified']
    import tarfile
    with tarfile.open(root/'round21_review_light.tar.gz') as archive:
        assert not any(n.endswith(('.npz','.pt','.gz')) or n.endswith('selected_cells.csv') for n in archive.getnames())


def test_agents_code_and_iteration_feedback_paid_truncation(tmp_path,monkeypatch):
    import json
    from dataclasses import asdict
    from test_deepseek_agent import Client,reply
    from vcell.model_agent.research import Proposal
    from vcell.round21.agents import SignatureScientist,SignatureReviewer
    d=measured();m=d['meta'];metadata=tmp_path/'meta.csv';m.to_csv(metadata,index=False);held=['T11'];used=np.flatnonzero(~m.cell_line.eq('A549')&~m.perturbation.isin(held))
    p=parent_cache(tmp_path/'p',d,used[2:],used[:2],np.arange(24));cache=transfer.cache(p,metadata,held,tmp_path/'c');clients={}
    monkeypatch.setenv('DEEPSEEK_API_KEY','SECRET_NOT_SAVED')
    def scientist(folder,*args,**kwargs):
        arm=Path(folder).parent.name
        proposals=[Proposal('propose_trial',None,'Widen source correction',runner.BASE_HEAD.replace('32',str(w)),32,.0005,.01,'mse') for w in (64,128)]
        client=Client([reply(content=json.dumps(asdict(proposals[0]))),reply(finish='length')]);clients[arm]=client
        return SignatureScientist(folder,*args,**kwargs,client=client)
    def reviewer(folder,*args,**kwargs):
        client=Client([reply(content=json.dumps({'tool':'review','expert_id':None,'reason':'Source scores checked','next_hypothesis':'Try a wider correction while retaining donor rules'}))]*2);clients['review']=client
        return SignatureReviewer(folder,*args,**kwargs,client=client)
    monkeypatch.setattr(runner,'SearchAgent',scientist);monkeypatch.setattr(runner,'ShortReviewer',reviewer)
    monkeypatch.setattr(runner,'dispatch',lambda specs,*args:[transfer.worker(cfg,True) for cfg in specs.values()])
    a=SimpleNamespace(seed=17,device='cpu',max_steps=1,batch_size=4,save_every=1,parallel_jobs=1,agent_model='deepseek-v4-pro')
    baseline={'mse':10.,'strong_mse':10.,'specificity_gap':-10.}
    records,selected,ledger=runner.search(a,{'name':'dev','outer_context':'A549_IFNG'},d,{'f':cache},tmp_path,held,.7,baseline,2)
    assert len(records)==8 and all(s['attempted_requests']==2 for s in ledger)
    body=json.loads(clients['dual'].calls[1][1]['json']['messages'][1]['content']);assert body['previous_source_trials'][0]['review']['next_hypothesis'].startswith('Try a wider')
    assert len([r for r in records if r['arm'] in ('single','dual') and r['status']=='rejected'])==2
    assert not any(r['outer_used'] for r in records)
    runner.search(a,{'name':'dev','outer_context':'A549_IFNG'},d,{'f':cache},tmp_path,held,.7,baseline,2)
    assert all(len(c.calls)==0 for c in clients.values())
    assert 'SECRET_NOT_SAVED' not in ''.join(p.read_text() for p in (tmp_path/'search').rglob('*.json'))


def test_gate_requires_two_source_contexts_and_verified_raw_controls():
    rows=[]
    for context in ('SRC_A','SRC_B'):
        for method,mse,specificity in [('zero',1.,0.),('background_mean',1.1,0.),('string_ridge',1.2,.1),
            ('source_target_mean',.8,.2),('nearest_background',.9,.1),('mlp_correction',.85,.15),('shuffled_reference',1.1,-.2)]:
            for t in range(4):rows.append({'context':context,'target':'T'+str(t),'method':method,'mse':mse,'zero_mse':1.,'prediction_norm':.5,'truth_norm':1.,
                'strong_mse':mse,'specificity_gap':specificity,'common_support':True,'reference_available':True,'globally_unknown':False,'method_status':'TRAINED','fold':'fold_'+context})
    f=pd.DataFrame(rows);rf=f.loc[f.context.eq('SRC_A') & f.method.isin(['source_target_mean','shuffled_reference'])].copy()
    rf.method=rf.method.replace({'source_target_mean':'calibrated_rep1','shuffled_reference':'wrong_rep1'})
    assert runner.gates(f,rf,{'independent_repeat_controls_verified':True})['agent_search_allowed']
    assert not runner.gates(f,rf,{'independent_repeat_controls_verified':False})['agent_search_allowed']
    assert not runner.gates(f.loc[f.context.eq('SRC_A')],rf,{'independent_repeat_controls_verified':True})['agent_search_allowed']
    blocked=f.copy();blocked.loc[blocked.method.eq('mlp_correction'),'method_status']='BLOCKED_NO_RELIABLE_TRAIN_REFERENCE'
    assert not runner.gates(blocked,rf,{'independent_repeat_controls_verified':True})['agent_search_allowed']


def test_raw_audit_interface_counts_alignment_and_secret_stripping(tmp_path,monkeypatch):
    import hashlib
    # No RDS is deserialized in this mocked interface test. Emulate the server's
    # memory reserve instead of depending on the smaller hosted CI machine.
    original_read=Path.read_text
    def read_with_server_memory(path,*args,**kwargs):
        if path==Path('/proc/meminfo'):return 'MemAvailable: 120000000 kB\n'
        return original_read(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read_with_server_memory)
    raw=tmp_path/'raw';raw.mkdir();rds=raw/'Seurat_object_IFNG_Perturb_seq.rds';rds.write_bytes(b'SOFTWARE_FIXTURE_NOT_AN_RDS');sha=hashlib.sha256(rds.read_bytes()).hexdigest()
    write_json(raw/'INSPECT_STATUS.json',{'sha256':sha});rscript=tmp_path/'Rscript';rscript.write_text('MOCK_EXECUTABLE')
    adapter=tmp_path/'adapter';(adapter/'mapping').mkdir(parents=True);(adapter/'prepared').mkdir();(adapter/'aggregation').mkdir()
    cells=pd.DataFrame({'cell_id':['n1','n2','p1','p2'],'stratum_id':[1,2,1,2],'group_id':[1,2,3,4],
        'is_control':['1','1','0','0'],'Batch_info':['Rep1','Rep2','Rep1','Rep2'],'cell_type':['HAP1']*4,
        'gene':['NT','NT','BATF2','BATF2'],'sample':['Rep1','Rep2','Rep1','Rep2'],'orig.ident':['id']*4,'sample_ID':['id']*4,'guide':['g']*4})
    cells.to_csv(adapter/'mapping/cells.csv.gz',index=False)
    groups=pd.DataFrame({'group_id':[3,4],'cell_type':['HAP1']*2,'Batch_info':['Rep1','Rep2'],'gene':['BATF2']*2});groups.to_csv(adapter/'mapping/groups.csv',index=False)
    pd.DataFrame({'group_id':[3,4],'stratum_id':[1,2],'n_controls':[1,1]}).to_csv(adapter/'aggregation/matched_strata.csv',index=False)
    pd.DataFrame({'group_id':[3,4],'guide_eligible':[True,True]}).to_csv(adapter/'prepared/guide_qc.csv',index=False)
    pd.DataFrame({'cell_type':['HAP1']*2,'replicate':['Rep1','Rep2'],'target':['BATF2']*2,'n_retained':[1,1]}).to_csv(adapter/'prepared/target_qc.csv',index=False)
    write_json(adapter/'plan.json',{'source_sha256':sha});seal(adapter);monkeypatch.setenv('DEEPSEEK_API_KEY','NEVER_PASS_TO_R')
    calls=[]
    def fake_run(args,**kwargs):
        assert 'DEEPSEEK_API_KEY' not in kwargs['env'];calls.append(args)
        selected=pd.read_csv(args[3]);assert not selected.cell_type.eq('HT29').any()
        selected['library_size']=[10,20,30,40];selected.to_csv(Path(args[4])/'cell_library_qc.csv.gz',index=False)
        write_json(Path(args[4])/'raw_measurement.json',{'state':'SOFTWARE_FIXTURE'})
    monkeypatch.setattr(audit.subprocess,'run',fake_run)
    result=audit.raw_audit(adapter,raw,tmp_path/'out',REPO,str(rscript));assert result['independent_repeat_controls_verified'] and len(calls)==1
    marker=(tmp_path/'out/COMPLETE.json').read_bytes();audit.raw_audit(adapter,raw,tmp_path/'out',REPO,str(rscript));assert len(calls)==1 and marker==(tmp_path/'out/COMPLETE.json').read_bytes()


def test_status_reports_blocked_heads_and_no_false_completion(tmp_path):
    import subprocess
    root=tmp_path/'runs/status';write_json(root/'workers/x/training_status.json',{'status':'BLOCKED_NO_RELIABLE_TRAIN_REFERENCE'})
    write_json(root/'stage.json',{'stage':'source heads','test_evaluated':False})
    result=subprocess.run([sys.executable,str(REPO/'scripts/round21_status.py'),'--work-dir',str(tmp_path),'--name','status'],capture_output=True,text=True)
    assert result.returncode==0 and 'BLOCKED_NO_RELIABLE_TRAIN_REFERENCE' in result.stdout and 'ROUND21 COMPLETE' not in result.stdout


def test_post_freeze_hap1_vector_audit_distinguishes_squared_ratio(tmp_path):
    d=measured();m=d['meta'];mapping=dict(zip(['T0','T1','T2','T3'],audit.HAP_TARGETS));m.perturbation=m.perturbation.replace(mapping)
    take=m.context.eq('HAP1_IFNG')&m.perturbation.isin(audit.HAP_TARGETS)
    d['delta'][take & m.replicate.eq('Rep1')]*=3
    audit.response_audit(d,tmp_path);s=read(tmp_path/'hap1_response_summary.json');f=pd.read_csv(tmp_path/'hap1_gene_vector_audit.csv')
    assert 8<s['rep1_to_rep2_mean_squared_ratio']<10 and len(f)==4 and f.gene_vector_pearson.min()>.9
    assert 'post-freeze' in s['role'] and s['cause']=='not determined'
