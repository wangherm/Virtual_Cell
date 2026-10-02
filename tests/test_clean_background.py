import argparse
import json
import tarfile

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from vcell.background_inspect import digest, write_json
from vcell.clean_background import (export_shards, filter_reasons, gene_mapping,
    load_panel, obs_table, qc_dataset, read_rows, run, run_lock, select_cells, split_donors)


def fixture(path, tissue='blood', n=9):
    obs=pd.DataFrame({'donor_id':[f'D{i%3}' for i in range(n)],'tissue':tissue,
        'cell_type':'T cell','assay':"10x 3' v3",'disease':'normal','is_primary_data':False},
        index=[f'c{i}' for i in range(n)])
    var=pd.DataFrame({'feature_name':['G1','G2','MT-A','DUP','DUP'],
                      'feature_is_filtered':False},index=['e1','e2','mt','d1','d2'])
    a=ad.AnnData(sparse.csr_matrix(np.tile([2.,8.,1.,3.,4.],(n,1))),obs=obs,var=var)
    a.raw=a.copy()
    a=a[:,:2].copy();a.X=a.X*.123  # Main X is processed and has a different gene axis.
    a.write_h5ad(path)


def config():
    return dict(min_counts=1,min_genes=1,max_mito_pct=20,chunk_rows=2,shard_rows=3,
                per_stratum=2000,max_cells_per_dataset=60000,seed=17,reserve_gib=0)


def setup_run(tmp_path):
    source=tmp_path/'input';(source/'h5ad').mkdir(parents=True)
    records=[]
    for key,tissue in [('ts_blood','blood'),('ts_lung','lung'),('hlca_core','lung')]:
        path=source/'h5ad'/f'{key}.h5ad';fixture(path,tissue)
        records.append({'key':key,'status':'inspected_quarantine',
                        'download':{'sha256':digest(path).hexdigest()}})
    write_json(source/'inspect_report.json',{'stage':'inspect_complete','datasets':records})
    panel=tmp_path/'panel.npz'
    np.savez(panel,genes=np.array(['e1','G2','missing','DUP']),sealed_labels=np.array([object()]))
    return argparse.Namespace(input=str(source),output=str(tmp_path/'output'),panel=str(panel),expected_genes=4,**config())


def test_exact_alignment_ambiguity_and_filtered(tmp_path):
    path=tmp_path/'source.h5ad';fixture(path)
    with h5py.File(path,'r+') as f:
        mapping,_=gene_mapping(f,['e1','G2','missing','DUP'])
        assert mapping.status.tolist()==['exact_id','unique_symbol','missing','ambiguous_symbol']
        with pytest.raises(ValueError,match='same source gene'):gene_mapping(f,['e1','G1'])
        f['raw/var/feature_is_filtered'][1]=True
        mapping,_=gene_mapping(f,['G2'])
        assert mapping.status.tolist()==['source_feature_filtered'] and not mapping.available.any()


def test_all_rows_qc_and_masked_full_library_normalization(tmp_path):
    path=tmp_path/'source.h5ad';fixture(path)
    output=tmp_path/'out';cfg=config()
    obs,mapping,summary=qc_dataset(path,output,'ts_blood',['e1','G2','missing','DUP'],cfg)
    assert summary['qc_pass']==9 and obs.total_counts.eq(18).all() and obs.n_genes.eq(5).all()
    selected=select_cells(obs,'ts_blood',split_donors({'ts_blood':obs},17),cfg)
    shards=export_shards(path,output,selected,mapping,cfg)
    assert sum(s['cells'] for s in shards)==9
    for s in shards:
        a=ad.read_h5ad(output/s['file'])
        assert set(a.obs.split)=={s['split']}
        np.testing.assert_allclose(a.X.toarray()[0],np.log1p(np.array([2,8,0,0])*10000/18),rtol=1e-6)
        assert a.var.available.tolist()==[True,True,False,False]
        assert not a.obs.is_primary_data.astype(str).str.lower().eq('true').any()


@pytest.mark.parametrize('invalid',[np.nan,-1,.2])
def test_invalid_late_counts_are_not_missed_by_sampling(tmp_path,invalid):
    path=tmp_path/'source.h5ad';fixture(path,n=99)
    with h5py.File(path,'r+') as f:f['raw/X/data'][-1]=invalid
    with pytest.raises(ValueError,match='count checks'):
        qc_dataset(path,tmp_path/'out','ts_blood',['G1'],config())


def test_duplicate_sparse_values_checked_before_coalescing(tmp_path):
    path=tmp_path/'source.h5ad'
    with h5py.File(path,'w') as f:
        x=f.create_group('x');x.attrs['encoding-type']='csr_matrix';x.attrs['shape']=[1,1]
        x['indptr']=[0,2];x['indices']=[0,0];x['data']=[-.5,1.5]
        with pytest.raises(ValueError,match='count checks'):read_rows(x,0,1)


def test_annotations_and_sampling_are_distinct_reasons(tmp_path):
    path=tmp_path/'source.h5ad';fixture(path)
    with h5py.File(path) as f:obs=obs_table(f)
    obs.loc[0,'assay']='Smart-seq2';obs.loc[1,'lung_condition']='Healthy (tumor adjacent)'
    obs.loc[2,'Manually_curated_celltype']='MNP/T doublets'
    obs.loc[3,'donor_id']='unknown';obs.loc[4,'cell_type']='unknown'
    obs.loc[5,'cell_line']='Jurkat';obs.loc[6,'cell_id']=obs.loc[5,'cell_id']
    reasons=filter_reasons(obs,np.full(9,18),np.full(9,5),np.full(9,5),config())
    for i,reason in enumerate(['non_10x','tumor_adjacent','annotated_doublet','missing_donor',
                               'unknown_cell_type','heldout_background','duplicate_obs_id_within_file']):
        assert reason in reasons[i]
    assert reasons[7:]==['','']


def test_complete_resume_and_tamper_detection(tmp_path,capsys):
    args=setup_run(tmp_path);run(args)
    root=tmp_path/'output';r=json.loads((root/'report.json').read_text())
    assert r['stage']=='complete' and not r['training_started'] and r['donor_disjoint']
    assert not set(r['train_donors'])&set(r['val_donors'])
    assert r['datasets']['hlca_core']['split_counts']=={'external_candidate':9}
    blood=pd.read_csv(root/'ts_blood/membership.csv.gz')
    lung=pd.read_csv(root/'ts_lung/membership.csv.gz')
    assert blood.groupby('donor_id').split.first().equals(lung.groupby('donor_id').split.first())
    with tarfile.open(root/'background_clean_review.tar.gz') as t:
        assert 'report.html' in t.getnames() and 'summary.csv' in t.getnames()
        assert not any(n.endswith('.h5ad') or n.endswith('csv.gz') for n in t.getnames())
    before=digest(root/'ts_blood/membership.csv.gz').hexdigest()
    run(args)
    assert 'REUSE EXPORT: ts_blood' in capsys.readouterr().out
    assert digest(root/'ts_blood/membership.csv.gz').hexdigest()==before
    (root/'ts_blood/membership.csv.gz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Membership cache corrupted'):run(args)
    assert not (root/'COMPLETE.json').exists()
    assert json.loads((root/'report.json').read_text())['stage']=='failed'


def test_resume_after_qc_preserves_partition_and_export(tmp_path,monkeypatch):
    import vcell.clean_background as module
    args=setup_run(tmp_path);original=module.export_shards
    def interrupt(*args,**kwargs):raise RuntimeError('interrupted')
    monkeypatch.setattr(module,'export_shards',interrupt)
    with pytest.raises(RuntimeError,match='interrupted'):run(args)
    assert (tmp_path/'output/ts_blood/qc_complete.json').exists()
    monkeypatch.setattr(module,'export_shards',original)
    run(args)
    assert (tmp_path/'output/COMPLETE.json').exists()


def test_deterministic_caps_and_no_duplicate_writers(tmp_path):
    path=tmp_path/'source.h5ad';fixture(path)
    obs,_,_=qc_dataset(path,tmp_path/'out','ts_blood',['G1'],config())
    cfg=config();cfg.update(per_stratum=1,max_cells_per_dataset=3)
    a=select_cells(obs,'ts_blood',split_donors({'ts_blood':obs},17),cfg)
    b=select_cells(obs,'ts_blood',split_donors({'ts_blood':obs},17),cfg)
    assert a.equals(b) and a.selected.sum()==3 and a.loc[a.selected,'donor_group'].nunique()==3
    assert a.loc[~a.selected,'exclusion_reasons'].eq('sampling_cap').all()
    with run_lock(tmp_path/'lock'):
        with pytest.raises(RuntimeError,match='already active'):
            with run_lock(tmp_path/'lock'):pass
    with run_lock(tmp_path/'lock'):pass


def test_source_integrity_panel_and_configuration_guards(tmp_path):
    args=setup_run(tmp_path)
    assert load_panel(args.panel,4)==['e1','G2','missing','DUP']  # Sealed object labels are never loaded.
    with pytest.raises(ValueError,match='unique panel'):load_panel(args.panel,3)
    run(args);args.seed=29
    with pytest.raises(ValueError,match='Configuration changed'):run(args)
    args.seed=17
    with (tmp_path/'input/h5ad/ts_blood.h5ad').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='Source SHA256 mismatch'):run(args)
