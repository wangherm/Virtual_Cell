"""Real sparse R integration plus metadata, lineage, controls and holdout boundaries."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest

from test_round12 import dataset
from vcell.data import load_prepared
from vcell.round15_adapter import metadata_plan, prepare_extension, report, package, POLICY

REPO = Path(__file__).resolve().parents[1]


def metadata():
    rows=[]
    for line in ('BXPC3','A549','HT29'):
        for rep in ('Rep1','Rep2'):
            for gene in ('NT','P0','P25','NEW'):
                for guide in (1,2):
                    for _ in range(10):
                        rows.append(dict(cell_id='c'+str(len(rows)),cell_type=line,
                          sample=('BXCP3' if line=='BXPC3' else line)+'_IFNG',pathway='IFNG',
                          Batch_info=rep,**{'orig.ident':'01'},sample_ID='sample_1',gene=gene,guide=f'{gene}g{guide}',
                          mixscale_score=1e99))
    return pd.DataFrame(rows)


def test_metadata_alias_reservation_and_no_score(tmp_path):
    m=metadata(); audit=metadata_plan(m,['G0','G1'],['G1','G0'],tmp_path)
    cells=pd.read_csv(tmp_path/'cells.csv.gz')
    assert 'HT29' not in set(cells.cell_type)
    assert 'mixscale_score' not in cells
    assert audit['reserved_HT29_cells']==160 and audit['alias_cells']==160
    assert audit['missing_target_genes']==['NEW','P0','P25']
    assert (tmp_path/'panel.txt').read_text().splitlines()==['G1','G0']
    m.loc[0,'sample']='MCF7_IFNG'
    with pytest.raises(ValueError,match='sample/cell_type'):metadata_plan(m,['G0'],['G0'],tmp_path)


@pytest.mark.parametrize('change,match',[
    ('panel','Missing 1 frozen'),('guide','guide/gene'),('duplicate','Duplicate cell'),('missing','Empty required'),('stimulus','stimulus')])
def test_invalid_metadata_and_panel_block(tmp_path,change,match):
    m=metadata(); panel=['G0']
    if change=='panel':panel=['G0','absent']
    if change=='guide':m.loc[0,'guide']='WRONGg1'
    if change=='duplicate':m.loc[1,'cell_id']=m.loc[0,'cell_id']
    if change=='missing':m.loc[0,'sample_ID']=''
    if change=='stimulus':m.loc[0,'pathway']='TGFB'
    with pytest.raises(ValueError,match=match):metadata_plan(m,['G0'],panel,tmp_path)


def fake_aggregation(old,root):
    m=metadata(); metadata_plan(m,old['genes'],old['genes'],root)
    groups=pd.read_csv(root/'groups.csv'); n=len(groups)
    cells=pd.read_csv(root/'cells.csv.gz')
    links=cells[['group_id','stratum_id']].drop_duplicates();links['n_controls']=20
    links.to_csv(root/'matched_strata.csv',index=False)
    pd.DataFrame({'group_id':groups.group_id,'n_input':10,'n_positive':10,'n_matched':10}).to_csv(root/'group_qc.csv',index=False)
    for name,offset in [('mean',2.),('baseline',.5)]:
        values=np.ones((n,len(old['genes'])))*offset
        f=pd.DataFrame(values,columns=old['genes']);f.insert(0,'group_id',groups.group_id)
        f.to_csv(root/(name+'.csv.gz'),index=False)


def test_prepared_extension_preserves_old_rows_and_global_target_splits(tmp_path):
    old=dataset(tmp_path/'old'); fake_aggregation(old,tmp_path/'cache')
    result=prepare_extension(old,old,tmp_path/'cache',tmp_path/'prepared',POLICY)
    new=load_prepared(tmp_path/'prepared'); n=len(old['meta'])
    np.testing.assert_array_equal(old['delta'],new['delta'][:n])
    np.testing.assert_array_equal(old['baseline'],new['baseline'][:n])
    np.testing.assert_allclose(new['delta'][n:],1.5)
    assert result['appended_rows']==12 and result['appended_cells']==240
    assert new['meta'].iloc[n:].n_controls.eq(20).all()  # Shared NT cells counted once across guides.
    assert set(new['meta'].iloc[n:].context)=={'A549_IFNG','BXPC3_IFNG'}
    for protocol in ('context','target','double'):
        parts=json.loads((tmp_path/'prepared/partitions'/protocol/'partition.json').read_text())
        fit=set(new['meta'].iloc[parts['fit']].perturbation)
        cal=set(new['meta'].iloc[parts['calibration']].perturbation)
        outer=set(new['meta'].iloc[parts['outer']].perturbation)
        assert not fit&cal
        if protocol!='context':assert not fit&outer
        assert not new['meta'].iloc[sum(parts.values(),[])].context.isin(['HT29','HT29_IFNG','Jurkat']).any()


def test_low_counts_and_permuted_expression_fail(tmp_path):
    old=dataset(tmp_path/'old');fake_aggregation(old,tmp_path/'cache')
    qc=pd.read_csv(tmp_path/'cache/group_qc.csv');qc.n_matched=1;qc.to_csv(tmp_path/'cache/group_qc.csv',index=False)
    with pytest.raises(ValueError,match='No eligible'):prepare_extension(old,old,tmp_path/'cache',tmp_path/'out',POLICY)
    f=pd.read_csv(tmp_path/'cache/mean.csv.gz');f=f.iloc[::-1];f.to_csv(tmp_path/'cache/mean.csv.gz',index=False)
    with pytest.raises(ValueError,match='order mismatch'):prepare_extension(old,old,tmp_path/'cache',tmp_path/'out',POLICY)


def test_review_archive_hashes_and_exclusions(tmp_path):
    (tmp_path/'mapping').mkdir();(tmp_path/'mapping/cells.csv.gz').write_bytes(b'private cell metadata')
    (tmp_path/'dataset.npz').write_bytes(b'large expression')
    report(tmp_path,{'state':'BLOCKED','reason':'<unsafe>','test_evaluated':False});package(tmp_path)
    assert '&lt;unsafe&gt;' in (tmp_path/'report.html').read_text()
    with tarfile.open(tmp_path/'round15_D1_adapter_review.tar.gz') as t:
        assert not any(n.endswith(('.npz','.csv.gz')) for n in t.getnames())
        manifest=json.load(t.extractfile('ARCHIVE_MANIFEST.json'))
        for name,sha in manifest['files'].items():assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==sha


def test_r_sparse_matched_controls_full_library_and_ht29_exclusion(tmp_path):
    r=shutil.which('Rscript')
    if not r:
        if os.getenv('VCELL_REQUIRE_R_TEST')=='1':pytest.fail('CI requires R integration')
        pytest.skip('Rscript unavailable; set VCELL_REQUIRE_R_TEST=1 to require integration')
    probe=subprocess.run([r,'-e','stopifnot(all(vapply(c("SeuratObject","Matrix","jsonlite"),requireNamespace,logical(1),quietly=TRUE)))'],capture_output=True)
    if probe.returncode:
        if os.getenv('VCELL_REQUIRE_R_TEST')=='1':pytest.fail(probe.stderr.decode(errors='replace'))
        pytest.skip('SeuratObject, Matrix and jsonlite are required for R integration')
    # G2 is outside the panel, so normalization on the panel alone is detectably wrong.
    rows=[];matrix=[]
    for line in ('A549','HT29'):
        for stratum in ('a','b','without_control'):
            for gene in ('NT','P0'):
                if stratum=='without_control' and gene=='NT':continue
                for guide in (1,2):
                    for _ in range(2):
                        rows.append(dict(cell_id='c'+str(len(rows)),cell_type=line,sample=line+'_IFNG',pathway='IFNG',
                                         Batch_info='Rep1',**{'orig.ident':stratum},sample_ID='sample_1',gene=gene,guide=f'{gene}g{guide}'))
                        value=[1,2,7] if gene=='NT' else [4,2,14]
                        if stratum=='b':value=[3,1,16] if gene=='NT' else [5,1,4]
                        if line=='HT29':value=[1000000,999999,1]
                        matrix.append(value)
    zero=next(row.copy() for row in rows if row['cell_type']=='A549' and row['guide']=='P0g1')
    zero['cell_id']='zero_library';rows.append(zero);matrix.append([0,0,0])
    m=pd.DataFrame(rows);m.to_csv(tmp_path/'source_metadata.csv',index=False)
    pd.DataFrame(matrix,columns=['G0','G1','G2'],index=m.cell_id).T.to_csv(tmp_path/'counts.csv')
    build=tmp_path/'fixture.R';build.write_text('''args<-commandArgs(TRUE)
library(SeuratObject);library(Matrix)
m<-read.csv(file.path(args[1],"source_metadata.csv"),colClasses="character",check.names=FALSE);rownames(m)<-m$cell_id
x<-as.matrix(read.csv(file.path(args[1],"counts.csv"),row.names=1,check.names=FALSE))
obj<-CreateSeuratObject(counts=Matrix(x,sparse=TRUE),meta.data=m,min.cells=0,min.features=0)
saveRDS(obj,file.path(args[1],"input.rds"))
''')
    subprocess.run([r,str(build),str(tmp_path)],check=True)
    mapping=tmp_path/'mapping';metadata_plan(m,['G0','G1','G2'],['G1','G0'],mapping)
    out=tmp_path/'agg'
    subprocess.run([r,str(REPO/'scripts/aggregate_round15_mixscale.R'),str(tmp_path/'input.rds'),str(mapping),str(out),'2','3'],check=True)
    groups=pd.read_csv(mapping/'groups.csv');qc=pd.read_csv(out/'group_qc.csv')
    means=pd.read_csv(out/'mean.csv.gz').iloc[:,1:].to_numpy();bases=pd.read_csv(out/'baseline.csv.gz').iloc[:,1:].to_numpy()
    expected=np.mean([np.log1p(np.array([2,4])*10000/20),np.log1p(np.array([1,5])*10000/10)],axis=0)
    base=np.mean([np.log1p(np.array([2,1])*10000/10),np.log1p(np.array([1,3])*10000/20)],axis=0)
    for i in np.flatnonzero(groups.gene.eq('P0')):
        assert qc.iloc[i].n_input==(7 if groups.iloc[i].guide=='P0g1' else 6)
        assert qc.iloc[i].n_positive==6 and qc.iloc[i].n_matched==4
        np.testing.assert_allclose(means[i],expected,rtol=1e-12)
        np.testing.assert_allclose(bases[i],base,rtol=1e-12)
    assert not groups.cell_type.eq('HT29').any()
    assert not pd.read_csv(out/'control_qc.csv').cell_type.eq('HT29').any()


def test_launcher_resume_reuses_complete_export_and_rejects_changed_policy(tmp_path,monkeypatch):
    sys.path.insert(0,str(REPO/'scripts'))
    import adapt_round15_mixscale as runner
    old=dataset(tmp_path/'old');source=tmp_path/'raw';source.mkdir()
    raw=source/'Seurat_object_IFNG_Perturb_seq.rds';raw.write_bytes(b'fixture')
    metadata().to_csv(source/'metadata.csv.gz',index=False)
    (source/'RNA_counts_genes.txt').write_text('\n'.join(old['genes'])+'\n')
    sha=hashlib.sha256(raw.read_bytes()).hexdigest()
    (source/'INSPECT_STATUS.json').write_text(json.dumps({'state':'METADATA_INSPECTED','sha256':sha}))
    r=tmp_path/'Rscript';r.write_text('fake test runtime')
    # Pinned-file checking is independently covered by downloader tests; synthetic fixture substitutes only stat/digest.
    real_stat=Path.stat
    def stat(path,*args,**kwargs):
        result=real_stat(path,*args,**kwargs)
        if path==raw:
            from types import SimpleNamespace
            return SimpleNamespace(st_size=2915636149)
        return result
    monkeypatch.setattr(Path,'stat',stat)
    monkeypatch.setattr(runner,'digest',lambda _: '0fef1f14c36906e9c40e4d1c6aae6926')
    monkeypatch.setattr(runner,'check_resources',lambda *args:None)
    calls=[]
    def export(command,check):
        mapping=Path(command[3]);out=Path(command[4]);calls.append(command)
        fake_aggregation(old,out)
        # Fake R does not rewrite the authoritative mapping.
        assert (mapping/'groups.csv').read_bytes()==(out/'groups.csv').read_bytes()
        (out/'aggregation.json').write_text('{}');(out/'control_qc.csv').write_text('n_control_positive\n40\n')
    monkeypatch.setattr(runner.subprocess,'run',export)
    args=['--work-dir',str(tmp_path),'--input-dir',str(source),'--reference-data',str(tmp_path/'old'),
          '--original-data',str(tmp_path/'old'),'--rscript',str(r)]
    root=runner.main(args);assert len(calls)==1
    before=(root/'COMPLETE.json').read_bytes()
    runner.main(args);assert len(calls)==1
    assert (root/'COMPLETE.json').read_bytes()==before
    with pytest.raises(ValueError,match='Frozen artifact changed'):runner.main(args+['--min-controls','11'])
    assert json.loads((root/'ADAPTER_STATUS.json').read_text())['state']=='ADAPTER_READY'


@pytest.mark.skipif(sys.platform=='win32',reason='Linux shell')
def test_adapter_shell_syntax():
    subprocess.run(['bash','-n',str(REPO/'scripts/adapt_round15_autodl.sh')],check=True)
