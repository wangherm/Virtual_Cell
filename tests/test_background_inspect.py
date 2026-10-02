import argparse
import base64
import hashlib
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from vcell.background_inspect import BUCKET, candidates, download, inspect_h5ad, object_name, run


def uri(name):
    return f'gs://{BUCKET}/scbasecount/2026-01-12/h5ad/Gene/Homo_sapiens/{name}.h5ad'


def metadata():
    return pd.DataFrame([
        dict(srx_accession='SRX1', file_path=uri('SRX1'), cell_line='K562', perturbation='unsure'),
        dict(srx_accession='SRX2', file_path=uri('SRX2'), cell_line='K562, Jurkat', perturbation='none'),
        dict(srx_accession='SRX3', file_path=uri('SRX3'), cell_line='H1', perturbation='untreated'),
        dict(srx_accession='SRX4', file_path=uri('SRX4'), cell_line='RPE-1', perturbation='drug A'),
    ])


def test_candidates_fail_closed():
    x = candidates(metadata(), '2026-01-12', 'Gene').set_index('srx_accession')
    assert x.loc['SRX1', 'control_status'] == 'unverified'
    assert x.loc['SRX2', 'decision'] == 'excluded'
    assert x.loc['SRX3', 'training_admitted'] == False
    assert x.loc['SRX4', 'control_status'] == 'unverified'
    with pytest.raises(ValueError):
        object_name(uri('../escape'), '2026-01-12', 'Gene')
    with pytest.raises(ValueError):
        object_name(uri('x').replace(BUCKET, 'other'), '2026-01-12', 'Gene')


class Remote:
    def __init__(self):
        self.data = b'abc123'*50
        self.starts = []

    def chunk(self, identity, start, end):
        self.starts.append(start)
        return self.data[start:end+1]


def test_resume_checksum_and_identity(tmp_path):
    r = Remote()
    identity = dict(object='x', generation='1', size=len(r.data),
                    md5=base64.b64encode(hashlib.md5(r.data).digest()).decode())
    dest = tmp_path/'x.h5ad'
    (tmp_path/'x.h5ad.part').write_bytes(r.data[:40])
    (tmp_path/'x.h5ad.download.json').write_text(json.dumps({'identity': identity}))
    download(r, identity, dest, reserve_bytes=0)
    assert r.starts == [40] and dest.read_bytes() == r.data
    download(r, identity, dest, reserve_bytes=0)
    assert r.starts == [40]
    with pytest.raises(ValueError, match='identity changed'):
        download(r, {**identity, 'generation': '2'}, dest, reserve_bytes=0)
    dest.write_bytes(b'x'*len(r.data))
    with pytest.raises(ValueError, match='validation'):
        download(r, identity, dest, reserve_bytes=0)


@pytest.mark.parametrize('kind', ['dense','csr','csc'])
def test_h5ad_qc_bounded_and_panel(tmp_path, kind):
    x = np.array([[1.,0,3],[0,2,0],[4,0,0]])
    if kind == 'csr':
        x = sparse.csr_matrix(x)
    if kind == 'csc':
        x = sparse.csc_matrix(x)
    a = ad.AnnData(x, obs=pd.DataFrame({'condition':['control','control','drug']},index=['a','b','c']),
                   var=pd.DataFrame({'gene_symbols':['A','B','B']},index=['g1','g2','g3']))
    path = tmp_path/'x.h5ad'
    a.write_h5ad(path)
    result = inspect_h5ad(path, ['g1','B','MISSING'], sample_rows=2)
    assert result['duplicate_symbols'] == 1
    assert result['panel']['matched'] == 2
    assert result['sampled_values']['noninteger'] == 0
    assert not result['training_admitted']


def args(tmp_path, **extra):
    values = dict(output=str(tmp_path/'out'), billing_project=None, release='2026-01-12', feature='Gene',
        metadata=str(tmp_path/'metadata.csv'), local_dir=str(tmp_path), metadata_only=False,
        panel=None, per_background=2, max_file_gib=5, max_total_gib=20, sample_rows=256)
    values.update(extra)
    return argparse.Namespace(**values)


def test_offline_pipeline_and_heldout_exclusion(tmp_path):
    metadata().to_csv(tmp_path/'metadata.csv', index=False)
    for name in ('SRX1','SRX2','SRX3','SRX4'):
        ad.AnnData(sparse.csr_matrix(np.eye(3))).write_h5ad(tmp_path/f'{name}.h5ad')
    run(args(tmp_path))
    r = json.loads((tmp_path/'out/inspect_report.json').read_text())
    assert r['stage'] == 'inspect_complete'
    assert {x['accession'] for x in r['files']} == {'SRX1','SRX3','SRX4'}
    assert (tmp_path/'out/background_inspect_review.tar.gz').exists()
    run(args(tmp_path))


def test_failure_still_creates_report(tmp_path):
    metadata().to_csv(tmp_path/'metadata.csv', index=False)
    (tmp_path/'SRX1.h5ad').write_text('broken')
    with pytest.raises(RuntimeError):
        run(args(tmp_path))
    r = json.loads((tmp_path/'out/inspect_report.json').read_text())
    assert r['stage'] == 'partial_failure' and r['errors']


def test_metadata_only_and_empty_coverage(tmp_path):
    frame = metadata().iloc[:1].copy()
    frame['cell_line'] = 'unrelated'
    frame.to_csv(tmp_path/'metadata.csv', index=False)
    run(args(tmp_path, metadata_only=True))
    r = json.loads((tmp_path/'out/inspect_report.json').read_text())
    assert r['stage'] == 'metadata_inspected' and r['candidate_rows'] == 0


def test_corrupt_download_never_promoted(tmp_path):
    r = Remote()
    identity = dict(object='x', generation='1', size=len(r.data), md5='bad')
    with pytest.raises(IOError, match='checksum mismatch'):
        download(r, identity, tmp_path/'x', reserve_bytes=0)
    assert not (tmp_path/'x').exists() and (tmp_path/'x.part').exists()


def test_size_cap_and_metadata_mutation(tmp_path):
    metadata().to_csv(tmp_path/'metadata.csv', index=False)
    ad.AnnData(np.eye(3)).write_h5ad(tmp_path/'SRX1.h5ad')
    a = args(tmp_path, max_file_gib=1e-9, max_total_gib=1e-9)
    run(a)
    r = json.loads((tmp_path/'out/inspect_report.json').read_text())
    assert r['stage'] == 'no_files_inspected'
    metadata().iloc[:1].to_csv(tmp_path/'metadata.csv', index=False)
    with pytest.raises(ValueError, match='Metadata contents changed'):
        run(a)


def test_missing_project_writes_blocked_report(tmp_path):
    with pytest.raises(ValueError, match='billing-project'):
        run(args(tmp_path, metadata=None, local_dir=None))
    r = json.loads((tmp_path/'out/inspect_report.json').read_text())
    assert r['stage'] == 'blocked'


def test_parquet_metadata(tmp_path):
    pytest.importorskip('pyarrow')
    metadata().to_parquet(tmp_path/'sample_metadata.parquet', index=False)
    run(args(tmp_path, metadata=str(tmp_path/'sample_metadata.parquet'), metadata_only=True))
    r = json.loads((tmp_path/'out/inspect_report.json').read_text())
    assert r['candidate_rows'] == 3 and r['coverage']['K562'] == 1
