import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import requests
from scipy import sparse

from vcell.background_inspect import write_json
from vcell.public_background import annotations, identity, inspect_atlas, run, selection, transfer


class Response:
    def __init__(self, data=b'', status=206, headers=None, fail=False):
        self.data,self.status_code,self.headers,self.fail = data,status,headers or {},fail
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def raise_for_status(self):
        if self.status_code>=400:raise requests.HTTPError('failed')
    def iter_content(self,n):
        yield self.data[:5]
        if self.fail:raise requests.ConnectionError('interrupted')
        yield self.data[5:]


class Session:
    def __init__(self, mode='normal'):
        self.data=b'\x89HDF\r\n\x1a\n'+b'abc'*30
        self.mode=mode; self.calls=[]
    def head(self,*args,**kw):
        return Response(status=200,headers={'Content-Length':str(len(self.data)),'ETag':'"fixed"'})
    def get(self,url,headers,**kw):
        start,end=map(int,headers['Range'][6:].split('-'));self.calls.append(start)
        return Response(self.data[start:end+1],status=200 if self.mode=='ignored' else 206,
            headers={'Content-Range':f'bytes {start}-{end}/{len(self.data)}',
                     'ETag':'"changed"' if self.mode=='changed' else '"fixed"'},
            fail=self.mode=='interrupt_once' and len(self.calls)==1)


def test_resume_and_retry_no_duplicate_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr('vcell.public_background.time.sleep',lambda _:None)
    session=Session('interrupt_once')
    ident=identity(session,{'url':'https://example.invalid','size_bytes':len(session.data)})
    dest=tmp_path/'x.h5ad'
    dest.with_suffix('.h5ad.part').write_bytes(session.data[:12])
    write_json(dest.with_suffix('.download.json'),{'identity':ident})
    record=transfer(session,ident,dest,reserve=0,chunk_size=20)
    assert dest.read_bytes()==session.data and session.calls[:2]==[12,12]
    assert record['sha256']
    before=len(session.calls);transfer(session,ident,dest,reserve=0)
    assert len(session.calls)==before


@pytest.mark.parametrize('mode',['ignored','changed'])
def test_bad_range_never_appended(tmp_path,monkeypatch,mode):
    monkeypatch.setattr('vcell.public_background.time.sleep',lambda _:None)
    s=Session(mode);ident=identity(s,{'url':'https://example.invalid','size_bytes':len(s.data)})
    dest=tmp_path/'x.h5ad'
    with pytest.raises(ValueError):transfer(s,ident,dest,reserve=0)
    assert not dest.exists() and dest.with_suffix('.h5ad.part').stat().st_size==0


def fixture(path):
    a=ad.AnnData(sparse.csr_matrix([[1.,0,3],[2,1,0],[0,2,1]]),
        obs=pd.DataFrame({'donor_id':['A','A','B'],'assay':pd.Categorical(['10x 3\' v3','Smart-seq2','10x 3\' v3']),
                          'is_primary_data':[True,True,False]},index=['a','b','c']),
        var=pd.DataFrame(index=['g1','g2','g3']))
    a.raw=a.copy()
    a=a[:,:2].copy()
    a.layers['raw_counts']=a.X.copy()
    a.X=a.X*.25
    a.write_h5ad(path)


def test_raw_axis_and_count_layers(tmp_path):
    path=tmp_path/'x.h5ad';fixture(path)
    r=inspect_atlas(path,['g3'])
    matrices={x['matrix_path']:x for x in r['matrices']}
    assert matrices['X']['n_genes']==2 and matrices['raw/X']['n_genes']==3
    assert matrices['X']['panel']['matched']==0 and matrices['raw/X']['panel']['matched']==1
    assert matrices['layers/raw_counts']['sampled_values']['noninteger']==0
    assert matrices['X']['sampled_values']['noninteger']>0
    assert r['full_annotation_counts']['assay']['cells_labelled_10x']==2
    assert not r['training_admitted']


def test_manifest_and_offline_report(tmp_path):
    manifest=Path(__file__).resolve().parents[1]/'configs/public_backgrounds.json'
    data=json.loads(manifest.read_text())
    assert len(selection(data,['ts_blood','hlca_core']))==2
    with pytest.raises(ValueError):selection(data,['ts_blood','ts_blood'])
    a=argparse.Namespace(manifest=str(manifest),datasets=['ts_blood'],output=str(tmp_path),
        panel=None,max_total_gib=20,sample_rows=8,catalog_only=True)
    run(a)
    r=json.loads((tmp_path/'inspect_report.json').read_text())
    assert r['stage']=='catalog_only' and r['datasets'][0]['status']=='pending'
    assert (tmp_path/'public_background_review.tar.gz').exists()


def test_complete_pipeline_with_mock_transfer(tmp_path,monkeypatch):
    manifest=Path(__file__).resolve().parents[1]/'configs/public_backgrounds.json'
    def download(session,ident,dest):
        dest.parent.mkdir(parents=True,exist_ok=True);fixture(dest);return {'test_fixture':True}
    monkeypatch.setattr('vcell.public_background.identity',lambda session,item:{})
    monkeypatch.setattr('vcell.public_background.transfer',download)
    a=argparse.Namespace(manifest=str(manifest),datasets=['ts_blood'],output=str(tmp_path),
        panel=None,max_total_gib=20,sample_rows=8,catalog_only=False)
    run(a)
    r=json.loads((tmp_path/'inspect_report.json').read_text())
    assert r['stage']=='inspect_complete'
    assert r['datasets'][0]['status']=='inspected_quarantine'
