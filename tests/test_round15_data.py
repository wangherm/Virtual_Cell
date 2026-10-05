"""Data acquisition is pinned, resumable and never implicitly training-ready."""
import hashlib
import json
from pathlib import Path
import sys
import subprocess

import pytest
import requests

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import round15_data as data


@pytest.mark.skipif(sys.platform=='win32',reason='Server shell syntax is checked on Linux CI')
def test_server_shell_syntax():
    repo=Path(__file__).resolve().parents[1]
    subprocess.run(['bash','-n',str(repo/'scripts/run_round15_autodl.sh'),str(repo/'scripts/run_round15_mixscale.sh'),str(repo/'scripts/resume_round15_inspect.sh')],check=True)


def test_interrupted_download_range_and_corrupt_file(tmp_path,monkeypatch):
    content=b'abcdef';record=dict(name='test.rds',url='https://example.invalid/test',bytes=6,md5=hashlib.md5(content).hexdigest());calls=[]
    class Response:
        def __init__(self,offset):self.offset=offset;self.status_code=206 if offset else 200;self.headers={'Content-Range':f'bytes {offset}-5/6'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,size):
            if len(calls)==1:
                yield content[:3];raise requests.ConnectionError('interrupted')
            yield content[self.offset:]
    def get(url,headers,**kwargs):calls.append(headers);return Response(3 if 'Range' in headers else 0)
    monkeypatch.setattr(data.requests,'get',get);monkeypatch.setattr(data.time,'sleep',lambda _:None)
    path=data.download(record,tmp_path);assert path.read_bytes()==content and calls[1]['Range']=='bytes=3-'
    assert data.download(record,tmp_path)==path
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):data.download(record,tmp_path)


def test_metadata_only_reservation_precedes_download(tmp_path,monkeypatch):
    data.main(['--output',str(tmp_path)])
    reserved=json.loads((tmp_path/'test_v2_reservation.json').read_text())
    assert reserved['cell_line']=='HT29' and reserved['all_stimuli'] and not reserved['automatic_substitution']
    status=json.loads((tmp_path/'INSPECT_STATUS.json').read_text())
    assert status['state']=='PLANNED' and not status['training_ready'] and not status['response_analysis']
    assert not (tmp_path/'COMPLETE.json').exists()
    fake=tmp_path/'fake.rds';fake.write_bytes(b'abc')
    def acquire(record,root):
        assert (root/'test_v2_reservation.json').exists()
        return fake
    monkeypatch.setattr(data,'download',acquire);monkeypatch.setattr(data.shutil,'which',lambda _:None)
    data.main(['--output',str(tmp_path),'--download'])
    assert json.loads((tmp_path/'INSPECT_STATUS.json').read_text())['state']=='BLOCKED_R_RUNTIME'
