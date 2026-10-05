"""Mirror-only environment repair without touching training artifacts or failed prefixes."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import setup_round15_r as repair


@pytest.mark.parametrize('partial',[False,True])
def test_mirror_only_repair_and_no_repeat_install(tmp_path,monkeypatch,partial):
    work=tmp_path/'work';output=work/'raw';old=work/'envs/round15-r'
    if partial:old.mkdir(parents=True);(old/'keep.txt').write_text('original')
    monkeypatch.setattr(repair.shutil,'which',lambda _: '/fake/conda');monkeypatch.delenv('VCELL_CONDA_CHANNEL',raising=False)
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        if command[0]=='/fake/conda':
            assert '--override-channels' in command and '--strict-channel-priority' in command
            assert command[command.index('--channel')+1]==repair.CHANNEL
            assert 'defaults' not in command
            assert str(work) in kwargs['env']['CONDA_PKGS_DIRS']
            prefix=Path(command[command.index('--prefix')+1]);(prefix/'bin').mkdir(parents=True)
            (prefix/'bin/Rscript').write_text('fake')
        return SimpleNamespace(returncode=0,stdout='R packages loaded',stderr='')
    runtime=repair.setup_runtime(work,output,run)
    assert runtime.is_file() and len(calls)==2
    if partial:assert (old/'keep.txt').read_text()=='original' and runtime.parent.parent.name=='round15-r-recovery'
    calls.clear();assert repair.setup_runtime(work,output,run)==runtime
    assert len(calls)==1  # Only the package probe, no Conda call.


def test_managed_incomplete_environment_uses_install(tmp_path,monkeypatch):
    prefix=tmp_path/'envs/round15-r';(prefix/'conda-meta').mkdir(parents=True);(prefix/'conda-meta/history').write_text('existing')
    monkeypatch.setattr(repair.shutil,'which',lambda _: '/fake/conda')
    def failed(command,**kwargs):
        assert command[1]=='install';return SimpleNamespace(returncode=1)
    assert repair.setup_runtime(tmp_path,tmp_path/'output',failed) is None
    assert json.loads((tmp_path/'output/R_SETUP.json').read_text())['state']=='BLOCKED'
    assert (prefix/'conda-meta/history').read_text()=='existing'


def test_separate_inspection_package_excludes_raw_and_training(tmp_path):
    (tmp_path/'INSPECT_STATUS.json').write_text('{"state":"METADATA_INSPECTED"}')
    (tmp_path/'data.rds').write_bytes(b'large data')
    (tmp_path/'metadata.csv.gz').write_bytes(b'cell metadata')
    repair.package_inspection(tmp_path)
    with tarfile.open(tmp_path/'round15_D1_inspect_review.tar.gz') as archive:
        manifest=json.load(archive.extractfile('INSPECT_ARCHIVE_MANIFEST.json'))
        assert set(manifest['files'])=={'INSPECT_STATUS.json'}
        for name,digest in manifest['files'].items():assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest


def test_blocked_inspection_has_nonzero_outcome_and_still_packages(tmp_path,monkeypatch):
    monkeypatch.setattr(repair,'setup_runtime',lambda *args:None)
    def inspect(args):
        assert '--rscript' in args
        (tmp_path/'INSPECT_STATUS.json').write_text('{"state":"BLOCKED_R_RUNTIME"}')
    monkeypatch.setattr(repair.round15_data,'main',inspect)
    with pytest.raises(RuntimeError,match='BLOCKED_R_RUNTIME'):
        repair.main(['--work-dir',str(tmp_path/'work'),'--output',str(tmp_path)])
    assert (tmp_path/'round15_D1_inspect_review.tar.gz').exists()
