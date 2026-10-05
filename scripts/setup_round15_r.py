"""Repair the dedicated R inspector using one explicit channel, then resume D1 only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import hashlib
import io

from vcell.utils import write_json
import round15_data

CHANNEL='https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge'
PROBE='stopifnot(all(vapply(c("SeuratObject","Matrix","jsonlite"),requireNamespace,logical(1),quietly=TRUE))); print(sessionInfo())'


def choose_prefix(work):
    """Never remove an interrupted/non-Conda directory or alter the base environment."""
    for suffix in ['', '-recovery']+[f'-recovery-{i}' for i in range(2,100)]:
        prefix=Path(work)/'envs'/('round15-r'+suffix)
        if not prefix.exists() or (prefix/'conda-meta/history').is_file():return prefix
    raise RuntimeError('No unused dedicated R prefix available')


def setup_runtime(work,output,runner=None):
    runner=runner or subprocess.run;work=Path(work);output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for key,subdir in [('CONDA_PKGS_DIRS','cache/conda_pkgs'),('TMPDIR','tmp')]:
        value=work/subdir;value.mkdir(parents=True,exist_ok=True);env[key]=str(value)
    env['CONDA_REMOTE_CONNECT_TIMEOUT_SECS']='30';env['CONDA_REMOTE_READ_TIMEOUT_SECS']='120'
    def probe(prefix):
        rscript=prefix/'bin/Rscript'
        if not rscript.is_file():return False
        try:result=runner([str(rscript),'-e',PROBE],env=env,capture_output=True,text=True,timeout=120)
        except (OSError,subprocess.TimeoutExpired):return False
        (output/'r_environment.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
        return result.returncode==0
    # A working dedicated runtime needs no network request, including manual recovery.
    existing=sorted((work/'envs').glob('round15-r*')) if (work/'envs').exists() else []
    for prefix in existing:
        if probe(prefix):
            write_json(output/'R_SETUP.json',{'state':'READY','prefix':str(prefix),'action':'reuse validated runtime'})
            return prefix/'bin/Rscript'
    prefix=choose_prefix(work)
    conda=shutil.which('conda') or (str(Path('/root/miniconda3/bin/conda')) if Path('/root/miniconda3/bin/conda').is_file() else None)
    channel=os.environ.get('VCELL_CONDA_CHANNEL',CHANNEL)
    if not channel.startswith('https://'):raise ValueError('VCELL_CONDA_CHANNEL must be an explicit HTTPS channel URL')
    audit={'state':'BLOCKED','prefix':str(prefix),'channel':channel,'override_channels':True}
    if not conda:
        write_json(output/'R_SETUP.json',{**audit,'reason':'Conda executable unavailable'});return None
    action='install' if (prefix/'conda-meta/history').is_file() else 'create'
    command=[conda,action,'-y','--prefix',str(prefix),'--override-channels','--strict-channel-priority','--channel',channel,
             'r-base>=4.3,<4.6','r-seuratobject>=5,<6','r-matrix','r-jsonlite']
    print('R SETUP:',action,'prefix=',prefix,'channel=',channel,flush=True)
    result=runner(command,env=env,check=False)
    ready=result.returncode==0 and probe(prefix)
    write_json(output/'R_SETUP.json',{**audit,'state':'READY' if ready else 'BLOCKED',
        'action':action,'exit_code':result.returncode,'reason':None if ready else 'Environment solve/install or required-package probe failed; see log'})
    return prefix/'bin/Rscript' if ready else None


def package_inspection(output):
    """Separate D1 artifact; never rewrite a completed training run or its manifests."""
    output=Path(output);entries={}
    with tarfile.open(output/'round15_D1_inspect_review.tar.gz','w:gz') as archive:
        for p in sorted(output.iterdir()):
            if p.is_file() and p.suffix in ('.json','.txt'):
                content=p.read_bytes();member=tarfile.TarInfo(p.name);member.size=len(content)
                archive.addfile(member,io.BytesIO(content));entries[p.name]=hashlib.sha256(content).hexdigest()
        content=json.dumps({'files':entries,'scope':'D1 inspection only; no raw data or training artifacts'},indent=2).encode()
        info=tarfile.TarInfo('INSPECT_ARCHIVE_MANIFEST.json');info.size=len(content);archive.addfile(info,io.BytesIO(content))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',required=True);parser.add_argument('--output',required=True);a=parser.parse_args(argv)
    work=Path(a.work_dir).resolve();output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=True)
    try:
        runtime=setup_runtime(work,output)
        # An explicit absent path prevents falling back to an unrelated system R installation.
        missing=output/'unavailable_runtime'/'Rscript'
        round15_data.main(['--output',str(output),'--download','--rscript',str(runtime or missing)])
        status=json.loads((output/'INSPECT_STATUS.json').read_text())
        if status['state']!='METADATA_INSPECTED':raise RuntimeError('D1 inspection remains '+status['state'])
    finally:
        package_inspection(output)
        print('D1 REVIEW PACKAGE:',output/'round15_D1_inspect_review.tar.gz',flush=True)


if __name__=='__main__':main()
