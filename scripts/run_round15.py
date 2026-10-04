"""Round15 core screen and parallel Mixscale inspection; later stages are explicit gates."""
import argparse
from collections import deque
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time

import torch

from run_round7 import run_jobs
from vcell.clean_background import run_lock
from vcell.round11 import freeze, verify_files
from vcell.round15 import worker, report, seal, verify_marker
from vcell.round15_heads import ARMS, VARIANTS
from vcell.train import source_fingerprint
from vcell.utils import file_sha256, write_json

REPO=Path(__file__).resolve().parents[1]


def package(root):
    """Every packaged byte has its own hash, including truncated log tails."""
    entries={}
    with tarfile.open(root/'round15_review_light.tar.gz','w:gz') as archive:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix not in ('.json','.csv','.html','.log','.txt'):continue
            if 'inputs' in path.relative_to(root).parts:continue
            if path.suffix=='.log':
                with path.open(encoding='utf-8',errors='replace') as stream:content=''.join(deque(stream,maxlen=80)).encode()
            else:content=path.read_bytes()
            name=path.relative_to(root).as_posix();member=tarfile.TarInfo(name);member.size=len(content)
            archive.addfile(member,io.BytesIO(content))
            entries[name]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content),
                           'kind':'log_tail' if path.suffix=='.log' else 'full_file'}
        content=json.dumps({'files':entries,'note':'Archive bytes; full-server manifests remain separate.'},indent=2).encode()
        member=tarfile.TarInfo('ARCHIVE_MANIFEST.json');member.size=len(content);archive.addfile(member,io.BytesIO(content))


def hardware(work):
    result={'platform':platform.platform(),'python':sys.version,'cpu_count':os.cpu_count(),
            'disk_free_bytes':shutil.disk_usage(work).free,'torch':torch.__version__,'cuda':torch.version.cuda,
            'gpu_count':torch.cuda.device_count(),'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    for name,command in [('nvidia_smi',['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv']),
                          ('memory',['free','-b']),('git_head',['git','-C',str(REPO),'rev-parse','HEAD'])]:
        try:result[name]=subprocess.run(command,capture_output=True,text=True,timeout=10).stdout.strip()
        except (OSError,subprocess.TimeoutExpired):result[name]='unavailable'
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-config');p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work')
    p.add_argument('--name',default='round15_01');p.add_argument('--round14-run')
    p.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS))
    p.add_argument('--variants',nargs='*',choices=VARIANTS,default=list(VARIANTS))
    p.add_argument('--seeds',nargs='+',type=int,default=[17,29,43])
    p.add_argument('--protocols',nargs='+',choices=['target','context','double'],default=['target','context','double'])
    p.add_argument('--max-steps',type=int,default=2000);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--save-every',type=int,default=100);p.add_argument('--parallel-jobs',type=int,choices=[1,2,3,4],default=3)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--resume',action='store_true')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--skip-data',action='store_true')
    a=p.parse_args(argv)
    if a.worker_config:return worker(json.loads(Path(a.worker_config).read_text()),a.resume)
    if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.name):p.error('Use simple run name')
    if min(a.max_steps,a.batch_size,a.save_every)<1 or min(a.seeds)<0:p.error('Invalid counts')
    for k in ('arms','variants','seeds','protocols'):
        if len(getattr(a,k))!=len(set(getattr(a,k))):p.error('Duplicate '+k)
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    work=Path(a.work_dir).resolve();work.mkdir(parents=True,exist_ok=True)
    with run_lock(work/'runs'/('.'+a.name+'.lock')):return launch(a,work,work/'runs'/a.name)


def launch(a,work,root):
    previous=Path(a.round14_run or work/'runs/round14_01').resolve()
    print(f'ROUND15 AUDIT full Round14 parent: {previous}',flush=True);verify_files(previous)
    old=json.loads((previous/'plan.json').read_text())
    if old['experiment']!='round14':raise ValueError('Full completed Round14 required')
    if not set(a.seeds)<=set(old['seeds']) or not set(a.protocols)<=set(old['protocols']):raise ValueError('Historical folds/seeds unavailable')
    plan={'experiment':'round15_core','source':source_fingerprint(),'launcher':file_sha256(__file__),
          'parent':str(previous),'parent_sha256':file_sha256(previous/'COMPLETE.json'),
          **{k:getattr(a,k) for k in ('arms','variants','seeds','protocols','max_steps','batch_size','save_every','device','skip_data')},
          'training_parent':'local_factor_noid','selection':'predeclared from Round14 development; no automatic outer winner',
          'data_registry':file_sha256(REPO/'configs/round15_mixscale.json'),
          'test_v2':'HT29 reserved; metadata only; not scored','test_evaluated':False}
    if root.exists() and any(root.iterdir()) and not a.resume:raise ValueError('Use --resume or new name')
    if not root.exists() and shutil.disk_usage(work).free<10*1024**3:raise ValueError('Need 10 GiB core reserve; data download checks its own reserve')
    root.mkdir(parents=True,exist_ok=True);freeze(root/'plan.json',plan)
    if (root/'CORE_COMPLETE.json').exists():
        verify_marker(root,'CORE_COMPLETE.json');package(root);print('ROUND15 CORE VERIFIED',root);return root
    def stage(label):
        write_json(root/'stage.json',{'stage':label,'scope':'core screen; not full Round15','test_evaluated':False});print('ROUND15 STAGE: '+label,flush=True)
    started=time.perf_counter()
    try:
        stage('audit and queue preparation');write_json(root/'hardware.json',hardware(work))
        specs={};jobs={};(root/'configs').mkdir(exist_ok=True)
        for seed in a.seeds:
            for protocol in a.protocols:
                name=f'{protocol}_seed{seed}';parent_worker=previous/'models'/name;verify_files(parent_worker)
                cfg={k:plan[k] for k in ('arms','variants','max_steps','batch_size','save_every','device','training_parent')}
                cfg.update(seed=seed,protocol=protocol,parent_worker=str(parent_worker),output_dir=str(root/'models'/name))
                path=root/'configs'/(name+'.json');freeze(path,cfg);specs[name]=cfg
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-config',str(path)]
                if a.resume:command.append('--resume')
                jobs[name]={'cmd':command,'resource':'gpu','complete':str(Path(cfg['output_dir'])/'SCREEN_COMPLETE.json')}
                evidence=root/'splits'/name;evidence.mkdir(parents=True,exist_ok=True)
                for file in ('partition.json','membership.csv'):
                    source=previous/'splits'/name/file
                    if source.exists():shutil.copyfile(source,evidence/file)
        data_root=work/'raw/round15_D1/IFNG'
        if not a.skip_data:
            jobs['D1_inspect']={'cmd':['bash',str(REPO/'scripts/run_round15_mixscale.sh'),str(data_root),str(work)],
                'resource':'cpu','complete':str(data_root/'INSPECT_STATUS.json')}
        write_json(root/'jobs.json',jobs)
        write_json(root/'budget.json',{'screen_slots':len(specs)*(len(a.arms)+len(a.variants)),
            'P0_is_included_in_P1':True,'slots_are_not_executed_fits':True,'actual_counts_file':'execution_status.csv',
            'parallel_small_workers':a.parallel_jobs,'cpu_threads_per_worker':2,'native_gpu_workers':0,
            'plan_upper_bounds':{'full_training':675,'adaptation':180,'closed_form':36},
            'implemented_scope':'P0/P1/P2 available heads; D1 metadata inspection; remaining stages not executed'})
        gates={
            'P3':{'status':'WAITING_CANDIDATE_FREEZE','implemented':False,'required':'At most four candidates, then new split-specific B2/basis rebuilding and confirmation launcher'},
            'P4':{'status':'WAITING_D1_QC','implemented':False,'required':'Validated sparse cell data adapter, explicit conditions, matched controls and locked test-v2'},
            'P5':{'status':'ADAPTERS_NOT_IMPLEMENTED','implemented':False,'required':'Pinned native model adapters, audited single-cell splits, output alignment and license/use review'},
            'P6':{'status':'WAITING_SOURCE_OOF_TEACHER_EVIDENCE','implemented':False,'required':'Native source-OOF predictions that correct strong baseline residuals'},
            'FS':{'status':'WAITING_FROZEN_BASE_AND_SUPPORT_SPLITS','implemented':False,'required':'Fixed model, disjoint support/query targets and sufficient support pool'},
            'closed_form':{'status':'PLANNED_NOT_IMPLEMENTED','implemented':False,'required':'Explicit structured-baseline definitions and shared panel'},
            'test_v2':{'status':'RESERVED_NOT_EVALUATED','cell_line':'HT29','automatic_open':False}}
        write_json(root/'stage_readiness.json',gates)
        if a.prepare_only:stage('prepared; no training or downloads started');return root
        stage('P0/P1/P2 core screen; D1 inspection in parallel')
        result=run_jobs(jobs,root,a.parallel_jobs,False,label='ROUND15')
        if not a.skip_data:
            evidence=root/'D1';evidence.mkdir(exist_ok=True)
            for name in ('INSPECT_STATUS.json','test_v2_reservation.json','download_manifest.json','rds_inspect.json','r_environment.txt'):
                if (data_root/name).exists():shutil.copyfile(data_root/name,evidence/name)
        failed={k:v for k,v in result.items() if v}
        if failed:raise RuntimeError('Jobs failed; preserve completed heads and resume: '+str(failed))
        stage('core reports with completed/reused/blocked accounting');report(specs,root)
        write_json(root/'wall_time.json',{'orchestrator_wall_seconds':time.perf_counter()-started,'includes_audit_and_download':True})
        (root/'INCOMPLETE.json').unlink(missing_ok=True);stage('core_complete; later stages gated')
        seal(root,'CORE_COMPLETE.json',core_complete=True,full_round15_complete=False,test_evaluated=False)
        print('ROUND15 CORE COMPLETE:',root/'comparison_mean.csv',flush=True)
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error':str(exc),'test_evaluated':False});raise
    finally:
        package(root);print('REVIEW PACKAGE:',root/'round15_review_light.tar.gz',flush=True)
    return root


if __name__=='__main__':main()
