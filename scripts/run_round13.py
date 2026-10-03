"""Round13 stage A: frozen data/backgrounds, diffusion and dual-path interactions."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import torch

from run_round7 import run_jobs
from run_round9 import package
from vcell.clean_background import run_lock
from vcell.data import load_prepared
from vcell.round11 import freeze, verify_files
from vcell.round12 import save_complete
from vcell.round13 import ARMS, worker, report
from vcell.string_diffusion import prepare
from vcell.train import source_fingerprint
from vcell.utils import file_sha256, write_json

REPO=Path(__file__).resolve().parents[1]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-config')
    p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work')
    p.add_argument('--name',default='round13_01')
    p.add_argument('--round12-run')
    p.add_argument('--diffusion-dir',help='Versioned prepared graph directory; default work/knowledge/round13_string_v12')
    p.add_argument('--seeds',nargs='+',type=int,default=[17,29,43])
    p.add_argument('--protocols',nargs='+',choices=['target','context','double'],default=['target','context','double'])
    p.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS))
    p.add_argument('--max-steps',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--parallel-jobs',type=int,choices=[1,2],default=2)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args(argv)
    if a.worker_config:return worker(json.loads(Path(a.worker_config).read_text()),a.resume)
    if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.name):p.error('Use simple run name')
    if min(a.max_steps,a.batch_size,a.save_every)<1 or min(a.seeds)<0:p.error('Invalid training counts')
    for key in ('seeds','protocols','arms'):
        if len(getattr(a,key))!=len(set(getattr(a,key))):p.error('Duplicate '+key)
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    work=Path(a.work_dir).resolve()
    with run_lock(work/'runs'/('.'+a.name+'.lock')):return launch(a,work,work/'runs'/a.name)


def launch(a,work,root):
    previous=Path(a.round12_run or work/'runs/round12_01').resolve()
    print(f'ROUND13 AUDIT parent checksums: {previous}',flush=True)
    verify_files(previous)
    old_plan=json.loads((previous/'plan.json').read_text())
    if old_plan['experiment']!='round12':raise ValueError('Completed Round12 required')
    if not set(a.seeds)<=set(old_plan['seeds']) or not set(a.protocols)<=set(old_plan['protocols']):raise ValueError('Requested historical folds/seeds unavailable')
    graph=Path(a.diffusion_dir or work/'knowledge/round13_string_v12').resolve()
    plan={'experiment':'round13','source':source_fingerprint(),'launcher':file_sha256(__file__),
        'scheduler':file_sha256(REPO/'scripts/run_round7.py'),'parent':str(previous),
        'parent_complete_sha256':file_sha256(previous/'COMPLETE.json'),
        'graph_dir':str(graph),'graph_release_manifest':file_sha256(REPO/'configs/string_v12_human.json'),
        **{k:getattr(a,k) for k in ('seeds','protocols','arms','max_steps','batch_size','save_every','device')},
        'response_rank':32,'oracle_ranks':[32,64],'learning_rate':.0005,'weight_decay':.01,'factor_width':32,
        'selection':'source calibration only','data_expansion':False,'background_training':False,
        'Qwen_training':False,'test_evaluated':False}
    if root.exists() and any(root.iterdir()) and not a.resume:raise ValueError('Use --resume or new name')
    if not root.exists() and shutil.disk_usage(work).free<5*1024**3:raise ValueError('Need 5 GiB free; no artifacts deleted')
    root.mkdir(parents=True,exist_ok=True);freeze(root/'plan.json',plan)
    if (root/'COMPLETE.json').exists():verify_files(root);print(f'ROUND13 COMPLETE VERIFIED: {root}',flush=True);return root
    def stage(label):
        write_json(root/'stage.json',{'stage':label,'test_evaluated':False});print('ROUND13 STAGE: '+label,flush=True)
    try:
        stage('pinned human STRING and diffusion')
        first=previous/'benchmarks'/f'{a.protocols[0]}_expanded_seed{a.seeds[0]}'
        first_cfg=json.loads((first/'manifest.json').read_text())['config'];data=load_prepared(first_cfg['data_dir'])
        diffusion=prepare(REPO/'configs/string_v12_human.json',graph,data['perturbations'].tolist(),previous/'annotation_coverage.csv')
        for name in ('audit.json','mapping.csv','plan.json'):
            dest=root/'graph_evidence'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(graph/name,dest)
        freeze(root/'graph_identity.json',{'complete_sha256':file_sha256(graph/'COMPLETE.json'),
            'embeddings_sha256':file_sha256(diffusion),'parent_coverage':file_sha256(previous/'annotation_coverage.csv')})
        specs={};jobs={};(root/'configs').mkdir(exist_ok=True)
        for protocol in a.protocols:
            for seed in a.seeds:
                name=f'{protocol}_seed{seed}';parent_worker=previous/'benchmarks'/f'{protocol}_expanded_seed{seed}'
                verify_files(parent_worker)
                pc=json.loads((parent_worker/'manifest.json').read_text())['config']
                if pc['data_dir']!=first_cfg['data_dir']:raise ValueError('Historical workers disagree on data source')
                cfg={k:plan[k] for k in ('max_steps','batch_size','save_every','device','arms')}
                cfg.update(seed=seed,protocol=protocol,parent_worker=str(parent_worker),diffusion=str(diffusion),output_dir=str(root/'models'/name))
                path=root/'configs'/(name+'.json');freeze(path,cfg);specs[name]=cfg
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-config',str(path)]
                if a.resume:cmd.append('--resume')
                jobs[name]={'cmd':cmd,'resource':'gpu','complete':str(Path(cfg['output_dir'])/'COMPLETE.json')}
                evidence=root/'splits'/name;evidence.mkdir(parents=True,exist_ok=True)
                freeze(evidence/'partition.json',pc['partition'])
                meta=data['meta'].copy();meta['role']='excluded'
                for label,indices in pc['partition'].items():meta.loc[indices,'role']=label
                meta.to_csv(evidence/'membership.csv',index=False)
        write_json(root/'jobs.json',jobs)
        write_json(root/'budget.json',{'jobs':len(jobs),'new_heads':len(jobs)*len(a.arms),
            'factor_heads':len(jobs)*sum(ARMS[k][1]=='factor' for k in a.arms),'max_steps':a.max_steps,
            'batch_size':a.batch_size,'parallel_jobs':a.parallel_jobs,'threads_per_worker':2,
            'background_jobs':0,'Qwen_backbone_jobs':0,'parent_predictions_reused':True})
        print(f'ROUND13 READY: {len(jobs)} jobs; {len(jobs)*len(a.arms)} small heads; no background/Qwen retraining',flush=True)
        if a.prepare_only:stage('prepared; training pending');return root
        stage('small interaction models');result=run_jobs(jobs,root,a.parallel_jobs,False,label='ROUND13')
        if any(result.values()):raise RuntimeError('Worker failed; inspect logs and resume identical plan')
        stage('paired and specificity reports');report(specs,root)
        (root/'INCOMPLETE.json').unlink(missing_ok=True);stage('complete')
        save_complete(root,{'test_evaluated':False,'jobs':len(jobs),'new_heads':len(jobs)*len(a.arms)})
        print(f'ROUND13 COMPLETE: {root/"comparison_mean.csv"}',flush=True)
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error':str(exc),'test_evaluated':False});raise
    finally:
        package(root,'round13');print(f'REVIEW PACKAGE: {root/"round13_review_light.tar.gz"}',flush=True)
    return root


if __name__=='__main__':main()
