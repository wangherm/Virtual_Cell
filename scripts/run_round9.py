"""Full background/module comparisons on existing prepared perturbation data."""
import argparse
from collections import deque
import copy
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile

import numpy as np
import torch

from run_round7 import run_jobs
from vcell.background_training import prepare_background, pretrain
from vcell.clean_background import run_lock
from vcell.data import load_prepared
from vcell.knowledge import load_knowledge
from vcell.qwen import backbone_identity
from vcell.round8 import partitions, worker
from vcell.round9 import arm_matrix, diagnostics, report
from vcell.train import source_fingerprint
from vcell.utils import file_sha256, read_config, write_json

REPO=Path(__file__).resolve().parents[1]


def package(root,experiment='round9'):
    with tarfile.open(root/(experiment+'_review_light.tar.gz'),'w:gz') as tar:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or 'background_cache' in path.relative_to(root).parts:continue
            if path.suffix in ('.json','.csv','.html'):
                tar.add(path,arcname=path.relative_to(root).as_posix())
            elif path.suffix=='.log':
                with path.open(encoding='utf-8',errors='replace') as stream:text=''.join(deque(stream,maxlen=60)).encode()
                info=tarfile.TarInfo(path.relative_to(root).as_posix());info.size=len(text);tar.addfile(info,io.BytesIO(text))


def main(argv=None,experiment='round9'):
    if experiment not in ('round9','round10'):raise ValueError('Unknown experiment')
    if experiment=='round10':
        from vcell.round10 import arm_matrix as experiment_matrix, PROTOCOLS
    else:
        experiment_matrix=arm_matrix;PROTOCOLS=('context','target','double')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-config')
    p.add_argument('--work-dir',default=os.environ.get('VCELL_WORK','/root/autodl-tmp/vcell-work'))
    p.add_argument('--name',default=experiment+'_01')
    p.add_argument('--data',help='Existing Round8 plus_h1 prepared directory')
    p.add_argument('--background-dir')
    p.add_argument('--knowledge-npz')
    p.add_argument('--model-dir')
    p.add_argument('--student-config',default=str(REPO/'configs/qwen_three_teachers.yaml'))
    choices=['_'.join(a) for a in experiment_matrix()]
    p.add_argument('--arms',nargs='+',choices=choices,default=choices)
    p.add_argument('--protocols',nargs='+',choices=PROTOCOLS,default=list(PROTOCOLS) if experiment=='round10' else ['context','target'])
    if experiment=='round10':p.add_argument('--module-warmup-steps',type=int,default=500)
    p.add_argument('--seeds',nargs='+',type=int,default=[17,29,43])
    p.add_argument('--max-steps',type=int,default=2000)
    p.add_argument('--pretrain-epochs',type=int,default=10,help='B2 public-cell passes; determines equal B1/B2 updates')
    p.add_argument('--pretrain-steps',type=int,help='Explicit fixed update override for infrastructure tests')
    p.add_argument('--pretrain-batch',type=int,default=256)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--gradient-accumulation',type=int,default=1)
    p.add_argument('--parallel-students',type=int,choices=[1,2],default=2)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args(argv)
    if a.worker_config:
        cfg=read_config(a.worker_config)
        return pretrain(cfg,a.resume) if cfg.get('stage')=='background' else worker(cfg,a.resume)
    if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.name):p.error('Use a simple run name')
    if min(a.max_steps,a.pretrain_epochs,a.save_every,a.batch_size,a.gradient_accumulation)<1 or a.pretrain_batch<2 or min(a.seeds)<0:
        p.error('Invalid training counts')
    if a.pretrain_steps is not None and a.pretrain_steps<1:p.error('pretrain-steps must be positive')
    if any(len(v)!=len(set(v)) for v in (a.arms,a.protocols,a.seeds)):p.error('Duplicate arms/protocols/seeds')
    if experiment=='round10' and a.module_warmup_steps<1:p.error('Module warmup must be positive')
    with run_lock(Path(a.work_dir)/'runs'/('.'+a.name+'.lock')):return launch(a,experiment)


def launch(a,experiment='round9'):
    if experiment=='round10':
        from vcell.round10 import arm_matrix as experiment_matrix, arm_allowed, make_partition, partition_audit, prepare_aggregates, report as experiment_report
    else:
        experiment_matrix=arm_matrix;arm_allowed=lambda protocol,mode:True
        make_partition=partitions;experiment_report=report
    label=experiment.upper()
    if a.device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):raise ValueError('CUDA/BF16 unavailable')
    torch.set_num_threads(2)
    work=Path(a.work_dir).resolve();root=work/'runs'/a.name
    data_path=Path(a.data or work/'runs/round8_01/prepared/plus_h1').resolve()
    background=Path(a.background_dir or work/'background/human_background_v1').resolve()
    knowledge=Path(a.knowledge_npz or work/'knowledge/round8/knowledge.npz').resolve()
    data=load_prepared(data_path);assets=load_knowledge(knowledge,data)
    model=read_config(a.student_config)['model']
    model.update(model_id=str(Path(a.model_dir or work/'models/Qwen3-0.6B-Base').resolve()),revision=None,
                 local_files_only=True,dtype='bfloat16' if a.device=='cuda' else 'float32',gradient_checkpointing=False)
    for k in ('functional','background_interaction','response_rank'):model.pop(k,None)
    matrix={ '_'.join(v):v for v in experiment_matrix()}
    plan={'experiment':experiment,'source':source_fingerprint(),'launcher':file_sha256(__file__),
        'scheduler':file_sha256(REPO/'scripts/run_round7.py'),'data_path':str(data_path),'data':data['audit']['fingerprint'],
        'background_path':str(background),'background_report':file_sha256(background/'report.json'),
        'background_config':file_sha256(background/'run_config.json'),'knowledge':str(knowledge),
        'knowledge_sha256':file_sha256(knowledge),'model':model,'backbone':backbone_identity(model),
        **{k:getattr(a,k) for k in ('arms','protocols','seeds','max_steps','pretrain_epochs','pretrain_steps','pretrain_batch',
                                  'save_every','batch_size','gradient_accumulation')},'test_evaluated':False}
    if experiment=='round10':
        plan.update(module_warmup_steps=a.module_warmup_steps,entrypoint=file_sha256(REPO/'scripts/run_round10.py'))
    if root.exists() and any(root.iterdir()):
        if not a.resume or not (root/'plan.json').exists() or json.loads((root/'plan.json').read_text())!=plan:
            raise ValueError(f'Resume identical {label} plan or use a new name')
    students=sum(arm_allowed(p,matrix[arm][2]) for p in a.protocols for arm in a.arms)*len(a.seeds)
    if not students:raise ValueError('No allowed arms for the requested protocols')
    reserve=max(12,students*.6+5)*1024**3
    if shutil.disk_usage(work).free<reserve:raise ValueError(f'Need {reserve/1024**3:.1f} GiB free reserve; no files deleted')
    root.mkdir(parents=True,exist_ok=True);write_json(root/'plan.json',plan)
    (root/'COMPLETE.json').unlink(missing_ok=True)
    specs={}
    try:
        write_json(root/'stage.json',{'stage':'verify/cache backgrounds','test_evaluated':False})
        cache=prepare_background(background,root/'background_cache',data['genes'].tolist())
        write_json(root/'background_audit.json',json.loads((cache/'audit.json').read_text()))
        n_public=len(np.load(cache/'train.npy',mmap_mode='r'))
        bg_steps=a.pretrain_steps or math.ceil(n_public/(a.pretrain_batch//2))*a.pretrain_epochs
        aggregate_dir=None
        if experiment=='round10':
            if data['audit'].get('normalization')!='mean(log1p(10000*counts/full_library))' or data['audit'].get('target_sum')!=10000:
                raise ValueError('B3 requires the existing mean-of-log, full-library normalization')
            aggregate_dir=prepare_aggregates(cache,root/'background_aggregates')
        write_json(root/'budget.json',{'students':students,'student_optimizer_steps':a.max_steps,
            'student_effective_batch':a.batch_size*a.gradient_accumulation,'background_optimizer_steps':bg_steps,
            'background_batch':a.pretrain_batch,'public_training_cells':n_public,'test_evaluated':False})
        (root/'configs').mkdir(exist_ok=True)
        bgjobs={};jobs={};splits={}
        for protocol in a.protocols:
            split=make_partition(data,protocol);splits[protocol]=split
            folder=root/'splits'/protocol;folder.mkdir(parents=True,exist_ok=True)
            write_json(folder/'partition.json',split)
            meta=data['meta'].copy();meta[experiment+'_partition']='excluded'
            for partition_name,ix in split.items():meta.loc[ix,experiment+'_partition']=partition_name
            meta.to_csv(folder/'membership.csv',index=False)
            if experiment=='round10':partition_audit(data,split,folder)
            else:
                print(f'{label} DIAGNOSTICS: {protocol} basis oracle and representation probes',flush=True)
                diagnostics(data,assets,split,root/'diagnostics'/protocol)
            for seed in a.seeds:
                for arm in a.arms:
                    family,bg,mode=matrix[arm];name=f'{protocol}_{arm}_seed{seed}'
                    if not arm_allowed(protocol,mode):continue
                    encoder=None
                    if bg!='b0':
                        bn=f'{protocol}_{family}_{bg}_seed{seed}'
                        encoder=str(root/'background_pretraining'/bn/'encoder.pt')
                        if bn not in bgjobs:
                            bc={'stage':'background','data_dir':str(data_path),'partition':split,'background_cache':str(cache),
                                'background':bg,'family':family,'model':copy.deepcopy(model),'seed':seed,'device':a.device,
                                'pretrain_steps':bg_steps,'pretrain_batch':a.pretrain_batch,'save_every':a.save_every,
                                'output_dir':str(root/'background_pretraining'/bn)}
                            if aggregate_dir is not None:bc['aggregate_dir']=str(aggregate_dir)
                            path=root/'configs'/('background_'+bn+'.json');write_json(path,bc)
                            bgjobs[bn]=job(path,bc,a.resume)
                    cfg={'experiment':experiment,'data_dir':str(data_path),'knowledge':str(knowledge),'partition':split,
                        'protocol':protocol,'arm':arm,'family':family,'background':bg,'module_mode':mode.removesuffix('_warmup'),
                        'module_permutation_seed':818,'seed':seed,'model':copy.deepcopy(model),'device':a.device,
                        'rank':32,'learning_rate':.0002,'module_weight':.1,'kd_weight':0.,'teacher_cache':None,
                        'encoder':encoder,'diagnostic_every':250,'max_steps':a.max_steps,'save_every':a.save_every,
                        'batch_size':a.batch_size,'gradient_accumulation':a.gradient_accumulation,
                        'output_dir':str(root/'students'/name)}
                    if experiment=='round10':cfg['module_warmup_steps']=a.module_warmup_steps if mode.endswith('_warmup') else 0
                    path=root/'configs'/(name+'.json');write_json(path,cfg)
                    specs[name]=cfg;jobs[name]=job(path,cfg,a.resume)
        # Warmup diagnostics follow the core comparisons in the scheduler queue.
        jobs=dict(sorted(jobs.items(),key=lambda item:specs[item[0]].get('module_warmup_steps',0)>0))
        write_json(root/'background_jobs.json',bgjobs);write_json(root/'jobs.json',jobs)
        print(f'{label} READY: background jobs={len(bgjobs)}, students={len(jobs)}, parallel={a.parallel_students}',flush=True)
        if a.prepare_only:return root
        for stage,queue in [('background pretraining',bgjobs),('students',jobs)]:
            write_json(root/'stage.json',{'stage':stage,'jobs':len(queue),'test_evaluated':False})
            # Enter workers even on resume so completed artifact checksums are verified.
            finished=run_jobs(queue,root,a.parallel_students,False,label=label)
            failures=[n for n,code in finished.items() if code]
            if stage=='students':experiment_report(data,specs,root)
            if failures:raise RuntimeError(f'Failed {stage}: {failures}; inspect logs and resume')
        write_json(root/'COMPLETE.json',{'students':len(jobs),'background_jobs':len(bgjobs),'test_evaluated':False,
            'note':'Complete development comparison; no automatic winner or test release.'})
        write_json(root/'stage.json',{'stage':'complete','test_evaluated':False})
        (root/'INCOMPLETE.json').unlink(missing_ok=True)
        print(f'{label} COMPLETE: {root}',flush=True)
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error':str(exc),'test_evaluated':False});raise
    finally:
        package(root,experiment);print(f'REVIEW PACKAGE: {root/(experiment+"_review_light.tar.gz")}',flush=True)
    return root


def job(path,cfg,resume):
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-config',str(path)]
    if resume:cmd.append('--resume')
    return {'cmd':cmd,'resource':'gpu','complete':str(Path(cfg['output_dir'])/'COMPLETE.json')}


if __name__=='__main__':main()
