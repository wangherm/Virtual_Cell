"""Round8: semantic knowledge, typed relations, pathway supervision, reliable KD."""
import argparse
import copy
from collections import deque
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

import numpy as np
import torch

from run_round7 import run_jobs
from vcell.data import load_prepared
from vcell.expansion import prepare_expansion
from vcell.knowledge import build_knowledge, load_knowledge
from vcell.knowledge_teachers import prepare_kd
from vcell.qwen import backbone_identity
from vcell.round8 import ARMS, PROTOCOLS, partitions, report, worker
from vcell.round5_methods import load_features
from vcell.train import source_fingerprint
from vcell.utils import file_sha256, read_config, write_json

REPO = Path(__file__).resolve().parents[1]


@contextmanager
def run_lock(path):
    """OS lock releases on process exit, including abrupt SSH/instance shutdown."""
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell()==0:
            stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('This Round8 run is already active; inspect its logs before restarting') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name=='nt':
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def package(root):
    with tarfile.open(root/'round8_review.tar.gz', 'w:gz') as tar:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or 'prepared' in path.relative_to(root).parts:
                continue
            if path.suffix in ('.json', '.csv') or path.name in ('predictions.npz', 'evaluation.npz'):
                tar.add(path, arcname=path.relative_to(root).as_posix())
            elif path.suffix == '.log':
                with path.open(encoding='utf-8', errors='replace') as stream:
                    text = ''.join(deque(stream, maxlen=100)).encode()
                info = tarfile.TarInfo(path.relative_to(root).as_posix()); info.size = len(text)
                tar.addfile(info, io.BytesIO(text))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-config')
    parser.add_argument('--work-dir', default=os.environ.get('VCELL_WORK', '/root/autodl-tmp/vcell-work'))
    parser.add_argument('--name', default='round8_01')
    parser.add_argument('--data')
    parser.add_argument('--challenge-data')
    parser.add_argument('--knowledge-npz', help='Use an already prepared, aligned knowledge snapshot')
    parser.add_argument('--knowledge-dir', help='Resumable online request cache and frozen embedding directory')
    parser.add_argument('--model-dir')
    parser.add_argument('--student-config', default=str(REPO/'configs/qwen_three_teachers.yaml'))
    parser.add_argument('--teacher-features-json', help='JSON mapping five family names to original frozen features.npz')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--protocols', nargs='+', choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[17,29,43])
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--parallel-students', type=int, choices=[1,2], default=2)
    parser.add_argument('--head-epochs', type=int, default=10)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    if args.worker_config:
        return worker(read_config(args.worker_config), args.resume)
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):
        parser.error('Use a simple run name')
    if min(args.max_steps,args.save_every,args.batch_size,args.gradient_accumulation,args.head_epochs)<1 or min(args.seeds)<0:
        parser.error('Invalid counts/seeds')
    if any(len(x)!=len(set(x)) for x in (args.arms,args.protocols,args.seeds)):
        parser.error('Duplicate arms/protocols/seeds')
    with run_lock(Path(args.work_dir).resolve()/'runs'/('.'+args.name+'.lock')):
        return launch(args)


def launch(args):
    if args.device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError('CUDA/BF16 unavailable')
    torch.set_num_threads(2)
    work = Path(args.work_dir).resolve()
    original = load_prepared(args.data or work/'prepared/real_min10_01')
    challenge = load_prepared(args.challenge_data or work/'prepared/vcc2025_c_01')
    root = work/'runs'/args.name
    model = read_config(args.student_config)['model']
    model.update(model_id=str(Path(args.model_dir or work/'models/Qwen3-0.6B-Base').resolve()),
                 revision=None, local_files_only=True, dtype='bfloat16' if args.device=='cuda' else 'float32',
                 gradient_checkpointing=False)
    for key in ('functional','background_interaction','response_rank'):
        model.pop(key,None)
    teacher_paths = {}
    if 'qwen_kd' in args.arms:
        if args.teacher_features_json:
            teacher_paths = read_config(args.teacher_features_json)
        else:
            teacher_paths = {f:str(work/'teachers/four_teacher_01'/f/'features.npz')
                             for f in ('scgpt','state','scfoundation','geneformer')}
            teacher_paths['uce'] = str(work/'runs/round5_01/uce/features.npz')
        if set(teacher_paths) != {'scgpt','state','scfoundation','geneformer','uce'}:
            raise ValueError('Provide all five frozen feature families; no fabricated teacher fallback')
        for path in teacher_paths.values():
            load_features(path, original)
    plan = {'source':source_fingerprint(), 'launcher':file_sha256(__file__), 'scheduler':file_sha256(REPO/'scripts/run_round7.py'),
        'original':original['audit']['fingerprint'], 'challenge':challenge['audit']['fingerprint'],
        'model':model, 'backbone':backbone_identity(model), 'arms':args.arms,'protocols':args.protocols,'seeds':args.seeds,
        'max_steps':args.max_steps,'save_every':args.save_every,'batch_size':args.batch_size,
        'gradient_accumulation':args.gradient_accumulation,'head_epochs':args.head_epochs,
        'teacher_features':{n:[str(p),file_sha256(p),file_sha256(Path(p).with_suffix('.json'))] for n,p in teacher_paths.items()},
        'knowledge_override': [str(Path(args.knowledge_npz).resolve()),file_sha256(args.knowledge_npz)] if args.knowledge_npz else None,
        'knowledge_dir':str(Path(args.knowledge_dir or work/'knowledge/round8').resolve()),
        'annotation_sources':json.loads((REPO/'assets/function/sources.json').read_text()),
        'test_evaluated':False}
    if root.exists() and any(root.iterdir()):
        if not args.resume or not (root/'plan.json').exists() or json.loads((root/'plan.json').read_text()) != plan:
            raise ValueError('Resume identical Round8 plan or choose a new name')
    reserve = max(12, len(args.arms)*len(args.protocols)*len(args.seeds)*.6)*1024**3
    if shutil.disk_usage(work).free < reserve:
        raise ValueError(f'Round8 requires {reserve/1024**3:.1f} GiB free reserve; no files deleted')
    root.mkdir(parents=True,exist_ok=True); write_json(root/'plan.json',plan)
    try:
        write_json(root/'progress.json', {'stage':'prepare data/knowledge','test_evaluated':False})
        data_path = prepare_expansion(original, challenge, root/'prepared')['plus_h1']
        data = load_prepared(data_path)
        splits = {p:partitions(data,p) for p in args.protocols}
        for protocol, split in splits.items():
            folder = root/'splits'/protocol; folder.mkdir(parents=True,exist_ok=True)
            write_json(folder/'partition.json',split)
            meta = data['meta'].copy()
            meta['round8_partition'] = 'excluded'
            for k, rows in split.items():
                meta.loc[rows,'round8_partition'] = k
            meta.to_csv(folder/'membership.csv',index=False)
            ix = np.array(split['outer']); ci = np.array(split['calibration'])
            np.savez_compressed(folder/'evaluation.npz', genes=data['genes'], outer_truth=data['delta'][ix],
                outer_row_ids=data['meta'].iloc[ix].row_id.to_numpy(dtype='U'), calibration_truth=data['delta'][ci],
                calibration_row_ids=data['meta'].iloc[ci].row_id.to_numpy(dtype='U'))
        knowledge = Path(args.knowledge_npz).resolve() if args.knowledge_npz else build_knowledge(data,
            plan['knowledge_dir'],REPO/'assets/function',model,args.device)
        load_knowledge(knowledge,data)
        write_json(root/'knowledge_identity.json',{'path':str(knowledge),'sha256':file_sha256(knowledge)})
        audit=knowledge.parent/'audit.json'
        write_json(root/'knowledge_audit.json',json.loads(audit.read_text()) if audit.exists() else
            {'source':'user-supplied aligned NPZ; accompanying source audit unavailable','certified':False})
        kd = {}
        for protocol in args.protocols:
            if teacher_paths:
                write_json(root/'progress.json',{'stage':'crossfit teachers','protocol':protocol,'test_evaluated':False})
                kd[protocol] = str(prepare_kd(original,data,splits[protocol],teacher_paths,
                    root/'teachers'/protocol,args.head_epochs,device=args.device))
        specs,jobs = {},{}
        (root/'configs').mkdir(exist_ok=True)
        for protocol in args.protocols:
            for seed in args.seeds:
                for arm in args.arms:
                    name = f'{protocol}_{arm}_seed{seed}'
                    cfg = {'data_dir':str(data_path), 'knowledge':str(knowledge),'partition':splits[protocol],
                        'protocol':protocol,'arm':arm,'seed':seed,'model':copy.deepcopy(model),'device':args.device,
                        'rank':32,'learning_rate':.0002,'module_weight':.1,'kd_weight':.5,
                        'max_steps':args.max_steps,'save_every':args.save_every,'batch_size':args.batch_size,
                        'gradient_accumulation':args.gradient_accumulation,'output_dir':str(root/'students'/name),
                        'teacher_cache':kd.get(protocol) if arm=='qwen_kd' else None}
                    specs[name]=cfg
                    path=root/'configs'/(name+'.json');write_json(path,cfg)
                    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-config',str(path)]
                    if args.resume:
                        command.append('--resume')
                    jobs[name]={'cmd':command,'resource':'gpu','complete':str(Path(cfg['output_dir'])/'COMPLETE.json')}
        write_json(root/'jobs.json',jobs)
        print(f'ROUND8 READY: {len(jobs)} jobs; parallel={args.parallel_students}; fixed steps={args.max_steps}',flush=True)
        if args.prepare_only:
            return root
        # Always enter worker validation, including already-completed jobs on resume.
        finished=run_jobs(jobs,root,args.parallel_students,False,label='ROUND8')
        report(data,specs,root)
        failed=[n for n,code in finished.items() if code]
        if failed:
            raise RuntimeError(f'Round8 failed jobs: {failed}; inspect logs and resume identical command')
        write_json(root/'COMPLETE.json',{'students':len(jobs),'test_evaluated':False,
            'note':'Fixed endpoint development comparisons. No automatic final-model/test selection.'})
        write_json(root/'progress.json', {'stage':'complete','students':len(jobs),'test_evaluated':False})
        if (root/'INCOMPLETE.json').exists():
            (root/'INCOMPLETE.json').unlink()
        print(f'ROUND8 COMPLETE: {root}',flush=True)
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error':str(exc),'test_evaluated':False})
        if (root/'COMPLETE.json').exists():
            (root/'COMPLETE.json').unlink()
        raise
    finally:
        package(root)
        print(f'REVIEW PACKAGE: {root/"round8_review.tar.gz"}',flush=True)
    return root


if __name__=='__main__':
    main()
