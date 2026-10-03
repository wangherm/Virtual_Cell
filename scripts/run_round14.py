"""Round14: local biology, interaction, extrapolation and information controls."""
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
from vcell.round14 import ARMS, worker, report
from vcell.train import source_fingerprint
from vcell.utils import file_sha256, write_json

REPO = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-config')
    p.add_argument('--work-dir', default='/root/autodl-tmp/vcell-work')
    p.add_argument('--name', default='round14_01')
    p.add_argument('--round13-run')
    p.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    p.add_argument('--protocols', nargs='+', choices=['target', 'context', 'double'], default=['target', 'context', 'double'])
    p.add_argument('--arms', nargs='+', choices=list(ARMS), default=list(ARMS))
    p.add_argument('--max-steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--parallel-jobs', type=int, choices=[1, 2, 3, 4], default=3)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args(argv)
    if a.worker_config: return worker(json.loads(Path(a.worker_config).read_text()), a.resume)
    if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.name):
        p.error('Use a simple run name')
    if min(a.max_steps, a.batch_size, a.save_every) < 1 or min(a.seeds) < 0: p.error('Invalid counts')
    for key in ('seeds', 'protocols', 'arms'):
        if len(getattr(a, key)) != len(set(getattr(a, key))): p.error('Duplicate ' + key)
    if a.device == 'cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    work = Path(a.work_dir).resolve()
    with run_lock(work/'runs'/('.'+a.name+'.lock')): return launch(a, work, work/'runs'/a.name)


def launch(a, work, root):
    parent = Path(a.round13_run or work/'runs/round13_01').resolve()
    print(f'ROUND14 AUDIT full Round13 parent: {parent}', flush=True)
    verify_files(parent)
    old_plan = json.loads((parent/'plan.json').read_text())
    if old_plan['experiment'] != 'round13': raise ValueError('Completed full Round13 run required; light archives omit predictions')
    if not set(a.seeds) <= set(old_plan['seeds']) or not set(a.protocols) <= set(old_plan['protocols']):
        raise ValueError('Requested historical folds/seeds unavailable')
    plan = {'experiment': 'round14', 'source': source_fingerprint(), 'launcher': file_sha256(__file__),
        'scheduler': file_sha256(REPO/'scripts/run_round7.py'), 'parent': str(parent),
        'parent_complete_sha256': file_sha256(parent/'COMPLETE.json'),
        **{k: getattr(a, k) for k in ('seeds', 'protocols', 'arms', 'max_steps', 'batch_size', 'save_every', 'device')},
        'arm_definitions': {arm: ARMS[arm] for arm in a.arms}, 'response_rank': 32,
        'learning_rate': .0005, 'weight_decay': .01, 'selection': 'source calibration only',
        'data_expansion': False, 'background_training': False, 'Qwen_training': False, 'test_evaluated': False}
    if root.exists() and any(root.iterdir()) and not a.resume: raise ValueError('Use --resume or a new name')
    if not root.exists() and shutil.disk_usage(work).free < 5*1024**3:
        raise ValueError('Need 5 GiB free; no artifacts deleted')
    root.mkdir(parents=True, exist_ok=True); freeze(root/'plan.json', plan)
    if (root/'COMPLETE.json').exists():
        verify_files(root); print(f'ROUND14 COMPLETE VERIFIED: {root}', flush=True); return root
    def stage(label):
        write_json(root/'stage.json', {'stage': label, 'test_evaluated': False})
        print('ROUND14 STAGE: ' + label, flush=True)
    try:
        stage('audit existing data, models and fixed splits; no downloads')
        specs = {}; jobs = {}; (root/'configs').mkdir(exist_ok=True); data = None
        # Interleave protocols so the first concurrent workers cover different tasks.
        for seed in a.seeds:
            for protocol in a.protocols:
                name = f'{protocol}_seed{seed}'; parent_worker = parent/'models'/name
                verify_files(parent_worker)
                c13 = json.loads((parent_worker/'manifest.json').read_text())['config']
                parent12 = Path(c13['parent_worker']); verify_files(parent12)
                pc = json.loads((parent12/'manifest.json').read_text())['config']
                if data is None:
                    data_path = pc['data_dir']; data = load_prepared(data_path)
                elif data_path != pc['data_dir']: raise ValueError('Historical data paths disagree')
                for head in ('panel_additive_ridge', 'diffusion_additive', 'diffusion_factor', 'diffusion_factor_id'):
                    if not (parent_worker/head/'predictions.npz').is_file():
                        raise ValueError(f'Required full parent prediction missing: {parent_worker/head}')
                cfg = {k: plan[k] for k in ('max_steps', 'batch_size', 'save_every', 'device', 'arms')}
                cfg.update(seed=seed, protocol=protocol, parent_worker=str(parent_worker), output_dir=str(root/'models'/name))
                path = root/'configs'/(name+'.json'); freeze(path, cfg); specs[name] = cfg
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker-config', str(path)]
                if a.resume: cmd.append('--resume')
                jobs[name] = {'cmd': cmd, 'resource': 'gpu', 'complete': str(Path(cfg['output_dir'])/'COMPLETE.json')}
                evidence = root/'splits'/name; evidence.mkdir(parents=True, exist_ok=True)
                freeze(evidence/'partition.json', pc['partition'])
                meta = data['meta'].copy(); meta['role'] = 'excluded'
                for label, indices in pc['partition'].items(): meta.loc[indices, 'role'] = label
                meta.to_csv(evidence/'membership.csv', index=False)
        write_json(root/'jobs.json', jobs)
        write_json(root/'budget.json', {'jobs': len(jobs), 'new_heads': len(jobs)*len(a.arms),
            'optimizer_steps': len(jobs)*a.max_steps*sum(ARMS[k]['steps_multiplier'] for k in a.arms),
            'parallel_jobs': a.parallel_jobs, 'threads_per_worker': 2, 'background_jobs': 0,
            'Qwen_backbone_jobs': 0, 'downloads': 0, 'parent_predictions_reused': True})
        print(f'ROUND14 READY: {len(jobs)} jobs; {len(jobs)*len(a.arms)} small heads; concurrency={a.parallel_jobs}', flush=True)
        if a.prepare_only: stage('prepared; training pending'); return root
        stage('controlled heads and input sensitivity')
        result = run_jobs(jobs, root, a.parallel_jobs, False, label='ROUND14')
        if any(result.values()): raise RuntimeError('Worker failed; inspect logs and resume identical plan')
        stage('paired MSE, specificity, effect-size and extrapolation reports'); report(specs, root)
        (root/'INCOMPLETE.json').unlink(missing_ok=True); stage('complete')
        save_complete(root, {'test_evaluated': False, 'jobs': len(jobs), 'new_heads': len(jobs)*len(a.arms)})
        print(f'ROUND14 COMPLETE: {root/"comparison_mean.csv"}', flush=True)
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json', {'error': str(exc), 'test_evaluated': False}); raise
    finally:
        package(root, 'round14'); print(f'REVIEW PACKAGE: {root/"round14_review_light.tar.gz"}', flush=True)
    return root


if __name__ == '__main__': main()
