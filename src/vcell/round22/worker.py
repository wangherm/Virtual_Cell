"""Standalone copied into candidate workdir; never loads grading labels."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    root=Path(a.job).resolve();cfg=json.loads((root/'job.json').read_text())
    sys.path.insert(0,str(root))
    import ctypes
    # A scheduler timeout must not leave a detached candidate running.
    libc=ctypes.CDLL(None);parent=os.getppid()
    if libc.prctl(1,9,0,0,0)!=0 or os.getppid()!=parent:raise RuntimeError('Unable to bind candidate lifetime to grader')
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    if cfg['device']=='cuda':
        torch.cuda.set_per_process_memory_fraction(.25)
        torch.zeros(1,device='cuda');torch.cuda.reset_peak_memory_stats()
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    resource.setrlimit(resource.RLIMIT_CPU,(cfg['seconds'],cfg['seconds']+5))
    resource.setrlimit(resource.RLIMIT_FSIZE,(256*1024**2,256*1024**2))
    os.environ.clear();os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TMPDIR=str(root/'output'),CUDA_CACHE_PATH=str(root/'output/cuda_cache'))
    from sandbox import restrict,runtime_paths
    devices=[str(p) for p in Path('/dev').glob('nvidia*') if p.is_char_device()]
    restrict([*runtime_paths(),root/'input.npz',root/'candidate.py'],[root/'output','/dev/null','/dev/urandom','/dev/random',*devices])
    updates=[0];original_init=torch.optim.Optimizer.__init__
    def counted_init(optimizer,*args,**kwargs):
        original_init(optimizer,*args,**kwargs)
        def before_step(*_):
            updates[0]+=1
            if updates[0]>cfg['steps']:raise RuntimeError('Torch optimizer step budget exceeded')
        optimizer.register_step_pre_hook(before_step)
    torch.optim.Optimizer.__init__=counted_init
    # Any top-level generated code executes only after both kernel restrictions.
    with np.load(root/'input.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    spec=importlib.util.spec_from_file_location('candidate',root/'candidate.py');module=importlib.util.module_from_spec(spec)
    started=time.monotonic();spec.loader.exec_module(module)
    prediction=np.asarray(module.fit_predict(data,{'steps':cfg['steps'],'batch_size':16,'seed':cfg['seed'],'device':cfg['device']}),np.float32)
    expected=(len(data['query_targets']),len(data['genes']))
    if prediction.shape!=expected or not np.isfinite(prediction).all():raise ValueError('Invalid ordered mean-delta prediction')
    np.savez_compressed(root/'output/prediction.npz',prediction=prediction,genes=data['genes'],row_ids=data['query_ids'])
    (root/'output/runtime.json').write_text(json.dumps({'seconds':time.monotonic()-started,'peak_vram_bytes':torch.cuda.max_memory_allocated() if cfg['device']=='cuda' else 0,'torch_optimizer_steps':updates[0],'sandbox':'Landlock+seccomp','query_truth_loaded':False}))


if __name__=='__main__':main()
