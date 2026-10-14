"""Trusted evaluator: generated modules only execute in kernel-restricted workers."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from ..model_agent.artifacts import read
from ..round11 import freeze
from ..clean_background import run_lock
from ..model_agent.contracts import digest
from ..utils import file_sha256,write_json
from .metrics import score,fitness,macro

START='# EVOLVE-BLOCK-START';END='# EVOLVE-BLOCK-END'


def namespace(repo):
    repo=Path(repo);paths=[repo/'scripts/run_round22.py',repo/'scripts/run_shinka_round22.py',*sorted((repo/'src/vcell/round22').glob('*.py')),*sorted((repo/'experiments/round22_evolve').glob('*'))]
    return digest({p.relative_to(repo).as_posix():file_sha256(p) for p in paths if p.is_file()})


def validate_code(code,initial):
    if len(code.encode())>100000 or code.count(START)!=1 or code.count(END)!=1:raise ValueError('One bounded EVOLVE-BLOCK required')
    before,body=code.split(START);body,after=body.split(END)
    b,rest=initial.split(START);_,a=rest.split(END)
    if (before,after)!=(b,a):raise ValueError('Immutable candidate wrapper changed')
    tree=ast.parse(code)
    if not any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='fit_predict' for n in tree.body):raise ValueError('fit_predict entrypoint missing')


def worker_env():
    # Explicit allowlist. No inherited PYTHONPATH, provider key or user credentials.
    env={k:os.environ[k] for k in ('PATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','SYSTEMROOT','WINDIR') if k in os.environ}
    env.update(PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    return env


def execute(code,data,dest,budget,python):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);(dest/'output').mkdir(exist_ok=True)
    (dest/'candidate.py').write_text(code,encoding='utf-8');np.savez_compressed(dest/'input.npz',**data)
    here=Path(__file__).parent
    for name in ('worker.py','sandbox.py'):shutil.copyfile(here/name,dest/name)
    cfg={k:budget[k] for k in ('seed','device','steps','seconds')};write_json(dest/'job.json',cfg)
    identity={'code':hashlib.sha256(code.encode()).hexdigest(),'input':file_sha256(dest/'input.npz'),'budget':cfg,
        'worker':file_sha256(here/'worker.py'),'sandbox':file_sha256(here/'sandbox.py')}
    freeze(dest/'identity.json',identity)
    if (dest/'COMPLETE.json').exists():
        done=read(dest/'COMPLETE.json')
        if done['prediction_sha256']!=file_sha256(dest/'output/prediction.npz'):raise ValueError('Prediction modified')
    else:
        ledger_path=Path(budget['root'])/'evaluation_budget.json';prefix=str(dest.resolve())
        with run_lock(ledger_path.with_suffix('.lock')):
            ledger=read(ledger_path) if ledger_path.exists() else {'workers':{},'cap_seconds':budget['evaluation_seconds_max']}
            ticket=prefix+'#'+str(1+sum(k.startswith(prefix+'#') for k in ledger['workers']))
            if sum(r['charged_seconds'] for r in ledger['workers'].values())+budget['seconds']>budget['evaluation_seconds_max']:raise RuntimeError('Round22 evaluation-time cap exhausted')
            ledger['workers'][ticket]={'status':'inflight','charged_seconds':budget['seconds']};write_json(ledger_path,ledger)
        write_json(Path(budget['root'])/'worker_progress.json',{'worker':str(dest),'status':'running','seconds_max':budget['seconds']})
        attempt_started=time.monotonic()
        with (dest/'worker.log').open('a',encoding='utf-8') as log:
            proc=subprocess.Popen([python,'-I',str(dest/'worker.py'),'--job',str(dest)],cwd=dest,env=worker_env(),stdout=log,stderr=subprocess.STDOUT,close_fds=True,start_new_session=sys.platform=='linux')
            try:
                started=time.monotonic()
                while proc.poll() is None:
                    if time.monotonic()-started>budget['seconds']:raise TimeoutError('Candidate wall-time cap exceeded')
                    if sys.platform=='linux':
                        status=Path(f'/proc/{proc.pid}/status')
                        try:
                            rss=next((int(s.split()[1])*1024 for s in status.read_text().splitlines() if s.startswith('VmRSS:')),0)
                        except FileNotFoundError:rss=0
                        if rss>budget['ram_bytes']:raise MemoryError('Candidate RSS cap exceeded')
                    time.sleep(.1)
                if proc.returncode:
                    # No hidden truth or provider credential is loaded by the worker.
                    # Feed the actual numerical exception back so code evolution can debug it.
                    tail='\n'.join((dest/'worker.log').read_text(encoding='utf-8',errors='replace').splitlines()[-20:])[-4000:]
                    tail=tail.replace(str(Path(budget['root'])),'<run>').replace(str(dest),'<candidate>')
                    raise RuntimeError('Candidate worker failed: '+tail)
            finally:
                if proc.poll() is None:
                    if sys.platform=='linux':os.killpg(proc.pid,9)
                    else:proc.kill()
                    proc.wait(timeout=10)
                with run_lock(ledger_path.with_suffix('.lock')):
                    ledger=read(ledger_path);ledger['workers'][ticket]={'status':'complete' if proc.returncode==0 else 'failed',
                        'charged_seconds':min(budget['seconds'],time.monotonic()-attempt_started)};write_json(ledger_path,ledger)
                write_json(Path(budget['root'])/'worker_progress.json',{'worker':str(dest),'status':'complete' if proc.returncode==0 else 'failed'})
        write_json(dest/'COMPLETE.json',{'prediction_sha256':file_sha256(dest/'output/prediction.npz'),'runtime':read(dest/'output/runtime.json')})
    with np.load(dest/'output/prediction.npz',allow_pickle=False) as f:p={k:f[k] for k in f.files}
    if p['prediction'].shape!=(len(data['query_ids']),len(data['genes'])) or not np.isfinite(p['prediction']).all() or not np.array_equal(p['genes'],data['genes']) or not np.array_equal(p['row_ids'],data['query_ids']):raise ValueError('Grader prediction alignment/finiteness failed')
    return p['prediction']


def inputs(fold):
    if file_sha256(fold['input'])!=fold['input_sha256'] or file_sha256(Path(fold['private'])/'grade.npz')!=fold['grade_sha256'] or file_sha256(Path(fold['private'])/'metadata.csv')!=fold['metadata_sha256']:raise ValueError('Frozen fold changed')
    with np.load(fold['input'],allow_pickle=False) as f:d={k:f[k] for k in f.files}
    with np.load(Path(fold['private'])/'grade.npz',allow_pickle=False) as f:g={k:f[k] for k in f.files}
    return d,g,pd.read_csv(Path(fold['private'])/'metadata.csv')


def fixed_predictions(d):
    y=np.r_[d['train_y'],d['calibration_y']];x=np.r_[d['train_x'],d['calibration_x']];qx=d['query_x']
    mu=y.mean(0);q=d['query_support'];a=np.r_[d['train_support'],d['calibration_support']]
    grid=(0.,.01,.03,.1,.3,.7,1.)
    alpha=min(grid,key=lambda v:float(np.square(v*a-y).mean()))
    centered=min(grid,key=lambda v:float(np.square(mu+v*(a-a.mean(0))-y).mean()))
    out={'zero':np.zeros_like(q),'background_mean':np.tile(mu,(len(q),1)),'calibrated_support':alpha*q}
    if str(d['task'])=='repeat':out['demeaned_support']=mu+centered*(q-a.mean(0))
    # Refit numerical comparators on this fold's permitted source, not old cached heads.
    tx=d['train_x'];cx=d['calibration_x'];ty=d['train_y'];cy=d['calibration_y']
    design=np.c_[tx,np.ones(len(tx))];cal=np.c_[cx,np.ones(len(cx))];query=np.c_[qx,np.ones(len(qx))]
    _,_,basis=np.linalg.svd(ty-ty.mean(0),full_matrices=False)
    for rank in (16,32):
        b=basis[:min(rank,len(basis))];models=[]
        for penalty in (.1,1.,10.,100.):
            reg=np.eye(design.shape[1])*penalty;reg[-1,-1]=0
            coef=np.linalg.solve(design.T@design+reg,design.T@((ty-ty.mean(0))@b.T))
            models.append((float(np.square((cal@coef)@b+ty.mean(0)-cy).mean()),coef))
        coef=min(models,key=lambda v:v[0])[1];out['raw_control_STRING_factor'+str(rank)]=(query@coef)@b+ty.mean(0)
    # Full-output STRING ridge, same permitted train/calibration.
    models=[]
    for penalty in (.1,1.,10.,100.):
        reg=np.eye(design.shape[1])*penalty;reg[-1,-1]=0
        coef=np.linalg.solve(design.T@design+reg,design.T@ty)
        models.append((float(np.square(cal@coef-cy).mean()),coef))
    out['raw_control_STRING_ridge']=query@min(models,key=lambda v:v[0])[1]
    return out


def evaluate(code,root,task,phase,output,execute_fn=execute):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True);plan=read(root/'run_plan.json');data_plan=read(root/'data_plan.json')
    if namespace(plan['repo'])!=plan['code_sha256']:raise ValueError('Trusted grader/task code changed after freeze')
    initial=(Path(plan['repo'])/'experiments/round22_evolve/initial.py').read_text(encoding='utf-8');validate_code(code,initial)
    if phase!='search' and not (root/'search_frozen.json').exists():raise ValueError('Confirmation/outer forbidden before search freeze')
    if phase=='outer' and not (root/'confirmation_frozen.json').exists():raise ValueError('Outer forbidden before confirmation freeze')
    frames=[];bases=[];counter=[];fixed=[];runtime=[]
    for fold in data_plan['folds'][task][phase]:
        d,g,m=inputs(fold);pred=execute_fn(code,d,output/'workers'/fold['name']/'correct',plan['budget'],plan['python'])
        rp=output/'workers'/fold['name']/'correct/output/runtime.json'
        if rp.exists():runtime.append({'fold':fold['name'],'variant':'correct',**read(rp)})
        f=score(pred,g['truth'],g['background'],g['strong_mask'],m);f['fold']=fold['name'];f['method']='candidate'
        f['reference_available']=g['available'];f['common_support']=g['common'];f['globally_unknown']=g['unknown'];frames.append(f)
        per=[]
        for name,p in fixed_predictions(d).items():
            bf=score(p,g['truth'],g['background'],g['strong_mask'],m);bf['method']=name;bf['fold']=fold['name'];per.append(bf)
        fixed.extend(per)
        # Baseline selection must already be source-only frozen for confirmation/outer.
        selection=read(root/'baseline_selection.json') if (root/'baseline_selection.json').exists() else {}
        chosen=selection.get(task,'background_mean' if task=='repeat' else 'zero')
        bases.append(next(f for f in per if f.method.iloc[0]==chosen))
        # Counterfactuals run the same generated method, source labels and seed.
        for variant in ('wrong','shuffled'):
            alt={k:v.copy() for k,v in d.items()}
            alt['query_support']=g[variant+'_support'];alt['query_x']=g[variant+'_x'];alt['query_targets']=g[variant+'_targets']
            lookup={str(t):v for t,v in zip(np.r_[d['train_targets'],d['calibration_targets']],np.r_[d['train_prior'],d['calibration_prior']])}
            alt['query_prior']=np.array([lookup.get(str(t),d['query_prior'][i]) for i,t in enumerate(alt['query_targets'])])
            xx={str(t):v for t,v in zip(np.r_[d['train_targets'],d['calibration_targets']],np.r_[d['train_x'],d['calibration_x']])}
            control_dim=min(16,len(d['train_y'])+len(d['calibration_y']))
            for i,t in enumerate(alt['query_targets']):
                if str(t) in xx:alt['query_x'][i,control_dim:]=xx[str(t)][control_dim:]
            q=execute_fn(code,alt,output/'workers'/fold['name']/variant,plan['budget'],plan['python'])
            rp=output/'workers'/fold['name']/variant/'output/runtime.json'
            if rp.exists():runtime.append({'fold':fold['name'],'variant':variant,**read(rp)})
            cf=score(q,g['truth'],g['background'],g['strong_mask'],m);cf['variant']=variant;cf['fold']=fold['name'];cf['common_support']=g['common'];counter.append(cf)
    if not frames:raise ValueError('No legal folds for '+task+'/'+phase)
    frame=pd.concat(frames,ignore_index=True);base=pd.concat(bases,ignore_index=True);controls=pd.concat(counter,ignore_index=True)
    combined,public=fitness(frame,base)
    public['reference_rows']=int(frame.reference_available.sum());public['query_rows']=len(frame)
    common=frame.loc[frame.common_support];wrong=controls.loc[controls.variant.eq('wrong') & controls.common_support]
    public['same_method_correct_vs_wrong_gain']=1-macro(common,'mse')/max(macro(wrong,'mse'),1e-12) if len(common) else None
    shuffled=controls.loc[controls.variant.eq('shuffled') & controls.common_support]
    public['same_method_correct_vs_shuffled_gain']=1-macro(common,'mse')/max(macro(shuffled,'mse'),1e-12) if len(common) else None
    public['signal_screen_passed']=bool(public['signal_screen_passed'] and public['same_method_correct_vs_wrong_gain'] is not None and public['same_method_correct_vs_wrong_gain']>0 and
        public['same_method_correct_vs_shuffled_gain'] is not None and public['same_method_correct_vs_shuffled_gain']>0)
    combined+=10*(public['same_method_correct_vs_wrong_gain'] or 0.)
    frame.to_csv(output/'source_scores.csv',index=False);controls.to_csv(output/'counterfactual_scores.csv',index=False);pd.concat(fixed).to_csv(output/'fixed_scores.csv',index=False)
    subsets=[]
    for label,selection in {'all':np.ones(len(frame),bool),'common_reference_support':frame.common_support,'reference_available':frame.reference_available,
            'reference_unavailable':~frame.reference_available,'globally_unknown':frame.globally_unknown,'known_target':~frame.globally_unknown}.items():
        part=frame.loc[selection]
        if len(part):subsets.append({'subset':label,'rows':len(part),'targets':int(part.target.nunique()),**{k:macro(part,k) for k in ('mse','zero_mse','background_mse','specificity_gap','strong_mse')}})
    pd.DataFrame(subsets).to_csv(output/'subset_scores.csv',index=False)
    grouped=[]
    for context,part in frame.groupby('context'):
        cb=base.loc[base.context.eq(context)];_,scores=fitness(part,cb);grouped.append({'context':context,**scores})
    pd.DataFrame(grouped).to_csv(output/'context_scores.csv',index=False)
    pd.DataFrame(runtime,columns=None if runtime else ['fold','variant','seconds','peak_vram_bytes','torch_optimizer_steps']).to_csv(output/'candidate_runtime.csv',index=False)
    metrics={'combined_score':float(combined),'public':public,'private':{},'extra_data':{'task':task,'phase':phase,'code_sha256':hashlib.sha256(code.encode()).hexdigest(),
        'method_seconds':sum(r['seconds'] for r in runtime) if runtime else None,'peak_vram_bytes':max((r['peak_vram_bytes'] for r in runtime),default=None),
        'torch_optimizer_steps':sum(r['torch_optimizer_steps'] for r in runtime) if runtime else None,'cost_scope':'saved method runtime, including ablations; total charged process time is in evaluation_budget.json'},
        'text_feedback':json.dumps({'task':task,'scores':public,'by_source_context':grouped,
            'objective':'Improve over strongest fixed baseline and use correct target information. Amplitude and specificity are diagnostic; no external success inferred.'})}
    write_json(output/'metrics.json',metrics);write_json(output/'correct.json',{'correct':True,'error':None});return metrics
