"""Independent Round25 evaluator; candidate code runs only in protected workers."""
import hashlib
from pathlib import Path
import time
import numpy as np
import pandas as pd
from ..evolution.journal import append, classify
from ..model_agent.artifacts import read
from ..model_agent.contracts import digest
from ..round22.evaluator import execute, inputs
from ..round22.metrics import score, macro
from ..round24.modules import compose
from ..round24.runtime import public_data, counterfactual, metrics, fingerprint as previous_fingerprint
from ..utils import write_json, file_sha256


def fingerprint(repo):
    repo = Path(repo)
    paths = [*sorted((repo/'src/vcell/round25').glob('*.py')),
             repo/'src/vcell/evolution/self_correction.py',
             repo/'scripts/run_round25.py', repo/'scripts/run_shinka_round25.py',
             repo/'experiments/round25_evolve/evaluate.py']
    return digest({'numeric_base': previous_fingerprint(repo),
                   'round25': {p.relative_to(repo).as_posix(): file_sha256(p) for p in paths if p.is_file()}})


def charged(root, pool=None):
    path = Path(root)/'evaluation_budget.json'
    if not path.exists(): return 0.
    return sum(r['charged_seconds'] for key, r in read(path)['workers'].items()
               if pool is None or key.startswith(str(Path(pool).resolve())))


def shared_execute(code, data, dest, budget, python):
    dest = Path(dest).resolve(); root = Path(budget['root']).resolve()
    worker_root = Path(budget['worker_root']).resolve()
    if not dest.is_relative_to(worker_root): raise ValueError('Worker outside assigned artifact root')
    path = dest/'input.npz'
    if path.exists() and path.stat().st_nlink > 1: path.unlink()
    try:
        return execute(code, data, dest, budget, python)
    finally:
        if path.is_file():
            import os
            cache = root/'input_cache'/f'{file_sha256(path)}.npz'; cache.parent.mkdir(parents=True, exist_ok=True)
            if cache.exists() and file_sha256(cache) != file_sha256(path): raise ValueError('Input cache changed')
            try:
                if not cache.exists(): os.link(path, cache)
                else:
                    temporary = dest/'input.link.tmp'; temporary.unlink(missing_ok=True)
                    os.link(cache, temporary); os.replace(temporary, path)
            except OSError: pass  # Ordinary files remain valid on filesystems without hardlinks.


def measure(frame, baseline, controls, cfg, task):
    result = metrics(frame, baseline, controls, cfg); m = result['public']
    m['gain_over_zero'] = 1-m['mse']/max(macro(frame, 'zero_mse'), 1e-12)
    subsets = {}
    for name, mask in {'all': np.ones(len(frame), bool), 'common_reference_support': frame.common_support,
                       'reference_available': frame.reference_available, 'reference_unavailable': ~frame.reference_available,
                       'globally_unseen_target': frame.globally_unknown}.items():
        a = frame.loc[mask]; b = baseline.loc[mask]
        subsets[name] = {'rows': len(a), 'targets': int(a.target.nunique()),
                         'mse': macro(a, 'mse') if len(a) else None,
                         'gain_over_fixed': 1-macro(a, 'mse')/max(macro(b, 'mse'), 1e-12) if len(a) else None,
                         'gain_over_zero': 1-macro(a, 'mse')/max(macro(a, 'zero_mse'), 1e-12) if len(a) else None}
    m['reference_rows'] = int(frame.reference_available.sum())
    m['common_support_rows'] = int(frame.common_support.sum())
    # Explicit gene identity is a separate observation, not a requirement to use measured Rep1.
    support = [m.get('correct_vs_'+v+'_gain') for v in ('support_wrong', 'support_shuffled')]
    m['shrinkage_only'] = bool(m['amplitude_ratio'] < cfg['minimum_amplitude_ratio'] or
                               (m.get('specificity_gap') or 0) <= 0)
    m['target_response_candidate'] = bool(m['overall_error_candidate'] and m['strong_mse_ratio'] is not None
        and m['strong_mse_ratio'] <= cfg['maximum_strong_mse_ratio'] and not m['shrinkage_only']
        and m['baseline_specificity_gap'] is not None
        and m['specificity_gap']-m['baseline_specificity_gap'] >= cfg['minimum_specificity_change']
        and all(v is not None and v > 0 for v in support))
    common = subsets['common_reference_support']
    m['transfer_candidate'] = bool(task == 'transfer' and m['overall_error_candidate'] and m['gain_over_zero'] >= cfg['minimum_mse_gain']
        and common['rows'] > 0 and common['gain_over_fixed'] >= cfg['minimum_mse_gain'] and common['gain_over_zero'] >= cfg['minimum_mse_gain'])
    if task == 'transfer': m['target_response_candidate'] = bool(m['target_response_candidate'] and m['transfer_candidate'])
    result['subsets'] = subsets
    return result


def assess(pool, module, phase='search', controls=True, baseline=False, execute_fn=None):
    pool = Path(pool); plan = read(pool/'run_plan.json'); cfg = plan['config']; task = plan['task']
    if fingerprint(plan['repo']) != plan['framework_sha256']: raise ValueError('Round25 code changed; create a new campaign')
    if phase not in ('search', 'confirmation'): raise ValueError('No outer or sealed external access in Round25')
    if phase != 'search' and not baseline and not (pool/'search_frozen.json').exists(): raise ValueError('Freeze source finalists before confirmation')
    if not baseline and not (pool/'baseline_selection.json').exists(): raise ValueError('Freeze matched fixed baselines before search')
    identity = {'module': hashlib.sha256(module.encode()).hexdigest(), 'plan': file_sha256(pool/'run_plan.json'),
                'folds': file_sha256(pool/'folds.json'), 'phase': phase, 'task': task, 'controls': controls,
                'baseline': baseline, 'comparator': file_sha256(pool/'baseline_selection.json') if not baseline else None}
    dest = pool/'assessments'/digest(identity); cache = dest/'assessment.json'
    if cache.exists():
        old = read(cache)
        if old['identity'] != identity or digest({k:v for k,v in old.items() if k != 'sha256'}) != old['sha256']:
            raise ValueError('Assessment identity/checksum changed')
        for name, sha in old['files'].items():
            if file_sha256(dest/name) != sha: raise ValueError('Cached assessment payload changed: '+name)
        return old
    dest.mkdir(parents=True, exist_ok=True); (dest/'module.py').write_text(module, encoding='utf-8')
    budget = dict(plan['budget'])
    if phase == 'search' and not baseline: budget['evaluation_seconds_max'] *= cfg['search_fraction']
    run = execute_fn or shared_execute
    def bounded(program, d, location, cap=None):
        b = dict(budget)
        if cap is not None: b['steps'] = cap
        if not baseline and phase == 'search' and charged(b['root'], pool)+b['seconds'] > plan['budget']['evaluation_seconds_max']*cfg['search_fraction']*cfg['maximum_task_search_share']:
            raise RuntimeError('Task search compute quota exhausted')
        return run(program, d, location, b, plan['python'])
    frames=[]; bases=[]; counters=[]; behaviors=[]; runtimes=[]
    try:
        program = compose(module)
        for fold in read(pool/'folds.json')[phase]:
            d, g, metadata = inputs(fold); d = public_data(d, 'all', cfg['new_source_datasets'])
            if phase == 'search' and not baseline and not frames:
                tiny = {k:(v[:2] if k.startswith('query_') or k in ('reference_available','background_reference') else v) for k,v in d.items()}
                a = bounded(program, tiny, dest/'contract', min(5,budget['steps']))
                b = bounded(program, tiny, dest/'contract2', min(5,budget['steps']))
                if not np.allclose(a,b,rtol=1e-5,atol=1e-7): raise ValueError('Nonreproducible deterministic contract')
            for variant in cfg['counterfactual_variants'] if controls else ['correct']:
                worker = dest/'workers'/fold['name']/variant
                pred = bounded(program, counterfactual(d,g,variant), worker)
                if pred.shape != g['truth'].shape or not np.isfinite(pred).all(): raise ValueError('Prediction shape/nonfinite failure')
                f = score(pred,g['truth'],g['background'],g['strong_mask'],metadata)
                f['common_support']=g['common']; f['reference_available']=g['available']; f['globally_unknown']=g['unknown']
                f['fold']=fold['name']; f['variant']=variant
                if variant == 'correct':
                    frames.append(f)
                    reference = read(pool/'baseline_selection.json')[phase][fold['name']] if not baseline else {'path':fold['baseline'],'sha256':fold['baseline_sha256']}
                    if file_sha256(reference['path']) != reference['sha256']: raise ValueError('Baseline changed')
                    with np.load(reference['path'],allow_pickle=False) as bf: bp=bf['prediction']
                    bases.append(score(bp,g['truth'],g['background'],g['strong_mask'],metadata))
                    np.savez_compressed(dest/(fold['name']+'.npz'),prediction=pred,genes=d['genes'],row_ids=d['query_ids'])
                    # Quantized behavior diversity is disclosed; not proof of a distinct mechanism.
                    quantum=cfg['behavior_quantum']; clipped=np.clip(pred/quantum,-9e18,9e18)
                    behaviors.append(hashlib.sha256(np.rint(clipped).astype(np.int64).tobytes()).hexdigest())
                else: counters.append(f)
                rp=worker/'output/runtime.json'
                if rp.exists(): runtimes.append({'fold':fold['name'],'variant':variant,**read(rp)})
        if not frames: raise ValueError('No legal source folds')
        frame=pd.concat(frames,ignore_index=True); base=pd.concat(bases,ignore_index=True)
        cf=pd.concat(counters,ignore_index=True) if counters else pd.DataFrame(columns=['variant','common_support'])
        result=measure(frame,base,cf,cfg,task)
        frame.to_csv(dest/'scores.csv',index=False);base.to_csv(dest/'baseline.csv',index=False);cf.to_csv(dest/'controls.csv',index=False)
        pd.DataFrame(runtimes).to_csv(dest/'candidate_runtime.csv',index=False)
        result['text_feedback']=str({'scope':'source_search_only' if phase=='search' else 'development_retest',
            'task':task,'metrics':result['public']})
        record={'status':'valid','metrics':result,'behavior_sha256':digest(behaviors)}
    except Exception as exc:
        error=str(exc)[-4000:]
        record={'status':'invalid','category':classify(exc),'error':error,
            'metrics':{'combined_score':-1e6,'public':{},'private':{},'text_feedback':error}}
    record.update(identity=identity,module_sha256=identity['module'],module=str(dest/'module.py'),directory=str(dest),
                  code_lines=len(module.splitlines()),finished_at=time.time(),worker_seconds=charged(budget['root'],dest),
                  peak_vram_bytes=max((r.get('peak_vram_bytes',0) for r in runtimes),default=None))
    record['files']={p.relative_to(dest).as_posix():file_sha256(p) for p in dest.rglob('*') if p.is_file() and p.name!='assessment.json'}
    record['sha256']=digest(record);write_json(cache,record)
    append(pool/'journal.jsonl','assessment',phase=phase,task=task,status=record['status'],code_sha256=identity['module'],
           score=record['metrics']['combined_score'],error=record.get('error'),category=record.get('category'))
    return record
