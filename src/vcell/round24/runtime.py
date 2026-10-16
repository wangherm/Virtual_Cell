"""Trusted grading and input contracts; generated Python executes only in the old kernel worker."""
import hashlib
import os
from pathlib import Path
import numpy as np
import pandas as pd
from ..model_agent.artifacts import read
from ..model_agent.contracts import digest
from ..round22.evaluator import inputs,execute,namespace
from ..round22.metrics import score,macro
from ..evolution.identity import canonical
from ..evolution.journal import append,classify
from ..utils import file_sha256,write_json
from .modules import compose


def execute_shared(code,data,dest,budget,python):
    """Deduplicate identical public worker input files after legacy execution.

    Detach existing links before the legacy writer runs, including after interruption.
    Only source/public inputs are shared; private grader truths are never linked in.
    """
    dest=Path(dest).resolve();root=Path(budget['root']).resolve()
    if not dest.is_relative_to(root/'assessments'):raise ValueError('Worker input cache target outside assessments')
    path=dest/'input.npz'
    if path.exists() and path.stat().st_nlink>1:path.unlink()
    try:return execute(code,data,dest,budget,python)
    finally:
        if path.is_file():
            sha=file_sha256(path);cache=root/'input_cache'/(sha+'.npz');cache.parent.mkdir(exist_ok=True)
            if cache.exists() and file_sha256(cache)!=sha:raise ValueError('Shared public input checksum changed')
            try:
                if not cache.exists():os.link(path,cache)
                else:
                    temporary=dest/'input.link.tmp';temporary.unlink(missing_ok=True);os.link(cache,temporary);os.replace(temporary,path)
            except OSError:
                write_json(root/'input_dedup_unavailable.json',{'reason':'Filesystem does not support input hardlinks; ordinary worker files retained'})


def fingerprint(repo):
    repo=Path(repo)
    files=[*sorted((repo/'src/vcell/round24').glob('*.py')),*sorted((repo/'experiments/round24_evolve').glob('*.py')),
        repo/'scripts/run_round24.py',repo/'scripts/run_shinka_round24.py',repo/'configs/round23_hgnc_identity.json.gz',
        repo/'src/vcell/evolution/identity.py',repo/'src/vcell/evolution/journal.py']
    return digest({'worker_namespace':namespace(repo),'code':{p.relative_to(repo).as_posix():file_sha256(p) for p in files if p.is_file()}})


def public_data(d,arm,new_datasets):
    d={k:v.copy() for k,v in d.items()}
    n=len(d.get('aux_target',[]));keep=np.ones(n,bool)
    if arm=='none':keep[:]=False
    elif arm in ('historical','new'):
        new=np.isin(d.get('aux_dataset',np.repeat('',n)),new_datasets);keep=new if arm=='new' else ~new
    elif arm!='all':raise ValueError('Unknown auxiliary-data arm')
    for key in d:
        if key.startswith('aux_'):
            if len(d[key])!=n:raise ValueError('Auxiliary metadata alignment mismatch')
            d[key]=d[key][keep]
    symbols=[canonical(g) or '' for g in d['genes']]
    counts=pd.Series(symbols).value_counts()
    d['gene_symbols']=np.array([s if s and counts[s]==1 else '' for s in symbols],dtype='U')
    return d


def counterfactual(d,g,variant):
    alt={k:v.copy() for k,v in d.items()}
    if variant=='correct':return alt
    kind,which=variant.split('_')
    if kind not in ('support','identity','both') or which not in ('wrong','shuffled'):raise ValueError('Unknown counterfactual')
    if kind in ('support','both'):alt['query_support']=g[which+'_support'].copy()
    if kind in ('identity','both'):
        alt['query_targets']=g[which+'_targets'].copy()
        # Preserve the real control projection. Replace only the STRING/target block.
        control_dim=min(16,len(d['train_y'])+len(d['calibration_y']))
        xx={str(t):v for t,v in zip(np.r_[d['train_targets'],d['calibration_targets']],np.r_[d['train_x'],d['calibration_x']])}
        alt['query_x']=d['query_x'].copy()
        for i,t in enumerate(alt['query_targets']):
            if str(t) in xx:alt['query_x'][i,control_dim:]=xx[str(t)][control_dim:]
        lookup={str(t):v for t,v in zip(np.r_[d['train_targets'],d['calibration_targets']],np.r_[d['train_prior'],d['calibration_prior']])}
        alt['query_prior']=np.array([lookup.get(str(t),d['query_prior'][i]) for i,t in enumerate(alt['query_targets'])])
    return alt


def metrics(frame,baseline,controls,cfg):
    mse=macro(frame,'mse');bm=macro(baseline,'mse');gain=1-mse/max(bm,1e-12)
    strong=macro(frame,'strong_mse');bs=macro(baseline,'strong_mse')
    gap=macro(frame,'specificity_gap');bg=macro(baseline,'specificity_gap')
    ratio=strong/max(bs,1e-12) if np.isfinite(strong) and np.isfinite(bs) else None
    values={'mse':mse,'baseline_mse':bm,'gain_over_fixed':gain,'strong_mse':strong,'baseline_strong_mse':bs,
        'strong_mse_ratio':ratio,'specificity_gap':gap,'baseline_specificity_gap':bg,
        'amplitude_ratio':macro(frame,'prediction_norm')/max(macro(frame,'truth_norm'),1e-12),
        'residual_amplitude_ratio':macro(frame,'residual_norm')/max(macro(frame,'truth_residual_norm'),1e-12),
        'contexts':frame.context.nunique(),'targets':frame.target.nunique(),'query_rows':len(frame)}
    for variant in cfg['counterfactual_variants'][1:]:
        a=frame.loc[frame.common_support];b=controls.loc[controls.variant.eq(variant)&controls.common_support]
        values['correct_vs_'+variant+'_gain']=1-macro(a,'mse')/max(macro(b,'mse'),1e-12) if len(a) and len(b) else None
    overall=gain>=cfg['minimum_mse_gain']
    support=[values.get('correct_vs_'+v+'_gain') for v in ('support_wrong','support_shuffled','both_wrong','both_shuffled')]
    identities=[values.get('correct_vs_'+v+'_gain') for v in ('identity_wrong','identity_shuffled')]
    response=bool(overall and ratio is not None and ratio<=cfg['maximum_strong_mse_ratio'] and np.isfinite(gap) and np.isfinite(bg)
        and gap>0 and gap-bg>=cfg['minimum_specificity_change'] and all(v is not None and v>0 for v in support)
        and all(v is not None and v>=-1e-9 for v in identities))
    values.update(overall_error_candidate=bool(overall),target_response_candidate=response,
        explicit_identity_benefit=bool(all(v is not None and v>1e-6 for v in identities)))
    combined=100*gain-20*max(0,(ratio or 1)-1)-10*max(0,(bg-gap) if np.isfinite(bg-gap) else 0)
    values={k:(int(v) if isinstance(v,np.integer) else float(v) if isinstance(v,np.floating) and np.isfinite(v) else None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v) for k,v in values.items()}
    return {'combined_score':float(combined),'public':values,'private':{},'text_feedback':str(values)}


def assess(root,module,task='repeat',phase='search',arm='all',controls=True,audit=False,scope='fixed',execute_fn=None):
    execute_fn=execute_fn or execute_shared
    root=Path(root);plan=read(root/'run_plan.json');cfg=plan['config'];repo=Path(plan['repo'])
    if plan['framework_sha256']!=fingerprint(repo):raise ValueError('Round24 trusted code changed; use a new campaign')
    if scope=='evolved' and phase!='search' and not (root/'search_frozen.json').exists():raise ValueError('Evolved confirmation requires source-only search freeze')
    if scope in ('evolved','random') and not (root/'search_baseline.json').exists():raise ValueError('Freeze the strongest source-selected fixed comparator before evolution')
    if phase not in ('search','confirmation'):raise ValueError('Round24 does not open whole-line outer or external tests')
    specs=read(root/'folds.json')['tasks'][task][phase]
    identity={'module':hashlib.sha256(module.encode()).hexdigest(),'plan':file_sha256(root/'run_plan.json'),'folds':file_sha256(root/'folds.json'),
        'task':task,'phase':phase,'arm':arm,'controls':controls,'audit':audit,'scope':scope,
        'search_baseline':file_sha256(root/'search_baseline.json') if scope in ('evolved','random') else None}
    key=digest(identity);dest=root/'assessments'/key;cache=dest/'assessment.json'
    if cache.exists():
        old=read(cache)
        if digest({k:v for k,v in old.items() if k!='sha256'})!=old['sha256'] or old['identity']!=identity:raise ValueError('Assessment record changed')
        for name,sha in old['files'].items():
            if file_sha256(dest/name)!=sha:raise ValueError('Assessment payload changed: '+name)
        return old
    dest.mkdir(parents=True,exist_ok=True);(dest/'module.py').write_text(module,encoding='utf-8')
    budget=dict(plan['budget'])
    if phase=='search' or scope=='fixed':
        fraction=cfg['diagnostic_fraction'] if scope=='fixed' else cfg['random_fraction'] if scope=='random' else cfg['search_fraction']
        budget['evaluation_seconds_max']*=fraction
    frames=[];bases=[];counter=[];behavior=[];audits=[]
    try:
        program=compose(module,audit=audit)
        for fold in specs:
            d,g,m=inputs(fold);d=public_data(d,arm,cfg['new_source_datasets'])
            if scope=='evolved' and not frames and phase=='search':
                tiny={k:(v[:2] if k.startswith('query_') or k in ('reference_available','background_reference') else v) for k,v in d.items()}
                a=execute_fn(program,tiny,dest/'contract',dict(budget,steps=min(5,budget['steps'])),plan['python'])
                b=execute_fn(program,tiny,dest/'contract2',dict(budget,steps=min(5,budget['steps'])),plan['python'])
                if not np.allclose(a,b,rtol=1e-5,atol=1e-7):raise ValueError('Nonreproducible contract predictions')
            for variant in cfg['counterfactual_variants'] if controls else ['correct']:
                x=counterfactual(d,g,variant);worker=dest/'workers'/fold['name']/variant
                prediction=execute_fn(program,x,worker,budget,plan['python'])
                if prediction.shape!=g['truth'].shape or not np.isfinite(prediction).all():raise ValueError('Invalid prediction shape or finiteness')
                behavior.append(hashlib.sha256(np.asarray(prediction,np.float32).tobytes()).hexdigest())
                f=score(prediction,g['truth'],g['background'],g['strong_mask'],m);f['fold']=fold['name'];f['common_support']=g['common'];f['variant']=variant
                if variant=='correct':
                    frames.append(f)
                    baseline_path=fold['baseline'];baseline_sha=fold['baseline_sha256']
                    if scope in ('evolved','random'):
                        reference=read(root/'search_baseline.json')[phase][fold['name']];baseline_path=reference['path'];baseline_sha=reference['sha256']
                    if file_sha256(baseline_path)!=baseline_sha:raise ValueError('Frozen baseline changed')
                    with np.load(baseline_path,allow_pickle=False) as bf:baseline=bf['prediction']
                    bases.append(score(baseline,g['truth'],g['background'],g['strong_mask'],m))
                    np.savez_compressed(dest/(fold['name']+'.npz'),prediction=prediction,row_ids=d['query_ids'],genes=d['genes'])
                    ap=worker/'output/audit.json'
                    if ap.exists():audits.append({'fold':fold['name'],**read(ap)})
                else:counter.append(f)
        if not frames:raise ValueError('No legal source folds')
        frame=pd.concat(frames,ignore_index=True);baseline=pd.concat(bases,ignore_index=True)
        cf=pd.concat(counter,ignore_index=True) if counter else pd.DataFrame(columns=['variant','common_support'])
        result=metrics(frame,baseline,cf,cfg)
        frame.to_csv(dest/'scores.csv',index=False);cf.to_csv(dest/'controls.csv',index=False);baseline.to_csv(dest/'baseline.csv',index=False)
        write_json(dest/'branch_audit.json',audits)
        record={'status':'valid','metrics':result,'module':str(dest/'module.py'),'module_sha256':identity['module'],'behavior_sha256':digest(behavior),'directory':str(dest)}
    except Exception as exc:
        error=str(exc)[-4000:];record={'status':'invalid','category':classify(exc),'error':error,'module':str(dest/'module.py'),'module_sha256':identity['module'],
            'metrics':{'combined_score':-1000000.,'public':{},'private':{},'text_feedback':error}}
    record['identity']=identity;record['files']={str(p.relative_to(dest)):file_sha256(p) for p in dest.rglob('*') if p.is_file() and p.name!='assessment.json'}
    record['sha256']=digest(record);write_json(cache,record)
    append(root/'journal.jsonl','assessment',phase=phase,scope=scope,task=task,status=record['status'],code_sha256=identity['module'],
        score=record['metrics']['combined_score'],error=record.get('error'))
    return record
