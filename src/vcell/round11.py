"""Audited source-context calibration and bounded target-context adaptation.

No new representation is learned here. Cache reuse is explicit; Jurkat inference
requires an immutable release manifest written before accessing its responses.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from .background_training import fingerprint
from .fixed_training import response_basis
from .qwen import backbone_identity
from .round8 import partitions, validate_partition, load_checkpoint, predict
from .round10 import make_partition, paired_summary
from .selection import row_weights, mse
from .specialization import reference_responses
from .train import normalization
from .utils import file_sha256, write_json

SOURCES = {'K562', 'RPE1', 'H1_hESC'}
ARMS = ('mlp_b0_none', 'qwen_b1_none', 'qwen_b2_none')
SEEDS = (17, 29, 43)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Frozen artifact changed: {path}; use a new run name')
    else:
        write_json(path, value)


def verify_files(folder):
    folder = Path(folder)
    done = read_json(folder/'COMPLETE.json')
    if not done.get('files'):
        raise ValueError(f'Missing checksummed completion: {folder}')
    for name, sha in done['files'].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or file_sha256(path) != sha:
            raise ValueError(f'Cache checksum mismatch: {path}')
    return done


def context_partition(data, context):
    if context in ('RPE1', 'H1_hESC'):
        return make_partition(data, 'context_rpe1' if context == 'RPE1' else 'context_h1')
    if context != 'K562':
        raise ValueError('Only source contexts may enter OOF calibration')
    train = data['splits']['train']
    labels = data['meta'].iloc[train].context.to_numpy()
    local = copy.copy(data)
    local['splits'] = {**data['splits'], 'train': train[labels != context], 'val': train[labels == context]}
    result = partitions(local, 'context')
    validate_partition(data, result)
    return result


def audit_cache(data, folder, expected_partition, expected_source=None):
    """Verify artifacts and independently reconstruct normalization/reference/calibration.

    Historical source fingerprints are retained, not compared to the now extended
    package fingerprint. Partition and basis are checked against the current data.
    """
    folder = Path(folder)
    done = verify_files(folder)
    manifest = read_json(folder/'manifest.json')
    cfg = manifest['config']
    actual = {k: v for k, v in manifest.items() if k not in ('fingerprint', 'test_evaluated')}
    if fingerprint(actual) != manifest['fingerprint'] or done['fingerprint'] != manifest['fingerprint']:
        raise ValueError('Historical manifest fingerprint mismatch')
    if expected_source is not None and manifest['source'] != expected_source:
        raise ValueError('Historical source differs from parent plan')
    if manifest['data'] != data['audit']['fingerprint'] or cfg['partition'] != expected_partition:
        raise ValueError('Historical data/partition mismatch')
    validate_partition(data, cfg['partition'])
    if cfg['arm'] not in ARMS or cfg['module_mode'] != 'none' or cfg['kd_weight'] != 0:
        raise ValueError('Round11 reuses only the fixed no-module, no-KD candidates')
    fit, cal, outer = [np.asarray(cfg['partition'][k]) for k in ('fit', 'calibration', 'outer')]
    if not set(data['meta'].iloc[np.r_[fit, cal]].context) <= SOURCES:
        raise ValueError('Non-source labels entered historical fitting')
    saved = torch.load(folder/'endpoint.pt', map_location='cpu', weights_only=True)
    if saved['config'] != cfg or saved['fingerprint'] != done['fingerprint']:
        raise ValueError('Endpoint/config mismatch')
    if saved['genes'] != data['genes'].tolist() or saved['perturbations'] != data['perturbations'].tolist():
        raise ValueError('Checkpoint vocabulary mismatch')
    if saved['backbone'] != manifest['backbone'] or backbone_identity(cfg['model']) != saved['backbone']:
        raise ValueError('Frozen backbone dependency changed')
    if saved['knowledge_sha256'] != manifest['knowledge'] or file_sha256(cfg['knowledge']) != saved['knowledge_sha256']:
        raise ValueError('Knowledge dependency changed')
    local = {**data, 'splits': {**data['splits'], 'train': fit}}
    norm = normalization(local)
    for key in norm:
        np.testing.assert_allclose(saved['normalization'][key].numpy(), norm[key], rtol=1e-6, atol=1e-8)
    basis, basis_audit = response_basis(data, fit, cfg['rank'])
    # Compare subspaces: SVD signs/rotations in tied singular spaces may differ.
    old = saved['basis'].numpy()
    if old.shape != basis.shape or not np.allclose((old @ basis.T) @ basis, old, atol=2e-4, rtol=2e-4):
        raise ValueError('Basis is not the fit-only response subspace')
    if read_json(folder/'basis_audit.json')['fit_rows'] != fit.tolist():
        raise ValueError('Basis fit-row provenance mismatch')
    background = None
    if cfg['background'] != 'b0':
        bg = Path(cfg['encoder']).parent
        bg_done = verify_files(bg)
        bm = read_json(bg/'manifest.json')
        if fingerprint({k: v for k, v in bm.items() if k != 'fingerprint'}) != bm['fingerprint']:
            raise ValueError('Background manifest fingerprint mismatch')
        bc = bm['config']
        enc = torch.load(bg/'encoder.pt', map_location='cpu', weights_only=True)
        if (bg_done['fingerprint'] != bm['fingerprint'] or enc['fingerprint'] != bm['fingerprint'] or
                bc['partition'] != expected_partition or enc['partition'] != expected_partition or
                bm['data'] != manifest['data'] or bm['source'] != manifest['source'] or
                file_sha256(bg/'encoder.pt') != manifest['background_encoder'] or
                enc['genes'] != data['genes'].tolist() or enc['background'] != cfg['background'] or
                bc['background'] != cfg['background'] or bc['family'] != cfg['family'] or
                enc['family'] != cfg['family'] or bc['seed'] != cfg['seed'] or bc['model'] != cfg['model']):
            raise ValueError('Background provenance/holdout mismatch')
        for key in norm:
            np.testing.assert_allclose(enc['normalization'][key].numpy(), norm[key], rtol=1e-6, atol=1e-8)
        cache = Path(bc['background_cache'])
        verify_files(cache)
        if file_sha256(cache/'COMPLETE.json') != bm['background']:
            raise ValueError('Background cache identity changed')
        background = {'encoder_sha256': file_sha256(bg/'encoder.pt'), 'config': bc,
                      'manifest_sha256': file_sha256(bg/'manifest.json')}
    with np.load(folder/'predictions.npz', allow_pickle=False) as f:
        predictions = {k: f[k] for k in ('outer', 'calibration', 'reference', 'calibrated', 'shrunk')}
        for key, ix in [('outer_row_ids', outer), ('calibration_row_ids', cal)]:
            if not np.array_equal(f[key], data['meta'].iloc[ix].row_id.to_numpy(dtype='U')):
                raise ValueError('Cached prediction row order mismatch')
        if not np.array_equal(f['genes'], data['genes']):
            raise ValueError('Cached prediction gene order mismatch')
    ref, seen = reference_responses(data, fit)
    np.testing.assert_allclose(predictions['reference'], ref[data['pert_idx'][outer]], atol=1e-7)
    from .round7 import calibrated_mix, scalar_shrink
    weights = row_weights(data['meta'].iloc[cal])
    old_cal = read_json(folder/'calibration.json')
    alpha = scalar_shrink(predictions['calibration'], data['delta'][cal], weights)
    mix = calibrated_mix([predictions['calibration'], ref[data['pert_idx'][cal]],
                          np.zeros_like(predictions['calibration'])], data['delta'][cal], weights)
    np.testing.assert_allclose(mix, old_cal['mix'], atol=1e-6)
    np.testing.assert_allclose(alpha, old_cal['alpha'], atol=1e-6)
    np.testing.assert_allclose(predictions['calibrated'], mix[0]*predictions['outer']+mix[1]*predictions['reference'], atol=1e-7)
    meta = data['meta'].iloc[outer].copy().reset_index(drop=True)
    meta['seen_in_fit'] = seen[data['pert_idx'][outer]]
    meta['data_row'] = outer
    identity = {'folder': str(folder), 'checkpoint_sha256': file_sha256(folder/'endpoint.pt'),
                'predictions_sha256': file_sha256(folder/'predictions.npz'),
                'manifest_sha256': file_sha256(folder/'manifest.json'), 'source': manifest['source'],
                'backbone': saved['backbone'], 'knowledge_sha256': saved['knowledge_sha256'],
                'config': cfg, 'basis': basis_audit, 'background': background,
                'raw_mse_recomputed': mse(predictions['outer'], data['delta'][outer], row_weights(meta)),
                'old_calibrated_mse_recomputed': mse(predictions['calibrated'], data['delta'][outer], row_weights(meta))}
    return {'cfg': cfg, 'meta': meta, 'rows': outer, 'predictions': predictions, 'identity': identity}


def moments(meta, truth, prediction, reference):
    """Per-row quadratic sufficient statistics, including centered correlations."""
    y, p, r = [np.asarray(a, dtype=np.float64) for a in (truth, prediction, reference)]
    if y.shape != p.shape or r.shape != y.shape or len(y) != len(meta) or not all(np.isfinite(a).all() for a in (y,p,r)):
        raise ValueError('Nonfinite or misaligned response arrays')
    frame = meta.copy().reset_index(drop=True)
    for name, x in [('t', y*y), ('q', p*p), ('r', r*r), ('c', p*y), ('h_ref', r*y), ('g_cross', p*r)]:
        frame[name] = x.mean(1)
    for name, x in [('mean_y', y), ('mean_p', p), ('mean_ref', r)]:
        frame[name] = x.mean(1)
    frame['row_weight'] = row_weights(frame)
    return frame


def mixture_error(frame, a, b=0.):
    return (a*a*frame.q + b*b*frame.r + 2*a*b*frame.g_cross - 2*a*frame.c - 2*b*frame.h_ref + frame.t).to_numpy()


def fit_coefficients(frame, alpha_only=False):
    w = row_weights(frame)
    if alpha_only:
        return [float(np.clip(np.dot(w, frame.c)/max(np.dot(w, frame.q), 1e-30), 0, 1)), 0.]
    gram = np.array([[w@frame.q, w@frame.g_cross, 0], [w@frame.g_cross, w@frame.r, 0], [0,0,0]])
    cross = np.array([w@frame.c, w@frame.h_ref, 0])
    gram += 1e-6*np.eye(3)  # Identical penalty/simplex family to previous calibration.
    opt = minimize(lambda x: x@gram@x-2*x@cross, np.ones(3)/3,
                   jac=lambda x: 2*(gram@x-cross), method='SLSQP', bounds=[(0,1)]*3,
                   constraints={'type': 'eq', 'fun': lambda x: x.sum()-1}, options={'ftol':1e-12,'maxiter':200})
    if not opt.success:
        raise RuntimeError(opt.message)
    coef = np.clip(opt.x, 0, 1); coef /= coef.sum()
    return coef[:2].tolist()


def assert_oof_eligible(caches, destination):
    if destination not in ('HepG2', 'Jurkat'):
        raise ValueError('These OOF caches cannot calibrate an unseen source-context evaluation')
    contexts = set()
    for cache in caches:
        cfg, meta = cache['cfg'], cache['meta']
        held = set(meta.context)
        if len(held) != 1 or not held <= SOURCES or contexts & held:
            raise ValueError('OOF folds must cover each source context exactly once')
        contexts |= held
        fitted = set(cache['identity']['fit_contexts'])
        if fitted != SOURCES-held or destination in fitted:
            raise ValueError('OOF source contamination')
    if contexts != SOURCES:
        raise ValueError('Missing source OOF context')


def support_plan(targets, query_count=100, pool_count=40):
    targets = sorted(set(map(str, targets)))
    if len(targets) != query_count+pool_count or pool_count < 16:
        raise ValueError('H1 target panel changed; expected fixed 100 query + 40 support targets')
    ordered = sorted(targets, key=lambda t: hashlib.sha256(('round11-query-v1:'+t).encode()).hexdigest())
    query, pool = ordered[:query_count], ordered[query_count:]
    draws = {str(d): sorted(pool, key=lambda t: hashlib.sha256(f'round11-support-v1:{d}:{t}'.encode()).hexdigest())
             for d in (101, 202, 303)}
    return {'query': query, 'support_pool': pool, 'draws': draws, 'sizes': [0,4,8,16],
            'zero_support_alpha': 1., 'selection': 'Hash of identifiers only; nested whole-target subsets'}


def score_rows(frame, a, b, **labels):
    err = mixture_error(frame, a, b)
    q = a*a*frame.q+b*b*frame.r+2*a*b*frame.g_cross
    c = a*frame.c+b*frame.h_ref
    mean = a*frame.mean_p+b*frame.mean_ref
    cov = c-mean*frame.mean_y
    denom = np.sqrt(np.maximum(q-mean*mean, 0)*np.maximum(frame.t-frame.mean_y**2, 0))
    corr = np.divide(cov, denom, out=np.full(len(frame),np.nan), where=denom>1e-20)
    cosine = np.divide(c, np.sqrt(np.maximum(q*frame.t, 0)), out=np.full(len(frame),np.nan), where=q*frame.t>1e-20)
    values=frame[['context','perturbation','seen_in_fit']].copy()
    values=values.assign(mse_delta=err,truth_energy=frame.t,prediction_energy=q,pearson=corr,cosine=cosine)
    grouped=values.groupby(['context','perturbation'],sort=True)
    result=grouped.agg(seen_in_fit=('seen_in_fit','first'),n_rows=('mse_delta','size'),
                       mse_delta=('mse_delta','mean'),truth_energy=('truth_energy','mean'),
                       prediction_energy=('prediction_energy','mean'),pearson=('pearson','mean'),cosine=('cosine','mean'))
    result=result.reset_index().rename(columns={'perturbation':'target'})
    result=result.assign(**labels,alpha=float(a),reference_weight=float(b))
    return result.to_dict('records')


def few_shot(frame, plan, seed):
    query = frame[frame.perturbation.isin(plan['query'])].reset_index(drop=True)
    support = frame[frame.perturbation.isin(plan['support_pool'])]
    results, coefficients = [], []
    for draw, order in plan['draws'].items():
        for n in plan['sizes']:
            subset = support[support.perturbation.isin(order[:n])].reset_index(drop=True)
            alpha = fit_coefficients(subset, True)[0] if n else 1.
            coefficients.append({'seed':seed,'draw':draw,'support_targets':n,'alpha':alpha,
                                 'target_ids':order[:n], 'support_rows':len(subset)})
            for method, a, b in [('adapted',alpha,0), ('unchanged',1.,0), ('zero',0.,0), ('reference',0.,1.)]:
                results += score_rows(query,a,b,protocol='h1_few_shot',seed=seed,draw=draw,support_targets=n,method=method)
    return results, coefficients


def query_target_specificity(meta, truth, prediction, plan, seed):
    """Fixed-query target-mean retrieval/shuffle diagnostic; never fits a parameter."""
    targets = sorted(plan['query'])
    lookup = meta.groupby('perturbation').indices
    y = np.stack([truth[lookup[t]].mean(0) for t in targets]).astype(float)
    p = np.stack([prediction[lookup[t]].mean(0) for t in targets]).astype(float)
    yc, pc = y-y.mean(1,keepdims=True), p-p.mean(1,keepdims=True)
    denom = np.linalg.norm(pc,axis=1)[:,None]*np.linalg.norm(yc,axis=1)[None,:]
    cross = np.divide(pc@yc.T,denom,out=np.full((len(targets),len(targets)),np.nan),where=denom>1e-20)
    # Circular shift of hash-ordered target IDs has no fixed points and uses no labels.
    order = sorted(range(len(targets)),key=lambda i:hashlib.sha256(('round11-shuffle:'+targets[i]).encode()).hexdigest())
    shuffled=np.empty(len(targets),dtype=int)
    for i,j in zip(order,np.roll(order,1)):shuffled[i]=j
    rows=[]
    for i,target in enumerate(targets):
        scores=cross[i];own=scores[i]
        greater=int(np.sum(scores>own+1e-12));ties=int(np.sum(np.isclose(scores,own,atol=1e-12,rtol=0)))
        rows.append({'seed':seed,'target':target,'shuffled_prediction_target':targets[shuffled[i]],
            'target_mean_raw_mse':float(np.mean((p[i]-y[i])**2)),
            'target_mean_shuffled_mse':float(np.mean((p[shuffled[i]]-y[i])**2)),
            'target_mean_zero_mse':float(np.mean(y[i]**2)),
            'own_target_pearson':float(own) if np.isfinite(own) else None,
            'mean_other_target_pearson':float(np.nanmean(np.delete(scores,i))) if np.isfinite(np.delete(scores,i)).any() else None,
            'true_target_midrank':greater+(ties+1)/2 if np.isfinite(own) else None,
            'top1_tie_credit':1/ties if greater==0 and ties else 0.,
            'query_targets':len(targets),'note':'Frozen raw model, gene-centered target means. Positive alpha preserves correlation; alpha=0 has no direction.'})
    return rows


def summarize(rows, folder, pairs):
    """Equal-context/target MSE; bootstrap clusters targets after averaging seeds."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows); frame.to_csv(folder/'per_target.csv',index=False)
    dimensions = [c for c in ('protocol','method','seed','draw','support_targets') if c in frame]
    summaries = []
    for key, group in frame.groupby(dimensions, dropna=False):
        for coverage, part in [('all',group),('seen',group[group.seen_in_fit]),('unseen',group[~group.seen_in_fit])]:
            if part.empty: continue
            summaries.append({**dict(zip(dimensions,key)), 'coverage':coverage,
                              'mse_delta':float(part.groupby('context').mse_delta.mean().mean()),
                              'targets':part.target.nunique(), 'contexts':part.context.nunique()})
    scores = pd.DataFrame(summaries); scores.to_csv(folder/'comparison.csv',index=False)
    dims = [c for c in dimensions if c != 'seed']+['coverage']
    scores.groupby(dims, dropna=False).mse_delta.agg(['mean','std','count']).reset_index().to_csv(folder/'seed_summary.csv',index=False)
    paired = []
    condition = [c for c in ('protocol','draw','support_targets') if c in frame]
    for key, group in frame.groupby(condition,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        for base, candidate in pairs:
            joined = group[group.method==base].merge(group[group.method==candidate],on=['context','target','seed'],suffixes=('_base','_candidate'))
            if joined.empty:continue
            joined['gain'] = joined.mse_delta_base-joined.mse_delta_candidate
            paired.append({**dict(zip(condition,key)),'baseline':base,'candidate':candidate,
                           'paired_seeds':joined.seed.nunique(),**paired_summary(joined)})
    pd.DataFrame(paired).to_csv(folder/'paired_diagnostics.csv',index=False)
    return scores


def release_inference(data, cache, rows, release_path, device='cuda'):
    release = read_json(release_path)
    folder = Path(cache['identity']['folder'])
    sha = file_sha256(folder/'endpoint.pt')
    if sha not in release['checkpoint_sha256'] or release['data_fingerprint'] != data['audit']['fingerprint']:
        raise ValueError('Checkpoint/data not frozen for Jurkat release')
    if not np.array_equal(rows, data['splits']['test']) or set(data['meta'].iloc[rows].context) != {'Jurkat'}:
        raise ValueError('Unexpected final test panel')
    model, norm = load_checkpoint(folder/'endpoint.pt', data, device)
    # Inference does not construct a target tensor or read test delta values.
    ts = {'x':torch.tensor((data['baseline']-norm['mean'])/norm['std']),
          'p':torch.tensor(data['pert_idx'],dtype=torch.long)}
    values, _ = predict(model, ts, rows, device, cache['cfg']['batch_size'])
    del model
    if device == 'cuda':torch.cuda.empty_cache()
    ref, seen = reference_responses(data, cache['cfg']['partition']['fit'])
    return values*norm['scale'], ref[data['pert_idx'][rows]], seen[data['pert_idx'][rows]]
