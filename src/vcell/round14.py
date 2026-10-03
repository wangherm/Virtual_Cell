"""Controlled local-biology heads on frozen Round12/13 backgrounds and splits."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .background_training import fingerprint
from .data import load_prepared
from .knowledge import load_knowledge
from .round11 import freeze, verify_files
from .round12 import encode_background, projection, score, save_complete
from .round13 import FactorizedResponse, identity_indices, retrieval, report as historical_report
from .selection import row_weights
from .specialization import reference_responses
from .train import source_fingerprint
from .utils import atomic_torch_save, file_sha256, seed_all, write_json


# Build independent dictionaries so each change is explicit and frozen in the plan.
BASE = dict(representation='local', interaction=False, use_id=False, background='raw',
            activation='tanh', width=32, steps_multiplier=1)
ARMS = {
    'local_additive': {},
    'local_factor': {'interaction': True},
    'local_additive_id': {'use_id': True},
    'local_factor_id': {'interaction': True, 'use_id': True},
    'background_only': {'representation': 'none'},
    'target_only': {'background': 'zero'},
    'id_background': {'representation': 'none', 'use_id': True},
    **{f'shuffled_local_{j}_factor_id': {'representation': f'shuffled_{j}', 'interaction': True, 'use_id': True} for j in (1, 2, 3)},
    'random_factor_id': {'representation': 'random', 'interaction': True, 'use_id': True},
    'clipped_additive_id': {'background': 'clip95', 'use_id': True},
    'clipped_factor_id': {'background': 'clip95', 'interaction': True, 'use_id': True},
    'unit_additive_id': {'background': 'unit', 'use_id': True},
    'unit_factor_id': {'background': 'unit', 'interaction': True, 'use_id': True},
    'linear_additive_id': {'activation': 'linear', 'use_id': True},
    'linear_factor_id': {'activation': 'linear', 'interaction': True, 'use_id': True},
    'factor_id_width16': {'width': 16, 'interaction': True, 'use_id': True},
    'factor_id_width64': {'width': 64, 'interaction': True, 'use_id': True},
    'long_additive_id': {'steps_multiplier': 2, 'use_id': True},
    'long_factor_id': {'steps_multiplier': 2, 'interaction': True, 'use_id': True},
}
ARMS = {name: {**BASE, **changes} for name, changes in ARMS.items()}


def target_representation(data, assets, fit, kind):
    """Labels never enter features; shuffle within coverage strata before fit-only PCA."""
    raw = assets['relations'][:, 2, :].copy()
    present = raw.sum(1) > 0
    perm = np.arange(len(raw))
    if kind.startswith('shuffled_'):
        rng = np.random.default_rng(14800 + int(kind.rsplit('_', 1)[1]))
        for state in (False, True):
            ix = np.flatnonzero(present == state)
            perm[ix] = rng.permutation(ix)
        raw = raw[perm]
    elif kind == 'random':
        raw = np.stack([np.random.default_rng(int.from_bytes(hashlib.sha256(
            ('round14:random:' + str(g)).encode()).digest()[:8], 'little')).normal(size=32)
            for g in data['perturbations']])
    elif kind not in ('local', 'none'):
        raise ValueError('Unknown target representation')
    values, transform = projection(raw, np.unique(data['pert_idx'][fit]), 32)
    values = np.c_[values, present.astype(np.float32)]
    if kind == 'none': values = np.zeros_like(values)
    transform.update(permutation=perm, representation=kind,
                     definition='fit-only PCA; coverage-stratified permutations; random control retains real coverage flag')
    return values, transform, present


def background_transform(background, fit, mode):
    """Fixed transformations. Radius is estimated exclusively from fit covariates."""
    norms = np.linalg.norm(background, axis=1)
    radius = float(np.quantile(norms[fit], .95) if mode == 'clip95' else np.median(norms[fit]))
    if mode == 'raw': values = background.copy()
    elif mode == 'zero': values = np.zeros_like(background)
    elif mode == 'clip95': values = background * np.minimum(1., radius / np.maximum(norms, 1e-12))[:, None]
    elif mode == 'unit': values = background * (radius / np.maximum(norms, 1e-12))[:, None]
    else: raise ValueError('Unknown background transform')
    return values.astype(np.float32), {'mode': mode, 'radius': radius, 'fit_only': True}


class ResponseHead(FactorizedResponse):
    def __init__(self, target_dim, background_dim, n_ids, rank, arm_spec):
        super().__init__(target_dim, background_dim, n_ids, rank,
                         arm_spec['interaction'], arm_spec['use_id'], arm_spec['width'])
        self.activation = arm_spec['activation']

    def forward(self, target, background, ids):
        hp = self.target(target)
        if self.use_id: hp = hp + self.ids(ids)
        hc = self.context(background)
        if self.activation == 'tanh': hp, hc = hp.tanh(), hc.tanh()
        elif self.activation != 'linear': raise ValueError('Unknown activation')
        out = self.target_out(hp) + self.context_out(hc) + self.bias
        return out + self.cross_out(hp * hc) if self.interaction else out


@torch.no_grad()
def predict(model, target, background, ids, device):
    model.eval()
    outputs = []
    for start in range(0, len(target), 512):
        end = start + 512
        outputs.append(model(torch.as_tensor(target[start:end], device=device),
                             torch.as_tensor(background[start:end], device=device),
                             torch.as_tensor(ids[start:end], device=device)).cpu().numpy())
    result = np.concatenate(outputs)
    if not np.isfinite(result).all(): raise FloatingPointError('Nonfinite inference')
    return result


def fit_head(target, background, ids, coefficients, fit, cal, cfg, arm, dest, stamp):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    arm_spec = ARMS[arm]; seed_all(cfg['seed']); device = cfg['device']
    model = ResponseHead(target.shape[1], background.shape[1], int(ids.max()) + 1,
                         coefficients.shape[1], arm_spec).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0005, weight_decay=.01)
    tx, bx = torch.as_tensor(target, device=device), torch.as_tensor(background, device=device)
    ix, y = torch.as_tensor(ids, device=device), torch.as_tensor(coefficients, device=device)
    cw = torch.as_tensor(cfg['cal_weights'], dtype=torch.float32, device=device)
    first = 0; best = float('inf'); best_step = 0; best_state = None; history = []
    steps = cfg['max_steps'] * arm_spec['steps_multiplier']
    if (dest/'last.pt').exists():
        saved = torch.load(dest/'last.pt', weights_only=True, map_location='cpu')
        if saved['fingerprint'] != stamp: raise ValueError('Checkpoint fingerprint mismatch')
        model.load_state_dict(saved['state']); optimizer.load_state_dict(saved['optimizer'])
        first, best, best_step = saved['step'], saved['best'], saved['best_step']
        best_state, history = saved['best_state'], saved['history']
    for step in range(first, steps):
        rows = np.random.default_rng(cfg['seed'] + 9000 + step).choice(fit, cfg['batch_size'], p=cfg['fit_weights'])
        optimizer.zero_grad(set_to_none=True)
        loss = (model(tx[rows], bx[rows], ix[rows]) - y[rows]).square().mean()
        if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite training loss: {arm}')
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
        done = step + 1
        if done == 1 or done % 100 == 0:
            print(f'ROUND14 {arm} step={done}/{steps} loss={loss.item():.6f}', flush=True)
        if done % cfg['save_every'] == 0 or done == steps:
            with torch.no_grad():
                validation = float((cw * (model(tx[cal], bx[cal], ix[cal]) - y[cal]).square().mean(1)).sum())
            if not np.isfinite(validation): raise FloatingPointError(f'Nonfinite calibration loss: {arm}')
            if validation < best:
                best, best_step = validation, done
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.append({'step': done, 'train_loss': float(loss.item()), 'calibration_coefficient_mse': validation})
            atomic_torch_save({'fingerprint': stamp, 'state': model.state_dict(), 'optimizer': optimizer.state_dict(),
                               'step': done, 'best': best, 'best_step': best_step, 'best_state': best_state,
                               'history': history}, dest/'last.pt')
            write_json(dest/'progress.json', {'step': done, 'max_steps': steps, 'best_step': best_step})
            pd.DataFrame(history).to_csv(dest/'history.csv', index=False)
    model.load_state_dict(best_state); model.eval()
    atomic_torch_save({'state': best_state, 'spec': arm_spec, 'target_dim': target.shape[1],
                       'background_dim': background.shape[1], 'n_ids': int(ids.max()) + 1,
                       'rank': coefficients.shape[1], 'fingerprint': stamp, 'best_step': best_step}, dest/'model.pt')
    write_json(dest/'selection.json', {'best_step': best_step, 'max_steps': steps,
        'selection': 'source calibration only; no outer labels', 'spec': arm_spec,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)})
    return model


def response_diagnostics(meta, truth, prediction, method, fit_effect_threshold):
    records = []
    for (context, target), ix in meta.groupby(['context', 'perturbation']).indices.items():
        p, y = prediction[ix].mean(0), truth[ix].mean(0)
        # Threshold fixed from fit responses. Outer masks are evaluation metrics, never training inputs.
        strong = np.abs(y) >= fit_effect_threshold
        records.append({'context': context, 'target': target, 'method': method,
            'prediction_rms': float(np.sqrt(np.mean(p*p))), 'truth_rms': float(np.sqrt(np.mean(y*y))),
            'strong_genes': int(strong.sum()),
            'strong_mse': float(np.mean((p[strong]-y[strong])**2)) if strong.any() else np.nan,
            'strong_sign_accuracy': float(np.mean(np.sign(p[strong]) == np.sign(y[strong]))) if strong.any() else np.nan,
            'fit_effect_threshold': fit_effect_threshold})
    return records


def sensitivity(model, targets, background, ids, meta, reference, basis, response_scale, device, arm):
    """Counterfactual input interventions: diagnostics, not valid alternate predictions."""
    cases = {'zero_target_features': (np.zeros_like(targets), background, ids),
             'zero_background': (targets, np.zeros_like(background), ids),
             'zero_id': (targets, background, np.zeros_like(ids))}
    # One fixed mapping per target and context; all rows of a target get the same donor features.
    shuffled = targets.copy()
    rng = np.random.default_rng(14017)
    for _, ix in meta.groupby('context').indices.items():
        groups = meta.iloc[ix].reset_index(drop=True).groupby('perturbation', sort=True).indices
        groups = list(groups.values()); order = rng.permutation(len(groups))
        for j, rows in enumerate(groups): shuffled[ix[rows]] = targets[ix[groups[order[j]][0]]]
    cases['shuffled_target_features'] = (shuffled, background, ids)
    records = []
    for name, (t, b, i) in cases.items():
        p = predict(model, t, b, i, device) @ basis * response_scale
        for (context, target), ix in meta.groupby(['context', 'perturbation']).indices.items():
            records.append({'method': arm, 'intervention': name, 'context': context, 'target': target,
                'prediction_change_rms': float(np.sqrt(np.mean((p[ix]-reference[ix])**2))),
                'raw_prediction_rms': float(np.sqrt(np.mean(reference[ix]**2)))})
    return records


def worker(cfg, resume=False):
    torch.set_num_threads(2); dest = Path(cfg['output_dir']); dest.mkdir(parents=True, exist_ok=True)
    parent13 = Path(cfg['parent_worker']); verify_files(parent13)
    c13 = json.loads((parent13/'manifest.json').read_text())['config']
    parent = Path(c13['parent_worker']); verify_files(parent)
    pm = json.loads((parent/'manifest.json').read_text()); pc = pm['config']
    if pc['variant'] != 'expanded' or pc['protocol'] != cfg['protocol'] or pc['seed'] != cfg['seed']:
        raise ValueError('Historical worker identity mismatch')
    if c13['protocol'] != cfg['protocol'] or c13['seed'] != cfg['seed']:
        raise ValueError('Round13 worker identity mismatch')
    data = load_prepared(pc['data_dir'])
    if data['audit']['fingerprint'] != pm['data']: raise ValueError('Historical data changed')
    if file_sha256(pc['knowledge']) != pm['knowledge'] or file_sha256(pc['encoder']) != pm['encoder']:
        raise ValueError('Historical dependencies changed')
    identity = {'config': cfg, 'source': source_fingerprint(),
                'parent13_complete': file_sha256(parent13/'COMPLETE.json'),
                'parent12_complete': file_sha256(parent/'COMPLETE.json'), 'arms': {a: ARMS[a] for a in cfg['arms']}}
    stamp = fingerprint(identity)
    if (dest/'manifest.json').exists() and not resume: raise ValueError('Use --resume or a new name')
    freeze(dest/'manifest.json', {'fingerprint': stamp, **identity})
    if (dest/'COMPLETE.json').exists(): verify_files(dest); return dest
    assets = load_knowledge(pc['knowledge'], data)
    fit, cal, outer = [np.asarray(pc['partition'][k], int) for k in ('fit', 'calibration', 'outer')]
    with np.load(parent/'background_transform.npz', allow_pickle=False) as f:
        basis = f['basis']; bt = {k: f[k] for k in ('mean', 'axes', 'scale')}; response_scale = float(f['response_scale'])
    latent, norm = encode_background(data, {**pc, 'device': cfg['device']})
    if not np.isclose(response_scale, norm['scale']): raise ValueError('Response normalization changed')
    bg = ((latent-bt['mean'])@bt['axes'].T/bt['scale']).astype(np.float32)
    ids, seen = identity_indices(data, fit)
    coeff = np.zeros((len(data['meta']), len(basis)), np.float32)
    coeff[np.r_[fit, cal]] = data['delta'][np.r_[fit, cal]] @ basis.T / response_scale
    meta = data['meta'].iloc[outer].reset_index(drop=True); truth = data['delta'][outer]
    effect_threshold = max(float(np.quantile(np.abs(data['delta'][fit]), .9)), 1e-6)
    fc = {**cfg, 'fit_weights': row_weights(data['meta'].iloc[fit]), 'cal_weights': row_weights(data['meta'].iloc[cal])}
    records = []; ranks = []; diagnostics = []; probes = []
    ref, _ = reference_responses(data, fit)
    baselines = {'zero': np.zeros_like(truth), 'mean_transfer': ref[data['pert_idx'][outer]]}
    cached = [(parent, n, 'r12_'+n) for n in ('string_ridge', 'id_ridge', 'none_ridge')]
    cached += [(parent13, n, 'r13_'+n) for n in ('panel_additive_ridge', 'diffusion_additive', 'diffusion_factor', 'diffusion_factor_id')]
    for folder, old, name in cached:
        with np.load(folder/old/'predictions.npz', allow_pickle=False) as f:
            if not np.array_equal(f['row_ids'], meta.row_id.to_numpy(dtype='U')) or not np.array_equal(f['genes'], data['genes']):
                raise ValueError('Baseline row/panel mismatch')
            baselines[name] = f['prediction']
    for name, pred in baselines.items():
        records += score(meta, truth, pred, name); ranks += retrieval(meta, truth, pred, name)
        diagnostics += response_diagnostics(meta, truth, pred, name, effect_threshold)
    # Reuse diagnostics, not oracle labels, from the completed parent.
    for name in ('oracle_sweep.csv', 'background_shift.csv'):
        (dest/name).write_bytes((parent13/name).read_bytes())
    geometry = []
    for arm in cfg['arms']:
        s = ARMS[arm]; folder = dest/arm; folder.mkdir(exist_ok=True)
        target, transform, available = target_representation(data, assets, fit, s['representation'])
        b, b_audit = background_transform(bg, fit, s['background'])
        for label, ix in [('fit', fit), ('calibration', cal), ('outer', outer)]:
            for context, jj in data['meta'].iloc[ix].reset_index(drop=True).groupby('context').indices.items():
                norms = np.linalg.norm(b[ix[jj]], axis=1)
                geometry.append({'method': arm, 'partition': label, 'context': context,
                    'mean_norm': float(norms.mean()), 'p95_norm': float(np.quantile(norms, .95)), **b_audit})
        if (folder/'COMPLETE.json').exists():
            verify_files(folder)
            if json.loads((folder/'COMPLETE.json').read_text())['fingerprint'] != stamp: raise ValueError('Head fingerprint changed')
            records += pd.read_csv(folder/'per_target.csv').to_dict('records')
            ranks += pd.read_csv(folder/'retrieval.csv').to_dict('records')
            diagnostics += pd.read_csv(folder/'response_diagnostics.csv').to_dict('records')
            if (folder/'sensitivity.csv').exists(): probes += pd.read_csv(folder/'sensitivity.csv').to_dict('records')
            continue
        print(f'ROUND14 HEAD {cfg["protocol"]} seed={cfg["seed"]} {arm}', flush=True)
        t = target[data['pert_idx']]
        model = fit_head(t, b, ids, coeff, fit, cal, fc, arm, folder, stamp)
        pred = predict(model, t[outer], b[outer], ids[outer], cfg['device']) @ basis * response_scale
        rows = score(meta, truth, pred, arm, available[data['pert_idx'][outer]])
        rr = retrieval(meta, truth, pred, arm); dd = response_diagnostics(meta, truth, pred, arm, effect_threshold)
        for name, rows_to_save in [('per_target', rows), ('retrieval', rr), ('response_diagnostics', dd)]:
            pd.DataFrame(rows_to_save).to_csv(folder/(name+'.csv'), index=False)
        if arm in ('local_additive', 'local_factor', 'local_additive_id', 'local_factor_id'):
            pr = sensitivity(model, t[outer], b[outer], ids[outer], meta, pred, basis, response_scale, cfg['device'], arm)
            pd.DataFrame(pr).to_csv(folder/'sensitivity.csv', index=False); probes += pr
        np.savez_compressed(folder/'predictions.npz', prediction=pred, row_ids=meta.row_id.to_numpy(dtype='U'), genes=data['genes'])
        np.savez_compressed(folder/'transforms.npz', **transform, basis=basis, response_scale=response_scale,
            background_mean=bt['mean'], background_axes=bt['axes'], background_scale=bt['scale'], fit_target_indices=seen)
        write_json(folder/'background_transform.json', b_audit)
        write_json(folder/'coverage.json', {'fit_targets': len(seen), 'outer_targets': int(meta.perturbation.nunique()),
            'outer_seen_targets': int(meta.loc[ids[outer]>0, 'perturbation'].nunique()),
            'outer_covered_targets': int(meta.loc[available[data['pert_idx'][outer]], 'perturbation'].nunique())})
        save_complete(folder, {'fingerprint': stamp}); records += rows; ranks += rr; diagnostics += dd
        del model
        if cfg['device'] == 'cuda': torch.cuda.empty_cache()
    for name, values in [('per_target', records), ('retrieval', ranks), ('response_diagnostics', diagnostics), ('geometry', geometry)]:
        pd.DataFrame(values).to_csv(dest/(name+'.csv'), index=False)
    pd.DataFrame(probes, columns=['method', 'intervention', 'context', 'target', 'prediction_change_rms', 'raw_prediction_rms']).to_csv(dest/'sensitivity.csv', index=False)
    save_complete(dest, {'fingerprint': stamp, 'test_evaluated': False})
    print(f'ROUND14 COMPLETE WORKER: {dest}', flush=True)
    return dest


def contrasts(methods):
    pairs = [('zero', m) for m in methods if m != 'zero']
    pairs += [('local_additive', 'local_factor'), ('local_additive_id', 'local_factor_id'),
              ('local_additive', 'local_additive_id'), ('local_factor', 'local_factor_id'),
              ('background_only', 'local_additive'), ('target_only', 'local_additive'),
              ('id_background', 'local_additive_id'), ('r12_string_ridge', 'local_factor_id'),
              ('r13_diffusion_factor_id', 'local_factor_id')]
    for suffix in ('additive_id', 'factor_id'):
        pairs += [(f'local_{suffix}', f'{kind}_{suffix}') for kind in ('clipped', 'unit', 'linear', 'long')]
    pairs += [('local_factor_id', f'factor_id_width{w}') for w in (16, 64)]
    pairs += [(f'shuffled_local_{j}_factor_id', 'local_factor_id') for j in (1, 2, 3)]
    pairs += [('random_factor_id', 'local_factor_id')]
    return [(a, b) for a, b in pairs if a in methods and b in methods]


def paired_report(frame, value, pairs, smaller=True):
    averaged = frame.groupby(['protocol', 'method', 'context', 'target'])[value].mean()
    out = []
    for protocol in frame.protocol.unique():
        for left, right in pairs:
            aligned = pd.concat([averaged.loc[(protocol, left)].rename('a'), averaged.loc[(protocol, right)].rename('b')], axis=1)
            if aligned.isna().any().any(): raise ValueError('Unpaired comparison')
            gain = ((aligned.a-aligned.b) if smaller else (aligned.b-aligned.a)).unstack('context')
            rng = np.random.default_rng(1701); v = gain.to_numpy(); draws = []
            for _ in range(2000):
                sample = v[rng.integers(len(v), size=len(v))]
                if np.isfinite(sample).any(0).all(): draws.append(float(np.nanmean(sample, axis=0).mean()))
            low, high = np.quantile(draws, [.025, .975]); target = gain.mean(1)
            largest = target.idxmax()
            out.append({'protocol': protocol, 'baseline': left, 'candidate': right, 'metric': value,
                'gain': float(gain.mean().mean()), 'ci_low': low, 'ci_high': high,
                'target_wins': int((target>0).sum()), 'targets': len(target),
                'largest_gain_target': largest,
                'gain_without_largest_target': float(gain.drop(index=largest).mean().mean()) if len(gain)>1 else np.nan})
    return pd.DataFrame(out)


def report(specs, root):
    root = Path(root)
    historical_report(specs, root)  # Identical per-context/per-target/seed aggregation.
    f = pd.read_csv(root/'per_target.csv'); r = pd.read_csv(root/'retrieval.csv')
    pairs = contrasts(sorted(f.method.unique()))
    pp = paired_report(f, 'mse', pairs)
    pp.to_csv(root/'paired_comparisons.csv', index=False)
    paired_report(r, 'specificity_gap', pairs, smaller=False).to_csv(root/'paired_specificity.csv', index=False)
    for name in ('response_diagnostics', 'sensitivity', 'geometry'):
        frames = []
        for cfg in specs.values():
            frame = pd.read_csv(Path(cfg['output_dir'])/(name+'.csv'))
            frame['protocol'], frame['seed'] = cfg['protocol'], cfg['seed']; frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(root/(name+'.csv'), index=False)
    # Coverage-stratified summaries retain missing targets in the main result.
    # Historical baseline rows carry a placeholder True, not actual annotation coverage.
    coverage = f[f.method.isin(ARMS)].groupby(['protocol','method','seed','context','annotation_available']).agg(mse=('mse','mean'), targets=('target','size')).reset_index()
    coverage.to_csv(root/'coverage_scores.csv', index=False)
    scope = {'test_evaluated': False, 'jurkat_scored': False, 'automatic_winner': False,
        'status': 'previously inspected development splits; three optimizer seeds are not independent dataset replications',
        'selection': 'source calibration only; no posthoc gate or output shrinkage',
        'bootstrap': '2000 target resamples after averaging optimizer seeds; joint across contexts; exploratory, no multiplicity correction',
        'sensitivity': 'counterfactual inputs, not valid alternate predictions or causal biology; zero features are diagnostic only',
        'strong_effect_metrics': 'fixed 90th percentile absolute fit-response threshold; descriptive outer masks, no selection',
        'background': 'frozen B2; radii estimated from fit covariates only',
        'controls': 'shuffle within coverage strata; random has real coverage flag; these test profile mapping, not topology in isolation',
        'id': 'unseen targets have zero ID; improvements there are indirect training effects',
        'long_training': 'twice optimizer steps, explicitly budget-mismatched diagnostic',
        'success': 'raw MSE, specificity, biological nulls and background-only controls must be read together'}
    write_json(root/'evaluation_scope.json', scope)
    summary = pd.read_csv(root/'comparison_mean.csv')
    (root/'report.html').write_text('<html><meta charset="utf-8"><title>Round14</title><h1>Round14 controlled diagnostics</h1>'
        '<p>Development results. No automatic winner. Lower MSE alone does not prove target recognition.</p>'
        + summary.to_html(index=False) + '<h2>Paired exploratory contrasts</h2>' + pp.to_html(index=False)
        + '<h2>Scope</h2><pre>' + json.dumps(scope, indent=2) + '</pre></html>', encoding='utf-8')
