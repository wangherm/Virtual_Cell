"""Refit existing frozen encoders on each Round8 fit partition, never old task heads."""
import json
from pathlib import Path

import numpy as np

from .round5_methods import load_features, fit_head
from .round7 import calibrated_mix, scalar_shrink
from .selection import row_weights, mse
from .specialization import reference_responses
from .train import source_fingerprint
from .utils import file_sha256, write_json


def prepare_kd(original, expanded, partition, paths, output, epochs=10, device='cpu'):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    path = output/'teachers.npz'
    identity = {'original': original['audit']['fingerprint'], 'expanded': expanded['audit']['fingerprint'],
        'fit': partition['fit'], 'epochs': epochs, 'source': source_fingerprint(),
        'sources': {name:[file_sha256(p),file_sha256(Path(p).with_suffix('.json'))] for name,p in paths.items()}}
    if (output/'COMPLETE.json').exists():
        done = json.loads((output/'COMPLETE.json').read_text())
        if done['identity'] != identity or done['sha256'] != file_sha256(path):
            raise ValueError('KD preparation changed')
        return path
    omap = {r:i for i,r in enumerate(original['meta'].row_id)}
    emap = {r:i for i,r in enumerate(expanded['meta'].row_id)}
    original_fit = np.array([omap[expanded['meta'].iloc[i].row_id] for i in partition['fit']
                            if expanded['meta'].iloc[i].row_id in omap], dtype=int)
    contexts = original['meta'].iloc[original_fit].context.to_numpy()
    if len(set(contexts)) < 2:
        raise ValueError('Teacher OOF requires at least two covered fit contexts')
    columns = np.array([list(original['genes']).index(g) for g in expanded['genes']])
    eligible = np.zeros(len(original['meta']), bool)
    references = np.zeros_like(original['delta'])
    folds = []
    for context in sorted(set(contexts)):
        fit, held = original_fit[contexts != context], original_fit[contexts == context]
        reference, seen = reference_responses(original, fit)
        eligible[held] = seen[original['pert_idx'][held]]
        references[held] = reference[original['pert_idx'][held]]
        folds.append((fit,held))
    predictions, provenance = [], {}
    for family, feature_path in paths.items():
        features, audit = load_features(feature_path, original)
        prediction = np.zeros_like(original['delta'])
        for fold, (fit, held) in enumerate(folds):
            cache = output/f'{family}_fold{fold}.npz'
            # Partial caches are guarded by an exact identity sidecar, including fit/held IDs.
            signature = {**identity, 'family': family, 'fold': fold, 'fit': fit.tolist(), 'held': held.tolist()}
            if cache.exists() and cache.with_suffix('.json').exists() and json.loads(cache.with_suffix('.json').read_text()).get('identity') == signature:
                saved = json.loads(cache.with_suffix('.json').read_text())
                if saved['sha256'] != file_sha256(cache):
                    raise ValueError('Partial teacher cache modified')
                with np.load(cache) as f:
                    values = f['delta']
            else:
                print(f'ROUND8 TEACHER {family} fold={fold+1}/{len(folds)} fit={len(fit)} held={len(held)}', flush=True)
                values, _ = fit_head(original, features, fit, held, epochs=epochs, device=device)
                np.savez_compressed(cache, delta=values)
                write_json(cache.with_suffix('.json'), {'identity': signature, 'sha256': file_sha256(cache)})
            prediction[held] = values
        predictions.append(prediction)
        provenance[family] = audit
    ix = np.flatnonzero(eligible)
    if len(ix) < 2:
        raise ValueError('Too few supported OOF teacher rows')
    truth = original['delta'][np.ix_(ix, columns)]
    weights = row_weights(original['meta'].iloc[ix])
    candidates = [a[np.ix_(ix,columns)] for a in predictions]
    # Reference shrinkage is also fit only inside the fit pool; this is a tuning diagnostic.
    reference = references[np.ix_(ix,columns)]
    alpha = scalar_shrink(reference, truth, weights)
    base = reference*alpha
    mix = calibrated_mix([*candidates, base], truth, weights)
    corrected = sum(w*a for w,a in zip(mix[:-1],candidates)) + mix[-1]*base
    reference_error, error = mse(base,truth,weights), mse(corrected,truth,weights)
    strength = float(np.clip(1-error/max(reference_error,1e-12),0,1))
    delta = np.zeros_like(expanded['delta']); gate = np.zeros(len(delta),dtype=np.float32)
    rows = np.array([emap[original['meta'].iloc[i].row_id] for i in ix])
    delta[rows] = corrected; gate[rows] = strength
    np.savez_compressed(path, delta=delta, gate=gate, fit=np.asarray(partition['fit']),
        row_ids=expanded['meta'].row_id.to_numpy(dtype='U'), genes=expanded['genes'])
    write_json(output/'audit.json', {'identity': identity, 'families': list(paths),
        'weights': mix.tolist(), 'kd_strength': strength, 'reference_oof_mse': reference_error,
        'mixture_oof_mse': error, 'eligible_rows': len(ix), 'unavailable_fit_rows': len(partition['fit'])-len(ix),
        'folds': [{'fit':a.tolist(),'held':b.tolist()} for a,b in folds], 'teacher_provenance': provenance,
        'pretraining_overlap': 'unverified; exploratory, including State SE (not native State ST)',
        'scope': 'Existing frozen features + newly fit VCell task heads; H1 and unsupported targets masked',
        'reliability': 'Fit-pool OOF tuning vs calibrated mean-transfer, NOT independent improvement over Qwen',
        'independent_test': 'Compare qwen_kd against qwen_modules on each untouched outer partition',
        'test_evaluated': False})
    write_json(output/'COMPLETE.json', {'identity': identity, 'sha256': file_sha256(path)})
    return path
