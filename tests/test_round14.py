"""Round14 controls, leakage boundaries, checkpoint recovery and server workflow."""
import json
from pathlib import Path
import shutil
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest
import torch

from test_round13 import parent_fixture
from vcell.knowledge import load_knowledge
from vcell.round11 import verify_files
from vcell.round12 import save_complete
from vcell.round13 import worker as previous_worker
from vcell.round14 import (ARMS, ResponseHead, background_transform, target_representation,
    fit_head, predict, response_diagnostics, worker, report, contrasts)
from vcell.utils import write_json


def test_background_radius_fit_only_and_modes():
    x = np.array([[3, 4], [0, 2], [100, 0], [0, 0]], np.float32); fit = np.array([0, 1])
    clipped, audit = background_transform(x, fit, 'clip95')
    assert audit['radius'] == pytest.approx(4.85)
    assert np.linalg.norm(clipped[2]) == pytest.approx(audit['radius'])
    unit, _ = background_transform(x, fit, 'unit')
    np.testing.assert_allclose(np.linalg.norm(unit[:3], axis=1), 3.5)
    assert not unit[3].any()
    x[2] = -10000
    assert background_transform(x, fit, 'clip95')[1] == audit
    assert not background_transform(x, fit, 'zero')[0].any()


def test_factorial_controls_and_unknown_id():
    a, b = ARMS['local_additive_id'], ARMS['local_factor_id']
    assert {k for k in a if a[k] != b[k]} == {'interaction'}
    assert len(ARMS) == 21
    assert sum(s['steps_multiplier'] for s in ARMS.values()) == 23
    assert ('local_additive_id', 'local_factor_id') in contrasts(list(ARMS))
    torch.manual_seed(8); model = ResponseHead(3, 2, 4, 2, b)
    with torch.no_grad():
        model.cross_out.weight.fill_(.2); model.target_out.weight.fill_(.1)
    x = torch.randn(5, 3); bg = torch.randn(5, 2); ids = torch.tensor([0, 1, 2, 0, 1])
    before = model(x, bg, ids).detach()
    with torch.no_grad(): model.ids.weight[1:].fill_(3)
    after = model(x, bg, ids).detach()
    torch.testing.assert_close(before[ids==0], after[ids==0])
    assert not torch.allclose(before[ids>0], after[ids>0])


def test_shuffle_preserves_missing_coverage_and_pca_is_fit_only():
    rng = np.random.default_rng(11)
    data = {'perturbations': np.array([f'G{i}' for i in range(12)]), 'pert_idx': np.arange(12)}
    raw = rng.uniform(size=(12, 3, 6)).astype(np.float32); raw[[0, 3, 9], 2] = 0
    fit = np.arange(8); assets = {'relations': raw}
    local, transform, present = target_representation(data, assets, fit, 'local')
    for j in (1, 2, 3):
        _, shuffled, mask = target_representation(data, assets, fit, f'shuffled_{j}')
        np.testing.assert_array_equal(mask, present[shuffled['permutation']])
        assert set(shuffled['permutation']) == set(range(12))
        assert not np.array_equal(shuffled['permutation'], np.arange(12))
    raw[8:] *= 1000
    changed, changed_transform, _ = target_representation(data, assets, fit, 'local')
    np.testing.assert_array_equal(local[fit], changed[fit])
    np.testing.assert_array_equal(transform['axes'], changed_transform['axes'])


def test_exact_resume_and_outer_labels_do_not_train(tmp_path, monkeypatch):
    import vcell.round14 as engine
    rng = np.random.default_rng(1); n = 35
    t = rng.normal(size=(n, 5)).astype(np.float32); b = rng.normal(size=(n, 3)).astype(np.float32)
    ids = np.arange(n) % 5; y = rng.normal(size=(n, 2)).astype(np.float32)
    fit = np.arange(20); cal = np.arange(20, 30)
    cfg = dict(seed=17, device='cpu', max_steps=6, batch_size=8, save_every=2,
               fit_weights=np.ones(20)/20, cal_weights=np.ones(10)/10)
    full = fit_head(t, b, ids, y, fit, cal, cfg, 'local_factor_id', tmp_path/'full', 'stamp')
    expected = predict(full, t, b, ids, 'cpu')
    original = engine.atomic_torch_save
    def interrupt(value, path):
        original(value, path)
        if Path(path).name == 'last.pt' and value['step'] == 2: raise InterruptedError('interrupt')
    monkeypatch.setattr(engine, 'atomic_torch_save', interrupt)
    with pytest.raises(InterruptedError): fit_head(t, b, ids, y, fit, cal, cfg, 'local_factor_id', tmp_path/'resume', 'stamp')
    monkeypatch.setattr(engine, 'atomic_torch_save', original)
    recovered = fit_head(t, b, ids, y, fit, cal, cfg, 'local_factor_id', tmp_path/'resume', 'stamp')
    np.testing.assert_array_equal(expected, predict(recovered, t, b, ids, 'cpu'))
    y[30:] = 1e6
    poisoned = fit_head(t, b, ids, y, fit, cal, cfg, 'local_factor_id', tmp_path/'poison', 'stamp')
    np.testing.assert_array_equal(expected, predict(poisoned, t, b, ids, 'cpu'))
    saved = torch.load(tmp_path/'full/model.pt', weights_only=True)
    reloaded = ResponseHead(saved['target_dim'], saved['background_dim'], saved['n_ids'], saved['rank'], saved['spec'])
    reloaded.load_state_dict(saved['state'])
    np.testing.assert_array_equal(expected, predict(reloaded, t, b, ids, 'cpu'))
    with pytest.raises(ValueError, match='fingerprint'):
        fit_head(t, b, ids, y, fit, cal, cfg, 'local_factor_id', tmp_path/'resume', 'wrong')


def fixture(tmp_path):
    data, pc, old = parent_fixture(tmp_path)
    old['arms'] = ['panel_additive_ridge', 'diffusion_additive', 'diffusion_factor', 'diffusion_factor_id']
    previous_worker(old)
    cfg = dict(parent_worker=old['output_dir'], seed=17, protocol='target', device='cpu',
               arms=list(ARMS), max_steps=2, save_every=1, batch_size=8, output_dir=str(tmp_path/'round14'))
    return data, pc, old, cfg


def test_all_heads_report_and_integrity(tmp_path):
    data, pc, old, cfg = fixture(tmp_path)
    assets = load_knowledge(pc['knowledge'], data); fit = pc['partition']['fit']
    local, _, available = target_representation(data, assets, fit, 'local')
    for kind in ['none', 'random', 'shuffled_1', 'shuffled_2', 'shuffled_3']:
        values, transform, coverage = target_representation(data, assets, fit, kind)
        assert values.shape == local.shape
        np.testing.assert_array_equal(coverage, available)
        np.testing.assert_array_equal(available[transform['permutation']], available)
        if kind == 'none': assert not values.any()
    worker(cfg); root = Path(cfg['output_dir']); verify_files(root); worker(cfg, True)
    dest = tmp_path/'reports'; dest.mkdir(); report({'one': cfg}, dest)
    summary = pd.read_csv(dest/'comparison_mean.csv')
    assert len(summary) == len(ARMS) + 9
    assert np.isfinite(summary.mse_mean).all()
    assert set(pd.read_csv(dest/'coverage_scores.csv').method) <= set(ARMS)
    assert (root/'long_factor_id/progress.json').is_file()
    assert json.loads((root/'long_factor_id/progress.json').read_text())['step'] == 4
    probes = pd.read_csv(dest/'sensitivity.csv')
    # Target holdout must have zero ID impact at inference.
    np.testing.assert_array_equal(probes.query("intervention=='zero_id'").prediction_change_rms, 0)
    pairs = pd.read_csv(dest/'paired_comparisons.csv')
    assert ((pairs.baseline=='local_additive_id') & (pairs.candidate=='local_factor_id')).any()
    diagnostics = pd.read_csv(dest/'response_diagnostics.csv')
    assert diagnostics.fit_effect_threshold.nunique() == 1
    (root/'local_factor/model.pt').write_bytes(b'bad')
    with pytest.raises(ValueError, match='checksum'): worker(cfg, True)


def test_response_amplitude_and_direction():
    meta = pd.DataFrame({'context':['A','A'], 'perturbation':['X','X']})
    y = np.array([[2., -.1], [2., -.1]])
    record = response_diagnostics(meta, y, -y, 'wrong_sign', 1.)[0]
    assert record['strong_genes'] == 1 and record['strong_sign_accuracy'] == 0
    assert record['strong_mse'] == 16
    assert record['prediction_rms'] == record['truth_rms']


def test_launcher_end_to_end_and_light_archive(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    import run_round14 as launcher
    _, _, old, cfg = fixture(tmp_path)
    previous = tmp_path/'parent13'; parent_worker = previous/'models/target_seed17'
    parent_worker.parent.mkdir(parents=True); shutil.copytree(old['output_dir'], parent_worker)
    write_json(previous/'plan.json', {'experiment':'round13', 'seeds':[17], 'protocols':['target']})
    save_complete(previous)
    def inline(jobs, root, *args, **kwargs):
        result = {}
        for name, job in jobs.items():
            command = job['cmd']; current = json.loads(Path(command[command.index('--worker-config')+1]).read_text())
            worker(current, '--resume' in command); result[name] = 0
        return result
    monkeypatch.setattr(launcher, 'run_jobs', inline)
    work = tmp_path/'work'; work.mkdir()
    args = ['--work-dir', str(work), '--round13-run', str(previous), '--seeds', '17', '--protocols', 'target',
            '--device', 'cpu', '--arms', 'local_additive_id', 'local_factor_id', '--max-steps', '2', '--save-every', '1']
    root = launcher.main(args); verify_files(root); launcher.main(args+['--resume'])
    budget = json.loads((root/'budget.json').read_text())
    assert budget['optimizer_steps'] == 4 and budget['downloads'] == 0
    with tarfile.open(root/'round14_review_light.tar.gz') as t:
        assert 'sensitivity.csv' in t.getnames()
        assert not any(n.endswith(('.npz', '.npy', '.pt')) for n in t.getnames())
    with pytest.raises(ValueError): launcher.main(args+['--resume', '--max-steps', '3'])
