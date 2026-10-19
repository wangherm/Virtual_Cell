"""Published model integrity, portable adapter reload and public input boundaries."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

from vcell import release_inference as release
from test_qwen import setup


def manifest(root, path, kind='qwen_adapter', **extra):
    root.mkdir(parents=True, exist_ok=True)
    model = {'id': 'qwen_student_a' if kind == 'qwen_adapter' else 'test_method',
             'kind': kind, 'file': path.relative_to(root).as_posix(), 'sha256': release.digest(path), **extra}
    (root / 'manifest.json').write_text(json.dumps({'models': [model]}))
    return model


def test_model_integrity_and_path_confinement(tmp_path):
    root = tmp_path / 'release'
    root.mkdir()
    path = root / 'adapter.pt'
    path.write_bytes(b'not deserialized in integrity inspection')
    entry = manifest(root, path)
    assert release.verified_model(root, 'qwen_student_a')[0] == entry
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        release.verified_model(root, 'qwen_student_a')
    with pytest.raises(ValueError, match='relative'):
        release.confined(root, '../outside')


def test_published_source_catalog_matches_frozen_bytes():
    root = Path(__file__).resolve().parents[1] / 'assets/research_release_20261019'
    catalog = json.loads((root / 'catalog.json').read_text())
    assert len(catalog['models']) == 8
    algorithms = [r for r in catalog['models'] if r['kind'] == 'evolved_algorithm']
    assert len(algorithms) == 7
    for item in algorithms:
        assert release.digest(root / item['file']) == item['sha256']
    config = json.loads((root / 'qwen_student_a/config.json').read_text())
    assert config['genes'] == 2000 and config['perturbations'] == 196
    assert len((root / 'qwen_student_a/genes.txt').read_text().splitlines()) == 2000
    assert len((root / 'qwen_student_a/perturbations.txt').read_text().splitlines()) == 196


def test_control_query_rejects_panel_targets_labels_and_bad_normalization(tmp_path):
    saved = {'genes': ['g1', 'g2'], 'perturbations': ['T1'],
             'normalization': {'mean': torch.zeros(2), 'std': torch.ones(2), 'scale': 0.1}}
    d = {'baseline': np.ones((1, 2)), 'genes': np.array(['g1', 'g2']),
         'targets': np.array(['T1']), 'row_ids': np.array(['query'])}
    path = tmp_path / 'query.npz'
    np.savez(path, **d)
    _, x, p, scale = release.query_arrays(path, saved)
    assert x.shape == (1, 2) and p.tolist() == [0] and scale == 0.1
    for key, value in [('genes', np.array(['g2', 'g1'])), ('targets', np.array(['unknown'])),
                       ('query_y', np.zeros((1, 2)))]:
        changed = {**d, key: value}
        np.savez(path, **changed)
        with pytest.raises(ValueError):
            release.query_arrays(path, saved)
    np.savez(path, **d)
    saved['normalization']['std'][0] = 0
    with pytest.raises(ValueError, match='normalization'):
        release.query_arrays(path, saved)


def test_relocated_qwen_adapter_matches_original_reload(setup, tmp_path):
    from vcell.qwen import run_qwen, load_qwen_checkpoint, predict
    from vcell.train import tensors
    cfg, data = setup
    cfg['epochs'] = 1
    run_qwen(cfg)
    source = Path(cfg['output_dir']) / 'supervised/best.pt'
    root = tmp_path / 'release'
    root.mkdir()
    checkpoint = root / 'adapter.pt'
    shutil.copyfile(source, checkpoint)
    manifest(root, checkpoint)
    model, saved = load_qwen_checkpoint(source, 'cpu')
    norm = {k: v.numpy() if torch.is_tensor(v) else v for k, v in saved['normalization'].items()}
    indices = data['splits']['val'][:3]
    expected = predict(model, tensors(data, norm), indices, 'cpu', 2) * norm['scale']
    moved = tmp_path / 'relocated_base'
    shutil.copytree(Path(cfg['model']['model_id']), moved)
    query = tmp_path / 'queries.npz'
    np.savez(query, baseline=data['baseline'][indices], genes=data['genes'],
             targets=data['perturbations'][data['pert_idx'][indices]],
             row_ids=data['meta'].iloc[indices].row_id.to_numpy(dtype='U'))
    output = release.qwen_predict(root, query, moved, tmp_path / 'prediction.npz', batch_size=2)
    with np.load(output) as f:
        np.testing.assert_allclose(f['delta'], expected, atol=1e-6)
        assert f['genes'].tolist() == data['genes'].tolist()
    assert json.loads(output.with_suffix('.provenance.json').read_text())['evaluation'] is False
    (moved / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='identity'):
        release.qwen_predict(root, query, moved, tmp_path / 'bad.npz')


def method_data():
    d = {'task': np.array('transfer'), 'genes': np.array(['g1', 'g2']),
         'gene_symbols': np.array(['G1', 'G2']), 'query_ids': np.array(['q']),
         'query_targets': np.array(['T']), 'query_support': np.zeros((1, 2)),
         'query_source_context': np.array(['BXPC3_IFNG'])}
    for prefix in ('train', 'calibration'):
        d.update({prefix + '_targets': np.array(['T']), prefix + '_y': np.ones((1, 2)),
                  prefix + '_support': np.ones((1, 2)), prefix + '_source_context': np.array(['MCF7_IFNG'])})
    return d


def test_method_public_input_and_transfer_boundaries(tmp_path):
    path = tmp_path / 'public.npz'
    d = method_data()
    np.savez(path, **d)
    assert release.method_input(path, 'transfer', 2)['query_support'].shape == (1, 2)
    for change in ({'query_y': np.zeros((1, 2))}, {'query_source_context': np.array(['MCF7_IFNG'])},
                   {'query_source_context': np.array(['HT29_IFNG'])}):
        np.savez(path, **{**d, **change})
        with pytest.raises(ValueError):
            release.method_input(path, 'transfer', 2)


def test_generated_methods_fail_closed_off_linux(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    with pytest.raises(RuntimeError, match='Linux'):
        release.method_predict(tmp_path, 'anything', tmp_path / 'none', tmp_path / 'out')


@pytest.mark.skipif(sys.platform != 'linux', reason='Actual generated code requires Linux kernel isolation')
def test_published_transfer_algorithm_in_real_protected_worker(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    sha = 'b3e0f6940bc5ab734007d1cd07641843f5d102e4656033d884012f9893a90c66'
    code = repo / 'assets/research_release_20261019/methods' / (sha + '.py')
    root = tmp_path / 'release'
    root.mkdir()
    target = root / 'method.py'
    shutil.copyfile(code, target)
    manifest(root, target, kind='evolved_algorithm', task='transfer', genes=1809)
    d = method_data()
    d['genes'] = np.array(['g' + str(i) for i in range(1809)])
    d['gene_symbols'] = d['genes'].copy()
    d['query_support'] = np.full((1, 1809), 0.5, np.float32)
    for prefix in ('train', 'calibration'):
        d[prefix + '_support'] = np.ones((1, 1809), np.float32)
        d[prefix + '_y'] = np.full((1, 1809), 2., np.float32)
    query = tmp_path / 'public.npz'
    np.savez(query, **d)
    result = release.method_predict(root, 'test_method', query, tmp_path / 'out', seconds=60)
    with np.load(result) as f:
        np.testing.assert_allclose(f['delta'], np.ones((1, 1809)), atol=1e-6)
    assert json.loads((tmp_path / 'out/provenance.json').read_text())['evaluation'] is False
