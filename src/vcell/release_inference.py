"""Verified inference for archived adapters and isolated research algorithms.

The archive contains research artifacts, not a new foundation model. No provider
API is called. Generated algorithms execute only in the existing Linux worker.
"""
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
import torch


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def confined(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Expected a confined relative release path')
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('Release symlinks are not accepted')
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError('Missing or unconfined release file')
    return path


def verified_model(root, model_id):
    root = Path(root).resolve()
    manifest = json.loads(confined(root, 'manifest.json').read_text(encoding='utf-8'))
    matches = [r for r in manifest['models'] if r['id'] == model_id or r['sha256'] == model_id]
    if len(matches) != 1:
        raise ValueError('Choose one exact model ID or SHA-256 from manifest.json')
    model = matches[0]
    path = confined(root, model['file'])
    if digest(path) != model['sha256']:
        raise ValueError('Released model checksum mismatch')
    return model, path


def query_arrays(path, saved):
    with np.load(path, allow_pickle=False) as f:
        if set(f.files) != {'baseline', 'genes', 'targets', 'row_ids'}:
            raise ValueError('Query NPZ must contain exactly baseline, genes, targets and row_ids')
        d = {k: f[k] for k in f.files}
    n, g = len(d['targets']), len(saved['genes'])
    if d['baseline'].shape != (n, g) or n == 0 or not np.isfinite(d['baseline']).all():
        raise ValueError('Invalid finite control-mean matrix')
    if d['genes'].ndim != 1 or d['genes'].tolist() != saved['genes']:
        raise ValueError('Exact checkpoint gene order is required; no panel substitution or zero fill')
    if d['targets'].ndim != 1 or d['targets'].dtype.kind not in 'US':
        raise ValueError('Targets must be a one-dimensional string array')
    if d['row_ids'].shape != (n,) or d['row_ids'].dtype.kind not in 'US' or len(set(d['row_ids'])) != n:
        raise ValueError('Use unique string row IDs')
    lookup = {p: i for i, p in enumerate(saved['perturbations'])}
    if any(str(t) not in lookup for t in d['targets']):
        raise ValueError('This adapter supports only its saved perturbation vocabulary')
    norm = saved['normalization']
    mean = np.asarray(norm['mean']); std = np.asarray(norm['std']); scale = float(norm['scale'])
    if (mean.shape != (g,) or std.shape != (g,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or not (std > 0).all() or not math.isfinite(scale) or scale <= 0):
        raise ValueError('Invalid saved normalization')
    x = (d['baseline'].astype(np.float32) - mean) / std
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite normalized controls')
    return d, torch.tensor(x, dtype=torch.float32), torch.tensor([lookup[str(t)] for t in d['targets']]), scale


def qwen_predict(root, query, backbone, output, device='cpu', batch_size=16, dtype=None):
    from .qwen import QwenResponse, backbone_identity
    from .utils import device_from
    output = Path(output)
    if output.suffix != '.npz' or output.exists() or output.with_suffix('.provenance.json').exists() or batch_size < 1:
        raise ValueError('Use a fresh .npz output and positive batch size')
    entry, checkpoint = verified_model(root, 'qwen_student_a')
    if entry['kind'] != 'qwen_adapter':
        raise ValueError('Not a released Qwen adapter')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('format') != 'vcell-qwen-adapter-v1':
        raise ValueError('Unsupported checkpoint format')
    d, x, p, scale = query_arrays(query, saved)
    if not Path(backbone).is_dir():
        raise ValueError('Prepare the pinned local base snapshot before prediction')
    spec = dict(saved['model_spec'])
    spec.update(model_id=str(Path(backbone).resolve()), revision=None,
                local_files_only=True, gradient_checkpointing=False)
    # Moving the base model is supported; replacing its content is not.
    if backbone_identity(spec) != saved['backbone_identity']:
        raise ValueError('Base snapshot content differs from the checkpoint identity')
    selected_device = device_from(device)
    spec['dtype'] = dtype or ('float32' if selected_device.type == 'cpu' else saved['model_spec']['dtype'])
    if spec['dtype'] not in ('float32', 'bfloat16'):
        raise ValueError('Unsupported inference dtype')
    if spec['dtype'] == 'bfloat16' and selected_device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('CUDA bfloat16 is unavailable; choose --dtype float32')
    model = QwenResponse(len(saved['genes']), saved['perturbations'], spec,
                         saved['prompt_ids'], saved['prompt_mask'], saved.get('functional_vectors'),
                         saved.get('reference_values'), saved.get('response_basis')).to(selected_device).eval()
    state = saved['trainable_state']
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Checkpoint has nonfinite trainable parameters')
    model.load_adapter_state(state)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(p), batch_size):
            batch = model(x[start:start + batch_size].to(selected_device), p[start:start + batch_size].to(selected_device))
            predictions.append(batch.float().cpu().numpy() * scale)
    result = np.concatenate(predictions)
    if result.shape != d['baseline'].shape or not np.isfinite(result).all():
        raise ValueError('Invalid model prediction')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, delta=result, genes=d['genes'], row_ids=d['row_ids'], targets=d['targets'])
    output.with_suffix('.provenance.json').write_text(json.dumps({
        'model_sha256': entry['sha256'], 'input_sha256': digest(query), 'device': str(selected_device),
        'dtype': spec['dtype'], 'evaluation': False, 'independent_validation': False,
        'input_semantics': 'Caller-supplied original-preprocessing control means; not raw counts'
    }, indent=2) + '\n', encoding='utf-8')
    return output


def method_input(path, task, gene_count):
    from .model_agent.contracts import protected
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    allowed = {'task', 'genes', 'gene_symbols', 'reference_available', 'background_reference',
               'target_alias_labels', 'target_alias_symbols'}
    for prefix in ('train', 'calibration', 'query'):
        allowed.update(prefix + '_' + k for k in ('x', 'support', 'targets', 'control', 'prior', 'source_context', 'conditions'))
    allowed.update({'train_y', 'calibration_y', 'query_ids'})
    allowed.update('aux_' + k for k in ('baseline', 'delta', 'target', 'context', 'cell_line', 'dataset',
                                       'stimulus', 'modality', 'batch', 'time', 'unit_id'))
    if set(d) - allowed:
        raise ValueError('Unexpected fields; query truth and private grader arrays are prohibited')
    required = {'task', 'genes', 'query_ids', 'query_targets', 'query_support', 'gene_symbols'}
    required.update(prefix + '_' + k for prefix in ('train', 'calibration', 'query') for k in ('source_context',))
    required.update(prefix + '_' + k for prefix in ('train', 'calibration') for k in ('y', 'support', 'targets'))
    if required - set(d) or str(d['task']) != task or len(d['genes']) != gene_count:
        raise ValueError('Use the original prepared public worker input and matching task/panel')
    if len(set(d['genes'])) != gene_count or len(d['query_ids']) != len(d['query_targets']):
        raise ValueError('Invalid gene or query alignment')
    for prefix in ('train', 'calibration', 'query'):
        n = len(d[prefix + '_targets'])
        if n < 1 or len(d[prefix + '_source_context']) != n or d[prefix + '_support'].shape != (n, gene_count):
            raise ValueError('Invalid public source/query dimensions')
        if prefix != 'query' and d[prefix + '_y'].shape != (n, gene_count):
            raise ValueError('Invalid permitted source label dimensions')
    for key, value in d.items():
        if value.dtype.kind == 'O' or (value.dtype.kind in 'fc' and not np.isfinite(value).all()):
            raise ValueError('Object/nonfinite public data array: ' + key)
    if any(protected(str(c)) for key in ('train_source_context', 'calibration_source_context', 'query_source_context', 'aux_context')
           for c in d.get(key, [])):
        raise ValueError('Reserved contexts are unavailable through this research release')
    if task == 'transfer':
        cell = lambda x: str(x).upper().split('_')[0]
        source = {cell(c) for k in ('train_source_context', 'calibration_source_context', 'aux_context') for c in d.get(k, [])}
        if source & {cell(c) for c in d['query_source_context']}:
            raise ValueError('Transfer query cell line overlaps source labels')
    return d


def method_predict(root, model_id, input_path, output, device='cpu', steps=500, seconds=120):
    from .round22.evaluator import execute
    from .round24.modules import compose
    if sys.platform != 'linux':
        raise RuntimeError('Generated methods require the existing Linux Landlock/seccomp worker')
    output = Path(output).resolve()
    if output.exists() or steps < 1 or seconds < 1:
        raise ValueError('Use a fresh output directory and positive resource caps')
    entry, code = verified_model(root, model_id)
    if entry['kind'] != 'evolved_algorithm':
        raise ValueError('Not a released numerical algorithm')
    data = method_input(input_path, entry['task'], entry['genes'])
    budget = {'root': str(output), 'worker_root': str(output), 'seed': 17, 'device': device,
              'steps': steps, 'seconds': seconds, 'ram_bytes': 12 * 1024**3,
              'evaluation_seconds_max': seconds}
    prediction = execute(compose(code.read_text(encoding='utf-8')), data, output / 'worker', budget, sys.executable)
    np.savez_compressed(output / 'prediction.npz', delta=prediction, genes=data['genes'], row_ids=data['query_ids'])
    (output / 'provenance.json').write_text(json.dumps({
        'module_sha256': entry['sha256'], 'input_sha256': digest(input_path), 'task': entry['task'],
        'evaluation': False, 'independent_validation': False, 'source_provenance': 'Caller-supplied public source template; not independently audited'
    }, indent=2) + '\n', encoding='utf-8')
    return output / 'prediction.npz'
