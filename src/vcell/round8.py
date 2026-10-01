"""Controlled semantic/relational perturbation experiments; sealed test labels."""
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import load_prepared
from .fixed_training import response_basis, step_rows
from .knowledge import load_knowledge
from .qwen import QwenResponse, backbone_identity
from .round7 import internal_partition, scalar_shrink, calibrated_mix
from .selection import mse, row_weights
from .specialization import reference_responses
from .train import normalization, source_fingerprint, tensors
from .utils import atomic_torch_save, file_sha256, seed_all, write_json

ARMS = ('qwen_base', 'mlp_base', 'mlp_semantic', 'qwen_semantic', 'qwen_graph',
        'qwen_modules', 'qwen_kd', 'shuffled_semantic', 'random_graph')
PROTOCOLS = ('context', 'target', 'double')


def partitions(data, protocol, seed=817):
    """Splits use identifiers only. Calibration and evaluation never enter fitting."""
    train, val = data['splits']['train'], data['splits']['val']
    p = data['meta'].perturbation.to_numpy()
    if protocol == 'context':
        internal = internal_partition(data)
        parts = {'fit': sorted(internal['fit']+internal['outer']),
                 'calibration': internal['calibration'], 'outer': val.tolist()}
    elif protocol in ('target', 'double'):
        targets = sorted(set(p[train]) & (set(p[val]) if protocol == 'double' else set(p[train])))
        if len(targets) < 10:
            raise ValueError(f'{protocol}: need >=10 eligible targets for disjoint target splits')
        order = np.random.default_rng(seed).permutation(targets)
        n = max(2, len(order)//5)
        held, cal = set(order[:n]), set(order[n:2*n])
        parts = {'fit': [int(i) for i in train if p[i] not in held|cal],
                 'calibration': [int(i) for i in train if p[i] in cal],
                 'outer': [int(i) for i in (val if protocol == 'double' else train) if p[i] in held]}
    else:
        raise ValueError('Unknown protocol')
    validate_partition(data, parts)
    return parts


def validate_partition(data, parts):
    groups = [set(parts[k]) for k in ('fit', 'calibration', 'outer')]
    allowed = set(data['splits']['train']) | set(data['splits']['val'])
    if any(not g or not g <= allowed or len(g) != len(parts[k])
           for k,g in zip(('fit', 'calibration', 'outer'), groups)):
        raise ValueError('Invalid/empty partition or access to sealed test rows')
    if not groups[0] <= set(data['splits']['train']) or not groups[1] <= set(data['splits']['train']):
        raise ValueError('Fitting and calibration must use training-pool rows only')
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Partition overlap')


class KnowledgeResponse(nn.Module):
    """Fixed semantic evidence + context-weighted typed neighbors; numeric module head.

    Module predictions are supervised only in E/F. They are not natural-language
    chains of thought, nor assertions of causal pathway activation.
    """
    def __init__(self, data, knowledge, spec, arm, basis):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown knowledge arm')
        self.arm = arm
        semantic = knowledge['semantic'].copy()
        relations = knowledge['relations'].copy()
        if arm == 'shuffled_semantic':
            semantic = semantic[np.random.default_rng(818).permutation(len(semantic))]
        if arm == 'random_graph':
            # Permuting node labels preserves each typed row's degree and weights.
            relations = relations[:, :, np.random.default_rng(818).permutation(relations.shape[2])]
        for name, value in [('semantic', semantic), ('gene_semantic', knowledge['gene_semantic']),
                            ('relations', relations), ('module_matrix', knowledge['modules']), ('basis', basis)]:
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
        self.use_semantic = arm not in ('qwen_base', 'mlp_base')
        self.use_graph = arm in ('qwen_graph', 'qwen_modules', 'qwen_kd', 'random_graph')
        if arm.startswith('mlp'):
            h = 256
            self.control = nn.Sequential(nn.Linear(len(data['genes']), h), nn.GELU(), nn.LayerNorm(h))
            self.target = nn.Embedding(len(data['perturbations']), h)
            self.combine = nn.Sequential(nn.Linear(2*h, h), nn.GELU(), nn.LayerNorm(h))
        else:
            self.qwen = QwenResponse(len(data['genes']), data['perturbations'], spec)
            # This wrapper supplies its own identical low-rank head in every arm.
            for param in self.qwen.output.parameters():
                param.requires_grad_(False)
            h = self.qwen.backbone.config.hidden_size
        dim = semantic.shape[1]
        self.semantic_projector = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.LayerNorm(h))
        self.relation_projector = nn.Sequential(nn.Linear(dim+1, h), nn.GELU(), nn.LayerNorm(h))
        self.relation_types = nn.Parameter(torch.randn(3, h)*.02)
        self.module_head = nn.Linear(h, len(self.module_matrix))
        self.output = nn.Sequential(nn.LayerNorm(h+len(self.module_matrix)), nn.Linear(h+len(self.module_matrix), len(basis)))

    def forward(self, x, p):
        if self.arm.startswith('mlp'):
            target = self.semantic_projector(self.semantic[p]) if self.use_semantic else self.target(p)
            hidden = self.combine(torch.cat([self.control(x), target], dim=1))
        else:
            q = self.qwen
            control = q.input_projector(x).reshape(len(x), q.n_tokens, -1)
            extra = []
            if self.use_semantic:
                extra.append(self.semantic_projector(self.semantic[p]).unsqueeze(1))
            if self.use_graph:
                # Relative control state modulates support without declaring low expression absent.
                edges = self.relations[p] * (.1 + torch.sigmoid(x[:, None, :]))
                denominator = edges.sum(2, keepdim=True)
                neighbors = (edges @ self.gene_semantic) / denominator.clamp_min(1e-8)
                available = (denominator > 0).float()
                extra.append(self.relation_projector(torch.cat([neighbors, available], dim=2)) + self.relation_types)
            words = q.backbone.get_input_embeddings()(q.prompt_ids[p])
            prefix = torch.cat([control, *extra], dim=1).to(words.dtype)
            inputs = torch.cat([prefix, words, q.prediction_token.expand(len(x), -1, -1).to(words.dtype)], dim=1)
            mask = torch.cat([torch.ones((len(x), len(prefix[0])), dtype=torch.long, device=x.device),
                              q.prompt_mask[p], torch.ones((len(x), 1), dtype=torch.long, device=x.device)], dim=1)
            hidden = q.backbone(inputs_embeds=inputs, attention_mask=mask,
                position_ids=(mask.cumsum(-1)-1).clamp_min(0), use_cache=False).last_hidden_state[:, -1].float()
        modules = self.module_head(hidden)
        delta = self.output(torch.cat([hidden, modules], dim=1)) @ self.basis
        return delta, modules

    def adapter_state(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}

    def restore(self, values):
        expected = set(self.adapter_state())
        if set(values) != expected:
            raise ValueError('Round8 trainable checkpoint keys changed')
        self.load_state_dict(values, strict=False)


@torch.no_grad()
def predict(model, ts, rows, device, batch):
    model.eval()
    result, modules = [], []
    for start in range(0, len(rows), batch):
        ix = rows[start:start+batch]
        p, m = model(ts['x'][ix].to(device), ts['p'][ix].to(device))
        result.append(p.float().cpu().numpy()); modules.append(m.float().cpu().numpy())
    return np.concatenate(result), np.concatenate(modules)


def worker(cfg, resume=False):
    torch.set_num_threads(2)
    data = load_prepared(cfg['data_dir'])
    validate_partition(data, cfg['partition'])
    assets = load_knowledge(cfg['knowledge'], data)
    actual = {'config': cfg, 'data': data['audit']['fingerprint'], 'source': source_fingerprint(),
              'knowledge': file_sha256(cfg['knowledge']), 'backbone': backbone_identity(cfg['model']),
              'teacher': file_sha256(cfg['teacher_cache']) if cfg.get('teacher_cache') else None}
    stamp = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
    dest = Path(cfg['output_dir'])
    if dest.exists() and any(dest.iterdir()):
        if not resume or not (dest/'manifest.json').exists() or json.loads((dest/'manifest.json').read_text())['fingerprint'] != stamp:
            raise ValueError('Resume requires identical code, knowledge, teacher, data and config')
        if (dest/'COMPLETE.json').exists():
            complete = json.loads((dest/'COMPLETE.json').read_text())
            for name, digest in complete['files'].items():
                if file_sha256(dest/name) != digest:
                    raise ValueError('Completed Round8 artifact changed: '+name)
            return dest
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest/'manifest.json', {'fingerprint': stamp, **actual, 'test_evaluated': False})
    fit, cal, outer = [np.asarray(cfg['partition'][k], dtype=int) for k in ('fit', 'calibration', 'outer')]
    local = copy.copy(data); local['splits'] = {**data['splits'], 'train': fit}
    norm = normalization(local)
    basis, audit = response_basis(data, fit, cfg['rank'])
    write_json(dest/'basis_audit.json', audit)
    seed_all(cfg['seed'])
    device = cfg['device']
    model = KnowledgeResponse(data, assets, cfg['model'], cfg['arm'], basis).to(device)
    ts = tensors(data, norm)
    kd, gate = None, None
    if cfg['arm'] == 'qwen_kd':
        with np.load(cfg['teacher_cache'], allow_pickle=False) as f:
            if not np.array_equal(f['fit'], fit) or not np.array_equal(f['row_ids'], data['meta'].row_id.to_numpy(dtype='U')) or not np.array_equal(f['genes'], data['genes']):
                raise ValueError('KD cache split/order mismatch')
            if f['delta'].shape != data['delta'].shape or f['gate'].shape != (len(data['meta']),) or not np.isfinite(f['delta']).all() or not np.isfinite(f['gate']).all() or (f['gate']<0).any() or (f['gate']>1).any():
                raise ValueError('Invalid KD values/mask')
            kd, gate = torch.tensor(f['delta']/norm['scale']), torch.tensor(f['gate'])
        if (gate.numpy()[np.setdiff1d(np.arange(len(gate)), fit)] != 0).any():
            raise ValueError('KD outside fit rows')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=.01)
    first, history = 0, []
    if resume and (dest/'last.pt').exists():
        saved = torch.load(dest/'last.pt', map_location='cpu', weights_only=True)
        if saved['fingerprint'] != stamp:
            raise ValueError('Resume checkpoint mismatch')
        model.restore(saved['state']); optimizer.load_state_dict(saved['optimizer'])
        first, history = saved['step'], saved['history']
    timer = time.monotonic()
    for step in range(first, cfg['max_steps']):
        seed_all(cfg['seed']+10000+step)
        rows = step_rows(fit, cfg['seed'], step, cfg['batch_size']*cfg['gradient_accumulation'])
        model.train(); optimizer.zero_grad(set_to_none=True)
        losses = np.zeros(3)
        for start in range(0, len(rows), cfg['batch_size']):
            ix = rows[start:start+cfg['batch_size']]
            y = ts['y'][ix].to(device)
            pred, module = model(ts['x'][ix].to(device), ts['p'][ix].to(device))
            sup = (pred-y).square().mean()
            aux = (module-y @ model.module_matrix.T).square().mean() if cfg['arm'] in ('qwen_modules', 'qwen_kd') else sup*0
            distill = ((pred-kd[ix].to(device)).square().mean(1)*gate[ix].to(device)).mean() if kd is not None else sup*0
            loss = sup + cfg['module_weight']*aux + cfg['kd_weight']*distill
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite Round8 loss')
            (loss*len(ix)/len(rows)).backward()
            losses += np.array([sup.item(), aux.item(), distill.item()])*len(ix)/len(rows)
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        done = step+1
        if done == 1 or done % 50 == 0:
            print(f'ROUND8 {cfg["arm"]} step={done}/{cfg["max_steps"]} sup={losses[0]:.5f} module={losses[1]:.5f} kd={losses[2]:.5f}', flush=True)
        if done % cfg['save_every'] == 0 or done == cfg['max_steps']:
            # No outer labels used for stopping or checkpoint selection.
            history.append({'step': done, 'supervised_loss': float(losses[0]), 'module_loss': float(losses[1]),
                            'kd_loss': float(losses[2]), 'interval_seconds': time.monotonic()-timer})
            atomic_torch_save({'fingerprint': stamp, 'state': model.adapter_state(), 'optimizer': optimizer.state_dict(),
                              'step': done, 'history': history}, dest/'last.pt')
            pd.DataFrame(history).to_csv(dest/'history.csv', index=False)
            write_json(dest/'progress.json', {'step': done, 'max_steps': cfg['max_steps'], 'test_evaluated': False})
            timer = time.monotonic()
    atomic_torch_save({'format': 'vcell-round8-v1', 'fingerprint': stamp, 'config': cfg,
        'state': model.adapter_state(), 'basis': torch.tensor(basis), 'backbone': actual['backbone'],
        'knowledge_sha256': actual['knowledge'], 'normalization': {k:torch.tensor(v) for k,v in norm.items()},
        'genes': data['genes'].tolist(), 'perturbations': data['perturbations'].tolist()}, dest/'endpoint.pt')
    cp, cm = predict(model, ts, cal, device, cfg['batch_size'])
    op, om = predict(model, ts, outer, device, cfg['batch_size'])
    cp *= norm['scale']; op *= norm['scale']; cm *= norm['scale']; om *= norm['scale']
    ref, seen = reference_responses(data, fit)
    cr, ore = ref[data['pert_idx'][cal]], ref[data['pert_idx'][outer]]
    weights = row_weights(data['meta'].iloc[cal])
    alpha = scalar_shrink(cp, data['delta'][cal], weights)
    mix = calibrated_mix([cp, cr, np.zeros_like(cp)], data['delta'][cal], weights)
    np.savez_compressed(dest/'predictions.npz', genes=data['genes'],
        calibration_row_ids=data['meta'].iloc[cal].row_id.to_numpy(dtype='U'),
        outer_row_ids=data['meta'].iloc[outer].row_id.to_numpy(dtype='U'),
        calibration=cp, outer=op, calibrated=mix[0]*op+mix[1]*ore, shrunk=alpha*op,
        reference=ore, module_prediction=om, module_names=assets['module_names'])
    write_json(dest/'calibration.json', {'alpha': alpha, 'mix': mix.tolist(),
        'fit_rows': fit.tolist(), 'calibration_rows': cal.tolist(), 'outer_rows': outer.tolist(),
        'reference_seen_outer': int(seen[data['pert_idx'][outer]].sum()),
        'module_mse': float(np.mean((om-data['delta'][outer]@assets['modules'].T)**2)),
        'selection': 'fixed final optimizer step, no outer early stopping', 'test_evaluated': False})
    write_json(dest/'COMPLETE.json', {'fingerprint': stamp, 'test_evaluated': False,
        'files': {n:file_sha256(dest/n) for n in ('endpoint.pt','predictions.npz','calibration.json')}})
    print(f'ROUND8 STUDENT COMPLETE: {dest}', flush=True)
    return dest


def load_checkpoint(path, data, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    cfg = saved['config']
    if saved['format'] != 'vcell-round8-v1' or saved['genes'] != data['genes'].tolist() or saved['perturbations'] != data['perturbations'].tolist():
        raise ValueError('Round8 checkpoint vocabulary mismatch')
    if file_sha256(cfg['knowledge']) != saved['knowledge_sha256'] or backbone_identity(cfg['model']) != saved['backbone']:
        raise ValueError('Round8 checkpoint dependency changed')
    model = KnowledgeResponse(data, load_knowledge(cfg['knowledge'], data), cfg['model'], cfg['arm'], saved['basis']).to(device)
    model.restore(saved['state']); model.eval()
    norm = {k:v.numpy() if v.ndim else float(v) for k,v in saved['normalization'].items()}
    return model, norm


def report(data, specs, root):
    records, grouped, arrays = [], [], {}
    for name, cfg in specs.items():
        folder = Path(cfg['output_dir'])
        if not (folder/'COMPLETE.json').exists():
            continue
        rows = np.asarray(cfg['partition']['outer'])
        meta = data['meta'].iloc[rows].reset_index(drop=True)
        truth = data['delta'][rows]
        with np.load(folder/'predictions.npz') as f:
            if not np.array_equal(f['outer_row_ids'], meta.row_id.to_numpy(dtype='U')) or not np.array_equal(f['genes'], data['genes']):
                raise ValueError('Report prediction alignment mismatch')
            for mode in ('outer', 'shrunk', 'calibrated', 'reference', 'zero'):
                prediction = f[mode] if mode != 'zero' else np.zeros_like(truth)
                error = ((prediction-truth)**2).mean(1)
                unit_errors = []
                for (context, target), ix in meta.groupby(['context','perturbation'], sort=True).indices.items():
                    e = float(error[ix].mean()); unit_errors.append(e)
                    grouped.append({'protocol': cfg['protocol'], 'arm': cfg['arm'], 'seed': cfg['seed'],
                        'mode': mode, 'context': context, 'target': target, 'n_rows': len(ix), 'mse_delta': e})
                records.append({'protocol': cfg['protocol'], 'arm': cfg['arm'], 'seed': cfg['seed'], 'mode': mode,
                    'mse_delta': float(error @ row_weights(meta)), 'unit_equal_mse': float(np.mean(unit_errors)),
                    'n_rows': len(rows), 'n_units': len(unit_errors), 'n_genes': len(data['genes'])})
                arrays[(cfg['protocol'], cfg['arm'], cfg['seed'], mode)] = error
    frame = pd.DataFrame(records)
    frame.to_csv(Path(root)/'comparison.csv', index=False)
    pd.DataFrame(grouped).to_csv(Path(root)/'per_target.csv', index=False)
    if len(frame):
        frame.groupby(['protocol','arm','mode']).mse_delta.agg(['mean','std','count']).reset_index().to_csv(Path(root)/'seed_summary.csv', index=False)
        pairs = [('mlp_base','mlp_semantic'),('qwen_base','qwen_semantic'),('mlp_semantic','qwen_semantic'),
                 ('qwen_semantic','qwen_graph'),('qwen_graph','qwen_modules'),('qwen_modules','qwen_kd'),
                 ('shuffled_semantic','qwen_semantic'),('random_graph','qwen_graph')]
        comparisons=[]
        for (protocol,seed,mode), group in frame.groupby(['protocol','seed','mode']):
            scores=group.set_index('arm').mse_delta.to_dict()
            for baseline,candidate in pairs:
                if baseline in scores and candidate in scores:
                    comparisons.append({'protocol':protocol,'seed':int(seed),'mode':mode,'baseline':baseline,
                        'candidate':candidate,'baseline_mse':scores[baseline],'candidate_mse':scores[candidate],
                        'mse_improvement':scores[baseline]-scores[candidate],
                        'relative_improvement_percent':100*(scores[baseline]-scores[candidate])/max(scores[baseline],1e-12)})
        pd.DataFrame(comparisons).to_csv(Path(root)/'paired_comparisons.csv',index=False)
    write_json(Path(root)/'evaluation_scope.json', {'test_evaluated': False,
        'context': 'All original validation rows; fit/calibration use disjoint training-pool batches',
        'target': 'Held-out targets in training contexts; calibration targets are separately held out',
        'double': 'Held-out targets in validation context; no same-target training labels',
        'primary': 'context/target macro MSE; unit-equal sensitivity also reported',
        'limitations': ['Batch identifiers may be pseudo-batches; inspect original aggregation provenance',
            'Development protocols, not a newly independent final test',
            'Public text and pretrained teacher overlap with held-out biology is unverified',
            'No claims of causal reasoning from module predictions or text embeddings']})
    return frame
