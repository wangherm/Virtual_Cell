"""Typed source-only research actions and measured feedback, without a population DB."""
from dataclasses import asdict, dataclass
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..model_agent.artifacts import read
from ..model_agent.contracts import digest
from ..utils import file_sha256
from ..round22.metrics import macro

ACTIONS = {'repair_candidate', 'refine_parent', 'crossover_methods',
           'explore_new_mechanism', 'run_source_probe', 'stop_unproductive_branch'}
PROBES = {'none', 'reference_coverage', 'context_losses', 'target_losses', 'strong_response'}
PREDICTION_METRICS = {'valid', 'mse', 'strong_mse', 'specificity_gap', 'gain_over_fixed'}


def quota_failure(record):
    """An exhausted campaign allowance cannot be repaired; a slow/OOM method can."""
    text=(record.get('error') or '').lower()
    return any(s in text for s in ('evaluation-time cap exhausted','task search compute quota exhausted'))


@dataclass(frozen=True)
class ResearchAction:
    action: str
    dominant_failure: str
    hypothesis: str
    proposed_method_family: str
    parent_code_sha256: str | None
    second_parent_sha256: str | None
    experiment_priority: str
    requested_source_diagnostic: str
    success_prediction: dict
    fallback_plan: str
    reason: str
    evidence_ids: list
    repeat_share: float = .6

    @classmethod
    def parse(cls, payload, evidence):
        if not isinstance(payload, dict):
            raise ValueError('Research action must be an object')
        value = cls(**payload)
        if value.action not in ACTIONS or value.requested_source_diagnostic not in PROBES:
            raise ValueError('Unknown bounded research action/probe')
        for name in ('dominant_failure', 'hypothesis', 'proposed_method_family',
                     'experiment_priority', 'fallback_plan', 'reason'):
            text = getattr(value, name)
            if not isinstance(text, str) or not 1 <= len(text) <= 1600:
                raise ValueError('Bounded nonempty research text required: ' + name)
        allowed = {r['module_sha256'] for r in evidence['candidates']}
        for parent in (value.parent_code_sha256, value.second_parent_sha256):
            if parent is not None and parent not in allowed:
                raise ValueError('Parent not present in this source-search evidence')
        if value.action in {'repair_candidate', 'refine_parent', 'stop_unproductive_branch'} and value.parent_code_sha256 is None:
            raise ValueError('This action requires a measured parent')
        if value.action == 'crossover_methods' and (value.second_parent_sha256 is None or value.second_parent_sha256 == value.parent_code_sha256 or value.parent_code_sha256 is None):
            raise ValueError('Crossover requires two different measured parents')
        if value.action == 'run_source_probe' and value.requested_source_diagnostic == 'none':
            raise ValueError('A source probe must name an executable diagnostic')
        if not isinstance(value.evidence_ids, list) or not value.evidence_ids or not set(value.evidence_ids).issubset(allowed):
            raise ValueError('Source evidence IDs are required')
        if not .4 <= value.repeat_share <= .7:
            raise ValueError('Repeat allocation must be within 40-70 percent')
        p = value.success_prediction
        if not isinstance(p, dict) or set(p) != {'metric', 'direction', 'minimum_change'} or p['metric'] not in PREDICTION_METRICS or p['direction'] not in ('increase', 'decrease'):
            raise ValueError('Typed success prediction required')
        if not isinstance(p['minimum_change'], (int, float)) or not np.isfinite(p['minimum_change']) or p['minimum_change'] < 0:
            raise ValueError('Nonnegative finite predicted change required')
        return value


def parse_json(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return json.loads(text)


def source_records(pool):
    """Read exactly search records, never confirmation or imported review prose."""
    result = []
    for path in sorted((Path(pool) / 'assessments').glob('*/assessment.json')):
        r = read(path)
        if r['identity']['phase'] == 'search':
            if digest({k:v for k,v in r.items() if k!='sha256'})!=r['sha256']:
                raise ValueError('Source assessment checksum changed')
            for name in ('module.py','scores.csv','baseline.csv'):
                if name in r['files'] and file_sha256(path.parent/name)!=r['files'][name]:
                    raise ValueError('Source assessment payload changed: '+name)
            result.append(r)
    return result


def distinct(records, excluded=()):
    seen = set(excluded); result = []
    for r in sorted(records, key=lambda r: r.get('finished_at', 0)):
        if r['status'] == 'valid' and r['behavior_sha256'] not in seen:
            seen.add(r['behavior_sha256']); result.append(r)
    return result


def frontier(records, limit=8):
    valid = [r for r in records if r['status'] == 'valid']
    def vector(r):
        m = r['metrics']['public']
        return (m['mse'], m.get('strong_mse') if m.get('strong_mse') is not None else float('inf'),
                -(m.get('specificity_gap') or 0),int(m.get('shrinkage_only',False)),
                r.get('worker_seconds',float('inf')),r.get('code_lines',100000))
    result = []
    for r in valid:
        v = vector(r)
        if not any(all(x <= y for x, y in zip(vector(s), v)) and any(x < y for x, y in zip(vector(s), v)) for s in valid):
            result.append(r)
    # A compact evidence view; native Shinka remains the only population/archive.
    chosen = sorted(result, key=lambda r: r['metrics']['public']['mse'])[:limit]
    for r in sorted(valid, key=lambda r: -(r['metrics']['public'].get('specificity_gap') or 0))[:2]:
        if r not in chosen: chosen.append(r)
    return chosen[:limit + 2]


def loss_card(record):
    folder = Path(record['directory'])
    for name in ('scores.csv','baseline.csv'):
        if file_sha256(folder/name)!=record['files'][name]:
            raise ValueError('Source diagnostic payload changed')
    frame = pd.read_csv(folder / 'scores.csv'); baseline = pd.read_csv(folder / 'baseline.csv')
    keys = ['row_id', 'context', 'target']
    a = frame.groupby(keys, as_index=False).mse.mean()
    b = baseline.groupby(keys, as_index=False).mse.mean().rename(columns={'mse': 'baseline_mse'})
    joined = a.merge(b, on=keys, validate='one_to_one')
    joined['gain'] = 1 - joined.mse / joined.baseline_mse.clip(lower=1e-12)
    paired = joined.groupby(['context', 'target'], as_index=False)[['mse', 'baseline_mse', 'gain']].mean()
    return {'best_targets': paired.nlargest(3, 'gain').to_dict('records'),
            'worst_targets': paired.nsmallest(3, 'gain').to_dict('records'),
            'by_context': joined.groupby('context')[['mse', 'baseline_mse', 'gain']].mean().reset_index().to_dict('records'),
            'reference_coverage': frame.groupby('context').agg(rows=('row_id', 'size'),
                available=('reference_available', 'sum'), common=('common_support', 'sum'), unseen=('globally_unknown', 'sum')).reset_index().to_dict('records')}


def evidence(pool):
    rows = source_records(pool); selected = frontier(rows)
    failures = [r for r in rows if r['status'] != 'valid'][-3:]
    candidates = []
    for r in [*selected, *failures]:
        item = {'module_sha256': r['module_sha256'], 'status': r['status'],
                'category': r.get('category'), 'error': (r.get('error') or '')[-1600:],
                'metrics': r['metrics']['public'], 'code_lines': r.get('code_lines'),
                'worker_seconds':r.get('worker_seconds'),'peak_vram_bytes':r.get('peak_vram_bytes')}
        if r['status'] == 'valid': item['diagnosis'] = loss_card(r)
        candidates.append(item)
    pool=Path(pool);plan=read(pool/'run_plan.json')
    first=read(pool/'folds.json')['search'][0]
    if file_sha256(first['input'])!=first['input_sha256']:raise ValueError('Source input contract changed')
    with np.load(first['input'],allow_pickle=False) as public:
        contract={k:{'shape':list(public[k].shape),'dtype':str(public[k].dtype)} for k in public.files}
    from .journal import events
    all_history=[r for r in events(pool.parents[2]/'research_decisions.jsonl')
                 if r.get('phase')=='search' and r.get('arm')==plan['arm'] and r.get('task')==plan['task']]
    # Do not let ordinary mutation/no-action events evict the last scientific hypothesis.
    history=[r for r in all_history if r.get('action') is not None or r.get('prediction') is not None][-8:]
    return {'scope': 'source_search_only', 'task': plan['task'], 'recent_research_experience':history,
            'input_contract':contract,'contract_note':'One public source fold; row counts vary, and the contract smoke uses two query rows. No query truth supplied.',
            'recent_trial_outcomes':[r for r in all_history if r['kind']=='measured_action_outcome'][-3:],
            'failure_counts':dict(Counter(r.get('category','runtime_or_contract') for r in rows if r['status']!='valid')),
            'candidates': candidates, 'candidate_count': len(rows),
            'interpretation': 'Method-family labels and causal explanations are hypotheses, not verified mechanisms.'}


def critic_prompt(data):
    schema = asdict(ResearchAction('explore_new_mechanism', 'source stagnation',
        'Test one new falsifiable response mechanism', 'shared residual calibration',
        None, None, 'high', 'none', {'metric': 'mse', 'direction': 'decrease', 'minimum_change': .00001},
        'Switch to a different measured parent', 'Explain the measured evidence', [], .6))
    return ('Return exactly one JSON ResearchAction. Use only the source-search evidence below. '
        'Distinguish runtime repair from valid-but-weak science. Choose a measured parent SHA and evidence IDs. '
        'Actions: ' + ', '.join(sorted(ACTIONS)) + '. Probes: ' + ', '.join(sorted(PROBES)) + '. '
        'Do not request labels, change budgets/evaluation or claim causality. A scientific revision must name '
        'a testable mechanism beyond hidden width/learning rate. Prediction metric is valid/mse/strong_mse/'
        'specificity_gap/gain_over_fixed. Parent SHA may be null only for exploration or a source probe. '
        'Schema example (replace placeholders): ' + json.dumps(schema) + '\nSOURCE EVIDENCE:\n' + json.dumps(data, allow_nan=False))


def prediction_outcome(action, before, after):
    metric = action.success_prediction['metric']
    def value(r):
        return float(r['status'] == 'valid') if metric == 'valid' else r['metrics']['public'].get(metric)
    a, b = value(before), value(after)
    if a is None or b is None:
        return {'status': 'undefined', 'metric': metric}
    change = b - a if action.success_prediction['direction'] == 'increase' else a - b
    return {'status': 'supported' if change >= action.success_prediction['minimum_change'] else 'not_supported',
            'metric': metric, 'before': a, 'after': b, 'directed_change': change,
            'claim': 'Source-search observation after an action; not causal proof or independent confirmation'}
