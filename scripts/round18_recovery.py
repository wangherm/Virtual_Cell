"""Research-only rejection feedback and a narrowly scoped legacy resume migration.

No model-agent modules change: existing candidate worker identities stay valid.
Routing still fails closed. Rejections consume the original paid request budget.
"""
import json
from pathlib import Path
from vcell.model_agent.artifacts import read
from vcell.model_agent.contracts import digest
from vcell.model_agent.iteration import IterationBudget
from vcell.round11 import freeze
from vcell.utils import write_json


LEGACY_LAUNCHERS = {
    '25fb9e2b4fb82b47a23d6a1d27e10ea2c2f15b83152c7cf35da27101cc61f542',
    '20dfca59e19e6b905e1e524b00bd82f7947b678635563be35a6188ee71da01b0',
}


class ResearchIterationBudget(IterationBudget):
    """Larger new-run limits; identical measured-improvement stopping semantics."""
    def __init__(self, cycles, trials, patience):
        if not 1 <= cycles <= 8 or not 1 <= trials <= 4 or not 1 <= patience <= 8:
            raise ValueError('Research budget requires cycles 1..8, trials 1..4, patience 1..8')
        super().__init__(min(cycles, 3), min(trials, 2), min(patience, 3))
        self.cycles, self.trials, self.patience = cycles, trials, patience


def extended_control(arm, trial, seed, fold):
    from vcell.model_agent.scientist import control_proposal, head_code
    from vcell.model_agent.research import Proposal, parse_proposal
    if arm != 'fixed' or trial <= 6:
        return control_proposal(arm, trial, seed, fold)
    # Keep the original six controls; expand with distinct predeclared recipes.
    index = trial - 7
    rank = 32 if index < 12 else 16
    width = (32, 64, 128)[(index // 4) % 3]
    dropout = (0., .1)[index % 2]
    activation = ('tanh', 'gelu')[(index // 2) % 2]
    rate = .001 if index < 12 else .0005 if index < 24 else .0002
    value = Proposal('propose_trial', None, 'fixed expanded predeclared head',
                     head_code(width, True, dropout, activation), rank, rate, .01, 'mse')
    from dataclasses import asdict
    return parse_proposal(json.dumps(asdict(value)))
RULES = (
    'Duplicate proposal key', 'Exact proposal keys required', 'Invalid proposal action',
    'Unsupported proposal recipe', 'Recipe exceeds fixed hyperparameter budget', 'Stop has no code',
    'Duplicate scientific proposal key', 'Exact scientific proposal keys required',
    'Bounded testable hypothesis required', 'Bounded real literature citations required',
    'Unknown paper citation', 'Evidence quote is not a literal retrieved-abstract span',
    'Candidate source exceeds the bounded head language', 'Candidate head too complex',
    'Exactly one CandidateHead class required', 'CandidateHead must derive directly from nn.Module',
    'Only __init__ and forward methods are allowed', 'Private/dunder access forbidden',
    'Private name access forbidden', 'Unbounded/string literals forbidden',
    'Candidate call budget exceeded', 'Only one bounded concatenation permitted',
    'Exact bounded method signature required', 'Single assignment required',
    'Initialize only named self layers', 'Only bounded nn layers may be assigned to self',
    'Forward may assign local tensors only', 'Builtin/function calls forbidden',
    'Chained call access forbidden', 'Create layers only in __init__',
    'Two explicit Linear dimensions required', 'Unbounded Linear dimension expression',
    'Bounded LayerNorm dimension required', 'Bounded Dropout probability required',
    'Concatenate exactly two tensors along feature dimension',
    'Normalize features only; no cross-query normalization',
    'Candidate call is not allowlisted', 'Dynamic call forbidden',
)


def safe_rule(exc):
    # Only known local messages; never persist arbitrary exception/provider text.
    if isinstance(exc, json.JSONDecodeError): return 'Invalid proposal JSON'
    if isinstance(exc, SyntaxError): return 'Candidate code is not valid Python syntax'
    value = str(exc)
    if value in RULES: return value
    for prefix in ('Unsupported candidate syntax: ', 'Unknown candidate name: '):
        if value.startswith(prefix): return prefix.rstrip(': ')
    return 'Proposal rejected by local field or code validation'


def install_diagnostics(planner):
    """Wrap only research parsing, preserving prompt/configuration/cache identity."""
    original = planner.parse_action
    def parse(content, obs):
        try: return original(content, obs)
        except (ValueError, SyntaxError, TypeError, KeyError, AttributeError) as exc:
            number = read(planner.audit/'counter.json')['reserved_attempts']
            path = planner.audit/f'attempt_{number:04d}.json'
            if path.exists():
                record = read(path)
                if record['status'] == 'started' and record['observation_sha256'] == digest(obs):
                    freeze(planner.audit/f'validation_{number:04d}.json', {
                        'attempt': number, 'observation_sha256': digest(obs),
                        'error_type': type(exc).__name__, 'validation_rule': safe_rule(exc),
                        'raw_response_saved': False, 'proposal_executed': False,
                    })
            if isinstance(exc, json.JSONDecodeError): raise
            raise ValueError(safe_rule(exc)) from None
    planner.parse_action = parse
    return planner


def rejected_proposal(planner, obs):
    """Recover an already-paid rejection without requesting it again."""
    summary = planner.summary()  # validates the original ledger/counter
    matching = [read(p) for p in sorted(planner.audit.glob('attempt_*.json'))
                if read(p)['observation_sha256'] == digest(obs)]
    if matching:
        record = matching[-1]
        if not (record['status'] == 'failed' and record.get('http_status') == 200
                and record.get('finish_reason') == 'stop'
                and record.get('failure_category') in ('invalid_action', 'invalid_json')):
            return None  # auth/network/unknown outcomes remain explicit errors
        diagnostic = planner.audit/f'validation_{record["attempt"]:04d}.json'
        reason = 'Legacy proposal rejected; the old audit did not retain the specific validation rule'
        if diagnostic.exists():
            detail = read(diagnostic)
            if detail['observation_sha256'] != digest(obs) or detail['attempt'] != record['attempt']:
                raise ValueError('Proposal diagnostic identity changed')
            reason = detail['validation_rule']
        call = {k: record.get(k) for k in ('attempt', 'failure_category', 'http_status',
            'input_tokens', 'output_tokens', 'reasoning_tokens', 'latency_seconds')}
        return {'status': 'failed', 'error_type': record.get('error_type'), 'reason': reason,
                'provider_call': call, 'observation_sha256': digest(obs), 'outer_used': False,
                'proposal_executed': False, 'candidate_fits': 0}
    if summary['attempted_requests'] >= summary['max_api_calls']:
        return {'status': 'failed', 'error_type': 'RequestBudgetExhausted',
                'reason': 'Original request budget exhausted; no new request or candidate fit',
                'observation_sha256': digest(obs), 'outer_used': False,
                'proposal_executed': False, 'candidate_fits': 0}
    return None


def freeze_plan(root, plan, repair=False, resume=False):
    """Migrate only the known pre-recovery launcher, before routing/evaluation."""
    root = Path(root); path = root/'plan.json'
    if not path.exists() or read(path) == plan:
        freeze(path, plan); return
    old = read(path)
    if not repair or not resume:
        raise ValueError('Frozen plan changed; use --resume --repair-invalid-proposals for the supported recovery, or a new run')
    if old.get('launcher') not in LEGACY_LAUNCHERS:
        raise ValueError('Recovery accepts only the known 80692cd Round18 launcher')
    keys = ('launcher', 'proposal_recovery')
    if {k:v for k,v in old.items() if k not in keys} != {k:v for k,v in plan.items() if k not in keys}:
        raise ValueError('Recovery cannot change data, source code, seeds, prompts or budgets')
    if any((root/n).exists() for n in ('COMPLETE.json', 'search_and_routing_sealed.json', 'evaluation', 'episodes')):
        raise ValueError('Recovery is allowed only before outer routing/evaluation')
    attempts = list((root/'search').glob('*/*/api/attempt_*.json'))
    if not any(read(p).get('status') == 'failed' and read(p).get('http_status') == 200
               and read(p).get('finish_reason') == 'stop'
               and read(p).get('failure_category') in ('invalid_action', 'invalid_json') for p in attempts):
        raise ValueError('No completed invalid-proposal response exists for this recovery')
    freeze(root/'recovery'/'original_plan.json', old)
    freeze(root/'recovery'/'migration.json', {
        'old_plan_sha256': digest(old), 'new_plan_sha256': digest(plan),
        'original_launcher': old['launcher'], 'recovered_launcher': plan['launcher'],
        'source_identity_unchanged': old['source'] == plan['source'],
        'scope': 'Research proposal rejection diagnostics/feedback only; original attempts retained',
        'data_and_budgets_unchanged': True, 'outer_used': False,
    })
    write_json(path, plan)
    print('ROUND18 RECOVERY: plan migration audited; completed fits and paid attempts retained', flush=True)
