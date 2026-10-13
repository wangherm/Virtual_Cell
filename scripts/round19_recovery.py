"""Account for token-limit truncation without retrying or changing model training."""
from pathlib import Path
from vcell.model_agent.artifacts import read
from vcell.model_agent.contracts import digest
from vcell.round19.agents import guarded_call
from vcell.round11 import freeze
from vcell.utils import file_sha256,write_json

LEGACY_CODES={
    'd848df19638e352029b1454aa899fa9291ff78829a50444500bd84246cc63ad0',
    '3f2c1b990db44bd3f13841531a0af373d5ea6f64ce80ce63081bf87c486e805a',
}
LEGACY_MODEL_NAMESPACE='3645d8307db2dd3578bb13b0a8e3ffda5f1dfda11b95503de7d5bdae855b6ca2'


def namespace_identity():
    path=Path(__file__).resolve().parents[1]/'src/vcell/round19'
    return digest({p.name:file_sha256(p) for p in sorted(path.glob('*.py'))})


def is_truncated(record):
    return (record.get('status')=='failed' and record.get('http_status')==200
        and record.get('failure_category')=='incomplete_response'
        and record.get('finish_reason')=='length')


def rejection(record,role):
    return {'attempt':record['attempt'],'failure_category':'incomplete_response','finish_reason':'length',
        'validation_rule':'Output token limit reached before a valid final decision; no partial response accepted',
        'role':role,'observation_sha256':record['observation_sha256'],'executed':False,
        'retry_sent':False,'paid_request_retained':True,
        'output_tokens':record.get('output_tokens'),'reasoning_tokens':record.get('reasoning_tokens'),
        'feedback_available':False}


def call(planner,obs,role):
    """A truncated decision is missing evidence, never an action or invented advice."""
    planner.summary()  # check ledger/counter integrity before resuming
    key=digest(obs)
    attempts=[read(p) for p in sorted(planner.audit.glob('attempt_*.json'))
              if read(p)['observation_sha256']==key]
    if attempts and is_truncated(attempts[-1]):
        return None,rejection(attempts[-1],role)
    try:return guarded_call(planner,obs)
    except RuntimeError:
        record=planner.last_call or {}
        if is_truncated(record) and record.get('observation_sha256')==key:
            return None,rejection(record,role)
        raise  # authentication, transport, filtering and unknown outcomes still pause


def freeze_plan(root,plan,repair=False,resume=False):
    root=Path(root);path=root/'plan.json'
    if not path.exists() or read(path)==plan:
        freeze(path,plan);return
    old=read(path)
    if not repair or not resume:
        raise ValueError('Frozen plan changed; use --resume --repair-truncated-responses for the supported recovery, or a new name')
    if old.get('code') not in LEGACY_CODES or namespace_identity()!=LEGACY_MODEL_NAMESPACE:
        raise ValueError('Recovery requires the known 95d0882 launcher and unchanged model/cache/agent modules')
    ignored=('code','truncation_recovery')
    if {k:v for k,v in old.items() if k not in ignored}!={k:v for k,v in plan.items() if k not in ignored}:
        raise ValueError('Recovery cannot change data, seeds, folds, training steps or API configuration/budgets')
    if any((root/n).exists() for n in ('COMPLETE.json','search_frozen.json','final_decisions_sealed.json','outer_per_query.csv','comparison.csv')):
        raise ValueError('Recovery requires source-only search before final selection or outer evaluation')
    records=[read(p) for p in (root/'api').glob('*/attempt_*.json')]
    if not any(is_truncated(r) for r in records):raise ValueError('No completed token-limit response exists to recover')
    freeze(root/'recovery/original_plan.json',old)
    freeze(root/'recovery/migration.json',{'old_plan_sha256':digest(old),'new_plan_sha256':digest(plan),
        'model_namespace_sha256':namespace_identity(),'scope':'Treat API length-truncated decisions as missing feedback; no retry or partial execution',
        'training_and_api_budgets_unchanged':True,'paid_attempts_retained':True,'outer_used':False})
    write_json(path,plan)
    print('ROUND19 RECOVERY: migration audited; original trained workers and paid API attempts retained',flush=True)
