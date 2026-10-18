"""Short, evidence-linked decisions; one explicit local-schema correction, never transport retry."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from ..evolution.self_correction import ResearchAction,parse_json,frontier,loss_card,ACTIONS,PROBES
from ..evolution.journal import events
from ..model_agent.artifacts import read
from ..model_agent.contracts import digest
from ..round11 import freeze
from ..utils import write_json,file_sha256
from .runtime import source_records
from .memory import experience
from .domain import context


def evidence(pool):
    pool=Path(pool);plan=read(pool/'run_plan.json');rows=source_records(pool)
    selected=frontier(rows,limit=5)+sorted(rows,key=lambda r:r['finished_at'])[-1:]+[r for r in rows if r['status']!='valid'][-2:]
    cards=[]
    keys=('mse','gain_over_fixed','strong_mse','strong_mse_ratio','specificity_gap','amplitude_ratio',
          'baseline_specificity_gap','target_response_candidate','transfer_candidate')
    for r in {r['module_sha256']:r for r in selected}.values():
        card={'module_sha256':r['module_sha256'],'status':r['status'],
            'metrics':{k:r['metrics']['public'].get(k) for k in keys},'error':(r.get('error') or '')[-700:]}
        if r['status']=='valid':
            d=loss_card(r)
            card['diagnosis']={**d,'best_targets':d['best_targets'][:2],'worst_targets':d['worst_targets'][:2]}
        cards.append(card)
    history=[r for r in events(pool.parents[2]/'research_decisions.jsonl')
        if r.get('phase')=='search' and r.get('arm')==plan['arm'] and r.get('task')==plan['task']][-4:]
    root=pool.parents[2];memory=experience(root,plan['arm'],plan['task'])
    historical=read(root/'historical_memory.json') if (root/'historical_memory.json').exists() else {}
    from .runtime import novel
    from .knowledge import lookup
    targets=[r['target'] for c in cards for r in c.get('diagnosis',{}).get('worst_targets',[])][:6]
    return {'scope':'source_search_only','task':plan['task'],'candidates':cards,
        'virtual_cell':context(pool),'research_memory':memory,
        'gene_context':lookup(root,targets),
        'fixed_comparator':read(pool/'baseline_selection.json')['module_sha256'],
        'required_revision_id':memory['pending_revisions'][-1]['ticket'] if memory['pending_revisions'] else None,
        'historical_source_lessons':historical.get('source_lessons',[]),
        'exploration':{'distinct_new':len(novel(pool)),'minimum_before_branch_stop':plan['config'].get('minimum_branch_exploration',3)},
        'interpretation':'Mechanisms and imported lessons are hypotheses. Confirmation and outer results are unavailable.'}


def parse_action(payload,info,allowed_actions=None,required_evidence=None,banned=(),repair_counts=None,repair_origins=None):
    action=ResearchAction.parse(payload,info)  # One shared typed parser; no second ResearchAction definition.
    if action.action not in (allowed_actions or ACTIONS):raise ValueError('Action is unavailable in this stage')
    if required_evidence and required_evidence not in action.evidence_ids:raise ValueError('The latest measured child must appear in evidence_ids')
    lookup={r['module_sha256']:r for r in info['candidates']}
    for parent in (action.parent_code_sha256,action.second_parent_sha256):
        if parent in banned:raise ValueError('Stopped branch is unavailable')
        if parent and lookup[parent]['status']!='valid' and action.action!='repair_candidate':raise ValueError('Scientific parent must be valid')
    if action.action=='repair_candidate':
        parent=action.parent_code_sha256;origin=(repair_origins or {}).get(parent,parent)
        if lookup[parent]['status']=='valid' or (repair_counts or {}).get(origin,0)>=2:
            raise ValueError('Repair requires an invalid parent with remaining repair attempts')
    if action.action=='stop_unproductive_branch':
        if info['exploration']['distinct_new']<info['exploration']['minimum_before_branch_stop']:
            raise ValueError('Task needs actual new exploration before stopping a branch')
        if action.parent_code_sha256==info['fixed_comparator']:
            raise ValueError('A comparator matching itself is not a reason to abandon its task')
    if action.action=='run_source_probe':
        valid_ids={c['module_sha256'] for c in info['candidates'] if c['status']=='valid'}
        for prior in info.get('research_memory',{}).get('hypothesis_history',[]):
            probe=prior.get('probe') or {}
            if probe.get('kind')==action.requested_source_diagnostic and valid_ids.issubset({r['module_sha256'] for r in probe.get('summary',[])}):
                raise ValueError('This diagnostic already ran on these source methods; revise the hypothesis or test a new method')
    if isinstance(action.repeat_share,bool) or not np.isfinite(action.repeat_share):raise ValueError('Finite fractional repeat_share required')
    if isinstance(action.success_prediction['minimum_change'],bool) or action.success_prediction['minimum_change']<=0:raise ValueError('Declare a positive measurable minimum_change')
    reference=lookup.get(action.parent_code_sha256)
    metric=action.success_prediction['metric']
    if reference and metric!='valid' and reference['metrics'].get(metric) is None:raise ValueError('Prediction metric is undefined for this parent')
    return action


def contract(info,allowed_actions=None):
    valid=[r for r in info['candidates'] if r['status']=='valid']
    if not valid:raise ValueError('A measured source parent is required before scientific review')
    parent=min(valid,key=lambda r:r['metrics']['mse'])['module_sha256']
    example=asdict(ResearchAction('refine_parent','measured source error',
        'Separate shared response from a reliability-weighted target residual','shared residual calibration',
        parent,None,'high','none',{'metric':'mse','direction':'decrease','minimum_change':.00001},
        'Explore another source-tested parent','Use the source comparison',[parent],.6))
    pending=info.get('required_revision_id')
    link={'predecessor_experiment_id':pending,'failure_explanation':'The measured change did not support the previous prediction',
          'changed_assumption':'Test a shared calibrated residual rather than the previous unregularized correction'} if pending else None
    return {'required_fields':['action','revision'],'action_required_fields':list(example),'example':{'action':example,'revision':link},'allowed_actions':sorted(allowed_actions or ACTIONS),
        'allowed_probes':sorted(PROBES),'repeat_share':{'minimum':.4,'maximum':.7,'default':.6},
        'allowed_evidence_ids':[r['module_sha256'] for r in info['candidates']],
        'required_revision_id':pending,
        'rules':'Return action plus revision. A pending experiment requires revision with the exact predecessor_experiment_id, a failure_explanation and changed_assumption. Otherwise revision is null. Use source evidence IDs exactly; fractional repeat_share. Only invalid parents can be repaired. Do not infer absent biological signal from a fixed model equalling itself.'}


def parse_decision(payload,info,allowed,required_evidence,validation):
    if not isinstance(payload,dict) or set(payload)!={'action','revision'}:raise ValueError('Return exactly action and revision')
    action=parse_action(payload['action'],info,allowed,required_evidence,**validation)
    link=payload['revision'];required=info.get('required_revision_id')
    if required:
        if not isinstance(link,dict) or set(link)!={'predecessor_experiment_id','failure_explanation','changed_assumption'}:
            raise ValueError('Explicit measured-failure revision link is required')
        if link['predecessor_experiment_id']!=required:raise ValueError('Respond to the exact pending experiment')
        for key in ('failure_explanation','changed_assumption'):
            if not isinstance(link[key],str) or not 10<=len(link[key])<=1200:raise ValueError('A concise substantive revision is required: '+key)
    elif link is not None:raise ValueError('No pending experiment; revision must be null')
    return action,link


def decide(pool,gateway,ticket,cap,*,info=None,allowed_actions=None,required_evidence=None,**validation):
    folder=Path(pool)/'decisions'/ticket
    frozen=folder/'input.json'
    if not frozen.exists():freeze(frozen,{'evidence':info or evidence(pool),'allowed_actions':sorted(allowed_actions or ACTIONS),
        'required_evidence':required_evidence,'validation':validation})
    saved=read(frozen);info=saved['evidence'];allowed=saved['allowed_actions']
    result_path=folder/'decision.json'
    if result_path.exists():
        result=read(result_path)
        if result['input_sha256']!=file_sha256(frozen) or digest({k:v for k,v in result.items() if k!='sha256'})!=result['sha256']:
            raise ValueError('Saved scientific decision changed')
        if result['attribution']=='critic':parse_decision({'action':result['action'],'revision':result.get('revision')},info,allowed,saved['required_evidence'],saved['validation'])
        return result
    prompt='Return one concise strict JSON object containing action and revision. Use only SOURCE evidence. '+json.dumps(contract(info,allowed),allow_nan=False)
    messages=[{'role':'system','content':prompt},{'role':'user','content':json.dumps(info,allow_nan=False)}]
    status='provider_failed';error='';action=None;revision=None
    for attempt in range(2):
        audit=folder/f'attempt_{attempt+1}.json'
        if audit.exists():row=read(audit)
        else:
            gateway.scope=f'{read(Path(pool)/"run_plan.json")["task"]}/{ticket}/critic/{attempt+1}'
            gateway.role='critic' if attempt==0 else 'critic_schema_correction'
            http,body=gateway.forward({'model':gateway.model,'messages':messages,'max_tokens':cap,'stream':False})
            content=(body.get('choices') or [{}])[0].get('message',{}).get('content','') if http==200 else ''
            row={'http_status':http,'response':content,'response_sha256':hashlib.sha256(content.encode()).hexdigest(),
                 'ticket':gateway.scope,'role':gateway.role,'status':'received' if http==200 else 'provider_failed'}
            # Provider errors, truncation and transport failures are not schema retries.
            if http!=200:row['error']=body.get('error',{}).get('type','provider_failure')
            write_json(audit,row)
        if hashlib.sha256(row['response'].encode()).hexdigest()!=row['response_sha256']:raise ValueError('Saved critic response changed')
        if row['http_status']!=200:
            status='provider_failed';error=row.get('error','provider_failure');break
        try:
            action,revision=parse_decision(parse_json(row['response']),info,allowed,saved['required_evidence'],saved['validation'])
            row['status']='accepted';write_json(audit,row);status='accepted';break
        except (ValueError,TypeError,KeyError) as exc:
            error=str(exc)[:700];row.update(status='schema_rejected',error=error);write_json(audit,row);status='schema_rejected'
            messages+= [{'role':'assistant','content':row['response'][:3500]},
                {'role':'user','content':'One schema correction is permitted. Fix this exact local validation error: '+error+
                    '. Copy valid IDs and fields from the contract; do not change the source evidence. Return JSON only.'}]
    result={'status':status,'attribution':'critic' if action is not None else 'fallback',
        'action':asdict(action) if action else None,'revision':revision,'error':error if action is None else None,
        'scope':'source_search_only','attempts':len(list(folder.glob('attempt_*.json'))),'input_sha256':file_sha256(frozen)}
    result['sha256']=digest(result)
    freeze(result_path,result);return result
