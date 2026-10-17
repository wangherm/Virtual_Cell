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
    return {'scope':'source_search_only','task':plan['task'],'candidates':cards,
        'recent_research_experience':history,'interpretation':'Mechanisms are hypotheses. Historical confirmation and outer labels are unavailable.'}


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
    return {'required_fields':list(example),'example':example,'allowed_actions':sorted(allowed_actions or ACTIONS),
        'allowed_probes':sorted(PROBES),'repeat_share':{'minimum':.4,'maximum':.7,'default':.6},
        'allowed_evidence_ids':[r['module_sha256'] for r in info['candidates']],
        'rules':'Use nonempty evidence_ids copied exactly; fractional repeat_share, not a percentage. Exactly one JSON object; concise strings, no code. Only invalid parents can be repaired. Each prediction needs a positive minimum_change.'}


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
        if result['attribution']=='critic':parse_action(result['action'],info,allowed,saved['required_evidence'],**saved['validation'])
        return result
    prompt='Return one concise, strict JSON ResearchAction. Use only SOURCE evidence. '+json.dumps(contract(info,allowed),allow_nan=False)
    messages=[{'role':'system','content':prompt},{'role':'user','content':json.dumps(info,allow_nan=False)}]
    status='provider_failed';error='';action=None
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
            action=parse_action(parse_json(row['response']),info,allowed,saved['required_evidence'],**saved['validation'])
            row['status']='accepted';write_json(audit,row);status='accepted';break
        except (ValueError,TypeError,KeyError) as exc:
            error=str(exc)[:700];row.update(status='schema_rejected',error=error);write_json(audit,row);status='schema_rejected'
            messages+= [{'role':'assistant','content':row['response'][:3500]},
                {'role':'user','content':'One schema correction is permitted. Fix this exact local validation error: '+error+
                    '. Copy valid IDs and fields from the contract; do not change the source evidence. Return JSON only.'}]
    result={'status':status,'attribution':'critic' if action is not None else 'fallback',
        'action':asdict(action) if action else None,'error':error if action is None else None,
        'scope':'source_search_only','attempts':len(list(folder.glob('attempt_*.json'))),'input_sha256':file_sha256(frozen)}
    result['sha256']=digest(result)
    freeze(result_path,result);return result
