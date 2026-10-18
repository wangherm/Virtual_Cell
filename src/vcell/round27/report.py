"""Matched and extension scores stay separate; fallbacks never count as critic success."""
from collections import Counter,deque
import hashlib
import io
import json
from pathlib import Path
import tarfile
import pandas as pd
from ..evolution.journal import events
from ..evolution.self_correction import frontier
from ..model_agent.artifacts import read
from ..round24.stats import cluster_gain
from ..round25.runtime import charged
from ..utils import write_json
from .runtime import assess,source_records,novel


def report(root,cfg,stage):
    if not (root/'search_frozen.json').exists():raise ValueError('Freeze every source selection before any confirmation retest')
    arms=cfg['arms']+(['meta_extension'] if read(root/'search_frozen.json')['extension_executed'] else [])
    comparison=[];source=[];confirmation=[];trace=[];usage=[];paired=[];memory=[]
    actions=events(root/'research_decisions.jsonl')
    for arm in arms:
        ar=root/'arms'/arm;calls=read(ar/'api_ledger.json') if (ar/'api_ledger.json').exists() else []
        reviews=[read(p) for p in ar.glob('*/decisions/*/decision.json')]
        outcomes=[r for r in actions if r['kind']=='measured_action_outcome' and r.get('arm')==arm]
        repairs=[]
        for result in outcomes:
            a=result.get('action')
            if not a or a['action']!='repair_candidate':continue
            parent=next((r for r in source_records(ar/result['task']) if r['module_sha256']==a['parent_code_sha256']),None)
            repairs.append(bool(parent and parent['status']=='invalid' and any(c['status']=='valid' and c['module_sha256']!=parent['module_sha256'] for c in result['children'])))
        done=read(ar/'SEARCH_COMPLETE.json');valid_total=sum(len(novel(ar/t)) for t in cfg['tasks'])
        usage.append({'arm':arm,'comparison_scope':'extension' if arm=='meta_extension' else 'matched',
            'requests':len(calls),'requests_max':done['requests_max'],'unspent_request_slots':done['unspent_requests'],
            'roles':dict(Counter(r.get('role') for r in calls)),'charged_output_tokens':sum(r.get('charged_output_tokens',0) for r in calls),
            'input_tokens':sum(r.get('usage',{}).get('prompt_tokens',0) for r in calls),
            'provider_latency_seconds':sum(r.get('latency_seconds',0) for r in calls),
            'critic_reviews':len(reviews),'accepted_critic_decisions':sum(r['attribution']=='critic' for r in reviews),
            'fallback_decisions':sum(r['attribution']=='fallback' for r in reviews),
            'schema_correction_requests':sum(r.get('role')=='critic_schema_correction' for r in calls),
            'critic_implemented_actions':sum(r.get('attribution')=='critic' and bool(r['children']) for r in outcomes),
            'critic_measured_hypotheses':dict(Counter(r['prediction']['status'] for r in outcomes if r.get('attribution')=='critic' and r.get('prediction'))),
            'repair_attempts':len(repairs),'successful_code_repairs':sum(repairs),'distinct_new_behaviors':valid_total,
            'assessment_cache_reuses':sum(len(events(ar/task/'cache_reuse.jsonl')) for task in cfg['tasks']),
            'provider_cache_reuses':len(events(ar/'provider_cache_reuse.jsonl')),
            'behavior_target':done['valid_target'],'valid_per_api_request':valid_total/len(calls) if calls else None,
            'worker_seconds':charged(ar),'worker_hours_max':done['worker_hours_max'],'stop_reason':done['reason'],
            'currency_cost':None,'mechanism_discovery_count':None,
            'mechanism_note':'Agent family labels are hypotheses; no unvalidated automatic biological mechanism count.'})
        for task in cfg['tasks']:
            pool=ar/task;baseline=read(pool/'baseline_selection.json');base=baseline['confirmation_record']['metrics']['public']
            comparison.append({'arm':arm,'scope':'extension' if arm=='meta_extension' else 'matched','task':task,'method':'fixed/'+baseline['method'],
                'phase':'historical_development_confirmation',**base,'gain_over_fixed':0.,'scientific_pass':False,
                'baseline_mse':base['mse'],'baseline_strong_mse':base['strong_mse'],
                'baseline_specificity_gap':base['specificity_gap'],'strong_mse_ratio':1. if base['strong_mse'] is not None else None,
                'overall_error_candidate':False,'target_response_candidate':False,'transfer_candidate':False})
            new_ids={r['module_sha256'] for r in novel(pool)}
            for record in source_records(pool):
                trace.append({'arm':arm,'task':task,'module_sha256':record['module_sha256'],'status':record['status'],
                    'is_new_behavior':record['module_sha256'] in new_ids,'worker_seconds':record['worker_seconds'],**record['metrics']['public']})
                if record['status']=='valid':source.append(pd.read_csv(Path(record['directory'])/'scores.csv').assign(
                    arm=arm,task=task,module_sha256=record['module_sha256'],is_new_behavior=record['module_sha256'] in new_ids))
            front=frontier(source_records(pool));write_json(pool/'source_frontier.json',[{'module_sha256':r['module_sha256'],'metrics':r['metrics']['public']} for r in front])
            for r in front:memory.append({'arm':arm,'task':task,'scope':'source_only','module_sha256':r['module_sha256'],
                'module':r['module'],'source_metrics':r['metrics']['public'],'source_input_hashes':[f['input_sha256'] for f in read(pool/'folds.json')['search']]})
            for selected in read(pool/'search_frozen.json')['finalists']:
                from .manager import checkpoint
                checkpoint(root.parents[1],root.name)
                stage(root,'frozen development retest',arm=arm,task=task,module_sha256=selected['module_sha256'])
                result=assess(pool,Path(selected['module']).read_text(encoding='utf-8'),phase='confirmation')
                m=result['metrics']['public'];gate='target_response_candidate' if task=='repeat' else 'transfer_candidate'
                passed=bool(m.get(gate) and selected['metrics']['public'].get(gate))
                comparison.append({'arm':arm,'scope':'extension' if arm=='meta_extension' else 'matched','task':task,
                    'method':selected['module_sha256'],'tier':selected['tier'],'phase':'historical_development_confirmation',
                    'status':result['status'],**m,'scientific_pass':passed})
                if result['status']=='valid':
                    frame=pd.read_csv(Path(result['directory'])/'scores.csv');baseframe=pd.read_csv(Path(result['directory'])/'baseline.csv')
                    confirmation.append(frame.assign(arm=arm,task=task,module_sha256=selected['module_sha256']))
                    paired.append({'arm':arm,'task':task,'module_sha256':selected['module_sha256'],**cluster_gain(frame,baseframe)})
    pd.DataFrame(comparison).to_csv(root/'comparison.csv',index=False);pd.DataFrame(trace).to_csv(root/'evolution_trace.csv',index=False)
    for name,frames in [('source_scores.csv',source),('confirmation_scores.csv',confirmation)]:
        (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=['arm','task','module_sha256','row_id','context','target','mse'])).to_csv(root/name,index=False)
    write_json(root/'efficiency.json',usage);write_json(root/'paired_sensitivity.json',paired)
    write_json(root/'source_memory_library.json',{'methods':memory,'scope':'source_only','confirmation_included':False})
    pilot=read(root/'LIVE_SMOKE_COMPLETE.json')
    from .manager import revision_summary
    revisions=revision_summary(root);write_json(root/'revision_chains.json',revisions)
    coverage={arm:read(root/'arms'/arm/'SEARCH_COMPLETE.json')['coverage_complete'] for arm in cfg['arms']}
    repeated={task:sum(r['arm']=='meta' and r['task']==task and r['measured'] for r in revisions) for task in cfg['tasks']}
    registry=[]
    for arm in cfg['arms']:
        for task in cfg['tasks']:
            pool=root/'arms'/arm/task;fold=read(pool/'folds.json')['search'][0]
            for r in source_records(pool):
                if r['status']!='valid':continue
                registry.append({'arm':arm,'task':task,'module_sha256':r['module_sha256'],'module':r['module'],
                    'source_metrics':r['metrics']['public'],'template_input':fold['input'],
                    'template_sha256':fold['input_sha256'],'status':'research_candidate',
                    'independent_validation':False,'interface':'fit_predict(public_source_and_query_arrays,budget)'})
    write_json(root/'model_registry.json',{'models':registry,'scope':'source_methods; no independent deployment qualification'})
    summary={'search_complete':True,'task_coverage_complete':coverage,
        'framework_pass':all(coverage.values()) and all(repeated.values()),
        'measured_meta_revision_chains':repeated,'agent_self_improvement_proven':False,
        'scientific_pass':all(coverage.values()) and any(r.get('scientific_pass',False) for r in comparison),
        'independent_confirmation':False,'test_evaluated':False,'HT29_evaluated':False,'outer_evaluated':False,
        'pilot':pilot,'matched_arms':cfg['arms'],'extension_reported_separately':'meta_extension' in arms,'usage':usage,
        'pilot_requests_counted_separately':pilot['api_requests'],
        'total_requests_including_gate':pilot['api_requests']+sum(r['requests'] for r in usage),
        'total_assessment_cache_reuses':sum(len(events(p)) for p in root.rglob('cache_reuse.jsonl')),
        'total_provider_cache_reuses':sum(len(events(p)) for p in root.rglob('provider_cache_reuse.jsonl')),
        'prior_recovery_worker_seconds':pilot.get('recovery',{}).get('prior_worker_seconds',0.),
        'total_worker_seconds':charged(root/'baseline_budget')+charged(root/'pilot/meta')+sum(charged(root/'arms'/arm) for arm in arms)+pilot.get('recovery',{}).get('prior_worker_seconds',0.),
        'interpretation':'Historical development retest, one seed. Equal caps do not imply equal actual requests or candidates. A source signal candidate is not a confirmed transfer discovery.'}
    write_json(root/'summary.json',summary)
    lines=['# Round27 persistent Virtual Cell research agent','',
        'Search COMPLETE and Scientific PASS are separate. Confirmation is historically viewed development data.',
        'The live gate is excluded from arm comparisons, and included in total provider/compute usage.','',
        '| Arm | Scope | Requests | Accepted critic reviews | Fallbacks | New behaviors | Worker hours |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in usage:lines.append(f"| {r['arm']} | {r['comparison_scope']} | {r['requests']}/{r['requests_max']} | {r['accepted_critic_decisions']} | {r['fallback_decisions']} | {r['distinct_new_behaviors']} | {r['worker_seconds']/3600:.3f} |")
    lines+=['',f'Balanced task coverage: {coverage}. Measured Meta revision chains: {repeated}.',
        'Framework pass requires task coverage in all arms and a measured revision chain in each Meta task; it is not prediction superiority.',
        'Algorithm family labels are proposed hypotheses. See source_frontier.json, independent donor/identity controls,',
        'comparison.csv, paired_sensitivity.json, native sampling and action logs for measured outcomes.']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def package(root):
    path=root/'round27_review_light.tar.gz';entries={}
    with tarfile.open(path,'w:gz') as tar:
        for p in sorted(root.rglob('*')):
            rel=p.relative_to(root)
            if not p.is_file() or p.suffix not in ('.json','.jsonl','.csv','.md','.py','.log','.diff'):continue
            if set(rel.parts)&{'private','inputs','workers','contract','contract2','preflight','input_cache','provider_responses','provider_tickets'}:continue
            if p.suffix=='.log':
                with p.open(encoding='utf-8',errors='replace') as f:payload=''.join(deque(f,maxlen=60)).encode()
            else:payload=p.read_bytes()
            info=tarfile.TarInfo(rel.as_posix());info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
            entries[rel.as_posix()]={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
        payload=json.dumps({'files':entries,'scope':'source actions including accepted/rejected critic responses, generated code, scores and usage; no matrices, weights or credentials'}).encode()
        info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
    print('REVIEW PACKAGE:',path,flush=True);return path
