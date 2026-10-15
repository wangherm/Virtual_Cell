"""Candidate deduplication, real-schema contract smoke, and full source scoring."""
import hashlib
from pathlib import Path
import numpy as np
from ..model_agent.artifacts import read
from ..round22.evaluator import evaluate,execute,inputs,validate_code
from ..utils import write_json,file_sha256
from .journal import append,classify,fingerprint
from ..model_agent.contracts import digest


def assess(code,pool,phase='search',execute_fn=execute,scope='shinka'):
    pool=Path(pool);plan=read(pool/'run_plan.json');task=plan['task'];repo=Path(plan['repo'])
    if fingerprint(repo)!=plan['framework_sha256']:raise ValueError('Evolution framework changed after campaign freeze')
    if phase!='search' and not (pool/'search_frozen.json').exists():raise ValueError('Confirmation requires search freeze')
    if phase=='outer' and not (pool/'confirmation_frozen.json').exists():raise ValueError('Outer grading requires confirmation freeze')
    key=hashlib.sha256(code.encode()).hexdigest();directory='random' if phase=='search' and scope=='random' else phase;dest=pool/'evaluations'/directory/key
    cache=dest/'assessment.json';memory=pool/'events.jsonl'
    identity={'data_plan':file_sha256(pool/'data_plan.json'),'runtime_plan':file_sha256(pool/'run_plan.json'),
        'baseline':file_sha256(pool/'baseline_selection.json') if (pool/'baseline_selection.json').exists() else None,'code':key}
    if cache.exists():
        record=read(cache)
        if digest({k:v for k,v in record.items() if k!='sha256'})!=record.get('sha256'):raise ValueError('Assessment record checksum changed')
        if record['identity']!=identity:raise ValueError('Assessment inputs or baseline identity changed; use a new campaign')
        if record['code_sha256']!=key:raise ValueError('Candidate cache identity changed')
        for name,sha in record['files'].items():
            if file_sha256(dest/name)!=sha:raise ValueError('Cached assessment changed: '+name)
        append(memory,'cache_reused',phase=phase,code_sha256=key)
        return record
    dest.mkdir(parents=True,exist_ok=True);(dest/'candidate.py').write_text(code,encoding='utf-8')
    def bounded(code,data,directory,budget,python):
        limited=dict(budget)
        if phase=='search':limited['evaluation_seconds_max']=min(budget.get('evaluation_seconds_max',float('inf')),budget.get('random_seconds_max' if scope=='random' else 'source_seconds_max',float('inf')))
        return execute_fn(code,data,directory,limited,python)
    try:
        initial=(repo/'experiments/round22_evolve/initial.py').read_text(encoding='utf-8');validate_code(code,initial)
        if phase=='search':
            # Real training schema, but only two query rows and five optimizer steps.
            # This is a shape/resource contract, not a scientific score or promotion.
            fold=read(pool/'data_plan.json')['folds'][task]['search'][0];d,_,_=inputs(fold)
            for k in d:
                if k.startswith('query_') or k in ('reference_available','background_reference'):d[k]=d[k][:2]
            tiny={**plan['budget'],'steps':min(5,plan['budget']['steps'])}
            first=bounded(code,d,dest/'contract',tiny,plan['python'])
            second=bounded(code,d,dest/'contract_repeat',tiny,plan['python'])
            if not np.allclose(first,second,rtol=1e-5,atol=1e-7):raise ValueError('Contract predictions did not reproduce with the same source inputs and seed; seed all random generators')
        result=evaluate(code,pool,task,phase,dest/'full',execute_fn=bounded)
        behavior={str(p.relative_to(dest/'full')):file_sha256(p) for p in sorted((dest/'full/workers').glob('*/*/output/prediction.npz'))}
        record={'code_sha256':key,'status':'valid','metrics':result,'code':str(dest/'candidate.py'),'score_dir':str(dest/'full'),'behavior_sha256':digest(behavior)}
        append(memory,'valid_candidate',phase=phase,code_sha256=key,score=result['combined_score'],hypothesis_status='source_score_only')
    except Exception as exc:
        error=str(exc)[-4000:];category=classify(exc)
        record={'code_sha256':key,'status':'invalid','category':category,'error':error,'code':str(dest/'candidate.py'),
            'metrics':{'combined_score':-1000000.,'public':{},'private':{},'text_feedback':f'{category}: {error}'}}
        append(memory,'invalid_candidate',phase=phase,code_sha256=key,category=category,error=error)
    record['files']={str(p.relative_to(dest)):file_sha256(p) for p in dest.rglob('*') if p.is_file() and p.name not in ('assessment.json',)}
    record['identity']=identity
    record['sha256']=digest(record)
    write_json(cache,record);return record


def valid_candidates(pool):
    pool=Path(pool);rows=[]
    for p in sorted((pool/'evaluations/search').glob('*/assessment.json')):
        r=read(p)
        if digest({k:v for k,v in r.items() if k!='sha256'})!=r.get('sha256'):raise ValueError('Assessment record checksum changed')
        if r['status']=='valid':rows.append(r)
    return rows


def write_result(record,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/'correct.json',{'correct':record['status']=='valid','error':record.get('error')})
    write_json(dest/'metrics.json',record['metrics'])
