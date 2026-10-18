"""Reuse the protected numerical evaluator, with a separate frozen code namespace."""
from pathlib import Path
import hashlib
from ..round25 import runtime as previous
from ..model_agent.artifacts import read
from ..model_agent.contracts import digest
from ..utils import file_sha256
from ..evolution.self_correction import source_records as local_records, distinct
from ..evolution.journal import append


def fingerprint(repo):
    repo=Path(repo)
    files=[*sorted((repo/'src/vcell/round26').glob('*.py')),
        repo/'scripts/run_round26.py',repo/'scripts/run_shinka_round26.py',
        repo/'experiments/round26_evolve/evaluate.py']
    return digest({'protected_base':previous.fingerprint(repo),
        'round26':{p.relative_to(repo).as_posix():file_sha256(p) for p in files}})


def assess(pool,module,phase='search',controls=True,baseline=False,execute_fn=None):
    pool=Path(pool);plan=read(pool/'run_plan.json')
    if fingerprint(plan['repo'])!=plan['framework_sha256']:raise ValueError('Research framework code changed; create a new campaign')
    # The native runner evaluates its generation-zero source seed again. Reuse
    # the already protected, numerically identical common replay instead of
    # consuming each arm/pilot's allowance on a duplicate baseline fit.
    if phase=='search' and controls and not baseline:
        module_sha=hashlib.sha256(module.encode()).hexdigest()
        refs=read(pool/'common_sources.json') if (pool/'common_sources.json').exists() else []
        for item in refs:
            if file_sha256(item['record'])!=item['sha256']:raise ValueError('Common source record changed')
            record=read(item['record'])
            if record['module_sha256']!=module_sha:continue
            original_pool=Path(record['directory']).parents[1];original=read(original_pool/'run_plan.json')
            if file_sha256(original_pool/'run_plan.json')!=record['identity']['plan'] or \
                file_sha256(original_pool/'folds.json')!=record['identity']['folds'] or \
                file_sha256(original_pool/'baseline_selection.json')!=record['identity']['comparator']:
                raise ValueError('Common source plan, folds or comparator changed')
            cfg_keys=('new_source_datasets','counterfactual_variants','behavior_quantum','minimum_mse_gain',
                'maximum_strong_mse_ratio','minimum_specificity_change','minimum_amplitude_ratio')
            same=original['framework_sha256']==plan['framework_sha256'] and original['task']==plan['task'] and record['status']=='valid' and \
                record['identity']['phase']=='search' and record['identity']['controls'] and not record['identity']['baseline'] and \
                read(original_pool/'folds.json')['search']==read(pool/'folds.json')['search'] and \
                read(original_pool/'baseline_selection.json')['search']==read(pool/'baseline_selection.json')['search'] and \
                all(original['budget'][k]==plan['budget'][k] for k in ('seed','device','steps','seconds','ram_bytes')) and \
                all(original['config'][k]==plan['config'][k] for k in cfg_keys)
            if same:
                if digest({k:v for k,v in record.items() if k!='sha256'})!=record['sha256']:raise ValueError('Common source checksum changed')
                for name,sha in record['files'].items():
                    if file_sha256(Path(record['directory'])/name)!=sha:raise ValueError('Common source numerical payload changed')
                append(pool/'cache_reuse.jsonl','shared_source_cache_hit',phase='search',module_sha256=module_sha,
                    source_record=item['record'],source_record_sha256=item['sha256'])
                return record
    existing={p.resolve() for p in (pool/'assessments').glob('*/assessment.json')}
    result=previous.assess(pool,module,phase=phase,controls=controls,baseline=baseline,execute_fn=execute_fn,framework_fingerprint=fingerprint)
    if (Path(result['directory'])/'assessment.json').resolve() in existing:
        append(pool/'cache_reuse.jsonl','assessment_cache_hit',phase=result['identity']['phase'],module_sha256=result['module_sha256'])
    return result


def source_records(pool):
    rows=[];path=Path(pool)/'common_sources.json'
    for item in read(path) if path.exists() else []:
        if file_sha256(item['record'])!=item['sha256']:raise ValueError('Common source record changed')
        record=read(item['record'])
        if record['identity']['phase']!='search' or record['status']!='valid':raise ValueError('Only valid source methods can seed the native archive')
        if digest({k:v for k,v in record.items() if k!='sha256'})!=record['sha256']:raise ValueError('Common source checksum changed')
        for name,sha in record['files'].items():
            # Common records were sealed after numerical evaluation; validate the public evidence used here.
            if name in ('module.py','scores.csv','baseline.csv') and file_sha256(Path(record['directory'])/name)!=sha:
                raise ValueError('Common source evidence changed')
        rows.append(record)
    rows+=local_records(pool)
    return list({r['module_sha256']:r for r in rows}.values())


def novel(pool):
    path=Path(pool)/'common_sources.json'
    excluded=[read(item['record'])['behavior_sha256'] for item in read(path)] if path.exists() else []
    return distinct(source_records(pool),excluded=excluded)
