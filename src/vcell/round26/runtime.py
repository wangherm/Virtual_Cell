"""Reuse the protected numerical evaluator, with a separate frozen code namespace."""
from pathlib import Path
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


def assess(*args,**kwargs):
    pool=Path(args[0] if args else kwargs['pool'])
    existing={p.resolve() for p in (pool/'assessments').glob('*/assessment.json')}
    result=previous.assess(*args,**kwargs,framework_fingerprint=fingerprint)
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
