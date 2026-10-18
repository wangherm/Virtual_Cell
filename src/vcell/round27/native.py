"""Seed and select in Shinka's own archive; no duplicate evolutionary population."""
import hashlib
import json
from pathlib import Path
from ..round25.shinka_hooks import select_archive_action,check_pinned_api


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def select_frontier(database,original,action,banned,records,turn,*args,**kwargs):
    if action:return select_archive_action(database,original,action,banned,*args,**kwargs)
    programs={hashlib.sha256(p.code.encode()).hexdigest():p for p in database.get_all_programs() if p.correct}
    # The trusted numerical controller freezes this Pareto view. The Shinka-only
    # environment must not import Torch, the grader, or private numerical inputs.
    candidates=[r for r in records if r['module_sha256'] in programs and r['module_sha256'] not in banned]
    if not candidates:return select_archive_action(database,original,None,banned,*args,**kwargs)
    chosen=candidates[turn%len(candidates)]
    return select_archive_action(database,original,{'action':'refine_parent','parent_code_sha256':chosen['module_sha256']},banned,*args,**kwargs)


def configure_generation_counter(runner):
    """0.0.7's row-count counter mistakes imported generation-zero roots for trials."""
    if not hasattr(runner,'_count_completed_generations_from_db'):raise ValueError('Pinned generation counter unavailable')
    async def count():
        ids=await runner.async_db.get_persisted_generation_ids_async()
        return min(len(set(ids)),runner.evo_config.num_generations)
    runner._count_completed_generations_from_db=count


def evaluation_walltime(pool):
    """Native job timeout must contain every bounded worker plus grader overhead."""
    pool=Path(pool);cfg=read(pool/'run_plan.json')['config'];folds=read(pool/'folds.json')['search']
    seconds=(2+len(folds)*len(cfg['counterfactual_variants']))*cfg['worker_seconds']+120
    hours,remainder=divmod(int(seconds),3600);minutes,seconds=divmod(remainder,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def install(pool,instruction):
    from shinka.database import Program,ProgramDatabase,DatabaseConfig
    check_pinned_api();pool=Path(pool)
    # Sampling connections are read-only in 0.0.7. Register shared source roots
    # once through the native writable API before the runner opens its readers.
    database=ProgramDatabase(DatabaseConfig(db_path=str(pool/'evolution/programs.sqlite'),num_islands=2,
        archive_size=60,migration_interval=5,migration_rate=.2),embedding_model=None)
    try:
        known={hashlib.sha256(p.code.encode()).hexdigest() for p in database.get_all_programs()}
        for index,item in enumerate(read(pool/'common_sources.json')):
            if sha_file(item['record'])!=item['sha256']:raise ValueError('Common source record changed')
            record=read(item['record']);sha=record['module_sha256']
            if sha in known:continue
            if record['status']!='valid' or record['identity']['phase']!='search':raise ValueError('Only valid source methods can enter the native archive')
            for name in ('module.py','scores.csv','baseline.csv'):
                if sha_file(Path(record['directory'])/name)!=record['files'][name]:raise ValueError('Common source payload changed')
            database.add(Program(id='r27-common-'+sha,code=Path(record['module']).read_text(encoding='utf-8'),
                generation=0,island_idx=index%2,correct=True,combined_score=record['metrics']['combined_score'],
                public_metrics=record['metrics']['public'],text_feedback=record['metrics']['text_feedback'],
                metadata={'common_source_record':item['sha256'],'new_candidate':False}))
            known.add(sha)
    finally:database.close()
    original=ProgramDatabase.sample_with_fix_mode
    def sample(database,*args,**kwargs):
        selected=select_frontier(database,original,instruction.get('action'),set(instruction.get('banned',[])),
            read(pool/'source_feedback.json')['native_frontier'],instruction['turn'],*args,**kwargs)
        parent,archive,top,fix=selected
        audit={'parent_id':parent.id,
            'parent_code_sha256':hashlib.sha256(parent.code.encode()).hexdigest(),'island':parent.island_idx,
            'needs_fix':fix,'inspiration_ids':[p.id for p in archive],
            'requested_action':instruction.get('action'),'scope':'source_search_only'}
        (pool/'evolution'/f'sampling_{instruction["turn"]:03d}.json').write_text(json.dumps(audit,allow_nan=False),encoding='utf-8')
        return selected
    ProgramDatabase.sample_with_fix_mode=sample
