"""One absolute native generation, with common source starts and measured Pareto parents."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vcell.round25.shinka_hooks import check_pinned_api
from vcell.round26.native import install,configure_generation_counter


def main():
    p=argparse.ArgumentParser();p.add_argument('--pool');p.add_argument('--generations',type=int);p.add_argument('--check-api',action='store_true');a=p.parse_args()
    capabilities=check_pinned_api()
    from shinka.core import ShinkaEvolveRunner,EvolutionConfig
    from shinka.database.async_dbase import AsyncProgramDatabase
    if not hasattr(ShinkaEvolveRunner,'_count_completed_generations_from_db') or not hasattr(AsyncProgramDatabase,'get_persisted_generation_ids_async'):
        raise ValueError('Pinned imported-root generation accounting API missing')
    capabilities['round26_imported_root_accounting']='distinct native persisted generation IDs'
    if a.check_api:print(json.dumps(capabilities));return
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig
    pool=Path(a.pool);plan=json.loads((pool/'run_plan.json').read_text());cfg=plan['config']
    instruction=json.loads((pool/'active_action.json').read_text());install(pool,instruction)
    feedback=json.loads((pool/'source_feedback.json').read_text())
    prompt='''Develop a numerical perturbation predictor as a complete editable Python module with one fit_predict(data,budget).
NumPy/SciPy/PyTorch only. No files, network, subprocesses or hidden labels. Return finite ordered mean-delta values.
train_y and calibration_y are permitted source labels. train/calibration/query_support are measured Rep1 for repeat;
for transfer they are legal same-target source donors, zero if absent. No held-cell-line perturbation truth is supplied.
train/calibration/query_x, _control, _prior, _targets, _source_context and ordered genes/gene_symbols are available.
aux_delta/baseline/target/dataset/context/stimulus/modality/batch/time are legal auxiliary sources.
Tune mixtures/scales/thresholds on train and calibration only; query is another source fold. Never tune on query labels.
Try common response plus target residual, source reliability denoising, conditional donor transformations or a calibrated
nonzero response branch. These are hypotheses, not compulsory architectures. A tiny query can contain just two rows.
Preserve target information: strong-gene MSE, response magnitude, and correct/wrong/shuffled donor versus identity-only
ablations are scored separately. A zero-like transfer output is not evidence of perturbation-specific success.
Seed random generators, cast dtypes and obey budget optimizer steps. The native archive retains both error and signal
Pareto candidates; a source signal candidate with worse MSE is only an exploration parent, not a validated discovery.
'''
    prompt+='\nTASK: '+plan['task']+'\nSOURCE: '+json.dumps(feedback,allow_nan=False)
    prompt+='\nEXECUTABLE ACTION AND SOURCE PROBE: '+json.dumps(instruction,allow_nan=False)
    model=f'local/{cfg["model"]}@http://127.0.0.1:{cfg["port"]}/v1?api_key_env=ROUND22_GATEWAY_KEY'
    os.environ['ROUND26_POOL']=str(pool)
    runner=ShinkaEvolveRunner(evo_config=EvolutionConfig(num_generations=a.generations,
        init_program_path=str(pool/'seed.py'),results_dir=str(pool/'evolution'),llm_models=[model],
        llm_kwargs={'max_tokens':cfg['api_max_tokens']},task_sys_msg=prompt,patch_types=['full'],patch_type_probs=[1.],
        max_patch_resamples=1,max_patch_attempts=1,max_novelty_attempts=1,embedding_model=None,
        meta_rec_interval=None,use_text_feedback=True),
        db_config=DatabaseConfig(num_islands=2,archive_size=60,migration_interval=5,migration_rate=.2,
            num_archive_inspirations=2,num_top_k_inspirations=1),
        job_config=LocalJobConfig(eval_program_path=str(Path(plan['repo'])/'experiments/round26_evolve/evaluate.py'),
            python_executable=plan['python'],time='01:00:00',numeric_threads_per_job=2),
        max_evaluation_jobs=1,max_proposal_jobs=1,max_db_workers=1)
    configure_generation_counter(runner);runner.run()


if __name__=='__main__':main()
