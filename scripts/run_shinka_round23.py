"""Pinned framework runner; continuation preserves one archive/database per pool."""
import argparse
import json
import os
from pathlib import Path
from importlib.metadata import version


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--pool',required=True);p.add_argument('--generations',type=int,required=True);a=p.parse_args()
    if version('shinka-evolve')!='0.0.7':raise RuntimeError('Install pinned Shinka 0.0.7 with the existing installer')
    from shinka.core import ShinkaEvolveRunner,EvolutionConfig
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig
    root=Path(a.root);pool=Path(a.pool);plan=json.loads((root/'campaign.json').read_text());cfg=plan['config'];runtime=json.loads((pool/'run_plan.json').read_text())
    model=f'local/{cfg["model"]}@http://127.0.0.1:{cfg["port"]}/v1?api_key_env=ROUND22_GATEWAY_KEY'
    memory=(pool/'source_feedback.json').read_text() if (pool/'source_feedback.json').exists() else '[]'
    task=runtime['task'];os.environ['ROUND23_POOL']=str(pool)
    prompt=f'''Autonomously improve the full prediction method for {task}; data arm {runtime['arm']}.
    The strongest fixed baseline is source-selected and is the starting program; it is often demeaned Rep1 support or zero.
    Study repeatable TARGET-SPECIFIC residuals, shared calibration across genes and context-aware transfer.
    Choose your own algorithm, features, losses, training and authorized data selection; no mandated neural architecture.
    All ordered output genes are in data['genes']; do not assume a row count, fixed projection width, or availability of donors.
    data has train_y/calibration_y, train_control/query_control, train_prior/query_prior,
    train_x/query_x, train_targets/query_targets, train_support/query_support and source_context.
    Expanded pools also supply aux_baseline, aux_delta, aux_target, aux_context, aux_dataset,
    aux_stimulus, aux_modality, aux_batch and aux_time. These are authorized source rows, not query truth.
    target_alias_labels/target_alias_symbols provide annotation-only canonical gene-name joins; use them when matching targets across studies.
    You may choose, weight, reject or combine these sources, accounting explicitly for CRISPR modality/stimulus differences.
    Aux inputs may be empty. STRING features are optional; target labels may need matching to gene identifiers.
    Input NumPy arrays may be float64: explicitly cast for Torch modules. Respect budget['steps'] even when five.
    Return a finite NumPy array of shape (len(query_ids), len(genes)). No network, subprocess or file access beyond worker data.
    An early contract evaluation checks two real-schema queries before full source OOF. Confirmation/outer scores are unavailable.
    Source-only history, including failures and previous hypotheses: {memory}
    Use this history to repair errors and explore different mechanisms, rather than repeat broken code.
    '''
    evo=EvolutionConfig(num_generations=a.generations,init_program_path=str(pool/'seed.py'),results_dir=str(pool/'evolution'),
        llm_models=[model],llm_kwargs={'max_tokens':cfg['api_max_tokens']},task_sys_msg=prompt,
        patch_types=['diff','full'],patch_type_probs=[.5,.5],max_patch_resamples=cfg['max_patch_resamples'],max_patch_attempts=cfg['max_patch_attempts'],
        embedding_model=None,meta_rec_interval=None,max_novelty_attempts=1,use_text_feedback=True)
    job=LocalJobConfig(eval_program_path=str(Path(plan['repo'])/'experiments/round23_evolve/evaluate.py'),python_executable=runtime['python'],time='01:00:00',numeric_threads_per_job=2)
    runner=ShinkaEvolveRunner(evo_config=evo,db_config=DatabaseConfig(num_islands=3,archive_size=100,migration_interval=5,migration_rate=.2),
        job_config=job,max_evaluation_jobs=1,max_proposal_jobs=1,max_db_workers=1)
    runner.run()
    (pool/'framework_state.json').write_text(json.dumps({'requested_generation_slots':a.generations,'stored_generations':runner.completed_generations,
        'proposals_generated':runner.total_proposals_generated,'scope':'Slots include failed proposals and initial clones; valid NEW behaviors are counted separately'}))


if __name__=='__main__':main()
