"""Continue the same Shinka module archive; only source feedback is sent to the model."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--generations',type=int,required=True);a=p.parse_args()
    if version('shinka-evolve')!='0.0.7':raise ValueError('Pinned Shinka 0.0.7 required')
    from shinka.core import ShinkaEvolveRunner,EvolutionConfig
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig
    root=Path(a.root);plan=json.loads((root/'run_plan.json').read_text());cfg=plan['config'];os.environ['ROUND24_ROOT']=str(root)
    memory=(root/'source_feedback.json').read_text(encoding='utf-8')
    prompt='''Improve supported Rep1-to-Rep2 target-specific residual prediction. Return an entire Python ALGORITHM MODULE.
The whole module is editable, including helper functions and imports. Keep one fit_predict(data,budget) entrypoint.
EVOLVE markers surround the module for patching; there is no immutable import/header shell to reproduce.
The trusted evaluator assembles imports and executes your module only in a kernel-isolated worker.
Never access files/network/subprocesses. Query truth is unavailable. Return finite ordered mean-delta predictions.
Use train_y,calibration_y,train_support,calibration_support,query_support,train_x,calibration_x,query_x.
Context arrays are train_source_context,calibration_source_context,query_source_context (not source_context).
Targets use train_targets,calibration_targets,query_targets. Ordered gene IDs are genes; gene_symbols gives unambiguous HGNC identities.
The aux_baseline,aux_delta,aux_target,aux_dataset,aux_context,aux_stimulus,aux_modality,aux_batch,aux_time arrays are legal auxiliary sources.
You may select or reject sources and invent features, residual models, calibration, ensembles or losses. No mandated neural architecture.
Queries may contain two rows in the contract smoke. Respect budget steps; explicitly cast NumPy/Torch dtypes.
Use the strongest source-selected fixed comparator, including the old winner. Optimize MSE without degrading strong-gene error or specificity.
Support-only and identity-only wrong/shuffled controls are independent. Amplitude is diagnostic, not a request to inflate predictions.
All available feedback below is SOURCE SEARCH only, never confirmation or external labels:
'''+memory
    model=f'local/{cfg["model"]}@http://127.0.0.1:{cfg["port"]}/v1?api_key_env=ROUND22_GATEWAY_KEY'
    runner=ShinkaEvolveRunner(evo_config=EvolutionConfig(num_generations=a.generations,init_program_path=str(root/'seed.py'),results_dir=str(root/'evolution'),
        llm_models=[model],llm_kwargs={'max_tokens':cfg['api_max_tokens']},task_sys_msg=prompt,patch_types=['full'],patch_type_probs=[1.],
        max_patch_resamples=1,max_patch_attempts=1,embedding_model=None,meta_rec_interval=None,use_text_feedback=True,max_novelty_attempts=1),
        db_config=DatabaseConfig(num_islands=2,archive_size=60,migration_interval=5,migration_rate=.2),
        job_config=LocalJobConfig(eval_program_path=str(Path(plan['repo'])/'experiments/round24_evolve/evaluate.py'),python_executable=plan['python'],time='01:00:00',numeric_threads_per_job=2),
        max_evaluation_jobs=1,max_proposal_jobs=1,max_db_workers=1)
    runner.run()


if __name__=='__main__':main()
