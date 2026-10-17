"""Run one resumable native Shinka generation with a frozen source action."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vcell.round25.shinka_hooks import check_pinned_api,install_action


def main():
    p=argparse.ArgumentParser();p.add_argument('--pool');p.add_argument('--generations',type=int);p.add_argument('--check-api',action='store_true');a=p.parse_args()
    capabilities=check_pinned_api()
    if a.check_api: print(json.dumps(capabilities));return
    from shinka.core import ShinkaEvolveRunner,EvolutionConfig
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig
    pool=Path(a.pool);plan=json.loads((pool/'run_plan.json').read_text());cfg=plan['config']
    instruction=json.loads((pool/'active_action.json').read_text());action=instruction.get('action')
    install_action(action,instruction.get('banned',[]))
    memory=json.loads((pool/'source_feedback.json').read_text())
    prompt='''Invent a complete numerical perturbation-response method, not just a weight-grid variant.
Return a full Python module with exactly one fit_predict(data,budget); the whole module and imports are editable.
NumPy, SciPy and PyTorch are available; do not assume extra dependencies or downloads.
Never access files, network, subprocesses or hidden labels. Return finite ordered mean-delta predictions.
Data: train_y/calibration_y are legal responses; train_support/calibration_support/query_support are measured Rep1
for repeat, and condition-matched source donors (or zeros if absent) for whole-cell-line transfer.
train_x/calibration_x/query_x contain legal control and target-prior features; train_control, calibration_control,
query_control, train_prior/calibration_prior/query_prior are also available. train_source_context,
calibration_source_context,query_source_context are cell backgrounds. train_targets/calibration_targets/query_targets
are HGNC identities; genes are ordered Ensembl outputs; gene_symbols supplies unambiguous HGNC names.
aux_delta,aux_baseline,aux_target,aux_dataset,aux_context,aux_stimulus,aux_modality,aux_batch,aux_time are legal sources.
You may invent residual decomposition, reliability estimation, background adaptation, graph features, low-rank/sparse
responses, optimization or ensembles. No architecture is required. Query may have only two rows in contract smoke.
Seed all random generators and cast dtypes explicitly; obey budget optimizer steps. Do not inflate output amplitude.
Correct-vs-wrong support and explicit target identity are tested separately. Same-cell supported repeat is not zero-shot transfer.
The typed action below is executable parent selection/fix/crossover policy, not permission to change evaluation.
All metrics below are source search only. Strong-gene undefined values are unknown, not zero.
'''
    prompt+='\nTASK: '+plan['task']+'\nSOURCE EVIDENCE: '+json.dumps(memory,allow_nan=False)
    prompt+='\nCURRENT RESEARCH ACTION: '+json.dumps(instruction,allow_nan=False)
    os.environ['ROUND25_POOL']=str(pool)
    model=f'local/{cfg["model"]}@http://127.0.0.1:{cfg["port"]}/v1?api_key_env=ROUND22_GATEWAY_KEY'
    runner=ShinkaEvolveRunner(evo_config=EvolutionConfig(num_generations=a.generations,
        init_program_path=str(pool/'seed.py'),results_dir=str(pool/'evolution'),llm_models=[model],
        llm_kwargs={'max_tokens':cfg['api_max_tokens']},task_sys_msg=prompt,patch_types=['full'],patch_type_probs=[1.],
        max_patch_resamples=1,max_patch_attempts=1,max_novelty_attempts=1,embedding_model=None,
        meta_rec_interval=None,use_text_feedback=True),
        db_config=DatabaseConfig(num_islands=2,archive_size=60,migration_interval=5,migration_rate=.2,
            num_archive_inspirations=2,num_top_k_inspirations=1),
        job_config=LocalJobConfig(eval_program_path=str(Path(plan['repo'])/'experiments/round25_evolve/evaluate.py'),
            python_executable=plan['python'],time='01:00:00',numeric_threads_per_job=2),
        max_evaluation_jobs=1,max_proposal_jobs=1,max_db_workers=1)
    runner.run()


if __name__=='__main__':main()
