"""Runs in the pinned Shinka venv, using its real evolutionary runner."""
import argparse
from importlib.metadata import version
from pathlib import Path
import os
import yaml


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--repo',required=True);p.add_argument('--task',required=True);p.add_argument('--python',required=True);p.add_argument('--generations',type=int,required=True);p.add_argument('--port',type=int,default=18022);p.add_argument('--model',default='deepseek-v4-pro');p.add_argument('--provider-smoke',action='store_true');a=p.parse_args()
    if version('shinka-evolve')!='0.0.7':raise RuntimeError('Install pinned shinka-evolve==0.0.7')
    from shinka.core import ShinkaEvolveRunner,EvolutionConfig
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig
    repo=Path(a.repo);root=Path(a.root);cfg=yaml.safe_load((repo/'experiments/round22_evolve/shinka.yaml').read_text())
    os.environ['ROUND22_ROOT']=str(root);os.environ['ROUND22_TASK']=a.task
    model=f'local/{a.model}@http://127.0.0.1:{a.port}/v1?api_key_env=ROUND22_GATEWAY_KEY'
    import json
    from importlib.metadata import distributions
    versions={d.metadata['Name']:d.version for d in distributions()}
    (root/'shinka_versions.json').write_text(json.dumps(versions,sort_keys=True,indent=2))
    if a.provider_smoke:
        import asyncio
        from shinka.llm import AsyncLLMClient
        result=asyncio.run(AsyncLLMClient(model_names=[model],max_tokens=4096,verbose=False).query(
            msg='Return exactly ROUND22_READY.',system_msg='This is a provider compatibility smoke test; no research data.'))
        if result is None or result.content.strip()!='ROUND22_READY':raise RuntimeError('Shinka/DeepSeek smoke response invalid')
        (root/'provider_verified.json').write_text(json.dumps({'framework':'0.0.7','model':a.model,'compatible':True,'research_data_sent':False}))
        return
    evo=EvolutionConfig(**cfg['evo'],num_generations=a.generations,llm_models=[model],
        init_program_path=str(repo/'experiments/round22_evolve/initial.py'),results_dir=str(root/'evolution'/a.task),
        task_sys_msg=f'''Develop a perturbation-specific mean-delta prediction method for task {a.task}.
        Evolve the entire marked fit_predict implementation, not only scalar hyperparameters.
        Inputs include source train/calibration labels, raw matched controls, immutable STRING priors,
        train-fit projections and legal same-target experimental support. For repeat, query Rep1 is permitted.
        For transfer, query perturbation response is never observed. No network, subprocess, repository,
        grader or hidden-label access is permitted. Output shape is query count x ordered 1834 genes.
        Follow supplied steps/seed/device budget. Optimize gains over strong fixed baselines and correct
        vs wrong target usage; merely collapsing output toward zero is not biological success.
        numpy and torch are available. Use historical measured feedback to revise hypotheses.''')
    job=LocalJobConfig(eval_program_path=str(repo/'experiments/round22_evolve/evaluate.py'),python_executable=a.python,time='00:45:00',numeric_threads_per_job=2)
    runner=ShinkaEvolveRunner(evo_config=evo,job_config=job,db_config=DatabaseConfig(**cfg['db']),
        max_evaluation_jobs=1,max_proposal_jobs=1,max_db_workers=1)
    runner.run()
    (root/'evolution'/a.task/'framework_state.json').write_text(json.dumps({'target_generations':a.generations,
        'stored_generations':runner.completed_generations,'proposals_generated':runner.total_proposals_generated,'framework_returned':True}))
    if runner.completed_generations<a.generations:raise RuntimeError('Framework stopped before requested generation budget; inspect failed proposals and API ledger')


if __name__=='__main__':main()
