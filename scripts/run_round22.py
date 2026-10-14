"""Prepare, protect, evolve, confirm, then grade an observed development cell line."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from vcell.clean_background import run_lock
from vcell.model_agent.artifacts import read,verify,seal
from vcell.model_agent.contracts import digest
from vcell.round11 import freeze
from vcell.round22.data import prepare
from vcell.round22.evaluator import evaluate,execute,worker_env,namespace
from vcell.round22.gateway import Gateway
from vcell.round22.metrics import macro
from vcell.utils import write_json,file_sha256


def stage(root,value,**kwargs):
    write_json(root/'stage.json',{'stage':value,'test_evaluated':False,**kwargs});print('ROUND22 STAGE:',value,kwargs,flush=True)


def preflight(root,budget,python):
    """Real adversarial IO probes in the same kernel boundary used for candidates."""
    forbidden=root/'private'/'sandbox_hidden.txt';forbidden.parent.mkdir(parents=True,exist_ok=True);forbidden.write_text('hidden grader sentinel')
    other=root/'outside_write.txt'
    code=f'''import numpy as np
import torch
# EVOLVE-BLOCK-START
def fit_predict(data,budget):
    import os, socket, subprocess
    from pathlib import Path
    rejected=0
    for action in (lambda: Path({str(forbidden)!r}).read_text(),lambda: Path({str(other)!r}).write_text('bad'),lambda:socket.socket(),lambda:subprocess.run(['/bin/true'])):
        try: action()
        except (PermissionError,OSError): rejected+=1
    if rejected!=4: raise RuntimeError('Candidate isolation did not reject all forbidden actions')
    model=torch.nn.Linear(2,1).to(budget['device'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    optimizer.zero_grad();model(torch.ones((2,2),device=budget['device'])).square().mean().backward();optimizer.step()
    return np.zeros((len(data['query_ids']),len(data['genes'])),dtype=np.float32)
# EVOLVE-BLOCK-END
'''
    data={'query_ids':np.array(['opaque']), 'query_targets':np.array(['probe']),'genes':np.array(['probe_gene'])}
    execute(code,data,root/'preflight',budget,python)
    if other.exists():raise RuntimeError('Outside write escaped sandbox')
    write_json(root/'sandbox_verified.json',{'boundary':'Landlock ABI>=3 + libseccomp; no prompt or AST sandbox claim',
        'denied':['hidden label read','outside write','network socket','subprocess execution'],'device':budget['device'],'verified':True})


def random_program(seed):
    rng=np.random.default_rng(seed);penalty=float(rng.choice([.1,1.,10.,100.]));rank=int(rng.choice([8,16,32]));center=bool(rng.integers(2))
    code=(REPO/'experiments/round22_evolve/initial.py').read_text()
    # Fixed random-mutation grammar is disclosed; not equal to unrestricted LLM code scope.
    code=code.replace('min(16,len(vh))',f'min({rank},len(vh))').replace('10.*penalty',f'{penalty}*penalty')
    if not center:code=code.replace('support=support-mu1;qs=qs-mu1','support=support;qs=qs')
    return code


def programs(root,task):
    records=[]
    for path in sorted((root/'evolution'/task).glob('gen_*/main.py')):
        metrics=path.parent/'results/metrics.json';correct=path.parent/'results/correct.json'
        if metrics.exists() and correct.exists() and read(correct).get('correct'):
            records.append({'arm':'shinka','task':task,'generation':int(path.parent.name.split('_')[1]),'code':str(path),'sha256':file_sha256(path),'metrics':read(metrics),'search_dir':str(path.parent/'results')})
    for path in sorted((root/'random'/task).glob('trial_*/candidate.py')):
        if (path.parent/'metrics.json').exists() and read(path.parent/'correct.json').get('correct'):
            records.append({'arm':'random','task':task,'code':str(path),'sha256':file_sha256(path),'metrics':read(path.parent/'metrics.json'),'search_dir':str(path.parent)})
    path=REPO/'experiments/round22_evolve/initial.py';out=root/'fixed'/task/'second'
    records.append({'arm':'fixed','task':task,'code':str(path),'sha256':file_sha256(path),'metrics':read(out/'metrics.json'),'search_dir':str(out)})
    return records


def package(root):
    import io,tarfile
    from collections import deque
    dest=root/'round22_review_light.tar.gz';entries={}
    with tarfile.open(dest,'w:gz') as tar:
        for p in sorted(root.rglob('*')):
            rel=p.relative_to(root)
            if not p.is_file() or p.suffix not in ('.json','.csv','.md','.py','.yaml','.log'):continue
            if any(s in rel.parts for s in ('private','inputs','workers','preflight')):continue
            payload=p.read_bytes()
            if p.suffix=='.log':
                with p.open(encoding='utf-8',errors='replace') as f:payload=''.join(deque(f,maxlen=50)).encode()
            m=tarfile.TarInfo(rel.as_posix());m.size=len(payload);tar.addfile(m,io.BytesIO(payload));entries[m.name]={'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
        payload=json.dumps({'files':entries,'scope':'Reports, generated code and log tails; no private labels, matrices, model weights or keys'}).encode();m=tarfile.TarInfo('ARCHIVE_MANIFEST.json');m.size=len(payload);tar.addfile(m,io.BytesIO(payload))
    return dest


def launch(a):
    work=Path(a.work_dir);root=work/'runs'/a.name;parent=Path(a.round21_run or work/'runs/round21_signal_01')
    if root.exists() and any(root.iterdir()) and not a.resume:raise ValueError('Use --resume with same frozen options or a new name')
    root.mkdir(parents=True,exist_ok=True)
    code_hash=namespace(REPO)
    budget={'steps':a.steps,'seconds':a.worker_seconds,'seed':a.seed,'device':a.device,'ram_bytes':12*1024**3,
        'evaluation_seconds_max':a.evaluation_hours*3600,'root':str(root)}
    freeze(root/'run_plan.json',{'repo':str(REPO),'python':sys.executable,'code_sha256':code_hash,'parent':str(parent),
        'parent_sha256':file_sha256(parent/'COMPLETE.json'),'budget':budget,'generations_per_task':a.generations,'tasks':a.tasks,'outer_context':a.outer_context,
        'seed':a.seed,'api_requests_max':a.api_requests,'api_output_tokens_max':a.api_output_tokens,'model':a.model,'port':a.port,
        'shinka_python':str(a.shinka_python or work/'envs/shinka/bin/python'),'shinka_version':'0.0.7',
        'training_versions':{name:version(name) for name in ('torch','numpy','pandas','scipy')},'test_evaluated':False})
    if (root/'COMPLETE.json').exists():verify(root);print('ROUND22 COMPLETE VERIFIED',root);return root
    try:
        stage(root,'rebuild source-only search and confirmation packs')
        plan=read(root/'data_plan.json') if (root/'data_plan.json').exists() else prepare(parent,root,a.outer_context,a.seed)
        if not plan['raw_controls_verified']:raise ValueError('Verified Round21 independent matched NT controls required')
        if a.mode=='prepare':stage(root,'prepared; no code execution or API');return root
        stage(root,'kernel isolation and deterministic fixed-method smoke')
        preflight(root,budget,sys.executable)
        initial=(REPO/'experiments/round22_evolve/initial.py').read_text()
        selections={}
        for task in a.tasks:
            first=evaluate(initial,root,task,'search',root/'fixed'/task/'first')
            second=evaluate(initial,root,task,'search',root/'fixed'/task/'second')
            if first!=second:raise ValueError('Fixed-method metrics did not reproduce exactly')
            f=pd.read_csv(root/'fixed'/task/'first'/'fixed_scores.csv')
            selections[task]=min(f.method.unique(),key=lambda m:macro(f[f.method.eq(m)],'mse'))
        freeze(root/'baseline_selection.json',selections)
        # Rescore seed against the selected strongest numerical baseline.
        for task in a.tasks:evaluate(initial,root,task,'search',root/'fixed'/task/'second')
        if a.mode=='smoke':stage(root,'numeric smoke complete; live provider and search not run');return root
        key=os.environ.get('DEEPSEEK_API_KEY')
        if not key:raise RuntimeError('Set DEEPSEEK_API_KEY in trusted orchestrator environment')
        token=secrets.token_hex(24);shinka=read(root/'run_plan.json')['shinka_python']
        if not Path(shinka).is_file():raise ValueError('Install isolated Shinka venv with scripts/install_round22_autodl.sh')
        with Gateway(root/'api_ledger.json',key,a.model,token,a.api_requests,a.api_output_tokens,a.port):
            for task in a.tasks:
                stage(root,'ShinkaEvolve method search',task=task,generations=a.generations)
                env=worker_env();env.update(ROUND22_GATEWAY_KEY=token,ROUND22_ROOT=str(root),ROUND22_TASK=task,SHINKA_PRICING_MODE='offline')
                command=[shinka,str(REPO/'scripts/run_shinka_round22.py'),'--root',str(root),'--repo',str(REPO),'--task',task,'--python',sys.executable,
                    '--generations',str(a.generations),'--model',a.model,'--port',str(a.port)]
                (root/'logs').mkdir(exist_ok=True)
                if not (root/'provider_verified.json').exists():
                    stage(root,'one tiny Shinka/DeepSeek provider smoke; no research data')
                    with (root/'logs/provider_smoke.log').open('a',encoding='utf-8') as log:
                        result=subprocess.run([*command,'--provider-smoke'],env=env,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode:raise RuntimeError('Provider compatibility smoke failed; inspect logs/provider_smoke.log')
                with (root/'logs'/f'shinka_{task}.log').open('a',encoding='utf-8') as log:
                    result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError('Shinka runner failed; inspect '+str(root/'logs'/f'shinka_{task}.log'))
                stage(root,'equal candidate-count random mutation control',task=task)
                for i in range(a.generations):
                    out=root/'random'/task/f'trial_{i:03d}';out.mkdir(parents=True,exist_ok=True);code=random_program(a.seed+i);freeze(out/'recipe.json',{'seed':a.seed+i,'scope':'rank/penalty/centering grammar; narrower than arbitrary generated code'})
                    (out/'candidate.py').write_text(code)
                    if not (out/'correct.json').exists():
                        try:evaluate(code,root,task,'search',out)
                        except Exception as exc:
                            write_json(out/'correct.json',{'correct':False,'error':str(exc)});write_json(out/'metrics.json',{'combined_score':-1000000.,'public':{},'private':{}})
        records=[r for task in a.tasks for r in programs(root,task)];chosen=[]
        for task in a.tasks:
            for arm in ('shinka','random','fixed'):
                candidates=sorted([r for r in records if r['task']==task and r['arm']==arm and (arm!='shinka' or r['generation']>0)],key=lambda r:r['metrics']['combined_score'],reverse=True)
                # Two evolution finalists; one random and one fixed comparator.
                seen=set()
                for r in candidates:
                    if r['sha256'] in seen:continue
                    seen.add(r['sha256']);chosen.append(r)
                    if len(seen)>=(2 if arm=='shinka' else 1):break
        freeze(root/'search_frozen.json',{'candidates':chosen,'baseline_selection':selections,'confirmation_used':False,'outer_used':False})
        confirmations=[]
        for i,r in enumerate(chosen):
            stage(root,'frozen source confirmation',task=r['task'],candidate=i+1,total=len(chosen))
            code=Path(r['code']).read_text()
            if file_sha256(r['code'])!=r['sha256']:raise ValueError('Finalist code changed')
            out=root/'confirmation'/r['task']/f'candidate_{i:03d}';metrics=evaluate(code,root,r['task'],'confirmation',out)
            confirmations.append({**r,'confirmation_metrics':metrics,'confirmation_dir':str(out)})
        winners={}
        for task in a.tasks:
            contenders=[r for r in confirmations if r['task']==task and r['arm']=='shinka']
            winners[task]=max(contenders,key=lambda r:r['confirmation_metrics']['combined_score']) if contenders else None
        freeze(root/'confirmation_frozen.json',{'records':confirmations,'winners':winners,'outer_used':False})
        outer=[]
        for r in confirmations:
            if r['task']!='transfer':continue
            # Include frozen fixed and random controls, and the confirmation-selected Shinka winner.
            if r['arm']=='shinka' and (winners['transfer'] is None or r['sha256']!=winners['transfer']['sha256']):continue
            stage(root,'private whole-line development evaluation',arm=r['arm'])
            metrics=evaluate(Path(r['code']).read_text(),root,'transfer','outer',root/'outer'/r['arm'])
            outer.append({'arm':r['arm'],'metrics':metrics})
        rows=[]
        for r in records:rows.append({'phase':'search','task':r['task'],'arm':r['arm'],'code_sha256':r['sha256'],**r['metrics']['public']})
        for r in confirmations:rows.append({'phase':'confirmation','task':r['task'],'arm':r['arm'],'code_sha256':r['sha256'],**r['confirmation_metrics']['public']})
        for r in outer:rows.append({'phase':'observed_development_outer','task':'transfer','arm':r['arm'],**r['metrics']['public']})
        pd.DataFrame(rows).to_csv(root/'comparison.csv',index=False)
        write_json(root/'summary.json',{'search_complete':True,'confirmation_complete':True,'outer_development_evaluated':bool(outer),
            'tasks_separate':True,'test_evaluated':False,'framework':'ShinkaEvolve 0.0.7','encoder':plan['scope'],
            'random_control_scope':'Same starting program and candidate-count/worker limits; fixed rank/penalty/centering mutation grammar, not unrestricted code-search equivalence',
            'no_Qwen_pretrained_inference':True,'repeat_is_not_zero_shot':True,'raw_data_changed':False,'HT29_evaluated':False,
            'confirmed_agent_signal':{t:r['confirmation_metrics']['public']['biological_success'] if r else False for t,r in winners.items()},
            'valid_evolved_candidates':{t:len([r for r in records if r['task']==t and r['arm']=='shinka' and r['generation']>0]) for t in a.tasks},
            'baseline_retained_unless_confirmed':True,'outer_candidates_exploratory':True,
            'agent_success':'Code search and confirmation are reported separately from biological success; tiny near-zero MSE gains do not meet promotion criteria'})
        (root/'INCOMPLETE.json').unlink(missing_ok=True);stage(root,'complete');seal(root,test_evaluated=False);return root
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error_type':type(exc).__name__,'error':str(exc),'test_evaluated':False});raise
    finally:print('REVIEW PACKAGE:',package(root),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round22_evolve_01')
    p.add_argument('--round21-run');p.add_argument('--outer-context',choices=['HAP1_IFNG','BXPC3_IFNG'],default='HAP1_IFNG');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--tasks',nargs='+',choices=['repeat','transfer'],default=['repeat','transfer']);p.add_argument('--generations',type=int,default=6)
    p.add_argument('--steps',type=int,default=500);p.add_argument('--worker-seconds',type=int,default=120);p.add_argument('--evaluation-hours',type=float,default=2.)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--api-requests',type=int,default=24);p.add_argument('--api-output-tokens',type=int,default=196608)
    p.add_argument('--model',default='deepseek-v4-pro');p.add_argument('--port',type=int,default=18022);p.add_argument('--shinka-python');p.add_argument('--mode',choices=['prepare','smoke','run'],default='run');p.add_argument('--resume',action='store_true');a=p.parse_args(argv)
    if not 2<=a.generations<=12 or min(a.steps,a.worker_seconds,a.evaluation_hours,a.api_requests,a.api_output_tokens)<1 or len(set(a.tasks))!=len(a.tasks):raise ValueError('Positive bounded budgets, 2..12 generations, distinct tasks required')
    with run_lock(Path(a.work_dir)/'runs'/('.'+a.name+'.lock')):return launch(a)


if __name__=='__main__':main()
