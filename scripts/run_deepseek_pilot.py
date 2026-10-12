"""Bounded live DeepSeek pilots on synthetic tools or completed P4 experts."""
import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))

import pandas as pd
from vcell.clean_background import run_lock
from vcell.model_agent.artifacts import read, seal, verify
from vcell.model_agent.contracts import load_query, safe_name
from vcell.model_agent.deepseek import DEFAULT_MODEL, DEFAULT_URL, DeepSeekPlanner
from vcell.model_agent.evidence import Profiles
from vcell.model_agent.experts import Registry, export_p4
from vcell.model_agent.runtime import code_identity, evaluate, execute
from vcell.model_agent.workflows import combine, package_review, query_from_prepared
from vcell.round11 import freeze, verify_files
from vcell.utils import file_sha256, write_json

BACKGROUNDS=('A549_IFNG','BXPC3_IFNG','HAP1_IFNG','MCF7_IFNG')


def prepare_p4(parent,root,protocol,seed):
    verify_files(parent);pool=root/'pool';registry=pool/'registry.json'
    if registry.exists():Registry(registry)
    paths=[];expanded_cfg=None
    for variant in ('reference','expanded'):
        worker=parent/'models'/f'{protocol}_{variant}_seed{seed}'
        verify_files(worker)
        cfg=read(worker/'manifest.json')['config']
        if cfg['variant']!=variant or cfg['seed']!=seed or cfg['protocol']!=protocol:raise ValueError('P4 worker identity mismatch')
        dest=pool/variant
        if (dest/'COMPLETE.json').exists():verify(dest);path=dest/'registry.json'
        else:path=export_p4(worker,dest)
        paths.append(path)
        if variant=='expanded':expanded_cfg=cfg
    if not registry.exists():registry=combine(paths,registry)
    # Deterministic identifier-only case selection, never based on response/errors.
    metadata=Path(expanded_cfg['data_dir'])/'metadata.csv'
    meta=pd.read_csv(metadata,dtype=str,keep_default_na=False);queries=root/'queries';queries.mkdir(exist_ok=True)
    chosen=[]
    for context in BACKGROUNDS:
        candidates=meta.loc[meta.context.eq(context)&meta.split.eq('train')].sort_values(['perturbation','row_id'])
        if candidates.empty:continue
        row_id=str(candidates.iloc[0].row_id);output=queries/(context+'.json')
        if not output.exists():query_from_prepared(expanded_cfg['data_dir'],row_id,output)
        q=load_query(output);chosen.append({'context':q.context,'target':q.target,'query_id':q.query_id,'file':str(output)})
    if not chosen:raise ValueError('No eligible IFNG development queries in completed P4 data')
    freeze(root/'cases.json',{'selection':'first identifier-sorted training row per development background; no response decoding',
        'cases':chosen,'metadata_sha256':file_sha256(metadata),'in_sample_possible':True,'test_evaluated':False})
    return registry


def main(argv=None,planner_factory=DeepSeekPlanner):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='deepseek_pilot_01')
    p.add_argument('--mode',choices=['synthetic','p4'],default='synthetic');p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--p4-run');p.add_argument('--protocol',choices=['context','target','double'],default='context')
    p.add_argument('--seed',type=int,default=17);p.add_argument('--profiles',help='Optional genuine source-OOF profiles for the real expert pool')
    p.add_argument('--model',default=DEFAULT_MODEL);p.add_argument('--base-url',default=DEFAULT_URL)
    p.add_argument('--reasoning-effort',choices=['low','high','max'],default='max')
    p.add_argument('--max-output-tokens',type=int,default=8192);p.add_argument('--max-api-calls',type=int,default=8)
    p.add_argument('--max-steps',type=int,default=8);p.add_argument('--budget-units',type=float,default=3.1)
    p.add_argument('--api-timeout',type=float,default=180.);p.add_argument('--resume',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args(argv);safe_name(a.name)
    if not 1<=a.rounds<=20 or a.seed<0 or a.max_steps<1 or a.max_api_calls<1 or a.max_output_tokens<256 or a.budget_units<=0:
        p.error('Invalid pilot limits')
    work=Path(a.work_dir).resolve();root=work/'runs'/a.name
    with run_lock(work/'runs'/('.'+a.name+'.lock')):return launch(a,root,work,planner_factory)


def launch(a,root,work,planner_factory):
    parent=Path(a.p4_run or work/'runs/round15_p4_02').resolve() if a.mode=='p4' else None
    if parent:verify_files(parent)
    if a.mode=='synthetic' and a.profiles:raise ValueError('Synthetic pilot builds its own synthetic source profiles')
    profile_path=Path(a.profiles).resolve() if a.profiles else None
    if profile_path:verify(profile_path)
    plan={'experiment':'deepseek_live_pilot','source':code_identity(),'launcher':file_sha256(__file__),
        **{k:getattr(a,k) for k in ('mode','rounds','protocol','seed','model','base_url','reasoning_effort','max_output_tokens',
                                    'max_api_calls','max_steps','budget_units','api_timeout')},
        'p4_parent':str(parent) if parent else None,'p4_parent_complete':file_sha256(parent/'COMPLETE.json') if parent else None,
        'profiles':str(profile_path) if profile_path else None,'profile_complete':file_sha256(profile_path/'COMPLETE.json') if profile_path else None,
        'scope':'API and model-execution pilot; repeated agent episodes are not biological replicates',
        'test_evaluated':False,'tool_costs':'declared units, not inference time or token charges',
        'max_requests_total':a.rounds*a.max_api_calls,'max_output_tokens_total':a.rounds*a.max_api_calls*a.max_output_tokens}
    if root.exists() and any(root.iterdir()) and not a.resume:raise ValueError('Use --resume or new pilot name')
    root.mkdir(parents=True,exist_ok=True);freeze(root/'plan.json',plan)
    if (root/'COMPLETE.json').exists():verify(root);print('DEEPSEEK PILOT COMPLETE VERIFIED:',root,flush=True);return root
    def stage(value,**extra):
        write_json(root/'stage.json',{'stage':value,'mode':a.mode,'test_evaluated':False,**extra});print('DEEPSEEK PILOT STAGE:',value,flush=True)
    try:
        stage('prepare executable tools and control-only queries')
        if a.mode=='synthetic':
            from vcell.model_agent.demo import make_demo
            base=root/'synthetic'
            if not (base/'READY.json').exists():
                if base.exists() and any(base.iterdir()):raise ValueError('Incomplete synthetic preparation; use a new name')
                make_demo(base);write_json(base/'READY.json',{'ready':True})
            registry_path=base/'registry.json';profile_path=base/'profiles';query_paths=[base/'query.json']
        else:
            registry_path=prepare_p4(parent,root,a.protocol,a.seed)
            query_paths=[Path(row['file']) for row in read(root/'cases.json')['cases']]
        registry=Registry(registry_path)
        if profile_path:Profiles(profile_path,registry)
        freeze(root/'inputs.json',{'registry':registry.fingerprint,'queries':{str(q):load_query(q).fingerprint for q in query_paths},
                                'profiles_complete':file_sha256(profile_path/'COMPLETE.json') if profile_path else None})
        if a.prepare_only:stage('prepared; no API requests sent');return root
        summaries=[]
        for i in range(a.rounds):
            label=f'round_{i+1:03d}';qpath=query_paths[i%len(query_paths)];q=load_query(qpath)
            episode=root/'episodes'/label;audit=root/'api'/label
            stage('live agent decisions',round=i+1,total=a.rounds,context=q.context,target=q.target)
            planner=planner_factory(audit,a.model,base_url=a.base_url,reasoning_effort=a.reasoning_effort,
                max_output_tokens=a.max_output_tokens,max_api_calls=a.max_api_calls,timeout=a.api_timeout)
            result=execute(registry_path,qpath,episode,profile_path=profile_path,budget_units=a.budget_units,
                max_steps=a.max_steps,resume=a.resume,planner=planner,planner_identity=planner.identity)
            summary={'round':i+1,'context':q.context,'target':q.target,'query_id':q.query_id,'status':result['status'],
                'selected_expert':result['selected_expert'],'tool_units':result['spent_units'],**result['provider_summary']}
            if a.mode=='synthetic':summary['synthetic_selected_mse']=evaluate(episode,qpath,root/'synthetic/private_truth.npz',registry_path)['selected_mse']
            summaries.append(summary);write_json(root/'summary.json',{'mode':a.mode,'episodes':summaries,
                'formal_effectiveness_evaluated':False,'test_evaluated':False})
            print(f'DEEPSEEK ROUND COMPLETE {i+1}/{a.rounds}: {result["status"]} expert={result["selected_expert"]}',flush=True)
        (root/'INCOMPLETE.json').unlink(missing_ok=True);stage('complete',rounds=a.rounds)
        write_json(root/'provider_summary.json',{'requested_model':a.model,'reasoning_effort':a.reasoning_effort,
            'attempted_requests':sum(row['attempted_requests'] for row in summaries),
            'known_input_tokens':sum(row['known_input_tokens'] for row in summaries),
            'known_output_tokens':sum(row['known_output_tokens'] for row in summaries),
            'unknown_usage_attempts':sum(row['unknown_usage_attempts'] for row in summaries),
            'currency_cost':'Not estimated; token costs follow provider billing and cache/peak status'})
        seal(root,live_provider='deepseek',formal_effectiveness_evaluated=False,test_evaluated=False)
        print('DEEPSEEK PILOT COMPLETE:',root,flush=True)
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error_type':type(exc).__name__,'message':'Pilot stopped; inspect provider audit or worker traceback',
            'test_evaluated':False});raise
    finally:print('REVIEW PACKAGE:',package_review(root),flush=True)
    return root


if __name__=='__main__':main()
