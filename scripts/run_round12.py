"""Round12: fold-safe target encoders, response diagnostics, controlled data expansion."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch

from run_round7 import run_jobs
from run_round9 import package, job
from vcell.background_training import prepare_background
from vcell.clean_background import run_lock
from vcell.data import load_prepared
from vcell.knowledge import build_knowledge, load_knowledge
from vcell.round11 import freeze, verify_files
from vcell.round12 import (REPRESENTATIONS, PROTOCOLS, assert_extension, make_partition,
                          worker, half_diagnostic, report, save_complete)
from vcell.round12_data import expand
from vcell.train import source_fingerprint
from vcell.utils import file_sha256, write_json

REPO=Path(__file__).resolve().parents[1]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-config')
    p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work')
    p.add_argument('--name',default='round12_01')
    p.add_argument('--round10-run')
    p.add_argument('--expanded-data',help='Optional already prepared and audited Round12 expansion')
    p.add_argument('--knowledge-npz',help='Optional verified knowledge aligned to expanded vocabulary')
    p.add_argument('--seeds',nargs='+',type=int,default=[17,29,43])
    p.add_argument('--protocols',nargs='+',choices=PROTOCOLS,default=list(PROTOCOLS))
    p.add_argument('--representations',nargs='+',choices=REPRESENTATIONS,default=list(REPRESENTATIONS))
    p.add_argument('--predictors',nargs='+',choices=['ridge','mlp'],default=['ridge','mlp'])
    p.add_argument('--max-steps',type=int,default=2000)
    p.add_argument('--pretrain-steps',type=int,help='Default: copy fixed B2 budget from Round10; override only for a declared new budget')
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--parallel-jobs',type=int,choices=[1,2],default=2)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--prepare-only',action='store_true',help='Finish raw/knowledge preparation and job configs, then stop before training')
    a=p.parse_args(argv)
    if a.worker_config: return worker(json.loads(Path(a.worker_config).read_text()),a.resume)
    if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.name):p.error('Use a simple run name')
    if min(a.max_steps,a.batch_size,a.save_every)<1 or min(a.seeds)<0 or (a.pretrain_steps is not None and a.pretrain_steps<1):p.error('Invalid training counts')
    for key in ('seeds','protocols','representations','predictors'):
        values=getattr(a,key)
        if len(values)!=len(set(values)):p.error('Duplicate '+key)
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    work=Path(a.work_dir).resolve()
    with run_lock(work/'runs'/('.'+a.name+'.lock')):return launch(a,work,work/'runs'/a.name)


def launch(a,work,root):
    previous=Path(a.round10_run or work/'runs/round10_01').resolve()
    parent=json.loads((previous/'plan.json').read_text())
    if parent['experiment']!='round10' or not (previous/'COMPLETE.json').exists():raise ValueError('Completed Round10 required')
    oldpath=Path(parent['data_path']);old=load_prepared(oldpath)
    if old['audit']['fingerprint']!=parent['data']:raise ValueError('Historical data changed')
    if old['audit'].get('normalization')!='mean(log1p(10000*counts/full_library))' or old['audit'].get('target_sum')!=10000:
        raise ValueError('Require the existing full-library mean-of-log normalization')
    if set(old['meta'].iloc[old['splits']['train']].context)!={'K562','RPE1','H1_hESC'}:
        raise ValueError('Require historical K562/RPE1/H1 sources')
    plan={'experiment':'round12','source':source_fingerprint(),'launcher':file_sha256(__file__),
          'scheduler':file_sha256(REPO/'scripts/run_round7.py'),'background_launcher':file_sha256(REPO/'scripts/run_round9.py'),
          'round10_plan_sha256':file_sha256(previous/'plan.json'),'old_data':old['audit']['fingerprint'],
          'metadata_sha256':file_sha256(oldpath/'metadata.csv'),
          'expanded_override':str(Path(a.expanded_data).resolve()) if a.expanded_data else None,
          'knowledge_override':file_sha256(a.knowledge_npz) if a.knowledge_npz else None,
          **{k:getattr(a,k) for k in ('seeds','protocols','representations','predictors','max_steps','pretrain_steps','batch_size','save_every','device')},
          'variants':['original','expanded'],'rank':32,'target_dim':32,'background_dim':16,
          'ridge_alphas':[.0001,.001,.01,.1,1.],
          'evaluation':'previously used development data; Jurkat not scored; no automatic winner',
          'normalization':'fixed historical panel; all learned transforms fit within each partition'}
    if root.exists() and any(root.iterdir()) and not a.resume:raise ValueError('Use --resume or new run name')
    if not root.exists() and shutil.disk_usage(work).free<15*1024**3:raise ValueError('Need 15 GiB free; no artifacts deleted')
    root.mkdir(parents=True,exist_ok=True);freeze(root/'plan.json',plan)
    if (root/'COMPLETE.json').exists():
        verify_files(root)
        print(f'ROUND12 COMPLETE VERIFIED: {root}',flush=True)
        return root
    def stage(name):
        write_json(root/'stage.json',{'stage':name,'test_evaluated':False});print('ROUND12 STAGE: '+name,flush=True)
    try:
        stage('raw data QC and uncapped training targets')
        expanded_path=Path(a.expanded_data).resolve() if a.expanded_data else root/'prepared/expanded'
        if a.expanded_data:verify_files(expanded_path);expanded=load_prepared(expanded_path)
        else:expanded=expand(work,old,expanded_path)
        assert_extension(old,expanded)
        freeze(root/'data_identity.json',{'original':old['audit']['fingerprint'],'expanded':expanded['audit']['fingerprint'],
               'expanded_complete':file_sha256(expanded_path/'COMPLETE.json')})
        half_diagnostic(expanded_path,root/'diagnostics')
        for name in ('expansion_qc.csv','data_audit.json'):
            shutil.copyfile(expanded_path/name,root/'diagnostics'/name)
        stage('frozen annotation-only target knowledge')
        model=copy.deepcopy(parent['model'])
        knowledge=Path(a.knowledge_npz).resolve() if a.knowledge_npz else build_knowledge(
            expanded,work/'knowledge/round12',REPO/'assets/function',model,a.device)
        verify_files(knowledge.parent);assets=load_knowledge(knowledge,expanded)
        freeze(root/'knowledge_identity.json',{'path':str(knowledge),'sha256':file_sha256(knowledge),
               'complete_sha256':file_sha256(knowledge.parent/'COMPLETE.json')})
        # Fix shuffle assignments on the union vocabulary before subsetting.
        # Otherwise a data expansion would also change the null representation.
        target_keys=['perturbations','semantic','relations']
        for j in (1,2,3):
            key=f'shuffled_string_{j}'
            perm=np.random.default_rng(817+j).permutation(len(expanded['perturbations']))
            assets[key]=assets['relations'][perm,2,:]
            assets[key+'_mapping']=expanded['perturbations'][perm]
            target_keys.extend([key,key+'_mapping'])
        cards_path=knowledge.parent/'cards.json'
        cards=json.loads(cards_path.read_text()) if cards_path.exists() else {}
        assets['text_available']=np.array([bool(cards.get(p,{}).get('summary')) for p in expanded['perturbations']])
        target_keys.append('text_available')
        coverage=pd.DataFrame({'target':expanded['perturbations'],'functional_text_available':assets['text_available'],
                      'string_panel_neighbors':(assets['relations'][:,2,:]>0).sum(1),
                      'canonical_symbol':[cards.get(p,{}).get('symbol') for p in expanded['perturbations']],
                      'card_status':[cards.get(p,{}).get('status','unverified') for p in expanded['perturbations']]})
        coverage.to_csv(root/'annotation_coverage.csv',index=False)
        mapped=coverage[coverage.card_status.eq('mapped') & coverage.canonical_symbol.notna()]
        if mapped.canonical_symbol.duplicated().any():
            raise ValueError('Multiple target labels map to one canonical gene; resolve aliases before global target holdout')
        # Both data conditions use exactly the same frozen per-gene vectors.
        expanded_k=root/'knowledge_expanded.npz';np.savez_compressed(expanded_k,**assets)
        original_k=root/'knowledge_original.npz'
        lookup={p:i for i,p in enumerate(expanded['perturbations'])};ix=[lookup[p] for p in old['perturbations']]
        np.savez_compressed(original_k,**{k:(v[ix] if k in target_keys else v) for k,v in assets.items()})
        stage('background cache and identifier-only partitions')
        cache=prepare_background(parent['background_path'],root/'background_cache',old['genes'].tolist())
        budget=json.loads((previous/'budget.json').read_text())
        steps=a.pretrain_steps or budget['background_optimizer_steps']
        specs={};bgjobs={};jobs={};(root/'configs').mkdir(exist_ok=True)
        datasets={'original':(old,oldpath,original_k),'expanded':(expanded,expanded_path,expanded_k)}
        for protocol in a.protocols:
            anchored=None
            for variant,(data,path,kpath) in datasets.items():
                parts=make_partition(old,data,protocol)
                rows=data['meta'].iloc[parts['outer']].row_id.tolist()
                if anchored is not None and rows!=anchored:raise ValueError('Evaluation rows changed with expansion')
                anchored=rows
                sf=root/'splits'/(protocol+'_'+variant);sf.mkdir(parents=True,exist_ok=True)
                freeze(sf/'partition.json',parts)
                membership=data['meta'].copy();membership['role']='excluded'
                for role,indices in parts.items():membership.loc[indices,'role']=role
                membership.to_csv(sf/'membership.csv',index=False)
                write_json(sf/'audit.json',{'fit_rows':len(parts['fit']),'calibration_rows':len(parts['calibration']),
                    'outer_rows':len(parts['outer']),'fit_targets':int(data['meta'].iloc[parts['fit']].perturbation.nunique()),
                    'outer_targets':int(data['meta'].iloc[parts['outer']].perturbation.nunique()),
                    'fit_contexts':sorted(data['meta'].iloc[parts['fit']].context.unique().tolist()),
                    'outer_contexts':sorted(data['meta'].iloc[parts['outer']].context.unique().tolist()),'global_target_exclusion':protocol!='context'})
                for seed in a.seeds:
                    name=f'{protocol}_{variant}_seed{seed}';bgout=root/'background_pretraining'/name
                    bc={'stage':'background','data_dir':str(path),'partition':parts,'background_cache':str(cache),
                        'background':'b2','family':'qwen','model':model,'seed':seed,'device':a.device,
                        'pretrain_steps':steps,'pretrain_batch':parent['pretrain_batch'],'save_every':a.save_every,'output_dir':str(bgout)}
                    bp=root/'configs'/('background_'+name+'.json');freeze(bp,bc);bgjobs[name]=job(bp,bc,a.resume)
                    cfg={k:plan[k] for k in ('rank','target_dim','background_dim','ridge_alphas','representations','predictors','max_steps','batch_size','save_every','device')}
                    cfg.update(data_dir=str(path),knowledge=str(kpath),partition=parts,protocol=protocol,variant=variant,
                               seed=seed,model=model,encoder=str(bgout/'encoder.pt'),output_dir=str(root/'benchmarks'/name))
                    cp=root/'configs'/(name+'.json');freeze(cp,cfg);specs[name]=cfg
                    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-config',str(cp)]
                    if a.resume:cmd.append('--resume')
                    jobs[name]={'cmd':cmd,'resource':'gpu','complete':str(Path(cfg['output_dir'])/'COMPLETE.json')}
        write_json(root/'background_jobs.json',bgjobs);write_json(root/'jobs.json',jobs)
        write_json(root/'budget.json',{'background_jobs':len(bgjobs),'background_steps_each':steps,
            'benchmark_jobs':len(jobs),'heads_each':len(a.representations)*len(a.predictors),'mlp_steps_each':a.max_steps,
            'mlp_batch_size':a.batch_size,'selection':'Ridge alpha from calibration; MLP fixed endpoint, no early stopping',
            'threads_per_worker':2,'concurrent_workers':a.parallel_jobs})
        print(f'ROUND12 READY: background jobs={len(bgjobs)}, benchmark jobs={len(jobs)}',flush=True)
        if a.prepare_only:stage('prepared; training pending');return root
        for label,queue in [('B2 pretraining',bgjobs),('target encoders',jobs)]:
            stage(label);result=run_jobs(queue,root,a.parallel_jobs,False,label='ROUND12')
            if any(result.values()):raise RuntimeError(f'{label} failed; inspect worker logs and use --resume')
        stage('paired reports');report(specs,root)
        (root/'INCOMPLETE.json').unlink(missing_ok=True);stage('complete')
        save_complete(root,{'test_evaluated':False,'benchmark_jobs':len(jobs),
                           'note':'Development only; no new external test, contrastive loss or confidence router trained'})
        print(f'ROUND12 COMPLETE: {root/"comparison_mean.csv"}',flush=True)
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error':str(exc),'test_evaluated':False});raise
    finally:
        package(root,'round12');print(f'REVIEW PACKAGE: {root/"round12_review_light.tar.gz"}',flush=True)
    return root


if __name__=='__main__':main()
