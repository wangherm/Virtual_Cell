"""Train the matched full-eligible-data P4 comparison; no external test opening."""
import argparse
import copy
from collections import deque
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile

import numpy as np
import pandas as pd
import torch

from run_round7 import run_jobs
from run_round15 import hardware
from vcell.background_training import prepare_background, pretrain
from vcell.clean_background import run_lock
from vcell.data import load_prepared
from vcell.round11 import freeze, verify_files
from vcell.round12 import make_partition
from vcell.round15 import seal
from vcell.round15_p4 import METHODS, assert_mixscale_extension, harmonize, prepare_relations, report, worker
from vcell.round15_target_annotation import SNAPSHOT
from vcell.selection import row_weights
from vcell.train import source_fingerprint
from vcell.utils import file_sha256, write_json

REPO=Path(__file__).resolve().parents[1]


def package_p4(root):
    """Small audit package; omit public-cell cache and harmonized expression data."""
    entries={}
    with tarfile.open(root/'round15_p4_review_light.tar.gz','w:gz') as archive:
        for path in sorted(root.rglob('*')):
            relative=path.relative_to(root)
            if (not path.is_file() or {'background_cache','harmonized'} & set(relative.parts)
                    or path.suffix not in ('.json','.csv','.html','.log','.txt')):continue
            if path.suffix=='.log':
                with path.open(encoding='utf-8',errors='replace') as stream:content=''.join(deque(stream,maxlen=80)).encode()
            else:content=path.read_bytes()
            name=relative.as_posix();member=tarfile.TarInfo(name);member.size=len(content)
            archive.addfile(member,io.BytesIO(content));entries[name]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
        content=json.dumps({'files':entries,'note':'Packaged bytes; omitted numerical artifacts remain checksummed on the server.'},indent=2).encode()
        member=tarfile.TarInfo('ARCHIVE_MANIFEST.json');member.size=len(content);archive.addfile(member,io.BytesIO(content))


def matched_parts(original, reference, expanded, adapter, protocols):
    parts={}
    raw=load_prepared(Path(adapter)/'prepared')
    for protocol in protocols:
        before=make_partition(original,reference,protocol)
        # Verify the adapter's saved split before applying any target-ID joins.
        saved=json.loads((Path(adapter)/'prepared/partitions'/protocol/'partition.json').read_text())
        if saved!=make_partition(original,raw,protocol):raise ValueError('Adapter partition mismatch')
        after=make_partition(original,expanded,protocol)
        for role in ('calibration','outer'):
            a=reference['meta'].iloc[before[role]].row_id.to_numpy()
            b=expanded['meta'].iloc[after[role]].row_id.to_numpy()
            if not np.array_equal(a,b):raise ValueError('Historical evaluation rows changed')
            for field in ('baseline','delta'):
                if not np.array_equal(reference[field][before[role]],expanded[field][after[role]]):
                    raise ValueError('Historical evaluation expression changed')
        parts[protocol]={'reference':before,'expanded':after}
    return parts


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-config');p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work')
    p.add_argument('--name',default='round15_p4_01');p.add_argument('--adapter-dir')
    p.add_argument('--round10-run');p.add_argument('--parent-knowledge');p.add_argument('--background-dir')
    p.add_argument('--seeds',nargs='+',type=int,default=[17,29,43])
    p.add_argument('--protocols',nargs='+',choices=['context','target','double'],default=['context','target','double'])
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--max-steps',type=int,default=2000);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--save-every',type=int,default=100);p.add_argument('--pretrain-steps',type=int)
    p.add_argument('--parallel-jobs',type=int,choices=[1,2,3,4],default=3)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--resume',action='store_true');p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args(argv)
    if a.worker_config:
        cfg=json.loads(Path(a.worker_config).read_text())
        return pretrain(cfg,a.resume) if cfg.get('stage')=='background' else worker(cfg,a.resume)
    if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.name):p.error('Simple output name required')
    if min(a.max_steps,a.batch_size,a.save_every)<1 or min(a.seeds)<0 or (a.pretrain_steps is not None and a.pretrain_steps<1):p.error('Invalid training counts')
    for key in ('methods','protocols','seeds'):
        if len(getattr(a,key))!=len(set(getattr(a,key))):p.error('Duplicate '+key)
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    work=Path(a.work_dir).resolve();work.mkdir(parents=True,exist_ok=True)
    with run_lock(work/'runs'/('.'+a.name+'.lock')):return launch(a,work,work/'runs'/a.name)


def launch(a,work,root):
    adapter=Path(a.adapter_dir or work/'prepared/round15_mixscale_ifng_04').resolve()
    verify_files(adapter)
    ready=json.loads((adapter/'training_handoff.json').read_text())
    status=json.loads((adapter/'ADAPTER_STATUS.json').read_text())
    if status.get('state')!='ADAPTER_READY' or not ready.get('training_data_ready') or ready.get('test_evaluated'):
        raise ValueError('Require completed, untested Mixscale adapter')
    reference_path=Path(ready['reference_data']).resolve();expanded_path=Path(ready['data_path']).resolve()
    if expanded_path != adapter/'prepared':raise ValueError('Adapter data path is not its completed prepared output')
    original_path=Path(ready['original_data']).resolve()
    reference=load_prepared(reference_path);raw_expanded=load_prepared(expanded_path);original=load_prepared(original_path)
    assert_mixscale_extension(reference,raw_expanded)
    previous=Path(a.round10_run or work/'runs/round10_01').resolve()
    # Round10's root marker predates checksummed completion manifests. No old
    # model is reused: consume and freeze only its split anchor/config/budget.
    previous_done=json.loads((previous/'COMPLETE.json').read_text())
    if previous_done.get('test_evaluated') is not False or (previous/'INCOMPLETE.json').exists():
        raise ValueError('Completed untested Round10 required')
    parent=json.loads((previous/'plan.json').read_text())
    if parent.get('experiment')!='round10' or parent.get('data')!=original['audit']['fingerprint']:
        raise ValueError('Round10 split anchor changed')
    parent_knowledge=Path(a.parent_knowledge or work/'knowledge/round12').resolve();verify_files(parent_knowledge)
    background=Path(a.background_dir or parent['background_path']).resolve()
    # prepare_background below verifies this legacy cleaner's shard manifests.
    model=copy.deepcopy(parent['model'])
    if not (Path(model['model_id'])/'config.json').is_file():raise ValueError('Local model config unavailable')
    steps=a.pretrain_steps or json.loads((previous/'budget.json').read_text())['background_optimizer_steps']
    plan={'experiment':'round15_p4','source':source_fingerprint(),'launcher':file_sha256(__file__),
        'scheduler':file_sha256(REPO/'scripts/run_round7.py'),'package':file_sha256(REPO/'scripts/run_round15.py'),
        'adapter':str(adapter),'adapter_complete':file_sha256(adapter/'COMPLETE.json'),
        'reference_path':str(reference_path),'reference_fingerprint':reference['audit']['fingerprint'],
        'reference_metadata':file_sha256(reference_path/'metadata.csv'),
        'expanded_fingerprint':raw_expanded['audit']['fingerprint'],
        'original_fingerprint':original['audit']['fingerprint'],
        'original_metadata':file_sha256(original_path/'metadata.csv'),
        'parent_knowledge':str(parent_knowledge),'parent_knowledge_complete':file_sha256(parent_knowledge/'COMPLETE.json'),
        'target_annotation_snapshot':file_sha256(SNAPSHOT),
        'background_path':str(background),'background_complete':file_sha256(background/'COMPLETE.json'),
        'background_report':file_sha256(background/'report.json'),
        'round10_plan':file_sha256(previous/'plan.json'),'round10_budget':file_sha256(previous/'budget.json'),
        'round10_complete':file_sha256(previous/'COMPLETE.json'),
        'model_config':file_sha256(Path(model['model_id'])/'config.json'),'model':model,
        'pretrain_steps':steps,'pretrain_batch':parent['pretrain_batch'],
        **{key:getattr(a,key) for key in ('methods','protocols','seeds','max_steps','batch_size','save_every','device')},
        'variants':['reference','expanded'],'ridge_alphas':[.0001,.001,.01,.1,1.],
        'background_dim':16,'target_dim':32,'weighting':'equal contexts / targets / rows; n_cells is not precision',
        'scope':'Matched historical development P4 data augmentation; not full expanded Round15',
        'test_evaluated':False,'HT29':'all sources/stimuli reserved','Jurkat':'not scored'}
    if root.exists() and any(root.iterdir()) and not a.resume:raise ValueError('Use --resume or new name')
    if not root.exists() and shutil.disk_usage(work).free<15*1024**3:raise ValueError('Need 15 GiB available; no artifacts deleted')
    root.mkdir(parents=True,exist_ok=True);freeze(root/'plan.json',plan)
    if (root/'COMPLETE.json').exists():
        verify_files(root);package_p4(root);print('P4 COMPLETE VERIFIED:',root,flush=True);return root
    def stage(label):
        write_json(root/'stage.json',{'stage':label,'scope':plan['scope'],'test_evaluated':False})
        print('P4 STAGE:',label,flush=True)
    try:
        write_json(root/'hardware.json',hardware(work))
        stage('incremental annotation and canonical target audit')
        knowledge_root=root/'knowledge';aliases=prepare_relations(reference,raw_expanded,parent_knowledge,knowledge_root)
        joined=harmonize(reference,raw_expanded,aliases,root/'harmonized')
        if joined is not None:expanded_path=joined
        expanded=load_prepared(expanded_path);assert_mixscale_extension(reference,expanded,allow_target_aliases=True)
        partitions=matched_parts(original,reference,expanded,adapter,a.protocols)
        freeze(root/'data_identity.json',{'reference':reference['audit']['fingerprint'],'expanded':expanded['audit']['fingerprint'],
            'relations':file_sha256(knowledge_root/'relations.npz'),'aliases':aliases,
            'expanded_metadata':file_sha256(expanded_path/'metadata.csv')})
        stage('background cache and fold-specific queues')
        cache=prepare_background(background,root/'background_cache',reference['genes'].tolist())
        specs={};bgjobs={};jobs={};datasets={'reference':(reference,reference_path),'expanded':(expanded,expanded_path)}
        (root/'configs').mkdir(exist_ok=True);weight_audits=[]
        for protocol in a.protocols:
            for variant,(data,path) in datasets.items():
                parts=partitions[protocol][variant];fit=parts['fit'];meta=data['meta'].iloc[fit]
                weights=row_weights(meta);membership=data['meta'][['row_id','context','perturbation']].copy();membership['role']='excluded'
                sf=root/'splits'/(protocol+'_'+variant);sf.mkdir(parents=True,exist_ok=True);freeze(sf/'partition.json',parts)
                for role,indices in parts.items():membership.loc[indices,'role']=role
                membership.to_csv(sf/'membership.csv',index=False)
                for context in sorted(set(meta.context)):
                    mask=meta.context.eq(context).to_numpy()
                    weight_audits.append({'protocol':protocol,'variant':variant,'context':context,'rows':int(mask.sum()),
                        'target_count':int(meta.loc[mask].perturbation.nunique()),'training_loss_mass':float(weights[mask].sum())})
                for seed in a.seeds:
                    name=f'{protocol}_{variant}_seed{seed}';bgout=root/'background_pretraining'/name
                    bc={'stage':'background','data_dir':str(path),'partition':parts,'background_cache':str(cache),
                        'background':'b2','family':'qwen','model':model,'seed':seed,'device':a.device,
                        'pretrain_steps':steps,'pretrain_batch':parent['pretrain_batch'],'save_every':a.save_every,'output_dir':str(bgout)}
                    bp=root/'configs'/('background_'+name+'.json');freeze(bp,bc)
                    cfg={key:plan[key] for key in ('methods','ridge_alphas','max_steps','batch_size','save_every','device')}
                    cfg.update(data_dir=str(path),relations=str(knowledge_root/'relations.npz'),partition=parts,protocol=protocol,
                        variant=variant,seed=seed,model=model,encoder=str(bgout/'encoder.pt'),output_dir=str(root/'models'/name))
                    cp=root/'configs'/(name+'.json');freeze(cp,cfg);specs[name]=cfg
                    for queue,config_path,complete in [(bgjobs,bp,bgout/'COMPLETE.json'),(jobs,cp,Path(cfg['output_dir'])/'COMPLETE.json')]:
                        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-config',str(config_path)]
                        if a.resume:command.append('--resume')
                        queue[name if queue is jobs else 'background_'+name]={'cmd':command,'resource':'gpu','complete':str(complete)}
        pd.DataFrame(weight_audits).to_csv(root/'training_weight_audit.csv',index=False)
        write_json(root/'background_jobs.json',bgjobs);write_json(root/'jobs.json',jobs)
        write_json(root/'budget.json',{'background_jobs':len(bgjobs),'background_steps_each':steps,
            'benchmark_jobs':len(jobs),'methods_each':len(a.methods),'fitted_heads':len(jobs)*len(a.methods),
            'small_head_steps_each':a.max_steps,'parallel_workers':a.parallel_jobs,'cpu_threads_each':2,
            'background_encoder':'Qwen-sized expression projector, not a pretrained Qwen language model',
            'no_new_model_downloads':True,'no_external_test':True})
        print(f'P4 READY: {len(bgjobs)} background fits; {len(jobs)} workers / {len(jobs)*len(a.methods)} heads',flush=True)
        if a.prepare_only:stage('prepared; training pending');return root
        for label,queue in [('B2 background training',bgjobs),('STRING Ridge and factor heads',jobs)]:
            stage(label)
            # Always dispatch resume jobs: workers verify checksums instead of
            # trusting the scheduler's existence-only completion skip.
            result=run_jobs(queue,root,a.parallel_jobs,False,label='P4')
            write_json(root/(('background' if queue is bgjobs else 'benchmark')+'_job_results.json'),result)
            if any(result.values()):raise RuntimeError(label+' failed; inspect worker logs and use --resume')
        stage('paired development reports');report(specs,root)
        (root/'INCOMPLETE.json').unlink(missing_ok=True);stage('complete')
        seal(root,test_evaluated=False,full_round15_complete=False,p4_historical_comparison_complete=True)
        print('P4 COMPLETE:',root/'comparison_mean.csv',flush=True)
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error':str(exc),'test_evaluated':False});raise
    finally:
        package_p4(root)
        print('REVIEW PACKAGE:',root/'round15_p4_review_light.tar.gz',flush=True)
    return root


if __name__=='__main__':main()
