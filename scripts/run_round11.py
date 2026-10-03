"""Round11: source-only OOF calibration, H1 alpha adaptation, frozen Jurkat release."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from run_round7 import run_jobs
from run_round9 import job, package
from vcell.clean_background import run_lock
from vcell.data import load_prepared
from vcell.perturbation_coverage import audit_available
from vcell.round8 import partitions
from vcell.round11 import (ARMS, SEEDS, SOURCES, read_json, freeze, verify_files, audit_cache,
    context_partition, moments, fit_coefficients, assert_oof_eligible, support_plan,
    score_rows, few_shot, summarize, release_inference, query_target_specificity)
from vcell.specialization import reference_responses
from vcell.train import source_fingerprint
from vcell.selection import row_weights
from vcell.utils import file_sha256, write_json


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work')
    p.add_argument('--round10-run')
    p.add_argument('--name',default='round11_01')
    p.add_argument('--parallel-students',type=int,choices=[1,2],default=2)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--development-only',action='store_true',help='Pause before the frozen Jurkat release; resume without this flag to release')
    args=p.parse_args(argv)
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):p.error('Use a simple run name')
    work=Path(args.work_dir).resolve();root=work/'runs'/args.name
    with run_lock(work/'runs'/('.'+args.name+'.lock')):
        return launch(args,work,root)


def launch(a,work,root):
    previous=Path(a.round10_run or work/'runs/round10_01').resolve()
    old_plan=read_json(previous/'plan.json')
    if old_plan['experiment']!='round10' or not (previous/'COMPLETE.json').exists():raise ValueError('Completed Round10 required')
    if not set(SEEDS)<=set(old_plan['seeds']):raise ValueError('All three paired seeds are required')
    if a.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    data=load_prepared(old_plan['data_path']);torch.set_num_threads(2)
    if (data['audit']['fingerprint']!=old_plan['data'] or set(data['meta'].iloc[data['splits']['train']].context)!=SOURCES or
        set(data['meta'].iloc[data['splits']['val']].context)!={'HepG2'} or set(data['meta'].iloc[data['splits']['test']].context)!={'Jurkat'}):
        raise ValueError('Round11 requires the fixed K562/RPE1/H1 -> HepG2/Jurkat panel')
    if data['audit'].get('normalization')!='mean(log1p(10000*counts/full_library))':raise ValueError('Unexpected target units')
    plan={'experiment':'round11','source':source_fingerprint(),'launcher':file_sha256(__file__),
          'scheduler':file_sha256(Path(__file__).with_name('run_round7.py')),
          'worker_launcher':file_sha256(Path(__file__).with_name('run_round9.py')),
          'round10':str(previous),'round10_plan_sha256':file_sha256(previous/'plan.json'),
          'data':data['audit']['fingerprint'],'metadata_sha256':file_sha256(Path(old_plan['data_path'])/'metadata.csv'),
          'seeds':list(SEEDS),'device':a.device,'new_training':'K562-out B2 only; budgets copied from audited Round10',
          'calibration':'one source-OOF convex pair and one alpha per optimizer seed; penalty=1e-6',
          'fewshot':'H1 hash-fixed 100 query/40 pool, sizes 0/4/8/16, draws 101/202/303',
          'release_candidates':['zero','reference','mlp_b0_none','qwen_b1_none','qwen_b2_none','b2_old_calibration','b2_oof_mix','b2_oof_alpha']}
    if root.exists() and any(root.iterdir()) and not a.resume:raise ValueError('Use --resume for the identical plan or a new name')
    if not root.exists() and shutil.disk_usage(work).free<8*1024**3:raise ValueError('Need 8 GiB free reserve; no old files deleted')
    root.mkdir(parents=True,exist_ok=True);freeze(root/'plan.json',plan)
    if (root/'COMPLETE.json').exists():
        verify_files(root)
        for marker in sorted(root.rglob('COMPLETE.json')):
            if marker.parent!=root:verify_files(marker.parent)
        print(f'ROUND11 COMPLETE VERIFIED: {root}',flush=True);return root
    def stage(name):
        write_json(root/'stage.json',{'stage':name,'test_evaluated':(root/'TEST_OPENED.json').exists()})
        print('ROUND11 STAGE: '+name,flush=True)
    try:
        caches={};stats={}
        (root/'evidence').mkdir(exist_ok=True);(root/'provenance').mkdir(exist_ok=True)
        (root/'configs').mkdir(exist_ok=True)
        def add(protocol,arm,seed,folder,parts,source=None):
            name=f'{protocol}_{arm}_seed{seed}'
            cache=audit_cache(data,folder,parts,source)
            if cache['cfg']['seed']!=seed or cache['cfg']['arm']!=arm:raise ValueError('Candidate identity mismatch')
            cache['identity']['fit_contexts']=sorted(set(data['meta'].iloc[parts['fit']].context))
            cache['identity']['calibration_contexts']=sorted(set(data['meta'].iloc[parts['calibration']].context))
            caches[name]=cache
            freeze(root/'provenance'/(name+'.json'),cache['identity'])
            cache['meta']['checkpoint_sha256']=cache['identity']['checkpoint_sha256']
            frame=moments(cache['meta'],data['delta'][cache['rows']],cache['predictions']['outer'],cache['predictions']['reference'])
            frame.to_csv(root/'evidence'/(name+'.csv'),index=False);stats[name]=frame
            cal=np.asarray(parts['calibration']);ref,seen=reference_responses(data,parts['fit'])
            cm=data['meta'].iloc[cal].copy().reset_index(drop=True)
            cm['data_row']=cal;cm['seen_in_fit']=seen[data['pert_idx'][cal]]
            cm['checkpoint_sha256']=cache['identity']['checkpoint_sha256']
            moments(cm,data['delta'][cal],cache['predictions']['calibration'],ref[data['pert_idx'][cal]]).to_csv(
                root/'evidence'/(name+'_old_calibration.csv'),index=False)
            membership=data['meta'].iloc[sum(parts.values(),[])].copy()
            membership['partition']=np.repeat(list(parts),[len(v) for v in parts.values()])
            membership['metric_row_weight']=np.concatenate([row_weights(data['meta'].iloc[ix]) for ix in parts.values()])
            membership.to_csv(root/'provenance'/(name+'_membership.csv'),index=False)
            print(f'ROUND11 CACHE VERIFIED: {name}',flush=True)
            return cache
        stage('audit existing predictions')
        expected=partitions(data,'context')
        for seed in SEEDS:
            for protocol,held,arms in [('context',None,ARMS),('context_rpe1','RPE1',('qwen_b2_none',)),('context_h1','H1_hESC',('qwen_b2_none',))]:
                parts=expected if held is None else context_partition(data,held)
                for arm in arms:add(protocol,arm,seed,previous/'students'/f'{protocol}_{arm}_seed{seed}',parts,old_plan['source'])
        pd.Series(data['genes']).to_csv(root/'evidence/genes.txt',index=False,header=False)
        write_json(root/'evidence/definitions.json',{
            'target':'perturbed mean(log1p(10000*count/full_library)) minus matched-control mean in the same units',
            'inverse_transform':'normalized response multiplied by fit-only RMS scale; no expm1 or counts reconstruction',
            'unknown_Qwen':'Same gene-name text prompt and tokenizer path as seen targets, no per-target random embedding',
            'unknown_MLP':'Target-specific embedding has no supervised fit examples in that fold; retained as a baseline limitation',
            'unknown_reference':'fit-only equal-context/equal-target weighted global delta mean',
            'moments':'t=E[y^2], q=E[p^2], r=E[ref^2], c=E[p*y], h_ref=E[ref*y], g_cross=E[p*ref]; candidate order [raw,reference,zero]',
            'control_dependence':'Prepared batch/control counts retained. Original control-cell membership is not present; shared pools may induce dependence.',
            'sealed_definition':'Prepared file integrity is checked in full; test labels are not used for fitting, tuning or scoring until release.'})
        h1=stats['context_h1_qwen_b2_none_seed17']
        support=support_plan(h1.perturbation)
        freeze(root/'h1_support_plan.json',support)
        h1[['row_id','context','perturbation','seen_in_fit']].assign(role=lambda f:np.where(f.perturbation.isin(support['query']),'query','support_pool')).to_csv(root/'h1_support_membership.csv',index=False)
        stage('original perturbation coverage audit')
        audit_available(work,data,root/'coverage')
        # Only one missing source fold is fitted. Copy every optimization setting.
        parts=context_partition(data,'K562');bgjobs={};jobs={}
        for seed in SEEDS:
            template=caches[f'context_h1_qwen_b2_none_seed{seed}'];cfg=copy.deepcopy(template['cfg'])
            bg=copy.deepcopy(template['identity']['background']['config'])
            bn=f'context_k562_qwen_b2_seed{seed}';name=f'context_k562_qwen_b2_none_seed{seed}'
            bg.update(partition=parts,device=a.device,output_dir=str(root/'background_pretraining'/bn))
            cfg.update(partition=parts,protocol='context_k562',device=a.device,encoder=str(root/'background_pretraining'/bn/'encoder.pt'),output_dir=str(root/'students'/name))
            for label,configuration,queue in [(bn,bg,bgjobs),(name,cfg,jobs)]:
                path=root/'configs'/(label+'.json');freeze(path,configuration);queue[label]=job(path,configuration,a.resume)
        write_json(root/'background_jobs.json',bgjobs);write_json(root/'jobs.json',jobs)
        if a.prepare_only:return root
        for label,queue in [('K562 background pretraining',bgjobs),('K562 students',jobs)]:
            stage(label)
            results=run_jobs(queue,root,a.parallel_students,False,label='ROUND11')
            if any(results.values()):raise RuntimeError(f'{label} failed; inspect logs and resume')
        for seed in SEEDS:
            add('context_k562','qwen_b2_none',seed,root/'students'/f'context_k562_qwen_b2_none_seed{seed}',parts)
        stage('source OOF calibration and rule freeze')
        coefficients={};training_scores=[]
        for seed in SEEDS:
            names=[f'{p}_qwen_b2_none_seed{seed}' for p in ('context_rpe1','context_h1','context_k562')]
            selected=[caches[n] for n in names];assert_oof_eligible(selected,'HepG2');assert_oof_eligible(selected,'Jurkat')
            frame=pd.concat([stats[n] for n in names],ignore_index=True);frame['row_weight']=row_weights(frame)
            frame.to_csv(root/'evidence'/f'source_oof_seed{seed}.csv',index=False)
            coefficients[str(seed)]={'mix':fit_coefficients(frame),'alpha':fit_coefficients(frame,True),
                                     'source_checkpoints':[c['identity']['checkpoint_sha256'] for c in selected]}
            for method,w in [('raw',[1,0]),('zero',[0,0]),('oof_mix',coefficients[str(seed)]['mix'])]:
                training_scores+=score_rows(frame,*w,protocol='source_oof_calibrator_fit_only',method=method,seed=seed)
        freeze(root/'coefficients.json',coefficients)
        # This manifest is frozen BEFORE development application and test scoring.
        release={'data_fingerprint':data['audit']['fingerprint'],'plan_sha256':file_sha256(root/'plan.json'),
                 'coefficients':coefficients,'coefficients_sha256':file_sha256(root/'coefficients.json'),
                 'checkpoint_sha256':[caches[f'context_{arm}_seed{s}']['identity']['checkpoint_sha256'] for s in SEEDS for arm in ARMS],
                 'source_provenance_sha256':{p.name:file_sha256(p) for p in sorted((root/'provenance').glob('*.json'))},
                 'gene_order':data['genes'].tolist(),'test_row_ids':data['meta'].iloc[data['splits']['test']].row_id.tolist(),
                 'training_contexts':sorted(SOURCES),'candidates':plan['release_candidates'],'seeds':list(SEEDS),
                 'primary_metric':'equal contexts, equal targets within context, equal rows within target MSE',
                 'preprocessing':read_json(root/'evidence/definitions.json'),
                 'old_calibration':{str(s):read_json(Path(caches[f'context_qwen_b2_none_seed{s}']['identity']['folder'])/'calibration.json')['mix'] for s in SEEDS},
                 'limitations':'Previously inspected development contexts. Jurkat held out within this pipeline; external pretraining overlap unverified. No model selection after release.'}
        freeze(root/'release_manifest.json',release)
        summarize(training_scores,root/'source_oof_fit',[])  # Not an outer performance claim.
        stage('HepG2 calibration transfer and H1 few-shot diagnostic')
        development=[];few=[];support_coeff=[];specificity=[]
        for seed in SEEDS:
            for arm in ARMS:
                frame=stats[f'context_{arm}_seed{seed}']
                development+=score_rows(frame,1,0,protocol='HepG2',method=arm,seed=seed)
                if arm=='qwen_b2_none':
                    old=release['old_calibration'][str(seed)]
                    for method,w in [('zero',[0,0]),('reference',[0,1]),('b2_old_calibration',old[:2]),('b2_oof_mix',coefficients[str(seed)]['mix']),('b2_oof_alpha',coefficients[str(seed)]['alpha'])]:
                        development+=score_rows(frame,*w,protocol='HepG2',method=method,seed=seed)
            result,coef=few_shot(stats[f'context_h1_qwen_b2_none_seed{seed}'],support,seed)
            few+=result;support_coeff+=coef
            hc=caches[f'context_h1_qwen_b2_none_seed{seed}']
            specificity+=query_target_specificity(hc['meta'],data['delta'][hc['rows']],hc['predictions']['outer'],support,seed)
        pairs=[('zero','b2_oof_mix'),('qwen_b2_none','b2_oof_mix'),('b2_old_calibration','b2_oof_mix'),('qwen_b2_none','b2_oof_alpha')]
        summarize(development,root/'development',pairs)
        summarize(few,root/'h1_few_shot',[('zero','adapted'),('unchanged','adapted')])
        write_json(root/'h1_few_shot/coefficients.json',support_coeff)
        pd.DataFrame(specificity).to_csv(root/'h1_few_shot/target_specificity.csv',index=False)
        if a.development_only:
            stage('development complete; Jurkat release pending');return root
        stage('frozen Jurkat evaluation')
        freeze(root/'TEST_OPENED.json',{'release_sha256':file_sha256(root/'release_manifest.json'),
                                      'note':'Subsequent changes guided by these scores require a new untouched final evaluation.'})
        final=[];test=data['splits']['test'];(root/'test_predictions').mkdir(exist_ok=True)
        for seed in SEEDS:
            for arm in ARMS:
                name=f'context_{arm}_seed{seed}';cache=caches[name];out=root/'test_predictions'/name;out.mkdir(exist_ok=True)
                if (out/'COMPLETE.json').exists():
                    done=verify_files(out)
                    if done['release_sha256']!=file_sha256(root/'release_manifest.json'):raise ValueError('Test prediction release changed')
                    with np.load(out/'predictions.npz') as f:pred,ref,seen=f['raw'],f['reference'],f['seen']
                else:
                    pred,ref,seen=release_inference(data,cache,test,root/'release_manifest.json',a.device)
                    np.savez_compressed(out/'predictions.npz',raw=pred,reference=ref,seen=seen,
                                        genes=data['genes'],row_ids=data['meta'].iloc[test].row_id.to_numpy(dtype='U'))
                    write_json(out/'COMPLETE.json',{'release_sha256':file_sha256(root/'release_manifest.json'),
                        'files':{'predictions.npz':file_sha256(out/'predictions.npz')}})
                meta=data['meta'].iloc[test].copy().reset_index(drop=True);meta['seen_in_fit']=seen;meta['data_row']=test
                meta['checkpoint_sha256']=cache['identity']['checkpoint_sha256']
                frame=moments(meta,data['delta'][test],pred,ref);frame.to_csv(root/'evidence'/f'jurkat_{arm}_seed{seed}.csv',index=False)
                final+=score_rows(frame,1,0,protocol='Jurkat',method=arm,seed=seed)
                if arm=='qwen_b2_none':
                    old=release['old_calibration'][str(seed)]
                    for method,w in [('zero',[0,0]),('reference',[0,1]),('b2_old_calibration',old[:2]),('b2_oof_mix',coefficients[str(seed)]['mix']),('b2_oof_alpha',coefficients[str(seed)]['alpha'])]:
                        final+=score_rows(frame,*w,protocol='Jurkat',method=method,seed=seed)
                print(f'ROUND11 JURKAT SCORED: {arm} seed={seed}',flush=True)
        summarize(final,root/'jurkat',pairs)
        (root/'INCOMPLETE.json').unlink(missing_ok=True)
        stage('complete')
        files={str(p.relative_to(root)):file_sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.json','.csv','.txt') and p!=root/'COMPLETE.json'}
        write_json(root/'COMPLETE.json',{'test_evaluated':True,'new_students':3,'new_background_jobs':3,'files':files,
            'note':'Fixed three-seed release; no winner selected. OOF fitting scores are not held-out calibrator scores.'})
        print(f'ROUND11 COMPLETE: {root}',flush=True)
        return root
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error':str(exc),'test_evaluated':(root/'TEST_OPENED.json').exists()});raise
    finally:
        package(root,'round11')
        print(f'REVIEW PACKAGE: {root/"round11_review_light.tar.gz"}',flush=True)


if __name__=='__main__':main()
