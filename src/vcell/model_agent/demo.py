"""Synthetic architectural exercise; no real-data or LLM effectiveness claim."""
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .artifacts import seal
from .contracts import NORMALIZATION, Query, digest
from .evidence import fit_profiles
from .runtime import evaluate, execute
from ..utils import write_json


def make_demo(root):
    root=Path(root)
    if root.exists() and any(root.iterdir()):raise ValueError('Choose new demonstration directory')
    root.mkdir(parents=True,exist_ok=True);genes=('G0','G1','G2');records=[];episodes=[]
    for name,multiplier in [('fast_linear',.1),('response_linear',.2)]:
        dest=root/name;dest.mkdir()
        np.savez_compressed(dest/'input.npz',genes=np.array(genes),control_mean=np.ones(3),control_std=np.ones(3))
        np.savez_compressed(dest/'model.npz',coef=np.eye(3)*multiplier,bias=np.zeros(3))
        recipe={'demo_multiplier':multiplier}
        write_json(dest/'expert.json',{'expert_id':name,'kind':'demo_linear','species':['human'],'modalities':['CRISPRi'],
            'normalization':NORMALIZATION,'synthetic':True,'recipe':recipe,'recipe_fingerprint':digest(recipe),
            'provenance':{'fit_contexts':['SYNTHETIC_TRAIN'],'calibration_contexts':[],
                'fit_targets':['DEMO'],'fit_query_ids':[],'calibration_query_ids':[]}})
        seal(dest);records.append({'expert_id':name,'bundle':name,'cost_units':1. if multiplier==.1 else 1.5,'cost_source':'synthetic'})
    write_json(root/'registry.json',{'schema_version':1,'experts':records})
    for i in range(8):
        q=Query(f'oof_{i}','SYNTHETIC_A' if i<4 else 'SYNTHETIC_B','DEMO','none',genes,tuple(np.full(3,.5+i*.1)),source_partition='train')
        write_json(root/(q.query_id+'.json'),asdict(q))
        np.savez_compressed(root/(q.query_id+'_private.npz'),genes=np.array(genes),delta=np.asarray(q.control)*(.1 if i<4 else .2),query_fingerprint=q.fingerprint)
        for record in records:episodes.append({'fold_id':q.context,'bundle':record['bundle'],
            'query':q.query_id+'.json','truth':q.query_id+'_private.npz'})
    write_json(root/'oof_spec.json',{'source_contexts':['SYNTHETIC_A','SYNTHETIC_B'],
        'excluded_contexts':['SYNTHETIC_OUTER'],'episodes':episodes})
    fit_profiles(root/'oof_spec.json',root/'profiles')
    q=Query('demo_query','SYNTHETIC_OUTER','DEMO','none',genes,(1.2,1.1,1.3))
    write_json(root/'query.json',asdict(q))
    np.savez_compressed(root/'private_truth.npz',genes=np.array(genes),delta=np.asarray(q.control)*.2,query_fingerprint=q.fingerprint)
    return root


def run_demo(root,resume=False):
    root=Path(root)
    if not resume:make_demo(root)
    for policy in ('fixed','rule','risk','workflow'):
        execute(root/'registry.json',root/'query.json',root/policy,policy,root/'profiles',budget_units=3.1,resume=resume)
        write_json(root/(policy+'_evaluation.json'),evaluate(root/policy,root/'query.json',root/'private_truth.npz',root/'registry.json'))
    write_json(root/'stage.json',{'stage':'architecture_demo_complete','synthetic':True,'llm_called':False,
        'formal_agent_experiment':False,'test_evaluated':False,'policies':['fixed','rule','risk','workflow']})
    from .workflows import package_review
    package_review(root)
    return root
