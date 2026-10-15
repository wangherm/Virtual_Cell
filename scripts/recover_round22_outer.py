"""Complete only frozen outer grading in a new artifact; never reopen source search."""
import argparse
import copy
from importlib.metadata import version
import os
from pathlib import Path
import sys
# Match the original server wrapper BEFORE NumPy/BLAS is imported: source pack
# reconstruction must reproduce its projection bytes, not merely similar scores.
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import pandas as pd

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from vcell.clean_background import run_lock
from vcell.model_agent.artifacts import read,seal,verify
from vcell.round11 import freeze
from vcell.round22.data import prepare
from vcell.round22.evaluator import evaluate,inputs,namespace
from vcell.utils import file_sha256,write_json
from run_round22 import package,preflight,stage

# Namespace of the affected published 39019e5 release (Git blob bytes, LF).
APPROVED_PRE_REPAIR_NAMESPACE='6195fbb6866da67d935b13f5b8d7d823d9c8c9cf20505150f3703ef9cc94f290'


def validate_source(source):
    source=Path(source)
    if (source/'COMPLETE.json').exists():raise ValueError('Source run is already complete; do not repair it')
    plan=read(source/'run_plan.json');data=read(source/'data_plan.json')
    if plan['code_sha256']!=APPROVED_PRE_REPAIR_NAMESPACE:raise ValueError('Recovery limited to the affected 39019e5 namespace')
    if data['folds']['transfer']['outer']:raise ValueError('Source has an outer fold; this specific repair does not apply')
    error=read(source/'INCOMPLETE.json')
    if error.get('error')!='No legal folds for transfer/outer':raise ValueError('Requires the specific empty-outer failure')
    search=read(source/'search_frozen.json');confirmation=read(source/'confirmation_frozen.json')
    if search.get('confirmation_used') is not False or search.get('outer_used') is not False or confirmation.get('outer_used') is not False:raise ValueError('Source stage boundaries changed')
    if search['baseline_selection']!=read(source/'baseline_selection.json'):raise ValueError('Source baseline selection changed')
    expected={(r['task'],r['arm'],r['sha256']) for r in search['candidates']}
    actual={(r['task'],r['arm'],r['sha256']) for r in confirmation['records']}
    if expected!=actual or len(actual)!=len(confirmation['records']):raise ValueError('Confirmation candidates do not match search freeze')
    for record in search['all_records']:
        if file_sha256(record['code'])!=record['sha256']:raise ValueError('Frozen candidate code changed')
        if read(Path(record['search_dir'])/'metrics.json')!=record['metrics']:raise ValueError('Frozen source search metrics changed')
    for record in confirmation['records']:
        if file_sha256(record['code'])!=record['sha256']:raise ValueError('Confirmed candidate code changed')
        if read(Path(record['confirmation_dir'])/'metrics.json')!=record['confirmation_metrics']:raise ValueError('Frozen confirmation metrics changed')
    for task,winner in confirmation['winners'].items():
        candidates=[r for r in confirmation['records'] if r['task']==task and r['arm']=='shinka']
        wanted=max(candidates,key=lambda r:r['confirmation_metrics']['combined_score']) if candidates else None
        if winner!=wanted:raise ValueError('Frozen confirmation winner changed')
    for task in ('repeat','transfer'):
        for phase in ('search','confirmation'):
            for fold in data['folds'][task][phase]:inputs(fold)
    for name,value in plan['training_versions'].items():
        if version(name)!=value:raise ValueError('Training package version changed: '+name)
    parent=Path(plan['parent']);verify(parent)
    if file_sha256(parent/'COMPLETE.json')!=plan['parent_sha256']:raise ValueError('Round21 parent changed')
    return plan,data,search,confirmation


def verify_source_rebuild(old,new):
    """No search or confirmation array, transform or unit may change in recovery."""
    for key in ('outer_context','seed','gene_order','confirmation_targets','global_held_targets','source_contexts','source_rows','parent_sha256','raw_controls_verified','data_sha256','relations_sha256'):
        if old[key]!=new[key]:raise ValueError('Source rebuild identity changed: '+key)
    for task in ('repeat','transfer'):
        for phase in ('search','confirmation'):
            before=old['folds'][task][phase];after=new['folds'][task][phase]
            if len(before)!=len(after):raise ValueError('Source fold count changed')
            for left,right in zip(before,after):
                # Paths relocate to a new artifact; all content hashes/row IDs stay equal.
                for key in set(left)|set(right):
                    if key not in ('input','private') and left.get(key)!=right.get(key):raise ValueError('Source fold changed: '+left['name']+'/'+key)
                inputs(right)
    if len(new['folds']['transfer']['outer'])!=1:raise ValueError('Repair requires exactly one held-line outer fold')


def recover(source,root):
    source=Path(source).resolve();root=Path(root).resolve()
    if source==root or source in root.parents or root in source.parents:raise ValueError('Recovery must use a separate sibling run directory')
    plan,old,search,confirmation=validate_source(source)
    names=['run_plan.json','data_plan.json','baseline_selection.json','search_frozen.json','confirmation_frozen.json','evaluation_budget.json','INCOMPLETE.json']
    if (source/'api_ledger.json').exists():names.append('api_ledger.json')
    fingerprints={n:file_sha256(source/n) for n in names}
    root.mkdir(parents=True,exist_ok=True)
    freeze(root/'recovery_plan.json',{'source_run':str(source),'source_files':fingerprints,'current_namespace':namespace(REPO),
        'recovery_script_sha256':file_sha256(__file__),'scope':'Only repair inherited outer query IDs and grade frozen finalists; no API/search/confirmation retraining',
        'test_evaluated':False})
    if (root/'COMPLETE.json').exists():verify(root);print('ROUND22 RECOVERY COMPLETE VERIFIED',root);return root
    try:
        stage(root,'validate unchanged source search and confirmation; rebuild outer queries')
        rebuilt=root/'rebuilt';rebuilt.mkdir(exist_ok=True)
        new=read(rebuilt/'data_plan.json') if (rebuilt/'data_plan.json').exists() else prepare(Path(plan['parent']),rebuilt,plan['outer_context'],plan['seed'])
        verify_source_rebuild(old,new)
        new_plan=copy.deepcopy(plan);new_plan.update(repo=str(REPO),python=sys.executable,code_sha256=namespace(REPO))
        new_plan['budget']['root']=str(root)
        freeze(root/'run_plan.json',new_plan);freeze(root/'data_plan.json',new)
        freeze(root/'baseline_selection.json',search['baseline_selection'])
        freeze(root/'search_frozen.json',search);freeze(root/'confirmation_frozen.json',confirmation)
        # Include source evidence in the light review, with explicitly inherited API usage.
        if (source/'api_ledger.json').exists():freeze(root/'api_ledger.json',read(source/'api_ledger.json'))
        freeze(root/'source_evidence/original_run_plan.json',plan)
        freeze(root/'source_evidence/original_data_plan.json',old)
        code_index={}
        for index,record in enumerate(search['all_records']):
            path=root/'source_evidence/candidates'/(record['sha256']+'.py');path.parent.mkdir(parents=True,exist_ok=True)
            payload=Path(record['code']).read_bytes()
            if path.exists() and path.read_bytes()!=payload:raise ValueError('Recovery candidate evidence changed')
            path.write_bytes(payload);code_index[record['sha256']]=str(path.relative_to(root))
            dest=root/'source_evidence/scores'/f's{index:03d}'
            freeze(dest/'identity.json',{'task':record['task'],'arm':record['arm'],'code_sha256':record['sha256']})
            for report in Path(record['search_dir']).glob('*'):
                if report.is_file() and report.suffix in ('.csv','.json'):freeze(dest/(report.name+'.snapshot.json'),{'sha256':file_sha256(report),'content':report.read_text(encoding='utf-8')})
        freeze(root/'source_evidence/code_index.json',code_index)
        if not (root/'evaluation_budget.json').exists():
            # Keep all original worker charges; this is not a fresh compute budget.
            write_json(root/'evaluation_budget.json',read(source/'evaluation_budget.json'))
        freeze(root/'source_reuse_verified.json',{'all_source_array_hashes_unchanged':True,'search_freeze_sha256':fingerprints['search_frozen.json'],
            'confirmation_freeze_sha256':fingerprints['confirmation_frozen.json'],'API_calls_added':0,'inherited_evaluation_charges':True})
        stage(root,'kernel protection before outer-only recovery')
        preflight(root,new_plan['budget'],sys.executable)
        outer=[]
        for record in confirmation['records']:
            if record['task']!='transfer':continue
            winner=confirmation['winners'].get('transfer')
            if record['arm']=='shinka' and (winner is None or record['sha256']!=winner['sha256']):continue
            stage(root,'private whole-line development evaluation; frozen candidate',arm=record['arm'])
            result=evaluate(Path(record['code']).read_text(),root,'transfer','outer',root/'outer'/record['arm'])
            outer.append({'arm':record['arm'],'metrics':result})
        if not outer:raise ValueError('No frozen transfer finalists to grade')
        rows=[]
        for record in search['all_records']:rows.append({'phase':'search','task':record['task'],'arm':record['arm'],'code_sha256':record['sha256'],**record['metrics']['public']})
        for record in confirmation['records']:rows.append({'phase':'confirmation','task':record['task'],'arm':record['arm'],'code_sha256':record['sha256'],**record['confirmation_metrics']['public']})
        for record in outer:rows.append({'phase':'observed_development_outer','task':'transfer','arm':record['arm'],**record['metrics']['public']})
        pd.DataFrame(rows).to_csv(root/'comparison.csv',index=False)
        for name,sha in fingerprints.items():
            if file_sha256(source/name)!=sha:raise ValueError('Original run changed during recovery: '+name)
        write_json(root/'summary.json',{'search_complete':True,'confirmation_complete':True,'outer_development_evaluated':True,
            'source_run':str(source),'recovery_only':True,'API_calls_added':0,'source_search_and_confirmation_unchanged':True,
            'test_evaluated':False,'HT29_evaluated':False,'tasks_separate':True,'repeat_is_not_zero_shot':True,'baseline_retained_unless_confirmed':True,
            'valid_evolved_candidates':{t:len([r for r in search['all_records'] if r['task']==t and r['arm']=='shinka' and r['generation']>0]) for t in plan['tasks']},
            'confirmation_signal_screen_passed':{t:r['confirmation_metrics']['public']['signal_screen_passed'] if r else False for t,r in confirmation['winners'].items()},
            'claim':'Development evaluation of previously frozen finalists; recovery is not a new independent experiment or evidence of biological success'})
        (root/'INCOMPLETE.json').unlink(missing_ok=True);stage(root,'complete');seal(root,test_evaluated=False);return root
    except BaseException as exc:
        write_json(root/'INCOMPLETE.json',{'error_type':type(exc).__name__,'error':str(exc),'test_evaluated':False});raise
    finally:print('REVIEW PACKAGE:',package(root),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--source-name',default='round22_evolve_03');p.add_argument('--name',default='round22_evolve_03_outerfix');a=p.parse_args()
    runs=Path(a.work_dir)/'runs'
    # Hold both run locks so source fingerprints cannot race a normal resume.
    with run_lock(runs/('.'+a.source_name+'.lock')):
        with run_lock(runs/('.'+a.name+'.lock')):recover(runs/a.source_name,runs/a.name)


if __name__=='__main__':main()
