"""Read-only import of a quota-failed pilot's paid decision/code into a new run."""
import hashlib
from pathlib import Path
import sqlite3
import shutil
from ..model_agent.artifacts import read
from ..model_agent.contracts import digest
from ..round11 import freeze
from ..round25.runtime import charged
from ..utils import file_sha256,write_json
from .critic import parse_action


def recover_pilot(root,source,cfg,shared):
    root=Path(root).resolve();source=Path(source).resolve()
    if root==source or root.is_relative_to(source) or source.is_relative_to(root):raise ValueError('Recovery needs a separate new run directory')
    if any((source/'arms').glob('*/api_ledger.json')):raise ValueError('Recovery only supports a pilot stopped before matched search')
    old=read(source/'campaign_plan.json');current=read(root/'campaign_plan.json')
    if old['parent_sha256']!=current['parent_sha256']:raise ValueError('Recovery Round25 parent changed')
    if old.get('versions')!=current.get('versions'):raise ValueError('Recovery numerical environment changed')
    if {k:v for k,v in old['config'].items() if k!='pilot_hours'}!={k:v for k,v in cfg.items() if k!='pilot_hours'}:
        raise ValueError('Only the dedicated pilot allowance may change during quota recovery')
    pilot=source/'pilot/meta';pool=pilot/'repeat'
    if read(pool/'folds.json')!=shared['folds'] or read(pool/'baseline_selection.json')!=shared['baseline']:
        raise ValueError('Recovery source folds or fixed comparator changed')
    old_plan=read(pool/'run_plan.json')
    for key in ('seed','device','steps','seconds'):
        wanted=cfg['worker_seconds'] if key=='seconds' else cfg[key]
        if old_plan['budget'][key]!=wanted:raise ValueError('Candidate-visible numerical budget changed: '+key)
    done=read(pilot/'turns/000/done.json');plan=read(pilot/'turns/000/plan.json')
    target=done['latest']
    if not target:raise ValueError('Recovery needs the actual previously generated candidate')
    matches=[p for p in (pool/'assessments').glob('*/assessment.json') if read(p).get('module_sha256')==target]
    if len(matches)!=1:raise ValueError('Ambiguous recovery candidate')
    record=read(matches[0])
    if digest({k:v for k,v in record.items() if k!='sha256'})!=record['sha256']:raise ValueError('Failed candidate record changed')
    if record['status']!='invalid' or record.get('error') not in ('Task search compute quota exhausted','Round22 evaluation-time cap exhausted'):
        raise ValueError('Only an evaluation-quota failure is eligible for resource-only recovery')
    module=Path(record['module']).resolve()
    if not module.is_relative_to(pool.resolve()/'assessments') or file_sha256(module)!=target or record['files'].get('module.py')!=target:
        raise ValueError('Previously generated module changed')
    first_dir=pool/'decisions/pilot_initial';initial=read(first_dir/'decision.json');info=read(first_dir/'input.json')
    if initial['attribution']!='critic' or initial['status']!='accepted' or initial['input_sha256']!=file_sha256(first_dir/'input.json') or \
        digest({k:v for k,v in initial.items() if k!='sha256'})!=initial['sha256']:
        raise ValueError('Paid scientific decision failed provenance validation')
    action=parse_action(initial['action'],info['evidence'],info['allowed_actions'],info['required_evidence'],**info['validation'])
    if plan['action']!=initial['action'] or done['attribution']!='critic':raise ValueError('Generated candidate was not linked to this critic action')
    ledger=read(pilot/'api_ledger.json')
    if not 2<=len(ledger)<=cfg['pilot_requests'] or any(r['status']!='complete' or r.get('http_status')!=200 or r.get('finish_reason')!='stop' for r in ledger):
        raise ValueError('Recovery refuses failed or ambiguous provider requests')
    if any(read(p).get('state')=='reserved' for p in (pilot/'provider_responses').glob('*.json')):
        raise ValueError('Unresolved old provider ticket; recovery cannot resend it')
    if (pool/'decisions/pilot_after_result').exists():raise ValueError('A post-result review already exists; this is not the supported quota-stop state')
    database=pool/'evolution/programs.sqlite';database_copy=root/'pilot/recovered_native/programs.sqlite'
    database_copy.parent.mkdir(parents=True,exist_ok=True);database_sources=[database]
    Path(str(database_copy)+'-shm').unlink(missing_ok=True)
    Path(str(database_copy)+'-wal').unlink(missing_ok=True)
    for suffix in ('','-wal'):
        original=Path(str(database)+suffix);copy=Path(str(database_copy)+suffix)
        if original.exists():
            shutil.copyfile(original,copy)
            if file_sha256(original)!=file_sha256(copy):raise ValueError('Native recovery database changed during copy')
            if suffix:database_sources.append(original)
    con=sqlite3.connect(database_copy.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        programs=con.execute('SELECT id,parent_id,island_idx,code,correct FROM programs').fetchall()
    finally:con.close()
    native=[p for p in programs if hashlib.sha256(p[3].encode()).hexdigest()==target]
    if len(native)!=1 or native[0][4]:raise ValueError('Missing native failed-child provenance')
    child=native[0];parent=next((p for p in programs if p[0]==child[1]),None)
    if parent is None:raise ValueError('Native parent link is missing')
    parent_sha=hashlib.sha256(parent[3].encode()).hexdigest()
    if action.parent_code_sha256 and parent_sha!=action.parent_code_sha256:raise ValueError('Native parent differs from the paid decision')
    files=[source/'campaign_plan.json',pool/'run_plan.json',pool/'folds.json',pool/'baseline_selection.json',
        pilot/'turns/000/done.json',pilot/'turns/000/plan.json',matches[0],module,pilot/'api_ledger.json',*database_sources,
        *sorted(first_dir.glob('*.json'))]
    for ledger_root in (source/'baseline_budget',pilot):
        p=ledger_root/'evaluation_budget.json'
        if p.exists():files.append(p)
    snapshot={'source_run':str(source),'files':{p.relative_to(source).as_posix():file_sha256(p) for p in files},
        'module_sha256':target,'action':initial['action'],'prediction_reference_sha256':plan['prediction_reference_sha256'],
        'native_child_id':child[0],'native_parent_id':child[1],'native_parent_sha256':parent_sha,'island':child[2],
        'imported_paid_requests':len(ledger),'prior_worker_seconds':charged(source/'baseline_budget')+charged(pilot),
        'reason':'Resource-only re-evaluation of the same native generated code; no regenerated critic/proposal',
        'source_only':True,'old_run_modified':False}
    freeze(root/'pilot_recovery.json',snapshot)
    newpilot=root/'pilot/meta';ledger_path=newpilot/'api_ledger.json'
    if ledger_path.exists():
        if read(ledger_path)[:len(ledger)]!=ledger:raise ValueError('Imported provider ledger prefix changed')
    else:write_json(ledger_path,ledger)
    new_first=newpilot/'repeat/decisions/pilot_initial';new_first.mkdir(parents=True,exist_ok=True)
    for p in sorted(first_dir.glob('*.json')):
        destination=new_first/p.name
        if destination.exists() and destination.read_bytes()!=p.read_bytes():raise ValueError('Recovered critic audit changed')
        if not destination.exists():destination.write_bytes(p.read_bytes())
    return {'action':action,'module':module.read_text(encoding='utf-8'),'snapshot':snapshot}
