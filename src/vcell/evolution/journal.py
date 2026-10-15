"""Append-only research memory and explicit experiment identities."""
import json
import hashlib
from pathlib import Path
import time
from ..clean_background import run_lock
from ..model_agent.contracts import digest
from ..utils import file_sha256


def events(path):
    path=Path(path);records=[];previous='root'
    if not path.exists():return records
    # An incomplete final append is uncommitted; earlier complete entries remain verifiable.
    lines=path.read_bytes().splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):lines=lines[:-1]
    for line in lines:
        record=json.loads(line);body={k:v for k,v in record.items() if k!='sha256'}
        if body['previous']!=previous or digest(body)!=record['sha256']:raise ValueError('Research journal chain changed')
        previous=record['sha256'];records.append(record)
    return records


def append(path,kind,**payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with run_lock(path.with_suffix('.lock')):
        old=events(path)
        raw=path.read_bytes() if path.exists() else b''
        if raw and not raw.endswith(b'\n'):
            head,_,tail=raw.rpartition(b'\n');temporary=path.with_suffix('.repair.tmp');temporary.write_bytes(head+b'\n' if head else b'');temporary.replace(path)
            payload['interrupted_append']={'uncommitted_bytes':len(tail),'sha256':hashlib.sha256(tail).hexdigest()}
        body={'sequence':len(old)+1,'time':time.time(),'previous':old[-1]['sha256'] if old else 'root','kind':kind,**payload}
        record={**body,'sha256':digest(body)}
        with path.open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
    return record


def classify(exc):
    text=str(exc).lower()
    if isinstance(exc,SyntaxError):return 'syntax'
    if 'reproduc' in text or 'nondetermin' in text:return 'reproducibility'
    if any(s in text for s in ('not permitted','permission','operation not','sandbox','landlock','seccomp')):return 'isolation_or_forbidden_operation'
    if any(s in text for s in ('dtype','float','double','mat1','mat2')):return 'dtype_or_linear_algebra'
    if any(s in text for s in ('shape','dimension','broadcast','size mismatch','aligned')):return 'shape'
    if any(s in text for s in ('time','budget','step cap','out of memory','rss cap')):return 'resource_limit'
    if 'nonfinite' in text or 'nan' in text:return 'nonfinite'
    return 'runtime_or_contract'


def fingerprint(repo):
    from ..round22.evaluator import namespace
    repo=Path(repo)
    paths=[*sorted((repo/'src/vcell/evolution').glob('*.py')),*sorted((repo/'experiments/round23_evolve').glob('*')),
        repo/'scripts/round23.py',repo/'scripts/run_shinka_round23.py',repo/'src/vcell/round15_gene_mapping.py',
        repo/'configs/round15_hgnc_previous_symbols.json',repo/'configs/round23_hgnc_identity.json.gz',repo/'src/vcell/round21/signal.py',repo/'src/vcell/data.py',repo/'src/vcell/model_agent/nested.py']
    return digest({'round22_runtime':namespace(repo),'campaign':{p.relative_to(repo).as_posix():file_sha256(p) for p in paths if p.is_file()}})


def feedback(path,limit=8):
    """Only source-side failures and scores; never confirmation/outer feedback."""
    records=[r for r in events(path) if r.get('phase')=='search'][-limit:]
    return [{'kind':r['kind'],'category':r.get('category'),'error':r.get('error','')[-1600:],
        'code_sha256':r.get('code_sha256'),'score':r.get('score'),'hypothesis_status':r.get('hypothesis_status')} for r in records]
