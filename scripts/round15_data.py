"""Pinned Mixscale acquisition and metadata-only QC, with prospective HT29 reservation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import requests

from vcell.round11 import freeze
from vcell.utils import file_sha256, write_json

REPO=Path(__file__).resolve().parents[1]


def digest(path,algorithm='md5'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def download(record,root):
    path=Path(root)/record['name'];part=path.with_suffix(path.suffix+'.part')
    if path.exists():
        if path.stat().st_size!=record['bytes'] or digest(path)!=record['md5']:raise ValueError('Existing file checksum mismatch')
        return path
    for attempt in range(5):
        offset=part.stat().st_size if part.exists() else 0
        if offset==record['bytes']:
            if digest(part)!=record['md5']:raise ValueError('Completed partial checksum mismatch; preserve file for inspection')
            part.replace(path);return path
        if offset>record['bytes']:raise ValueError('Oversized partial download')
        headers={'Accept-Encoding':'identity'}
        if offset:headers['Range']=f'bytes={offset}-'
        try:
            with requests.get(record['url'],headers=headers,stream=True,timeout=(20,120)) as r:
                r.raise_for_status()
                if r.status_code==206:
                    expected=f'bytes {offset}-'
                    if not r.headers.get('Content-Range','').startswith(expected):raise ValueError('Incorrect resume range')
                else:offset=0
                with part.open('ab' if offset else 'wb') as f:
                    last=offset
                    for block in r.iter_content(4*1024**2):
                        if block:f.write(block)
                        if f.tell()>record['bytes']:raise ValueError('Download exceeds pinned size')
                        if f.tell()-last>=128*1024**2:
                            print(f'MIXSCALE DOWNLOAD {f.tell()}/{record["bytes"]} bytes',flush=True);last=f.tell()
        except (requests.RequestException,OSError) as exc:
            if attempt==4:raise
            print(f'MIXSCALE RETRY {attempt+1}: {type(exc).__name__}',flush=True);time.sleep(min(2**attempt,15))
    if part.stat().st_size!=record['bytes'] or digest(part)!=record['md5']:raise ValueError('Downloaded checksum mismatch')
    part.replace(path);return path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--download',action='store_true')
    p.add_argument('--file');p.add_argument('--rscript');a=p.parse_args(argv)
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((REPO/'configs/round15_mixscale.json').read_text())
    freeze(root/'test_v2_reservation.json',{'cell_line':'HT29','all_stimuli':True,'all_sources':True,
        'status':'reserved candidate before response analysis; metadata/QC only',
        'excluded_from':['training','background_encoder','basis','normalization','retrieval','selection','adaptation'],
        'source_record':manifest['record'],'automatic_substitution':False})
    name=a.file or manifest['default_file'];record=next((r for r in manifest['files'] if r['name']==name),None)
    if record is None:raise ValueError('File not in pinned manifest')
    freeze(root/'download_manifest.json',record)
    status={'file':name,'reserved_cell_line':'HT29','training_ready':False,'response_analysis':False}
    try:
        path=root/name
        if not path.exists() and not a.download:
            status.update(state='PLANNED',reason='Run with --download to acquire the pinned processed object');return
        remaining=max(record['bytes']-(path.with_suffix(path.suffix+'.part').stat().st_size if path.with_suffix(path.suffix+'.part').exists() else 0),0)
        if not path.exists() and shutil.disk_usage(root).free<remaining+10*1024**3:raise ValueError('Insufficient reserve for pinned download; no files deleted')
        path=download(record,root);status.update(download_verified=True,md5=digest(path),sha256=file_sha256(path))
        rscript=a.rscript or shutil.which('Rscript')
        if not rscript or not Path(rscript).exists():
            status.update(state='BLOCKED_R_RUNTIME',reason='Rscript with SeuratObject, Matrix and jsonlite is required for sparse Seurat inspection');return
        if Path('/proc/meminfo').exists():
            mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split()[1].isdigit()}
            if mem.get('MemAvailable',0)<6*record['bytes']+8*1024**3:
                status.update(state='BLOCKED_MEMORY',reason='RDS inspection needs a conservative memory reserve; actual peak may be larger');return
        result=subprocess.run([str(rscript),str(REPO/'scripts/inspect_round15_rds.R'),str(path),str(root)],check=False)
        if result.returncode:raise RuntimeError('RDS inspector failed; see data worker log')
        status.update(state='METADATA_INSPECTED',reason='Field mapping and matched-control validation remain required before training')
    except BaseException as exc:
        status.update(state='FAILED',reason=str(exc));raise
    finally:
        write_json(root/'INSPECT_STATUS.json',status)
        print('MIXSCALE STATUS: '+json.dumps(status),flush=True)


if __name__=='__main__':main()
