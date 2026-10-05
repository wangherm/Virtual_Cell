"""Create a Mixscale QC report and all eligible development training aggregates."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from vcell.clean_background import run_lock
from vcell.data import load_prepared
from vcell.round11 import freeze, verify_files
from vcell.round15_adapter import POLICY, metadata_plan, prepare_extension, report, package
from vcell.round15_gene_mapping import collect_annotations
from vcell.utils import file_sha256, write_json
from round15_data import digest

REPO = Path(__file__).resolve().parents[1]


def discover_reference(work):
    """Follow the actual completed parent configuration; never guess a gene panel."""
    for path in sorted((work/'runs/round15_01/configs').glob('*.json')):
        cfg = json.loads(path.read_text())
        if 'parent_worker' not in cfg: continue
        for _ in range(3):
            manifest = Path(cfg['parent_worker'])/'manifest.json'
            cfg = json.loads(manifest.read_text())['config']
            if 'data_dir' in cfg: return Path(cfg['data_dir'])
    fallback = work/'runs/round12_01/prepared/expanded'
    if (fallback/'data_audit.json').is_file(): return fallback
    raise ValueError('Cannot identify historical prepared data; provide --reference-data')


def seal(folder, names):
    write_json(folder/'COMPLETE.json', {'files':{n:file_sha256(folder/n) for n in names}})


def check_resources(work, raw_bytes):
    if shutil.disk_usage(work).free < 5*1024**3: raise ValueError('Need 5 GiB available; no artifacts deleted')
    memfile=Path('/proc/meminfo')
    if memfile.exists():
        memory={line.split(':')[0]:int(line.split()[1])*1024 for line in memfile.read_text().splitlines()}
        if memory.get('MemAvailable',0)<6*raw_bytes+8*1024**3:
            raise ValueError('Insufficient memory reserve for deserializing RDS')


def launch(a):
    work = Path(a.work_dir).resolve(); source = Path(a.input_dir or work/'raw/round15_D1/IFNG').resolve()
    root = work/'prepared'/a.name; root.mkdir(parents=True,exist_ok=True)
    state = {'state':'STARTING', 'training_data_ready':False, 'full_training_started':False, 'test_evaluated':False}
    try:
        reference_path = Path(a.reference_data).resolve() if a.reference_data else discover_reference(work)
        oldpath = Path(a.original_data).resolve() if a.original_data else Path(json.loads((work/'runs/round10_01/plan.json').read_text())['data_path'])
        reference = load_prepared(reference_path); original = load_prepared(oldpath)
        if not np.array_equal(reference['genes'], original['genes']): raise ValueError('Historical panels differ')
        if reference['audit'].get('normalization')!='mean(log1p(10000*counts/full_library))' or reference['audit'].get('target_sum')!=10000:
            raise ValueError('Unsupported historical normalization')
        record = next(x for x in json.loads((REPO/'configs/round15_mixscale.json').read_text())['files'] if x['name']=='Seurat_object_IFNG_Perturb_seq.rds')
        raw = source/record['name']; metadata = source/'metadata.csv.gz'; genes = source/'RNA_counts_genes.txt'
        status = json.loads((source/'INSPECT_STATUS.json').read_text())
        if status.get('state')!='METADATA_INSPECTED': raise ValueError('Complete the metadata inspection first')
        print('MIXSCALE ADAPT: checking local RDS checksum (no download)', flush=True)
        if raw.stat().st_size!=record['bytes'] or digest(raw)!=record['md5']: raise ValueError('Pinned RDS checksum mismatch')
        if file_sha256(raw)!=status['sha256']: raise ValueError('RDS differs from inspected source')
        policy = {k:getattr(a,k) for k in POLICY}
        source_genes=genes.read_text().splitlines()
        annotations=(collect_annotations(work,reference['genes'],a.gene_map)
                     if a.gene_map or not set(reference['genes'])<=set(source_genes)
                     else {'records':[],'sources':[],'warnings':[]})
        identity = {'source_sha256':status['sha256'], 'metadata_sha256':file_sha256(metadata), 'genes_sha256':file_sha256(genes),
                    'reference':reference['audit']['fingerprint'], 'reference_metadata':file_sha256(reference_path/'metadata.csv'),
                    'original':original['audit']['fingerprint'], 'original_metadata':file_sha256(oldpath/'metadata.csv'),
                    'policy':policy, 'chunk_cells':a.chunk_cells,
                    'gene_annotations':annotations,
                    'code':{n:file_sha256(REPO/n) for n in ('scripts/adapt_round15_mixscale.py','scripts/aggregate_round15_mixscale.R','src/vcell/round15_adapter.py','src/vcell/round15_gene_mapping.py')},
                    'HT29':'reserved across all sources and stimuli; no expression export or evaluation'}
        freeze(root/'plan.json', identity)
        if (root/'COMPLETE.json').exists():
            verify_files(root); state=json.loads((root/'ADAPTER_STATUS.json').read_text())
            print('MIXSCALE ADAPTER VERIFIED:',root,flush=True); return root
        write_json(root/'test_v2_reservation.json', {'cell_line':'HT29','all_sources':True,'all_stimuli':True,'test_evaluated':False})
        print('MIXSCALE ADAPT: metadata mapping and frozen panel coverage',flush=True)
        mapping=root/'mapping'
        write_json(root/'gene_annotation_sources.json',annotations)
        metadata_audit=metadata_plan(pd.read_csv(metadata,dtype=str,keep_default_na=False),source_genes,reference['genes'],mapping,annotations)
        print(f'MIXSCALE PANEL VERIFIED: {metadata_audit["mapped_output_genes"]}/{metadata_audit["panel_genes"]}; original IDs/order preserved',flush=True)
        aggregation=root/'aggregation'; aggregation.mkdir(exist_ok=True)
        if (aggregation/'COMPLETE.json').exists():
            verify_files(aggregation)
        else:
            rscript = a.rscript
            if not rscript:
                setup=json.loads((source/'R_SETUP.json').read_text()); rscript=str(Path(setup['prefix'])/'bin/Rscript')
            if not Path(rscript).is_file(): raise ValueError('Validated Rscript missing; provide --rscript')
            check_resources(work, record['bytes'])
            state.update(state='AGGREGATING'); report(root,state)
            subprocess.run([str(rscript),str(REPO/'scripts/aggregate_round15_mixscale.R'),str(raw),str(mapping),str(aggregation),str(a.min_controls),str(a.chunk_cells)],check=True)
            seal(aggregation, ['mean.csv.gz','baseline.csv.gz','group_qc.csv','control_qc.csv','matched_strata.csv','aggregation.json'])
        # The same metadata mapping controls R and Python; immutable identity protects resume.
        shutil.copyfile(mapping/'groups.csv',aggregation/'groups.csv')
        state.update(state='PREPARING'); report(root,state)
        result=prepare_extension(reference,original,aggregation,root/'prepared',policy)
        ready={'training_data_ready':True,'data_path':str(root/'prepared'), 'reference_data':str(reference_path),
               'original_data':str(oldpath), 'partition_root':str(root/'prepared/partitions'),
               'comparison':'original vs appended data, identical historical outer/calibration rows',
               'next_required':['Rebuild annotation-only knowledge for the expanded target vocabulary',
                                'Fit background encoders, feature transforms and response bases on each saved fit partition',
                                'Run matched reference/expanded training; no HT29 or legacy test scoring'],
               'legacy_launcher_compatible':False,
               'note':'Round12 launcher intentionally only accepts K562/RPE1 expansion. Use these artifacts in a new P4 launcher; do not bypass its guard.',
               'full_training_started':False,'test_evaluated':False}
        write_json(root/'training_handoff.json',ready)
        state.update(state='ADAPTER_READY',training_data_ready=True,**result)
        report(root,state)
        names=[p.relative_to(root).as_posix() for p in sorted(root.rglob('*')) if p.is_file() and p.name!='COMPLETE.json' and not p.name.endswith('.tar.gz')]
        seal(root,names)
        print('MIXSCALE ADAPTER READY:',root/'prepared',flush=True)
        print('APPENDED:',json.dumps(result),flush=True)
    except BaseException as exc:
        # A failed resume must not overwrite an immutable successful run's report.
        if not (root/'COMPLETE.json').exists():
            state.update(state='BLOCKED',reason=str(exc));report(root,state)
        raise
    finally:
        package(root);print('REVIEW PACKAGE:',root/'round15_D1_adapter_review.tar.gz',flush=True)
    return root


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',default='/root/autodl-tmp/vcell-work');p.add_argument('--name',default='round15_mixscale_ifng_01')
    p.add_argument('--input-dir');p.add_argument('--reference-data');p.add_argument('--original-data');p.add_argument('--rscript')
    p.add_argument('--gene-map',help='Optional explicit CSV with panel_gene and source_gene; otherwise use local training annotations')
    p.add_argument('--chunk-cells',type=int,default=2048)
    for name,value in POLICY.items(): p.add_argument('--'+name.replace('_','-'),type=int,default=value)
    a=p.parse_args(argv)
    if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.name):p.error('Use a simple output name')
    if min([a.chunk_cells]+[getattr(a,k) for k in POLICY])<1:p.error('Counts must be positive')
    with run_lock(Path(a.work_dir)/'prepared'/('.'+a.name+'.lock')):return launch(a)


if __name__=='__main__': main()
