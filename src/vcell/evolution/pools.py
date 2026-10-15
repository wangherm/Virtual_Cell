"""Compile core/expanded source pools with identical sealed grading labels."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..data import load_prepared
from ..model_agent.artifacts import read
from ..round11 import freeze
from ..round22.data import prepare
from ..round22.evaluator import inputs,namespace
from ..utils import file_sha256
from ..model_agent.contracts import cell_line_key
from ..model_agent.contracts import digest
from .catalog import legal_auxiliary,load_source_bundle
from .identity import target_aliases,canonical


def frozen_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.pending.npz')
    np.savez_compressed(temporary,**arrays)
    if path.exists():
        if file_sha256(path)!=file_sha256(temporary):raise ValueError('Compiled input changed; original retained: '+str(path))
        temporary.unlink()
    else:temporary.replace(path)


def compile_pools(parent,root,tasks,arms,budget,python,framework_hash,catalog):
    base=root/'base';base.mkdir(exist_ok=True)
    plan=read(base/'data_plan.json') if (base/'data_plan.json').exists() else prepare(parent,base,budget['outer_context'],budget['seed'])
    lineage=read(Path(parent)/'lineage.json');historical=load_prepared(lineage['data_dir'])
    folder=Path(lineage['data_dir'])
    # Fixtures may supply in-memory source arrays; real prepared datasets always have these files.
    hashes={name:file_sha256(folder/name) for name in ('dataset.npz','metadata.csv','data_audit.json') if (folder/name).exists()}
    freeze(root/'canonical_source_inputs.json',{'path':str(folder),'files':hashes})
    genes=np.array(catalog.get('output_panel',{}).get('genes',historical['genes']),dtype='U')
    if len(set(genes))!=len(genes) or not set(genes)<=set(historical['genes']):raise ValueError('Invalid predeclared output panel')
    lookup={g:i for i,g in enumerate(historical['genes'])};index=np.array([lookup[g] for g in genes])
    blocks=[{'genes':genes,'baseline':historical['baseline'][:,index],'delta':historical['delta'][:,index],
        'unit_id':historical['meta'].row_id.astype(str).to_numpy(dtype='U'),
        'cell_line':historical['meta'].cell_line.fillna('unknown').astype(str).to_numpy(dtype='U') if 'cell_line' in historical['meta'] else np.array([cell_line_key(c) for c in historical['meta'].context],dtype='U'),
        'target':historical['meta'].perturbation.astype(str).to_numpy(dtype='U'),
        **{k:historical['meta'][k].fillna('unknown').astype(str).to_numpy(dtype='U') if k in historical['meta'] else np.repeat('unknown',len(historical['meta'])) for k in ('context','dataset','stimulus','modality','batch','time','split')}}]
    source_catalog=[]
    for entry in catalog.get('sources',[]):
        if entry.get('prepared') and entry.get('role')=='source':
            path=Path(entry['prepared'])
            for filename,key in (('dataset.npz','dataset_sha256'),('metadata.csv','metadata_sha256'),('data_audit.json','audit_sha256')):
                if file_sha256(path/filename)!=entry[key]:raise ValueError('Prepared auxiliary source changed: '+str(path))
            dataset=load_prepared(path);positions={g:i for i,g in enumerate(dataset['genes'])};idx=np.array([positions[g] for g in genes]);md=dataset['meta']
            block={'genes':genes,'baseline':dataset['baseline'][:,idx],'delta':dataset['delta'][:,idx],
                'unit_id':md.row_id.astype(str).to_numpy(dtype='U'),
                'target':md.perturbation.astype(str).to_numpy(dtype='U'),
                'cell_line':md.cell_line.fillna('unknown').astype(str).to_numpy(dtype='U') if 'cell_line' in md else np.array([cell_line_key(c) for c in md.context],dtype='U'),
                **{k:md[k].fillna('unknown').astype(str).to_numpy(dtype='U') if k in md else np.repeat('unknown',len(md)) for k in ('context','dataset','stimulus','modality','batch','time','split')}}
            blocks.append(block);source_catalog.append({'id':entry['id'],'sha256':entry['dataset_sha256'],'rows':len(md)});continue
        if entry.get('bundle') and entry.get('role')=='source':
            if entry.get('target_namespace')!='HGNC_symbol':raise ValueError('External targets require audited canonical HGNC_symbol namespace')
            block=load_source_bundle(entry,genes);block.setdefault('split',np.repeat('train',len(block['target'])))
            block.setdefault('unit_id',np.array([json.dumps([str(block[k][i]) for k in ('dataset','context','target','batch')]) for i in range(len(block['target']))],dtype='U'))
            blocks.append(block);source_catalog.append({'id':entry['id'],'sha256':entry['sha256'],'rows':len(block['target'])})
    # Concatenate public SOURCE candidates; fold-specific filtering happens before export.
    aux={k:np.concatenate([b[k] for b in blocks]) for k in ('baseline','delta','target','context','cell_line','dataset','stimulus','modality','batch','time','split','unit_id')}
    meta=pd.DataFrame({k:v for k,v in aux.items() if k not in ('baseline','delta')}).rename(columns={'target':'perturbation'})
    aliases=target_aliases();known=meta.perturbation.map(lambda s:canonical(s) is not None).to_numpy()
    canonical_values=np.array([canonical(s) or str(s) for s in aux['target']],dtype='U')
    annotation_labels=np.array(sorted(set(map(str,aux['target']))),dtype='U')
    annotation_symbols=np.array([canonical(s) or str(s) for s in annotation_labels],dtype='U')
    # Repeated imported artifacts cannot become repeated evidence.
    identity=meta[['dataset','unit_id']].astype(str).apply(lambda row:json.dumps(row.tolist()),axis=1)
    keep=~identity.duplicated(keep='first').to_numpy();duplicates=int((~keep).sum())
    # Deterministic canonical-first admission, never version choice by held scores.
    aux={k:v[keep] for k,v in aux.items()};meta=meta.loc[keep].reset_index(drop=True)
    known=known[keep];canonical_values=canonical_values[keep]
    result={}
    for task in tasks:
        for arm in arms:
            name=f'{task}_{arm}';pool=root/'pools'/name;pool.mkdir(parents=True,exist_ok=True)
            spec=copy.deepcopy(plan);spec['gene_order']=genes.tolist();spec['output_panel_policy']=catalog.get('output_panel',{'gene_count':len(genes),'policy':'Legacy reference panel'});report=[]
            for phase in ('search','confirmation','outer'):
                for fold in spec['folds'][task][phase]:
                    d,g,m=inputs(fold)
                    d['genes']=genes
                    for key in ('train_y','calibration_y','train_support','calibration_support','query_support','background_reference','train_control','calibration_control','query_control','train_prior','calibration_prior','query_prior'):
                        d[key]=d[key][:,index]
                    for key in ('truth','background','strong_mask','wrong_support','shuffled_support'):g[key]=g[key][:,index]
                    private=pool/'private'/phase/fold['name'];private.mkdir(parents=True,exist_ok=True)
                    frozen_npz(private/'grade.npz',**g)
                    metadata=(Path(fold['private'])/'metadata.csv').read_bytes();meta_path=private/'metadata.csv'
                    if meta_path.exists() and meta_path.read_bytes()!=metadata:raise ValueError('Compiled metadata changed; original retained')
                    if not meta_path.exists():meta_path.write_bytes(metadata)
                    fold.update(private=str(private),grade_sha256=file_sha256(private/'grade.npz'),metadata_sha256=file_sha256(private/'metadata.csv'),genes_sha256=digest(genes.tolist()))
                    selected=legal_auxiliary(meta,plan['outer_context'],m.context.unique(),m.target.unique(),plan['confirmation_targets'],plan['global_held_targets'],task,phase,aliases)
                    selected &= known
                    if arm=='core':selected[:]=False
                    for key in ('baseline','delta','target','context','cell_line','dataset','stimulus','modality','batch','time','unit_id'):
                        d['aux_'+key]=aux[key][selected]
                    d['aux_target']=canonical_values[selected]
                    d['target_alias_labels']=annotation_labels;d['target_alias_symbols']=annotation_symbols
                    path=pool/'inputs'/phase/fold['name']/'input.npz';frozen_npz(path,**d)
                    fold.update(input=str(path),input_sha256=file_sha256(path))
                    report.append({'phase':phase,'fold':fold['name'],'aux_rows':int(selected.sum()),'aux_datasets':sorted(set(aux['dataset'][selected]))})
            freeze(pool/'data_plan.json',spec)
            freeze(pool/'run_plan.json',{'repo':budget['repo'],'code_sha256':namespace(budget['repo']),'framework_sha256':framework_hash,
                'python':python,'task':task,'arm':arm,'budget':budget['worker'],'test_evaluated':False})
            freeze(pool/'auxiliary_audit.json',{'external_sources':source_catalog,'folds':report,
                'unresolved_auxiliary_target_rows_excluded':int((~known).sum()),
                'duplicate_units_excluded_canonical_first':duplicates,
                'scope':'Original canonical prepared source plus audited source bundles; current query whole cell line, global held targets and search-confirmation targets excluded; test/reserved labels never exported'})
            result[name]=pool
    return result
