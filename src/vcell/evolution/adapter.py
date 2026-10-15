"""Explicit h5ad-to-source-signature adapter; ambiguous mappings remain blocked."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from ..utils import write_json,file_sha256
from .identity import canonical,annotations as identity_annotations
from ..model_agent.contracts import digest


def adapt(path,config,genes,dest):
    import anndata as ad
    import pandas as pd
    genes=np.asarray(genes,dtype='U');dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'source_bundle.npz').exists():
        from ..model_agent.artifacts import read
        old=read(dest/'adapter_report.json')
        if old['config']!=config or old.get('reference_panel_sha256')!=digest(genes.tolist()) or old['source_sha256']!=file_sha256(path) or old['bundle_sha256']!=file_sha256(dest/'source_bundle.npz'):raise ValueError('Completed adapter artifact changed; use a new destination')
        return {'id':config['dataset'],'role':'source','bundle':str(dest/'source_bundle.npz'),'sha256':old['bundle_sha256'],'target_namespace':'HGNC_symbol'}
    a=ad.read_h5ad(path,backed='r')
    try:
        required=('target_column','control_values','stratum_columns','context_column','cell_line','stimulus','modality','time','dataset','target_namespace')
        if any(k not in config for k in required) or config['target_namespace']!='HGNC_symbol':raise ValueError('Explicit target/control/condition/context/HGNC mapping required')
        if config['context_column']=='__dataset_context__':a.obs['__dataset_context__']=config['cell_line']
        unit_columns=config.get('unit_columns',[])
        columns=list(dict.fromkeys([config['target_column'],config['context_column'],*config['stratum_columns'],*unit_columns]))
        if any(c not in a.obs for c in columns) or not config['stratum_columns']:raise ValueError('Missing explicit context or independent control stratum columns')
        included=np.ones(a.n_obs,bool);exclusions=[]
        missing_target=a.obs[config['target_column']].isna().to_numpy()
        if config.get('drop_missing_target') and missing_target.any():
            included[missing_target]=False;exclusions.append({'reason':'missing target label','cells':int(missing_target.sum())})
        for label,reason in config.get('exclude_target_labels',{}).items():
            mask=a.obs[config['target_column']].eq(label).fillna(False).to_numpy(bool)
            included[mask]=False;exclusions.append({'label':label,'reason':reason,'cells':int(mask.sum())})
        missing=a.obs.loc[included,columns].isna().sum();missing=missing[missing.gt(0)]
        metadata_audit={'input_cells':a.n_obs,'included_cells':int(included.sum()),'exclusions':exclusions,'missing_required_fields':missing.to_dict(),
            'unit_columns':unit_columns,'control_match_columns':config['stratum_columns'],'control_policy':'Only actual configured non-targeting labels; no missing labels or combination constructs converted to controls'}
        write_json(dest/'metadata_audit.json',metadata_audit)
        if len(missing):raise ValueError('Missing target/context/control stratum metadata: '+repr(missing.to_dict()))
        labels=a.var_names.astype(str).to_numpy() if config.get('gene_column','index')=='index' else a.var[config['gene_column']].astype(str).to_numpy()
        mapping=config.get('gene_map',{});labels=np.array([mapping.get(g,g).split('.')[0] if str(mapping.get(g,g)).startswith('ENSG') else mapping.get(g,g) for g in labels])
        if len(set(labels))!=len(labels):raise ValueError('Ambiguous duplicate canonical genes; explicit aggregation policy required')
        loc={g:i for i,g in enumerate(labels)};missing=[g for g in genes if g not in loc];annotation_sources=[]
        resolved=None
        if missing:
            from ..round15_gene_mapping import collect_annotations,resolve_panel
            annotations=identity_annotations(genes)
            if config.get('annotation_work'):
                old=collect_annotations(Path(config['annotation_work']),genes);annotations['records'].extend(old['records']);annotations['sources'].extend(old['sources'])
            resolved=resolve_panel(labels,genes,annotations);resolved.to_csv(dest/'gene_mapping.csv',index=False)
            missing=resolved.loc[~resolved.measured,'gene'].tolist();annotation_sources=annotations['sources']
        report={'source_sha256':file_sha256(path),'adapter_code_sha256':file_sha256(__file__),'reference_panel_sha256':digest(genes.tolist()),'reference_panel_genes':len(genes),'panel_genes':len(genes),'missing_panel_genes':missing,'config':config,'training_ready':False,'gene_annotation_sources':annotation_sources}
        write_json(dest/'adapter_report.json',report)
        if missing:
            if not config.get('allow_common_panel',False):raise ValueError(f'Missing {len(missing)} frozen expression genes; map IDs or use a separately frozen panel, no silent zero fill')
            kept=np.array([g not in set(missing) for g in genes]);genes=genes[kept]
            if len(genes)<int(config.get('min_common_genes',1000)):raise ValueError('Insufficient measured genes for the predeclared common-panel task')
            if resolved is not None:resolved=resolved.loc[kept]
            report.update(panel_genes=len(genes),output_panel=genes.tolist(),panel_policy='Annotation-only measured intersection before any model search; NOT the legacy full-panel benchmark')
        idx=resolved.source_index.to_numpy(int) if resolved is not None else np.array([loc[g] for g in genes]);targets=a.obs[config['target_column']].astype(str).to_numpy();contexts=a.obs[config['context_column']].astype(str).to_numpy()
        controls=set(map(str,config['control_values']));target_map=config.get('target_map',{})
        unknown=sorted({t for t in targets[included] if t not in controls and canonical(target_map.get(t,t)) is None})
        if unknown:raise ValueError('Unresolved/ambiguous target identity or missing control labels: '+repr(unknown[:15]))
        target=np.array(['__excluded__' if not included[i] else '__control__' if t in controls else canonical(target_map.get(t,t)) for i,t in enumerate(targets)])
        if any(any(x in t for x in ('+',';',',')) for t in set(target)-{'__control__'}):raise ValueError('Compound/guide labels require explicit single-target HGNC mapping')
        strata=a.obs[config['stratum_columns']].astype(str).apply(lambda row:tuple(row),axis=1).tolist()
        units=a.obs[unit_columns].astype(str).apply(lambda row:tuple(row),axis=1).tolist() if unit_columns else [()] * a.n_obs
        matrix=a.X if config.get('layer','X')=='X' else a.layers[config['layer']]
        sums={};counts=defaultdict(int);chunk=int(config.get('chunk_size',1024))
        for start in range(0,a.n_obs,chunk):
            if start%(chunk*10)==0:print('SOURCE ADAPTER:',config['dataset'],'cells',start,'/',a.n_obs,flush=True)
            block=matrix[start:start+chunk,:];values=block.data if hasattr(block,'toarray') else np.asarray(block).ravel()
            if not np.isfinite(values).all() or (values<0).any() or not np.allclose(values,np.round(values),rtol=0.,atol=1e-5):raise ValueError('Selected matrix is not verified raw counts; do not normalize logged expression again')
            library=np.asarray(block.sum(axis=1)).ravel();x=block[:,idx];x=x.toarray() if hasattr(x,'toarray') else np.asarray(x)
            valid=library>0;x=np.log1p(10000*x.astype(np.float64)/np.maximum(library[:,None],1))
            for j in np.flatnonzero(valid):
                i=start+j
                if not included[i]:continue
                # Guides remain separate perturbation units; NT guides share the exact matched-control pool.
                key=(contexts[i],strata[i],target[i],() if target[i]=='__control__' else units[i])
                if key not in sums:sums[key]=np.zeros(len(genes),np.float64)
                sums[key]+=x[j];counts[key]+=1
        rows=[];baseline=[];delta=[];skipped=0
        for (context,stratum,t,unit),total in sorted(sums.items()):
            if t=='__control__':continue
            ctrl=(context,stratum,'__control__',());n=counts[(context,stratum,t,unit)];nc=counts.get(ctrl,0)
            if n<int(config.get('min_cells',20)) or nc<int(config.get('min_controls',20)):skipped+=1;continue
            b=sums[ctrl]/nc;baseline.append(b);delta.append(total/n-b)
            rows.append({'context':context,'cell_line':config['cell_line'],'target':t,'batch':repr(stratum),'dataset':config['dataset'],
                'stimulus':config['stimulus'],'modality':config['modality'],'time':config['time'],'n_cells':n,'n_controls':nc,'unit':repr(unit)})
        if not rows:raise ValueError('No perturbation units with independently matched controls')
        meta=pd.DataFrame(rows);out=dest/'source_bundle.npz'
        np.savez_compressed(out,genes=genes,baseline=np.array(baseline,np.float32),delta=np.array(delta,np.float32),
            unit_id=np.array([json.dumps([row['dataset'],row['context'],row['target'],row['batch'],row['unit']]) for row in rows],dtype='U'),
            **{k:meta[k].to_numpy(dtype='U') for k in ('context','cell_line','target','dataset','batch','stimulus','modality','time')})
        report.update(training_ready=True,rows=len(rows),skipped_units=skipped,normalization='mean(log1p(10000 * raw counts / full RNA library)), exact context/stratum matched controls',bundle_sha256=file_sha256(out))
        meta.to_csv(dest/'units.csv',index=False);report['metadata_audit']=metadata_audit
        write_json(dest/'adapter_report.json',report)
        return {'id':config['dataset'],'role':'source','bundle':str(out),'sha256':file_sha256(out),'target_namespace':'HGNC_symbol'}
    finally:a.file.close()
