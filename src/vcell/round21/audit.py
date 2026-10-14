"""Read-only raw-library and inherited matched-control audit; no response-based filtering."""
from pathlib import Path
import os
import subprocess
import numpy as np
import pandas as pd
from ..model_agent.artifacts import read,seal,verify
from ..model_agent.contracts import digest
from ..round11 import freeze,verify_files
from ..utils import write_json,file_sha256

HAP_TARGETS=('BATF2','IFNGR2','MAFF','PRDM1')


def control_audit(cells,groups,links,library,guide_qc):
    """Exact matched strata; compare actual NT cell memberships between repetitions."""
    cells=cells.copy();cells['stratum_id']=cells.stratum_id.astype(int);cells['group_id']=cells.group_id.astype(int)
    sizes=library.set_index('cell_id').library_size.to_dict();nt=cells.loc[cells.is_control.astype(str).eq('1')].copy()
    nt=nt.loc[nt.cell_id.map(sizes).fillna(0)>0]
    if 'n_controls' in links:
        for stratum,part in links.groupby('stratum_id'):
            expected=set(part.n_controls.astype(int));actual=int(nt.stratum_id.eq(int(stratum)).sum())
            if expected!={actual}:raise ValueError('Raw positive NT membership differs from frozen matched-control count')
    eligible=set(guide_qc.loc[guide_qc.guide_eligible.astype(str).str.lower().isin(['true','1']),'group_id'].astype(int))
    records=[];pools={};grouped=groups.loc[groups.group_id.astype(int).isin(eligible)].groupby(['cell_type','Batch_info','gene'])
    for (line,rep,target),gs in grouped:
        ids=set(gs.group_id.astype(int));strata=set(links.loc[links.group_id.astype(int).isin(ids),'stratum_id'].astype(int))
        controls=nt.loc[nt.stratum_id.isin(strata)];pert=cells.loc[cells.group_id.isin(ids)]
        rep_values=set(controls.Batch_info);pool=set(controls.cell_id);pools[(line,rep,target)]=pool
        measured=library.loc[library.cell_id.isin(pert.cell_id)]
        control_measured=library.loc[library.cell_id.isin(pool)]
        r={'cell_line':line,'replicate':rep,'target':target,'eligible_guides':len(ids),'mapped_perturbation_cells':len(pert),
            'matched_control_cells_positive':len(pool),'control_strata':len(strata),
            'control_repeat_labels':','.join(sorted(rep_values)),'controls_match_repeat':rep_values=={rep},
            'control_membership_sha256':digest(sorted(pool)),
            'perturbation_library_measured_cells':len(measured),
            'perturbation_library_median':float(measured.library_size.median()) if len(measured) else None,
            'control_library_median':float(control_measured.library_size.median()) if len(control_measured) else None}
        records.append(r)
    for r in records:
        opposite='Rep2' if r['replicate']=='Rep1' else 'Rep1';other=pools.get((r['cell_line'],opposite,r['target']))
        r['controls_shared_with_other_repeat']=len(pools[(r['cell_line'],r['replicate'],r['target'])]&other) if other is not None else None
    frame=pd.DataFrame(records)
    paired=frame.controls_shared_with_other_repeat.dropna() if len(frame) else pd.Series(dtype=float)
    independent=bool(len(frame) and len(paired) and frame.controls_match_repeat.all() and paired.eq(0).all())
    return frame,independent


def raw_audit(adapter,raw_dir,dest,repo,rscript=None,skip=False):
    adapter=Path(adapter);raw_dir=Path(raw_dir);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    if skip:
        value={'state':'SKIPPED_EXPLICITLY','independent_repeat_controls_verified':False,'scope':'partial report; raw library audit not performed'}
        freeze(dest/'RAW_STATUS.json',value);return value
    verify_files(adapter);raw=raw_dir/'Seurat_object_IFNG_Perturb_seq.rds';status=read(raw_dir/'INSPECT_STATUS.json')
    if file_sha256(raw)!=status['sha256'] or status['sha256']!=read(adapter/'plan.json')['source_sha256']:raise ValueError('Raw RDS differs from frozen adapter')
    if not rscript:rscript=str(Path(read(raw_dir/'R_SETUP.json')['prefix'])/'bin/Rscript')
    if not Path(rscript).is_file():raise ValueError('Validated Rscript missing; provide --rscript or explicitly --skip-raw-audit')
    identity={'raw_sha256':status['sha256'],'adapter_sha256':file_sha256(adapter/'COMPLETE.json'),
        'script':file_sha256(Path(repo)/'scripts/audit_round21_counts.R'),'code':file_sha256(__file__),'rscript':str(rscript)}
    freeze(dest/'manifest.json',identity)
    if (dest/'COMPLETE.json').exists():verify(dest);return read(dest/'RAW_STATUS.json')
    # One R process only; no dense whole-genome matrix or second full RDS copy.
    mem=Path('/proc/meminfo')
    if mem.exists():
        values={s.split(':')[0]:int(s.split()[1])*1024 for s in mem.read_text().splitlines()}
        if values.get('MemAvailable',0)<6*raw.stat().st_size+8*1024**3:raise ValueError('Insufficient RAM reserve for a read-only RDS audit')
    cells=pd.read_csv(adapter/'mapping/cells.csv.gz',dtype=str,keep_default_na=False)
    selected=cells.loc[cells.is_control.eq('1') | (cells.cell_type.eq('HAP1') & cells.gene.isin(HAP_TARGETS))]
    if selected.cell_type.eq('HT29').any():raise ValueError('Reserved cell selected for raw audit')
    path=dest/'selected_cells.csv';selected.to_csv(path,index=False)
    if not (dest/'raw_measurement.json').exists():
        env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL'))}
        subprocess.run([str(rscript),str(Path(repo)/'scripts/audit_round21_counts.R'),str(raw),str(path),str(dest)],check=True,env=env)
    library=pd.read_csv(dest/'cell_library_qc.csv.gz',keep_default_na=False)
    if set(library.cell_id)!=set(selected.cell_id):raise ValueError('Raw library cell alignment changed')
    guides=pd.read_csv(adapter/'prepared/guide_qc.csv',keep_default_na=False)
    frame,independent=control_audit(cells,pd.read_csv(adapter/'mapping/groups.csv'),pd.read_csv(adapter/'aggregation/matched_strata.csv'),library,guides)
    frame.to_csv(dest/'control_pool_audit.csv',index=False)
    hap=frame.loc[frame.cell_line.eq('HAP1') & frame.target.isin(HAP_TARGETS)].copy()
    target_qc=pd.read_csv(adapter/'prepared/target_qc.csv',keep_default_na=False)
    target_qc=target_qc.rename(columns={'cell_type':'cell_line','target':'target'})
    hap=hap.merge(target_qc,on=['cell_line','replicate','target'],how='left',validate='one_to_one');hap.to_csv(dest/'batch_audit.csv',index=False)
    value={'state':'COMPLETE','independent_repeat_controls_verified':independent,'raw_sha256':status['sha256'],
        'selected_cells':len(selected),'HAP1_targets':list(HAP_TARGETS),'facts':'full-RNA library sizes, matched NT memberships and retained guide/cell counts',
        'hypotheses':['depth differences','control matching or shared pools','guide efficiency or cell-state composition'],
        'unresolved':'No causal attribution or normalization change; guide efficacy/cell-state effects require separate evidence',
        'pseudobulk_generation':'Inherited mean(log1p(10000*counts/full_library)); exact stratum-matched NT baselines; guide means weighted by matched-cell counts',
        'pseudobulk_recomputed':False,
        'HT29_expression_selected':False,'prepared_data_modified':False,'test_evaluated':False}
    write_json(dest/'RAW_STATUS.json',value);seal(dest);return value


def response_audit(data,dest):
    """Call only after search freeze; these four development outcomes never gate search."""
    m=data['meta'];take=m.context.eq('HAP1_IFNG')&m.perturbation.isin(HAP_TARGETS)&m.replicate.isin(['Rep1','Rep2']);rows=np.flatnonzero(take)
    frame=m.iloc[rows][['row_id','context','perturbation','replicate','n_cells','n_controls']].copy()
    frame['delta_norm']=np.linalg.norm(data['delta'][rows],axis=1);frame['mean_delta_squared']=np.square(data['delta'][rows]).mean(1)
    frame.to_csv(Path(dest)/'hap1_response_audit.csv',index=False)
    paired=frame.pivot_table(index='perturbation',columns='replicate',values='mean_delta_squared').dropna()
    ratio=float(paired.Rep1.mean()/paired.Rep2.mean()) if len(paired) and paired.Rep2.mean()>0 else None
    from .signal import pearson,cosine,top
    correlation=[]
    for target in paired.index:
        r={name:data['delta'][np.flatnonzero(take & m.perturbation.eq(target) & m.replicate.eq(name))].mean(0) for name in ('Rep1','Rep2')}
        a,b=r['Rep1'],r['Rep2'];ix=np.intersect1d(top(a,100),top(b,100))
        correlation.append({'target':target,'gene_vector_pearson':pearson(a,b),'gene_vector_cosine':cosine(a,b),
            'technical_mse':float(np.square(a-b).mean()),'top100_overlap':len(ix),
            'top100_sign_agreement':float((a[ix]*b[ix]>0).mean()) if len(ix) else None,
            'rep1_mean_squared':float(np.square(a).mean()),'rep2_mean_squared':float(np.square(b).mean())})
    pd.DataFrame(correlation,columns=['target','gene_vector_pearson','gene_vector_cosine','technical_mse','top100_overlap','top100_sign_agreement','rep1_mean_squared','rep2_mean_squared']).to_csv(Path(dest)/'hap1_gene_vector_audit.csv',index=False)
    write_json(Path(dest)/'hap1_response_summary.json',{'paired_targets':paired.index.tolist(),'rep1_to_rep2_mean_squared_ratio':ratio,
        'facts':'prepared response magnitude imbalance','cause':'not determined','role':'post-freeze development diagnosis, never source gate','test_evaluated':False})
