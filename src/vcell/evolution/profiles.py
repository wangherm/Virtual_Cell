"""Dataset-specific metadata contracts; no expression scores or LLM-invented aliases."""
import re
from .identity import canonical


def curated_config(entry,obs,work):
    profile=entry.get('adapter_profile')
    if not profile:return None
    approved={
        'papalexi_guides_v1':('papalexi_thp1','d9e8bfe20b0e7b0919be042e8fcf6d03'),
        'adamson_10x005_single_v1':('adamson_k562','8657391920e7f8b3e6fd52745777002a')}
    if profile not in approved or (entry['id'],entry['md5'])!=approved[profile]:raise ValueError('Curated metadata profile requires its exact approved dataset/checksum')
    cfg={'dataset':entry['id'],'cell_line':entry['cell_line'],'target_namespace':'HGNC_symbol','stimulus':entry.get('stimulus','unknown'),
         'modality':entry['modality'],'time':entry.get('time','unknown'),'annotation_work':str(work),'min_cells':20,'min_controls':20,
         'allow_common_panel':True,'min_common_genes':1000,'target_column':'perturbation','context_column':'cell_line',
         'gene_column':'index','layer':'X','stratum_columns':['cell_line'],'unit_columns':['perturbation'],
         'metadata_profile':profile,'drop_missing_target':True,'exclude_target_labels':{},'target_map':{}}
    if 'perturbation' not in obs or 'cell_line' not in obs:raise ValueError('Curated profile missing required metadata columns')
    observed=set(obs.perturbation.dropna().astype(str));excluded=cfg['exclude_target_labels'];mapping=cfg['target_map']
    for label in observed & {'','nan','None','*'}:excluded[label]='unassigned perturbation label'
    if profile=='papalexi_guides_v1':
        if 'hto' not in obs or 'guide_id' not in obs:raise ValueError('Papalexi requires original HTO strata and guide identities')
        cfg.update(control_values=['control'],stratum_columns=['cell_line','hto'],unit_columns=['guide_id'])
        for label in sorted(observed-{'control'}-set(excluded)):
            match=re.fullmatch(r'(.+)g([1-9][0-9]*)',label)
            if not match or canonical(match[1]) is None:
                parts=re.split(r'[+,;_]',label)
                parsed=[re.fullmatch(r'(.+)g([1-9][0-9]*)',part) for part in parts]
                if len(parts)>1 and all(m and canonical(m[1]) for m in parsed):
                    excluded[label]='multi-guide label outside the single-guide source task';continue
                raise ValueError('Unresolved Papalexi guide-to-gene identity: '+label)
            mapping[label]=canonical(match[1])
    else:
        cfg['control_values']=['3x_neg_ctrl_pMJ144-1','3x_neg_ctrl_pMJ144-2']
        mapping.update({'ATF6_only_pMJ145':'ATF6','PERK_only_pMJ146':'EIF2AK3','IRE1_only_pMJ148':'ERN1'})
        combinations={'ATF6_PERK_pMJ150','ATF6_IRE1_pMJ152','PERK_IRE1_pMJ154','ATF6_PERK_IRE1_pMJ158'}
        for label in observed & combinations:excluded[label]='multi-target construct outside this single-target prediction task'
        for label in sorted(observed-set(mapping)-set(excluded)-set(cfg['control_values'])):
            excluded[label]='outside the predeclared ATF6/PERK/IRE1 single-construct subset; no target identity assumed'
        # This is one accession/library. No independent replicate is invented.
        cfg['stratum_scope']='Within the single GSM2406677/10X005 library; cell_line is constant, not a biological replicate'
    actual=set(obs.cell_line.dropna().astype(str).str.upper().str.replace('-','',regex=False))
    if actual!={entry['cell_line'].upper().replace('-','')}:raise ValueError('Curated profile cell-line identity mismatch')
    cfg['observed_labels']=sorted(observed)
    return cfg
