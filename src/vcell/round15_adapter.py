"""Mixscale IFNG metadata mapping and matched-control training-data handoff.

No response-derived score, DE list, or HT29 expression is used by this adapter.
The cell-level manifest is deliberately separate from the lightweight report.
"""
import hashlib
import html
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

from .data import save_prepared
from .round12 import make_partition
from .round15_gene_mapping import resolve_panel
from .utils import file_sha256, write_json

LINES = ('A549', 'BXPC3', 'HAP1', 'K562', 'MCF7', 'HT29')
FIELDS = ('cell_id', 'cell_type', 'sample', 'pathway', 'Batch_info', 'orig.ident',
          'sample_ID', 'gene', 'guide')
MATCH = ['cell_type', 'pathway', 'Batch_info', 'orig.ident', 'sample_ID']
POLICY = dict(min_controls=10, min_guide_cells=5, min_target_cells=20, min_guides=2)


def key(values):
    return json.dumps(list(map(str, values)), separators=(',', ':'))


def metadata_plan(metadata, source_genes, panel, output, annotations=None):
    """Validate every mapping against actual rows, including metadata-only HT29."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    missing = set(FIELDS) - set(metadata)
    if missing:
        raise ValueError('Missing metadata fields: ' + str(sorted(missing)))
    m = metadata[list(FIELDS)].copy()
    if m.isna().any().any() or m.astype(str).apply(lambda s: s.str.strip().eq('')).any().any():
        raise ValueError('Empty required metadata field')
    m = m.astype(str)
    if m.cell_id.duplicated().any():
        raise ValueError('Duplicate cell IDs')
    if not set(m.cell_type) <= set(LINES) or set(m.pathway) != {'IFNG'}:
        raise ValueError('Unexpected cell line or stimulus; explicit new adapter required')
    if not set(m.Batch_info) <= {'Rep1', 'Rep2'}:
        raise ValueError('Unexpected replicate label')
    expected = m.cell_type + '_IFNG'
    alias = m['sample'].replace({'BXCP3_IFNG': 'BXPC3_IFNG'})
    if not alias.equals(expected):
        raise ValueError('sample/cell_type mismatch after explicit BXCP3 alias mapping')
    target = m.guide.str.extract(r'^(.+)g([0-9]+)$')[0]
    if target.isna().any() or not target.equals(m.gene):
        raise ValueError('guide/gene mapping mismatch')
    source_genes, panel = list(source_genes), list(panel)
    source_set, panel_set = set(source_genes), set(panel)
    if len(set(source_genes)) != len(source_genes) or len(set(panel)) != len(panel):
        raise ValueError('Duplicate gene identifiers')
    if not panel or any(not str(g).strip() for g in source_genes + panel):
        raise ValueError('Empty gene identifiers/panel')
    mapping = resolve_panel(source_genes,panel,annotations)
    mapping.to_csv(output/'gene_mapping.csv',index=False)
    mapping.to_csv(output/'panel_coverage.csv',index=False)
    mapped_symbols=set(mapping.loc[mapping.measured,'source_gene'])
    targets = sorted(set(m.gene) - {'NT'})
    pd.DataFrame({'target': targets, 'measured': [g in source_set for g in targets],
                  'in_output_panel': [g in panel_set or g in mapped_symbols for g in targets]}).to_csv(output/'target_coverage.csv', index=False)
    m.groupby(['sample', 'cell_type', 'Batch_info'], dropna=False).size().rename('n_cells').reset_index().to_csv(output/'sample_mapping.csv', index=False)
    m.groupby(['cell_type', 'pathway', 'Batch_info', 'gene', 'guide']).size().rename('n_cells').reset_index().to_csv(output/'cell_counts.csv', index=False)
    m.groupby(MATCH).agg(n_cells=('cell_id', 'size'), n_targets=('gene', 'nunique')).reset_index().to_csv(output/'technical_strata.csv', index=False)
    missing_panel = mapping.loc[~mapping.measured,'gene'].tolist()
    audit = {'cells_total': len(m), 'reserved_HT29_cells': int(m.cell_type.eq('HT29').sum()),
             'development_cells': int(m.cell_type.ne('HT29').sum()), 'panel_genes': len(panel),
             'missing_panel_genes': missing_panel, 'missing_target_genes': sorted(set(targets)-set(source_genes)),
             'direct_id_matches':int(mapping.status.eq('exact_id').sum()),
             'mapped_output_genes':int(mapping.measured.sum()), 'mapping_status':mapping.status.value_counts().to_dict(),
             'alias_cells': int(m['sample'].eq('BXCP3_IFNG').sum()), 'match_fields': MATCH,
             'excluded_response_fields': ['mixscale_score', 'author DE lists', 'Seurat normalized data'],
             'test_evaluated': False, 'response_analysis_HT29': False}
    write_json(output/'metadata_audit.json', audit)
    # A target absent from counts can remain a perturbation identity. Output genes cannot.
    if missing_panel:
        raise ValueError(f'Missing {len(missing_panel)} frozen output genes after ID mapping; inspect mapping/gene_mapping.csv; no silent zero fill or panel change')
    dev = m.loc[m.cell_type.ne('HT29')].copy()
    if dev.empty or not dev.gene.eq('NT').any():
        raise ValueError('No development cells or NT controls')
    dev['stratum_key'] = [key(v) for v in dev[MATCH].itertuples(index=False, name=None)]
    strata = sorted(dev.stratum_key.unique()); lookup = {v: i+1 for i, v in enumerate(strata)}
    dev['stratum_id'] = dev.stratum_key.map(lookup)
    # A row is ultimately a line/replicate/target aggregate. Guides are retained for QC.
    dev['group_key'] = [key(v) for v in dev[['cell_type','Batch_info','gene','guide']].itertuples(index=False, name=None)]
    groups = sorted(dev.group_key.unique()); lookup = {v: i+1 for i, v in enumerate(groups)}
    dev['group_id'] = dev.group_key.map(lookup)
    dev['is_control'] = dev.gene.eq('NT').astype(int)
    dev.to_csv(output/'cells.csv.gz', index=False)
    dev.drop_duplicates('group_id')[['group_id','cell_type','Batch_info','gene','guide']].sort_values('group_id').to_csv(output/'groups.csv', index=False)
    dev.drop_duplicates('stratum_id')[['stratum_id'] + MATCH].sort_values('stratum_id').to_csv(output/'strata.csv', index=False)
    (output/'panel.txt').write_text('\n'.join(panel)+'\n', encoding='utf-8')
    (output/'source_panel.txt').write_text('\n'.join(mapping.source_gene)+'\n',encoding='utf-8')
    return audit


def prepare_extension(reference, original, cache, dest, policy):
    """Pool eligible guides after cell-weighted, exact-stratum NT subtraction."""
    cache, dest = Path(cache), Path(dest); dest.mkdir(parents=True, exist_ok=True)
    groups = pd.read_csv(cache/'groups.csv')
    qc = pd.read_csv(cache/'group_qc.csv')
    if groups.group_id.duplicated().any() or qc.group_id.duplicated().any():
        raise ValueError('Duplicate group IDs')
    if set(groups.group_id) != set(qc.group_id):
        raise ValueError('QC group alignment mismatch')
    groups = groups.merge(qc, on='group_id', validate='one_to_one').sort_values('group_id').reset_index(drop=True)
    links = pd.read_csv(cache/'matched_strata.csv')
    if (links.duplicated(['group_id','stratum_id']).any() or not set(links.group_id)<=set(groups.group_id)
        or (links.n_controls<policy['min_controls']).any()
        or (links.groupby('stratum_id').n_controls.nunique()>1).any()):
        raise ValueError('Invalid matched-control lineage')
    if set(groups.loc[groups.n_matched>0,'group_id']) != set(links.group_id):
        raise ValueError('Matched-control lineage missing for retained cells')
    arrays = []
    for name in ('mean', 'baseline'):
        frame = pd.read_csv(cache/(name+'.csv.gz'))
        if list(frame.columns) != ['group_id'] + list(reference['genes']):
            raise ValueError('Expression panel order mismatch')
        if not np.array_equal(frame.group_id.to_numpy(), groups.group_id.to_numpy()):
            raise ValueError('Expression group order mismatch')
        x = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite aggregate expression')
        arrays.append(x)
    means, bases = arrays
    if groups.cell_type.eq('HT29').any():
        raise ValueError('Reserved HT29 in expression export')
    if ((groups.n_matched < 0) | (groups.n_matched > groups.n_positive) | (groups.n_positive > groups.n_input)).any():
        raise ValueError('Invalid group count accounting')
    groups['guide_eligible'] = groups.gene.ne('NT') & groups.n_matched.ge(policy['min_guide_cells'])
    groups.to_csv(dest/'guide_qc.csv', index=False)
    rows, baselines, deltas, summary = [], [], [], []
    for (line, rep, gene), frame in groups.loc[groups.gene.ne('NT')].groupby(['cell_type','Batch_info','gene'], sort=True):
        good = frame.loc[frame.guide_eligible]
        n = int(good.n_matched.sum())
        controls = links.loc[links.group_id.isin(good.group_id)].drop_duplicates('stratum_id')
        nc = int(controls.n_controls.sum())
        eligible = len(good) >= policy['min_guides'] and n >= policy['min_target_cells']
        summary.append({'cell_type': line, 'replicate': rep, 'target': gene, 'n_input': int(frame.n_input.sum()),
                        'n_positive': int(frame.n_positive.sum()), 'n_matched': int(frame.n_matched.sum()),
                        'n_retained': n if eligible else 0, 'n_controls':nc, 'eligible_guides': len(good), 'eligible': eligible,
                        'reason': 'eligible' if eligible else 'insufficient matched cells or distinct guides'})
        if not eligible:
            continue
        w = good.n_matched.to_numpy(dtype=float)/n
        base = w @ bases[good.index]; effect = w @ means[good.index] - base
        baselines.append(base); deltas.append(effect)
        rows.append({'row_id': 'mixscale_ifng:' + key([line,rep,gene]), 'dataset': 'GSE225775_IFNG',
                     'context': line+'_IFNG', 'cell_line': line, 'batch': rep, 'replicate': rep,
                     'perturbation': gene, 'split': 'train', 'n_cells': n, 'n_controls':nc, 'n_guides': len(good),
                     'stimulus': 'IFNG', 'time': '24h', 'dose': 'not_encoded_source_specific', 'modality': 'CRISPRi'})
    pd.DataFrame(summary).to_csv(dest/'target_qc.csv', index=False)
    if not rows:
        raise ValueError('No eligible perturbation units; inspect exact control strata, do not silently pool')
    # Historical rows are never relabeled or recomputed, and legacy test rows stay unscored.
    extra = pd.DataFrame(rows)
    extra.groupby(['context','replicate']).agg(n_rows=('row_id','size'), n_targets=('perturbation','nunique'),
                                              n_cells=('n_cells','sum')).reset_index().to_csv(dest/'context_summary.csv',index=False)
    pd.DataFrame({'target':sorted(set(extra.perturbation)),
                  'historical_target':[p in set(reference['perturbations']) for p in sorted(set(extra.perturbation))]}).to_csv(dest/'target_overlap.csv',index=False)
    if set(extra.row_id) & set(reference['meta'].row_id):
        raise ValueError('Duplicate appended row IDs')
    if reference['meta'].context.astype(str).str.upper().str.contains('HT29', regex=False).any():
        raise ValueError('Reference data contains reserved HT29')
    meta = pd.concat([reference['meta'], extra], ignore_index=True)
    vocab = sorted(set(meta.perturbation)); loc = {p:i for i,p in enumerate(vocab)}
    info = {k:reference['audit'][k] for k in ('normalization','target_sum')}
    info.update(historical_fingerprint=reference['audit']['fingerprint'], historical_rows=len(reference['meta']),
                appended_rows=len(extra), appended_cells=int(extra.n_cells.sum()), policy=policy,
                context_definition='cell line + IFNG; study and experimental replicate remain metadata',
                controls='Exact line/pathway/replicate/orig.ident/sample_ID; each target cell receives its stratum NT mean',
                aggregation='mean of per-cell log1p(10000*counts/full_library); eligible guides pooled by cell count',
                cell_qc='author-processed object; finite nonnegative integer counts, positive library; no new mitochondrial/doublet filter',
                precision_weighting=False, test_evaluated=False, HT29_expression_exported=False)
    save_prepared(dest, np.concatenate([reference['baseline'], np.asarray(baselines)]),
                  np.concatenate([reference['delta'], np.asarray(deltas)]),
                  np.array([loc[p] for p in meta.perturbation]), reference['genes'], vocab, meta, info)
    from .data import load_prepared
    combined = load_prepared(dest)
    for field in ('baseline','delta'):
        if not np.array_equal(combined[field][:len(reference['meta'])], reference[field]):
            raise ValueError('Historical expression changed')
    partitions = {}
    for protocol in ('context','target','double'):
        before = make_partition(original, reference, protocol)
        after = make_partition(original, combined, protocol)
        for role in ('calibration','outer'):
            if reference['meta'].iloc[before[role]].row_id.tolist() != combined['meta'].iloc[after[role]].row_id.tolist():
                raise ValueError('Historical evaluation membership changed')
        partitions[protocol] = after
        folder = dest/'partitions'/protocol; folder.mkdir(parents=True, exist_ok=True)
        write_json(folder/'partition.json', after)
        membership = combined['meta'][['row_id','context','perturbation']].copy(); membership['role'] = 'excluded'
        for role, indices in after.items():
            membership.loc[indices, 'role'] = role
        membership.to_csv(folder/'membership.csv', index=False)
    counts = {p:{role:len(indices) for role,indices in part.items()} for p,part in partitions.items()}
    return {'appended_rows':len(extra), 'appended_cells':int(extra.n_cells.sum()),
            'appended_targets':int(extra.perturbation.nunique()), 'contexts':sorted(extra.context.unique()),
            'partition_counts':counts, 'historical_rows_unchanged':True, 'test_evaluated':False}


def report(root, status):
    root = Path(root)
    write_json(root/'ADAPTER_STATUS.json', status)
    sections = ['<h1>Mixscale IFNG adaptation report</h1>', '<pre>'+html.escape(json.dumps(status, indent=2))+'</pre>']
    sections.append('<p>HT29 is reserved across sources/stimuli. No HT29 expression is exported, aggregated, fitted or scored. '
                    'Technical strata and guides are not independent biological replicates. Training has not started.</p>')
    for relative in ('mapping/metadata_audit.json','prepared/data_audit.json','training_handoff.json'):
        p = root/relative
        if p.exists(): sections.append('<h2>'+html.escape(relative)+'</h2><pre>'+html.escape(p.read_text())+'</pre>')
    for relative in ('mapping/sample_mapping.csv','mapping/gene_mapping.csv','mapping/panel_coverage.csv','mapping/target_coverage.csv',
                     'prepared/context_summary.csv','prepared/target_overlap.csv','prepared/target_qc.csv','aggregation/control_qc.csv'):
        p = root/relative
        if p.exists():
            df = pd.read_csv(p)
            sections.append('<h2>'+html.escape(relative)+f' ({len(df)} rows; first 100 shown)</h2>'+df.head(100).to_html(index=False, escape=True))
    (root/'report.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Mixscale adapter</title>'
                                  '<style>body{font:15px system-ui;margin:32px}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #ddd}pre{white-space:pre-wrap}</style>'
                                  '</head><body>'+''.join(sections)+'</body></html>', encoding='utf-8')


def package(root):
    root = Path(root); entries = {}
    with tarfile.open(root/'round15_D1_adapter_review.tar.gz','w:gz') as archive:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.suffix not in ('.json','.csv','.html','.txt'): continue
            # No per-cell manifest, numerical expression arrays, checkpoints, or old matrices.
            name = p.relative_to(root).as_posix()
            if name.endswith('/metadata.csv') or name.endswith('/membership.csv'): continue
            payload = p.read_bytes(); item = tarfile.TarInfo(name); item.size = len(payload)
            archive.addfile(item, io.BytesIO(payload)); entries[name] = hashlib.sha256(payload).hexdigest()
        payload = json.dumps({'files':entries}, indent=2).encode(); item = tarfile.TarInfo('ARCHIVE_MANIFEST.json'); item.size=len(payload)
        archive.addfile(item, io.BytesIO(payload))
