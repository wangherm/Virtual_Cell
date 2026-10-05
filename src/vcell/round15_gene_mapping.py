"""Offline, annotation-only translation of frozen output IDs to Mixscale row names."""
import hashlib
import json
from pathlib import Path
import re

import h5py
import pandas as pd

from .background_inspect import column
from .utils import file_sha256


def stable_id(value):
    value = str(value)
    return value.split('.')[0] if re.fullmatch(r'ENSG[0-9]+\.[0-9]+', value) else value


def collect_annotations(work, panel, explicit=None):
    """Read only var identifiers from K562/RPE1, plus existing mapped gene cards.

    No expression, cell observations, held-out data, or network calls are needed.
    The exact annotation pairs and provenance are frozen into the adapter plan.
    """
    work = Path(work); records = []; sources = []; warnings = []
    needed = {stable_id(g) for g in panel}
    if explicit:
        path = Path(explicit).resolve()
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if not {'panel_gene','source_gene'} <= set(frame):
            raise ValueError('--gene-map requires panel_gene and source_gene CSV columns')
        if frame[['panel_gene','source_gene']].apply(lambda s:s.str.strip().eq('')).any().any():
            raise ValueError('Empty explicit gene mapping')
        for row in frame.itertuples(index=False):
            records.append({'gene_id':row.panel_gene, 'symbol':row.source_gene, 'source':str(path)})
        sources.append({'kind':'explicit CSV', 'path':str(path), 'sha256':file_sha256(path)})
        return {'records':records,'sources':sources,'warnings':warnings}
    audit = work/'prepared/real_min10_01/data_audit.json'
    if audit.exists():
        info = json.loads(audit.read_text())
        paths = {s['id']:s['path'] for s in info.get('sources',[])}
        for entry in info.get('manifest',{}).get('datasets',[]):
            if entry.get('context') not in ('K562','RPE1'): continue
            path = Path(paths.get(entry['id'],entry['path']))
            if not path.is_file():
                candidates = sorted((work/'raw').rglob(path.name))
                if len(candidates)!=1:
                    warnings.append(f'Cannot uniquely locate training annotation: {path.name}');continue
                path = candidates[0]
            with h5py.File(path,'r') as f:
                var = f['var']; idx = entry.get('gene_key') or var.attrs.get('_index','_index')
                if isinstance(idx,bytes): idx=idx.decode()
                sym = next((k for k in ('gene_name','gene_symbols','feature_name','gene_symbol','symbol') if k in var),None)
                if sym is None:
                    warnings.append(f'No symbol annotation in {path.name}/var');continue
                ids, symbols = column(var,idx), column(var,sym)
                if len(ids)!=len(symbols): raise ValueError('Annotation axes disagree')
                # Record the whole ID-symbol table's digest without reading count matrices.
                pairs = list(zip(map(str,ids),map(str,symbols)))
                sha = hashlib.sha256(json.dumps(pairs,separators=(',',':')).encode()).hexdigest()
                source = str(path.resolve())+'/var/'+sym
                sources.append({'kind':'training H5AD var only','path':str(path.resolve()),'id_field':idx,'symbol_field':sym,'annotation_sha256':sha})
                for gene,symbol in pairs:
                    if stable_id(gene) in needed and symbol.strip() not in ('','<missing>','nan','None'):
                        records.append({'gene_id':gene,'symbol':symbol,'source':source})
    # These are already-resolved human gene annotations, not expression-derived features.
    cards_path=work/'knowledge/round12/cards.json'
    if cards_path.exists():
        cards=json.loads(cards_path.read_text())
        sources.append({'kind':'existing mapped human gene cards','path':str(cards_path),'sha256':file_sha256(cards_path)})
        for gene,card in cards.items():
            if stable_id(gene) in needed and card.get('status')=='mapped' and card.get('symbol'):
                records.append({'gene_id':gene,'symbol':str(card['symbol']),'source':str(cards_path)})
    if not sources: warnings.append('No local annotations found; --gene-map accepts an explicit audited mapping CSV')
    return {'records':records,'sources':sources,'warnings':warnings}


def resolve_panel(source_genes, panel, annotations=None):
    """Exact IDs first; otherwise require one annotated, measured source symbol.

    Only Ensembl version suffixes are normalized. No case folding, fuzzy aliases,
    duplicate summation, zero filling, or expression-based choices are allowed.
    """
    source_genes, panel = list(map(str,source_genes)), list(map(str,panel))
    if len(source_genes)!=len(set(source_genes)) or len(panel)!=len(set(panel)):
        raise ValueError('Duplicate gene identifiers')
    lookup = {g:i for i,g in enumerate(source_genes)}; by_id = {}
    for r in (annotations or {}).get('records',[]):
        by_id.setdefault(stable_id(r['gene_id']),[]).append(r)
    rows=[]
    for gene in panel:
        evidence=by_id.get(stable_id(gene),[])
        candidates=sorted({r['symbol'] for r in evidence if r['symbol'] in lookup})
        if gene in lookup:
            chosen=gene;status='exact_id'
        elif len(candidates)==1:
            chosen=candidates[0];status='unique_annotated_symbol'
        else:
            chosen='';status='ambiguous_mapping' if candidates else ('not_measured' if evidence else 'no_annotation')
        rows.append({'gene':gene,'source_gene':chosen,'source_index':lookup.get(chosen,-1),
                     'status':status,'measured':bool(chosen),
                     'candidate_symbols':json.dumps(sorted({r['symbol'] for r in evidence})),
                     'evidence':json.dumps(sorted({r['source'] for r in evidence}))})
    frame=pd.DataFrame(rows)
    # Distinct output IDs cannot be assigned the same measured expression row.
    collided=frame.measured & frame.source_gene.duplicated(keep=False)
    frame.loc[collided,'status']='many_panel_ids_to_one_source'
    frame.loc[collided,'measured']=False
    return frame
