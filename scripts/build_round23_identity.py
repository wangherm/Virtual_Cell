"""Build an annotation-only, ownership-checked human identifier snapshot."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def build(source,dest):
    source=Path(source)
    with source.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
    required={'symbol','hgnc_id','ensembl_gene_id','prev_symbol','status'}
    if not rows or not required<=rows[0].keys():raise ValueError('Full official HGNC table required')
    rows=[r for r in rows if r['status']=='Approved'];owners={};symbols={r['hgnc_id']:r['symbol'] for r in rows}
    for r in rows:
        for label in [r['symbol'],r['ensembl_gene_id'],*r['prev_symbol'].split('|')]:
            if label:owners.setdefault(label,set()).add(r['hgnc_id'])
    aliases={label:symbols[next(iter(ids))] for label,ids in owners.items() if len(ids)==1}
    genes={}
    for r in rows:
        gene=r['ensembl_gene_id']
        if gene and len(owners[gene])==1:genes[gene]=[s for s in [r['symbol'],*r['prev_symbol'].split('|')] if s and owners.get(s)=={r['hgnc_id']}]
    data={'schema_version':1,'source':{'url':'https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt','sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'approved_rows':len(rows)},
        'policy':'Approved symbols, Ensembl IDs and unique previous approved names only; informal aliases excluded','aliases':aliases,'genes':genes}
    Path(dest).write_bytes(gzip.compress(json.dumps(data,sort_keys=True,separators=(',',':')).encode(),mtime=0))
    print('HGNC SNAPSHOT:',len(aliases),'identifiers;',len(genes),'Ensembl genes;',Path(dest).stat().st_size,'bytes')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();build(a.source,a.output)
