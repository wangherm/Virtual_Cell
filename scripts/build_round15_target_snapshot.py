"""Build audited target identities from approved HGNC names and previous names."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SOURCE_URL='https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt'


def build(path, targets):
    path=Path(path)
    with path.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream,delimiter='\t'))
    fields={'hgnc_id','symbol','prev_symbol','ensembl_gene_id','entrez_id','status','name'}
    if not rows or not fields<=set(rows[0]):raise ValueError('Missing HGNC identity fields')
    owners={};approved={}
    for row in rows:
        if row['status']!='Approved':continue
        approved[row['hgnc_id']]=row
        for name in [row['symbol'],row['ensembl_gene_id']]+row['prev_symbol'].split('|'):
            if name:owners.setdefault(name,set()).add(row['hgnc_id'])
    records=[]
    for query in sorted(set(targets)):
        candidates=sorted(owners.get(query,()))
        if len(candidates)!=1:raise ValueError(f'Expected one approved identity for {query}; found {len(candidates)}')
        row=approved[candidates[0]]
        if not row['entrez_id'].isdigit():raise ValueError('Missing stable Entrez identity: '+query)
        records.append({'query':query,'symbol':row['symbol'],'name':row['name'],'status':'Approved',
            'hgnc_id':row['hgnc_id'],'entrez_id':row['entrez_id'],'ensembl_gene_id':row['ensembl_gene_id'],
            'prev_symbol':[name for name in row['prev_symbol'].split('|') if name],
            'query_owners':candidates,'report_url':'https://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/'+row['hgnc_id']})
    return {'schema_version':1,'scope':'Selected Mixscale IFNG target identities; approved and previous names only; no informal aliases',
        'snapshot_built_at':datetime.now(timezone.utc).isoformat(),
        'source':{'url':SOURCE_URL,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'rows':len(rows),'fields':sorted(fields),
                  'ownership_check':'All Approved symbol, prev_symbol and ensembl_gene_id fields in the full table'},
        'records':records}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hgnc-tsv',required=True);p.add_argument('--targets',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();snapshot=build(a.hgnc_tsv,a.targets)
    Path(a.output).write_text(json.dumps(snapshot,indent=2)+'\n',encoding='utf-8')
    print(f'HGNC TARGET SNAPSHOT: {len(snapshot["records"])} records')


if __name__=='__main__':main()
