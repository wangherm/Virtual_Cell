"""Offline, provenance-bearing identities for the selected Mixscale targets."""
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

SNAPSHOT=Path(__file__).resolve().parents[2]/'configs/round15_hgnc_targets.json'


def entrez_identity(value):
    """5920, '5920' and 5920.0 denote the same stable gene identity."""
    try:number=Decimal(str(value))
    except InvalidOperation as exc:raise ValueError('Invalid Entrez identity') from exc
    if not number.is_finite() or number<=0 or number!=number.to_integral_value():
        raise ValueError('Invalid Entrez identity')
    return str(int(number))


def load_snapshot(path=SNAPSHOT):
    snapshot=json.loads(Path(path).read_text(encoding='utf-8'))
    if snapshot.get('schema_version')!=1 or len(snapshot.get('source',{}).get('sha256',''))!=64:
        raise ValueError('Invalid HGNC target snapshot provenance')
    records={}
    for row in snapshot['records']:
        query=row['query'];owner=row['hgnc_id']
        if (query in records or row['status']!='Approved' or row['query_owners']!=[owner]
                or query not in [row['symbol'],row['ensembl_gene_id']]+row['prev_symbol']
                or not str(row['entrez_id']).isdigit()):
            raise ValueError('Ambiguous/invalid HGNC target identity: '+query)
        records[query]=row
    return snapshot,records


def apply_snapshot(labels,cards,path=SNAPSHOT):
    """Change only added target cards; reject a conflicting existing identity."""
    _,records=load_snapshot(path);result=dict(cards);audit=[]
    for label in labels:
        row=records.get(label);before=result.get(label,{})
        if row is None:
            audit.append({'target':label,'source':'parent/MyGene','status':before.get('status','unavailable'),
                          'symbol':before.get('symbol'),'entrezgene':before.get('entrezgene')})
            continue
        identity=int(row['entrez_id'])
        if before.get('status')=='mapped' and before.get('entrezgene') is not None:
            try:same=entrez_identity(before['entrezgene'])==str(identity)
            except ValueError:same=False
            if not same:raise ValueError('HGNC target identity conflicts with cached annotation: '+label)
        result[label]={**before,'id':label,'status':'mapped','symbol':row['symbol'],'name':row['name'],
                       'entrezgene':identity,'hgnc_id':row['hgnc_id'],'ensembl_gene_id':row['ensembl_gene_id'],
                       'source':row['report_url'],'identity_source':'frozen HGNC approved/previous name',
                       'summary':before.get('summary','')}
        audit.append({'target':label,'source':'frozen HGNC','status':'mapped','symbol':row['symbol'],
                      'entrezgene':identity,'hgnc_id':row['hgnc_id'],'previous_status':before.get('status','unavailable')})
    return result,audit
