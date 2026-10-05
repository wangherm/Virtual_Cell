"""Build a small, auditable naming snapshot from a locally downloaded HGNC TSV."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


SOURCE_URL = 'https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt'


def build(path, genes):
    path = Path(path)
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    required = {'ensembl_gene_id', 'hgnc_id', 'symbol', 'status', 'prev_symbol'}
    if not rows or not required <= set(rows[0]):
        raise ValueError('Missing required HGNC fields')
    owners = {}
    approved = [row for row in rows if row['status'] == 'Approved']
    for row in approved:
        # Informal alias_symbol is deliberately excluded: e.g. SEPT2 is an
        # alias of SEPTIN6 but a previous approved symbol of SEPTIN2.
        for symbol in [row['symbol']] + row['prev_symbol'].split('|'):
            if symbol:
                owners.setdefault(symbol, set()).add(row['hgnc_id'])
    records = []
    for gene in sorted(set(genes)):
        matched = [row for row in approved if row['ensembl_gene_id'] == gene]
        if len(matched) != 1:
            raise ValueError(f'Expected one approved HGNC record for {gene}; found {len(matched)}')
        row = matched[0]
        previous = [symbol for symbol in row['prev_symbol'].split('|') if symbol]
        records.append({**{key: row[key] for key in ('ensembl_gene_id', 'hgnc_id', 'symbol', 'status')},
                        'prev_symbol': previous,
                        'symbol_owners': {symbol: sorted(owners[symbol]) for symbol in [row['symbol']] + previous},
                        'report_url': 'https://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/' + row['hgnc_id']})
    return {'schema_version': 1,
            'scope': 'Annotation-only naming repair for the selected unresolved Ensembl IDs; not a general alias database',
            'source': {'url': SOURCE_URL, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'retrieved_at': datetime.now(timezone.utc).isoformat(), 'rows': len(rows),
                       'fields': sorted(required), 'ownership_check': 'All Approved symbol and prev_symbol fields in the complete set'},
            'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hgnc-tsv', required=True)
    parser.add_argument('--genes', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    snapshot = build(args.hgnc_tsv, args.genes)
    Path(args.output).write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')
    print(f'HGNC SNAPSHOT: {len(snapshot["records"])} records; source SHA256={snapshot["source"]["sha256"]}')


if __name__ == '__main__':
    main()
