"""Pinned canonical gene identities, shared by data admission and split guards."""
from functools import lru_cache
import gzip
import json
from pathlib import Path

SNAPSHOT=Path(__file__).resolve().parents[3]/'configs/round23_hgnc_identity.json.gz'


@lru_cache(maxsize=1)
def snapshot():
    with gzip.open(SNAPSHOT,'rt',encoding='utf-8') as f:data=json.load(f)
    if data['schema_version']!=1 or len(data['source']['sha256'])!=64:raise ValueError('Invalid HGNC identity snapshot')
    return data


def target_aliases():return snapshot()['aliases']


def canonical(label):
    label=str(label)
    if label.startswith('ENSG'):label=label.split('.')[0]
    return target_aliases().get(label)


def annotations(genes):
    data=snapshot();records=[]
    for gene in genes:
        for symbol in data['genes'].get(str(gene),[]):records.append({'gene_id':str(gene),'symbol':symbol,'source':data['source']['url']})
    return {'records':records,'sources':[{'kind':'bundled full HGNC identity snapshot','upstream_sha256':data['source']['sha256'],'url':data['source']['url']}],'warnings':[]}
