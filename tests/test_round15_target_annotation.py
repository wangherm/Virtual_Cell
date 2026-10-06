"""Approved-name resolution must repair legacy names without weakening identity guards."""
import csv
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from test_round15_p4 import adapter_fixture, annotated_parent
from vcell.round15_p4 import canonical_id, load_relations, prepare_relations, target_aliases
from vcell.round15_target_annotation import SNAPSHOT, apply_snapshot, load_snapshot
from vcell.utils import write_json


def test_bundled_all_59_unique_identities_and_rarres3_old_name():
    snapshot,records=load_snapshot()
    assert len(records)==59
    row=records['RARRES3']
    assert row['symbol']=='PLAAT4' and row['entrez_id']=='5920' and row['hgnc_id']=='HGNC:9869'
    assert row['prev_symbol']==['RARRES3']
    cards,audit=apply_snapshot(['RARRES3'],{'RARRES3':{'status':'missing'},'OLD':{'status':'mapped','symbol':'OLD'}})
    assert cards['RARRES3']['status']=='mapped' and cards['RARRES3']['entrezgene']==5920
    assert cards['OLD']=={'status':'mapped','symbol':'OLD'}
    assert audit[0]['previous_status']=='missing'
    assert len(snapshot['source']['sha256'])==64


def test_conflicting_identity_and_invalid_alias_evidence_are_rejected(tmp_path):
    with pytest.raises(ValueError,match='conflicts'):
        apply_snapshot(['RARRES3'],{'RARRES3':{'status':'mapped','entrezgene':23586}})
    snapshot,records=load_snapshot();record=copy.deepcopy(records['RARRES3'])
    for change in ({'query':'RIG1'},{'query_owners':['HGNC:9869','HGNC:19102']},{'entrez_id':''}):
        invalid=copy.deepcopy(snapshot);invalid['records']=[{**record,**change}]
        write_json(tmp_path/'bad.json',invalid)
        with pytest.raises(ValueError,match='invalid HGNC'):load_snapshot(tmp_path/'bad.json')


def test_stable_entrez_types_cannot_evade_identity_join():
    for value in (5920,'5920',5920.0,'5920.0'):
        assert canonical_id({'status':'mapped','entrezgene':value})=='entrez:5920'
    for value in ('NaN',5920.5,0,'unknown'):
        with pytest.raises(ValueError,match='Invalid Entrez'):
            canonical_id({'status':'mapped','entrezgene':value})


def test_rarres3_snapshot_repairs_inherited_missing_card_and_joins_held_gene(tmp_path,monkeypatch):
    import vcell.round15_p4 as engine
    old,expanded,_=adapter_fixture(tmp_path);parent=tmp_path/'parent'
    cards=annotated_parent(old,parent)
    # A newly added legacy label points to a historical globally held gene.
    # The frozen identity must join them even though the parent card was missing.
    from vcell.round12 import make_partition
    held=str(old['meta'].iloc[make_partition(old,old,'target')['outer'][0]].perturbation)
    cards[held]={**cards[held],'symbol':'PLAAT4','entrezgene':5920}
    cards['RARRES3']={'status':'missing'};write_json(parent/'cards.json',cards)
    from vcell.round15 import seal
    seal(parent)
    # The synthetic fixed-width dtype is shorter than this real target name.
    labels=np.array(['RARRES3' if str(p)=='NEW' else str(p) for p in expanded['perturbations']],dtype='U')
    expanded['perturbations']=labels
    expanded['meta'].loc[expanded['meta'].perturbation.eq('NEW'),'perturbation']='RARRES3'
    calls=[]
    monkeypatch.setattr(engine,'cached_request',lambda *args: calls.append(args) or [])
    aliases=prepare_relations(old,expanded,parent,tmp_path/'relations')
    assert aliases=={'RARRES3':held} and not calls
    np.testing.assert_array_equal(load_relations(tmp_path/'relations/relations.npz',old)['relations'],
                                  np.load(parent/'knowledge.npz')['relations'])
    resolved=json.loads((tmp_path/'relations/cards.json').read_text())
    assert resolved['RARRES3']['symbol']=='PLAAT4'
    assert target_aliases(old,expanded,resolved)[0]==aliases


def test_snapshot_builder_uses_full_table_ownership_and_excludes_informal_aliases(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from build_round15_target_snapshot import build
    path=tmp_path/'hgnc.tsv'
    fields=['hgnc_id','symbol','prev_symbol','ensembl_gene_id','entrez_id','status','name','alias_symbol']
    rows=[['HGNC:9869','PLAAT4','RARRES3','ENSG00000133321','5920','Approved','PLAAT4','RIG1'],
          ['HGNC:19102','DDX58','','ENSG00000107201','23586','Approved','DDX58','RIG1']]
    def save():
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(fields);w.writerows(rows)
    save();snapshot=build(path,['RARRES3']);assert snapshot['records'][0]['symbol']=='PLAAT4'
    with pytest.raises(ValueError,match='found 0'):build(path,['RIG1'])
    rows[1][2]='RARRES3';save()
    with pytest.raises(ValueError,match='found 2'):build(path,['RARRES3'])
