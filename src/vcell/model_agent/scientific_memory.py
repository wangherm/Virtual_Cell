"""Bounded public-literature ingestion and cell-line-scoped scientific memory."""
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path
import re
import sqlite3
import requests
from .contracts import cell_line_key,digest
from ..utils import write_json

ENDPOINT='https://www.ebi.ac.uk/europepmc/webservices/rest/search'
QUERIES=('(perturbation OR "Perturb-seq") AND (prediction OR generalization) AND ("gene expression" OR "single-cell" OR transcriptomics)',
         '(perturbation OR "Perturb-seq") AND ("gene expression" OR "single-cell" OR transcriptomics) AND ("low rank" OR bilinear OR regularization)')


def plain(text):return re.sub('<[^>]+>',' ',html.unescape(str(text))).strip()


class ScientificMemory:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS papers (id TEXT PRIMARY KEY, payload TEXT NOT NULL, sha TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, payload TEXT NOT NULL, sha TEXT NOT NULL)')
    def connect(self):return sqlite3.connect(self.path,timeout=30)
    def _put(self,table,key,value):
        body=json.dumps(value,sort_keys=True,allow_nan=False);sha=digest(value)
        with self.connect() as db:
            row=db.execute('SELECT sha FROM '+table+' WHERE id=?',(key,)).fetchone()
            if row and row[0]!=sha:raise ValueError('Immutable scientific memory record changed: '+key)
            db.execute('INSERT OR IGNORE INTO '+table+' VALUES (?,?,?)',(key,body,sha))
    def add_paper(self,paper):
        if paper.get('evidence_kind')!='abstract' or not paper.get('id') or not paper.get('source_url') or not paper.get('abstract'):
            raise ValueError('Traceable abstract record required')
        self._put('papers',paper['id']+':'+paper['snapshot_sha256'],{k:v for k,v in paper.items() if k!='retrieved_utc'})
    def add_experiment(self,key,value):
        allowed={'evaluation_role','outer_used','used_cell_lines','data_fingerprint','arm','status','source_mse','baseline_mse','reason','recipe_sha256'}
        if set(value)-allowed:raise ValueError('Unknown/outer fields cannot enter scientific memory')
        if value.get('evaluation_role')!='source_inner_oof' or value.get('outer_used') is not False or not value.get('used_cell_lines'):
            raise ValueError('Only explicitly scoped source-inner-OOF experiments enter memory')
        self._put('experiments',key,value)
    def retrieve(self,excluded_lines,data_fingerprint,limit=8):
        excluded={cell_line_key(c) for c in excluded_lines};items=[]
        with self.connect() as db:rows=db.execute('SELECT id,payload,sha FROM experiments ORDER BY id').fetchall()
        for key,body,sha in rows:
            value=json.loads(body)
            if digest(value)!=sha:raise ValueError('Scientific memory checksum mismatch')
            if value['data_fingerprint']!=data_fingerprint or excluded & {cell_line_key(c) for c in value['used_cell_lines']}:continue
            # The public projection has no paths, outer results or arbitrary fields.
            items.append({k:value[k] for k in ('arm','status','source_mse','baseline_mse','reason','recipe_sha256') if k in value})
        return items[-limit:]


def collect_literature(dest,cutoff,queries=QUERIES,page_size=6,client=None):
    """Fixed public queries; one page each, no API keys or arbitrary URLs."""
    from .artifacts import read,seal,verify
    from ..round11 import freeze
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',cutoff) or not 1<=page_size<=10 or not 1<=len(queries)<=4:raise ValueError('Bounded literature plan required')
    freeze(dest/'plan.json',{'endpoint':ENDPOINT,'queries':list(queries),'cutoff':cutoff,'page_size':page_size,
        'evidence_kind':'abstract','scope':'Public methods knowledge; not a perturbation-label retrieval tool'})
    if (dest/'COMPLETE.json').exists():verify(dest);return read(dest/'papers.json')
    client=client or requests.Session();papers={};statuses=[]
    for i,query in enumerate(queries):
        path=dest/f'response_{i+1}.json';params={'query':query+' FIRST_PDATE:[1900-01-01 TO '+cutoff+']',
            'resultType':'core','format':'json','pageSize':page_size}
        if path.exists():data=read(path)
        else:
            r=client.get(ENDPOINT,params=params,timeout=(15,45),allow_redirects=False)
            if r.status_code!=200:raise RuntimeError('Europe PMC HTTP '+str(r.status_code))
            data=r.json();write_json(path,data)
        if not isinstance(data.get('resultList',{}).get('result'),list):raise ValueError('Unexpected Europe PMC response')
        statuses.append({'query':query,'returned':len(data['resultList']['result'])})
        for row in data['resultList']['result']:
            source=str(row.get('source',''));identifier=str(row.get('id',''));abstract=plain(row.get('abstractText',''))
            date=str(row.get('firstPublicationDate',''))
            if not abstract or not re.fullmatch(r'[A-Za-z0-9._-]{1,64}',identifier) or source not in ('MED','PMC','PPR') or not date or date>cutoff:continue
            title=plain(row.get('title',''))
            if not re.search(r'single[ -]cell|transcriptom|gene expression|perturb[ -]seq|crispr|geneformer|scgpt',title+' '+abstract,re.I):continue
            paper={'id':source+':'+identifier,'title':title,'abstract':abstract[:6000],
                'doi':row.get('doi'),'publication_date':date,'source_url':'https://europepmc.org/article/'+source+'/'+identifier,
                'version':str(row.get('versionNumber','not_reported')),'evidence_kind':'abstract',
                'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                'code_links':sorted(set(re.findall(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',abstract))),
                'method_claim':'Abstract inspiration only; full-text methods and repository reproduction not verified'}
            paper['snapshot_sha256']=digest({k:v for k,v in paper.items() if k!='retrieved_utc'})
            papers[paper['id']]=paper
    if not papers:raise ValueError('No eligible abstracts found; use cached knowledge or disable literature arm explicitly')
    value=list(papers.values());write_json(dest/'papers.json',value);write_json(dest/'status.json',statuses);seal(dest);return value


def retrieve_papers(papers,question,limit=4):
    words=set(re.findall('[a-z]{3,}',question.lower()))
    def score(p):return len(words & set(re.findall('[a-z]{3,}',(p['title']+' '+p['abstract']).lower())))
    selected=sorted(papers,key=lambda p:(-score(p),p['id']))[:limit]
    return [{k:p[k] for k in ('id','title','abstract','doi','source_url','publication_date','version','snapshot_sha256','evidence_kind')} for p in selected]


def validate_citations(citations,papers,required=True):
    if not isinstance(citations,list) or len(citations)>3 or required and not citations:raise ValueError('Bounded real literature citations required')
    lookup={p['id']:p for p in papers}
    for item in citations:
        if not isinstance(item,dict) or set(item)!={'paper_id','quote'} or item['paper_id'] not in lookup:raise ValueError('Unknown paper citation')
        quote=item['quote'];p=lookup[item['paper_id']]
        if not isinstance(quote,str) or not 10<=len(quote)<=250 or quote not in p['abstract']:raise ValueError('Evidence quote is not a literal retrieved-abstract span')
    return citations
