"""Small metadata-only acquisition/adapter agent, separate from model selection."""
import json
from pathlib import Path
import re
import numpy as np
from ..model_agent.artifacts import read
from ..round22.gateway import Gateway
from ..utils import write_json,file_sha256
from ..round11 import freeze
from ..model_agent.contracts import digest
from .journal import fingerprint
from .catalog import fetch,inspect_h5ad
from .adapter import adapt
from .profiles import curated_config
from ..clean_background import run_lock
from ..model_agent.contracts import safe_name


def parse_proposal(content,inspection,entry,work):
    if content.strip().startswith('```'):
        match=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```\s*',content.strip(),re.S)
        if not match:raise ValueError('One complete JSON object required')
        content=match.group(1)
    proposal=json.loads(content)
    fields={'target_column','control_values','stratum_columns','context_column','gene_column','layer'}
    if not isinstance(proposal,dict) or not fields<=proposal.keys() or set(proposal)-fields:raise ValueError('Adapter agent may only choose declared metadata/matrix fields')
    obs=set(inspection['obs_columns']);var=set(inspection['var_columns'])
    if proposal['target_column'] not in obs or proposal['context_column'] not in obs|{'__dataset_context__'}:raise ValueError('Proposed observation field does not exist')
    if not isinstance(proposal['stratum_columns'],list) or not proposal['stratum_columns'] or not set(proposal['stratum_columns'])<=obs:raise ValueError('Independent matched-control strata must exist')
    if proposal['gene_column'] not in var|{'index'} or proposal['layer'] not in set(inspection['layers'])|{'X'}:raise ValueError('Proposed matrix/gene field does not exist')
    if not isinstance(proposal['control_values'],list) or not proposal['control_values'] or any(not isinstance(s,str) for s in proposal['control_values']):raise ValueError('Explicit observed control labels required')
    if not entry.get('cell_line'):raise ValueError('Catalog must provide a publication-supported canonical cell line')
    # The LLM cannot set gene aliases, cell-line identity, conditions, paths or thresholds.
    return {**proposal,'dataset':entry['id'],'cell_line':entry['cell_line'],'target_namespace':'HGNC_symbol','stimulus':entry.get('stimulus','unknown'),
        'modality':entry['modality'],'time':entry.get('time','unknown'),'annotation_work':str(Path(work).resolve()),'min_cells':20,'min_controls':20,
        'allow_common_panel':True,'min_common_genes':1000}


def _prepare_sources(work,catalog,panel,output,key,model='deepseek-v4-pro',request_fn=None):
    """At most four provider calls total; sealed external datasets are never sent."""
    work=Path(work);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    freeze(output/'data_agent_plan.json',{'catalog':catalog,'panel_sha256':file_sha256(panel),'model':model,'framework_sha256':fingerprint(Path(__file__).resolve().parents[3]),'scope':'Metadata-only, four requests and 131072 output tokens; distinct from research search budget'})
    if (output/'data_agent_complete.json').exists():
        done=read(output/'data_agent_complete.json')
        if file_sha256(output/'ready_catalog.json')!=done['ready_catalog_sha256']:raise ValueError('Completed data catalog changed')
        for entry in read(output/'ready_catalog.json')['sources']:
            if entry.get('bundle') and file_sha256(entry['bundle'])!=entry['sha256']:raise ValueError('Completed source bundle changed')
        return output/'ready_catalog.json'
    with np.load(panel,allow_pickle=False) as f:genes=f['genes']
    ready=[];reports=[]
    gateway=Gateway(output/'api_ledger.json',key,model,'metadata-only',requests_max=4,output_tokens_max=131072,max_tokens=32768,reasoning_effort='low',failure_limit=2,port=0)
    try:
        for entry in catalog['sources']:
            if entry.get('role')!='source':ready.append(entry);continue
            if entry.get('bundle') or entry.get('prepared'):ready.append(entry);continue
            safe_name(entry['id'])
            dest=output/entry['id'];dest.mkdir(exist_ok=True);record={'id':entry['id'],'status':'BLOCKED'}
            try:
                write_json(output/'progress.json',{'dataset':entry['id'],'stage':'download and inspect'});print('DATA AGENT:',entry['id'],'download and inspect',flush=True)
                path=fetch(entry,work/'external'/entry['id']);inspection=inspect_h5ad(path);write_json(dest/'inspect.json',inspection)
                if entry.get('adapter_profile'):
                    import anndata as ad
                    raw=ad.read_h5ad(path,backed='r')
                    try:cfg=curated_config(entry,raw.obs,work.resolve())
                    finally:raw.file.close()
                    write_json(dest/'curated_profile.json',cfg)
                    print('DATA AGENT:',entry['id'],'trusted dataset profile; no provider request',flush=True)
                    added=adapt(path,cfg,genes,dest/'curated_adapter')
                    ready.append({**entry,**added});record.update(status='READY',bundle=added['bundle'],adaptation='trusted dataset metadata profile; zero LLM calls')
                    reports.append(record);print('DATA AGENT:',entry['id'],'READY',flush=True);continue
                previous=None
                for attempt in range(1,3):
                    write_json(output/'progress.json',{'dataset':entry['id'],'stage':'metadata proposal','attempt':attempt});print('DATA AGENT:',entry['id'],'metadata proposal',attempt,flush=True)
                    prompt={'instruction':'Choose a raw-count matrix and explicit perturbation/control/context/matched-control metadata fields. Return ONLY a JSON object with target_column, control_values, stratum_columns, context_column (or __dataset_context__), gene_column (or index), layer (or X). Never invent gene mappings or use expression/validation scores. Preserve replicate/donor/stimulus strata when present.',
                        'publication':entry['publication'],'inspection':inspection,'last_adapter_error':previous}
                    body={'model':model,'messages':[{'role':'user','content':json.dumps(prompt)}],'max_tokens':32768}
                    status,response=(request_fn or gateway.forward)(body)
                    if status!=200:raise RuntimeError('Data-agent provider budget/circuit/response failed; inspect data-agent api_ledger.json')
                    try:
                        cfg=parse_proposal(response['choices'][0]['message']['content'],inspection,entry,work)
                        write_json(dest/f'proposal_{attempt:02d}.json',cfg)
                        write_json(output/'progress.json',{'dataset':entry['id'],'stage':'raw count and gene mapping adaptation','attempt':attempt})
                        added=adapt(path,cfg,genes,dest/f'adapter_{attempt:02d}')
                        ready.append({**entry,**added});record.update(status='READY',bundle=added['bundle']);break
                    except (ValueError,KeyError,TypeError) as exc:
                        previous=str(exc);write_json(dest/f'blocked_{attempt:02d}.json',{'reason':previous,'training_ready':False})
                else:ready.append(entry);record['reason']=previous
            except Exception as exc:
                ready.append(entry);record['reason']=str(exc)
            reports.append(record)
            print('DATA AGENT:',entry['id'],record['status'],record.get('reason',''),flush=True)
        measured=[genes]
        for entry in ready:
            if entry.get('role')=='source' and entry.get('bundle'):
                with np.load(entry['bundle'],allow_pickle=False) as f:measured.append(f['genes'])
        common=set(genes)
        for labels in measured:common &= set(labels)
        panel=np.array([g for g in genes if g in common],dtype='U')
        if len(panel)<1000:raise ValueError('Combined source panel has fewer than 1000 measured genes')
        for entry in ready:
            if entry.get('role')!='source' or not entry.get('bundle'):continue
            with np.load(entry['bundle'],allow_pickle=False) as f:d={k:f[k] for k in f.files}
            lookup={g:i for i,g in enumerate(d['genes'])};idx=np.array([lookup[g] for g in panel])
            d['genes']=panel;d['baseline']=d['baseline'][:,idx];d['delta']=d['delta'][:,idx]
            dest=output/'aligned'/entry['id'];dest.mkdir(parents=True,exist_ok=True);target=dest/'source_bundle.npz'
            np.savez_compressed(target,**d);entry.update(bundle=str(target),sha256=file_sha256(target))
        write_json(output/'ready_catalog.json',{**catalog,'sources':ready,'data_agent_audit':{'requests':len(gateway.records),'charged_output_tokens':sum(r.get('charged_output_tokens',0) for r in gateway.records),'ledger':str(output/'api_ledger.json')},'output_panel':{'genes':panel.tolist(),'reference_gene_count':len(genes),'gene_count':len(panel),
            'policy':'Identifier/annotation intersection frozen before search; both core and expanded arms use exactly this panel. Scores are not directly comparable to legacy full-panel MSE.'}})
        write_json(output/'summary.json',{'datasets':reports,'scope':'Metadata-only data acquisition/adapter agent; no target scoring; blocked sources remain untrained',
            'ready_source_bundles':sum(bool(e.get('bundle')) and e.get('role')=='source' for e in ready),'sealed_external_used':False,
            'provider_requests_max':4,'provider_output_tokens_max':131072})
        if any(e.get('role')=='source' and e.get('bundle') for e in ready):write_json(output/'data_agent_complete.json',{'ready_catalog_sha256':file_sha256(output/'ready_catalog.json')})
        ready_count=sum(bool(e.get('bundle')) and e.get('role')=='source' for e in ready)
        write_json(output/'progress.json',{'stage':'complete' if ready_count else 'blocked; no source ready; research search not started','ready_sources':ready_count,'gene_count':len(panel)})
        return output/'ready_catalog.json'
    finally:gateway.server.server_close()


def prepare_sources(work,catalog,panel,output,key,model='deepseek-v4-pro',request_fn=None):
    output=Path(output)
    with run_lock(output.parent/('.'+output.name+'.lock')):
        return _prepare_sources(work,catalog,panel,output,key,model,request_fn)
