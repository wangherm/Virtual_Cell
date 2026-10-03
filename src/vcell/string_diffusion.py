"""Label-free random-projected personalized PageRank on pinned human STRING."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
from scipy import sparse

from .round11 import freeze, verify_files
from .utils import file_sha256, write_json


def download_file(record, root):
    """Bounded streamed retry/resume; final size and committed SHA256 are mandatory."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    name=record['name']
    if Path(name).name!=name:raise ValueError('Invalid download filename')
    path=root/name;part=root/(name+'.part')
    if path.exists():
        if path.stat().st_size!=record['bytes'] or file_sha256(path)!=record['sha256']:
            raise ValueError('Existing STRING file checksum mismatch')
        return path
    for attempt in range(5):
        offset=part.stat().st_size if part.exists() else 0
        if offset>record['bytes']:raise ValueError('Oversized partial STRING download')
        if offset==record['bytes']:break
        try:
            headers={'Accept-Encoding':'identity'}
            if offset:headers['Range']=f'bytes={offset}-'
            with requests.get(record['url'],headers=headers,stream=True,timeout=(20,90)) as response:
                response.raise_for_status()
                if response.status_code==206:
                    if response.headers.get('Content-Range')!=f'bytes {offset}-{record["bytes"]-1}/{record["bytes"]}':
                        raise ValueError('Unexpected STRING byte range')
                elif response.status_code==200:offset=0
                else:raise ValueError('Unexpected STRING download response')
                with part.open('ab' if offset else 'wb') as output:
                    last=offset//(16*1024**2)
                    for chunk in response.iter_content(1024**2):
                        if offset+len(chunk)>record['bytes']:raise ValueError('STRING download exceeded pinned size')
                        output.write(chunk);offset+=len(chunk)
                        if offset//(16*1024**2)>last:
                            last=offset//(16*1024**2);print(f'STRING DOWNLOAD {name}: {offset}/{record["bytes"]}',flush=True)
            if offset!=record['bytes']:raise requests.ConnectionError('Incomplete STRING body')
            break
        except requests.RequestException:
            if attempt==4:raise
            time.sleep(min(2**attempt,8))
    if part.stat().st_size!=record['bytes'] or file_sha256(part)!=record['sha256']:
        raise ValueError('STRING checksum mismatch; retain partial for inspection, do not use it')
    part.replace(path);return path


def read_graph(info_path, links_path, threshold=700):
    info=pd.read_csv(info_path,sep='\t',usecols=['#string_protein_id','preferred_name'],dtype=str)
    info=info.sort_values('#string_protein_id').reset_index(drop=True)
    if info['#string_protein_id'].duplicated().any() or not info['#string_protein_id'].str.startswith('9606.').all():
        raise ValueError('Expected unique human STRING protein IDs')
    lookup=dict(zip(info['#string_protein_id'],range(len(info))))
    rows=[];cols=[];weights=[]
    for chunk in pd.read_csv(links_path,sep=r'\s+',chunksize=500000):
        chunk=chunk[(chunk.combined_score>=threshold)&chunk.protein1.ne(chunk.protein2)]
        a=chunk.protein1.map(lookup);b=chunk.protein2.map(lookup)
        if a.isna().any() or b.isna().any():raise ValueError('Edge node absent from pinned protein.info')
        if (chunk.combined_score>1000).any():raise ValueError('Invalid STRING confidence scale')
        rows.append(a.to_numpy(np.int32));cols.append(b.to_numpy(np.int32));weights.append(chunk.combined_score.to_numpy(np.float32)/1000)
    if not rows or not sum(len(r) for r in rows):raise ValueError('Empty STRING graph')
    a=sparse.csr_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(cols))),shape=(len(info),len(info)))
    if a.nnz!=sum(len(r) for r in rows):raise ValueError('Duplicate directed edges in STRING file')
    # AB and BA encode the same confidence. Never sum them into doubled weights.
    a=a.maximum(a.T).tocsr()
    return info,a


def diffuse(graph,dim=64,alpha=.85,tol=1e-6,max_iter=150,seed=1301):
    """H=(1-alpha)R+alpha T H, a random projection of full PPR rows.

    Isolated nodes are missing graph evidence (zero), not random pseudo-knowledge.
    The residual bound uses the row-stochastic infinity-norm contraction.
    """
    if not 0<alpha<1 or min(dim,max_iter)<1 or tol<=0:raise ValueError('Invalid diffusion parameters')
    graph=sparse.csr_matrix(graph,dtype=np.float64)
    if graph.shape[0]!=graph.shape[1] or not np.isfinite(graph.data).all() or (graph.data<0).any():raise ValueError('Invalid graph')
    degree=np.asarray(graph.sum(1)).ravel();active=degree>0
    transition=sparse.diags(1/np.maximum(degree,1e-30))@graph
    random=np.random.default_rng(seed).normal(size=(len(degree),dim))/np.sqrt(dim)
    random[~active]=0;values=random.copy()
    for iteration in range(1,max_iter+1):
        new=(1-alpha)*random+alpha*(transition@values)
        bound=float(np.max(np.abs(new-values))*alpha/(1-alpha));values=new
        if iteration==1 or iteration%10==0:print(f'STRING DIFFUSION iteration={iteration} max_error_bound={bound:.3g}',flush=True)
        if bound<=tol:break
    else:raise ValueError('Diffusion did not converge; no embedding released')
    return values.astype(np.float32),random.astype(np.float32),{'iterations':iteration,'error_bound':bound,
        'nodes':len(degree),'undirected_edges':graph.nnz//2,'isolated_nodes':int((~active).sum()),
        'dim':dim,'alpha':alpha,'tol':tol,'seed':seed,'self_component_retained':True}


def map_targets(targets,coverage,info,graph,embedding,random):
    if coverage.target.duplicated().any():raise ValueError('Duplicate annotation coverage targets')
    coverage=coverage.set_index('target');groups=info.groupby('preferred_name').indices
    degree=np.asarray(graph.sum(1)).ravel();vectors=[];null=[];records=[]
    for target in targets:
        if target not in coverage.index:raise ValueError('Missing Round12 mapping record')
        card=coverage.loc[target];symbol=card.get('canonical_symbol')
        if pd.isna(symbol) or not symbol:symbol=target
        indices=np.asarray(groups.get(symbol,[]),dtype=int)
        active=indices[degree[indices]>0];present=bool(len(active))
        vectors.append(embedding[active].mean(0) if present else np.zeros(embedding.shape[1]))
        null.append(random[active].mean(0) if present else np.zeros(random.shape[1]))
        records.append({'target':target,'symbol':symbol,'proteins':len(indices),'connected_proteins':len(active),
                        'available':present,'degree_sum':float(degree[active].sum()),
                        'mapping':'exact preferred_name; uniform average of connected proteins, no fuzzy/alias guessing'})
    return np.asarray(vectors,np.float32),np.asarray(null,np.float32),pd.DataFrame(records)


def prepare(manifest_path,root,targets,coverage_path):
    manifest_path,root,coverage_path=Path(manifest_path),Path(root),Path(coverage_path)
    manifest=json.loads(manifest_path.read_text());root.mkdir(parents=True,exist_ok=True)
    identity={'manifest_sha256':file_sha256(manifest_path),'builder_sha256':file_sha256(__file__),
              'targets':list(targets),'coverage_sha256':file_sha256(coverage_path),
              'threshold':700,'dim':64,'alpha':.85,'seed':1301,'tol':1e-6,'max_iter':150}
    freeze(root/'plan.json',identity)
    if (root/'COMPLETE.json').exists():verify_files(root);return root/'embeddings.npz'
    paths=[download_file(r,root/'downloads') for r in manifest['files']]
    info,graph=read_graph(paths[0],paths[1],700)
    full,random,audit=diffuse(graph)
    vectors,null,mapping=map_targets(targets,pd.read_csv(coverage_path),info,graph,full,random)
    if not mapping.available.any():raise ValueError('No connected target mapped to full STRING')
    np.savez_compressed(root/'embeddings.npz',targets=np.asarray(targets,dtype='U'),diffusion=vectors,
                        undiffused_random=null,available=mapping.available.to_numpy())
    mapping.to_csv(root/'mapping.csv',index=False)
    write_json(root/'audit.json',{**audit,'threshold':700,'target_coverage':int(mapping.available.sum()),
        'total_targets':len(targets),'source':manifest,'labels_used':False,
        'claim':'high-confidence full-human functional graph, not restricted to measured panel; not signed causal regulation',
        'null_scope':'gene/profile permutations test mapping; they do not isolate network topology from every other factor'})
    names=['plan.json','embeddings.npz','mapping.csv','audit.json']
    names += [p.relative_to(root).as_posix() for p in paths]
    write_json(root/'COMPLETE.json',{'files':{n:file_sha256(root/n) for n in names}})
    return root/'embeddings.npz'
