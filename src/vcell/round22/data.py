"""Rebuild search/confirmation transforms from raw prepared source units."""
from pathlib import Path
import numpy as np
import pandas as pd
from ..data import load_prepared
from ..model_agent.artifacts import read,verify
from ..model_agent.contracts import digest,protected
from ..model_agent.nested import lines
from ..round21.signal import pairs,signature_table,top
from ..round11 import freeze
from ..utils import file_sha256,write_json


def projection(train,query,rank=16):
    mean=train.mean(0);sd=np.maximum(train.std(0),.1)
    _,_,v=np.linalg.svd((train-mean)/sd,full_matrices=False);v=v[:min(rank,len(v))]
    return ((train-mean)/sd)@v.T,((query-mean)/sd)@v.T


def pack(root,name,task,train,query,y,truth,tx,qx,targets,qtargets,ids,contexts,genes,support,qsupport,wrong,wrong_x,mask,available,common,unknown,extra=None,wrong_targets=None):
    dest=root/'inputs'/name;dest.mkdir(parents=True,exist_ok=True)
    background=np.tile(y.mean(0),(len(query),1)).astype(np.float32)
    n=len(train);cal=np.array([int(digest(['cal',str(t)])[:8],16)%5==0 for t in targets])
    if cal.all() or not cal.any():cal=np.arange(n)%5==0
    if n<3 or len(query)==0:raise ValueError('Insufficient source pairs for '+name)
    values=dict(task=np.array(task),genes=np.asarray(genes,dtype='U'),train_x=tx[~cal],train_y=y[~cal],train_support=support[~cal],
        calibration_x=tx[cal],calibration_y=y[cal],calibration_support=support[cal],
        query_x=qx,query_support=qsupport,query_targets=np.asarray(qtargets,dtype='U'),query_ids=np.array([digest(['opaque_round22_query',str(s)]) for s in ids],dtype='U'),
        train_targets=np.asarray(targets,dtype='U')[~cal],calibration_targets=np.asarray(targets,dtype='U')[cal],
        reference_available=available,background_reference=background)
    for key,(train_value,query_value) in (extra or {}).items():
        values['train_'+key]=train_value[~cal];values['calibration_'+key]=train_value[cal];values['query_'+key]=query_value
    for key,v in values.items():
        if np.asarray(v).dtype.kind=='O':raise ValueError('Object array prohibited: '+key)
    # Candidate gets only these public arrays; truths live outside candidate allowlist.
    np.savez_compressed(dest/'input.npz',**values)
    private=root/'private'/name;private.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(private/'grade.npz',truth=truth,background=background,strong_mask=mask,available=available,common=common,unknown=unknown,
        wrong_support=wrong,wrong_x=wrong_x,wrong_targets=np.asarray(wrong_targets if wrong_targets is not None else qtargets,dtype='U'))
    pd.DataFrame({'row_id':ids,'context':contexts,'target':qtargets}).to_csv(private/'metadata.csv',index=False)
    lineage={'task':task,'source_train_rows':np.asarray(train).tolist(),'sealed_query_rows':np.asarray(query).tolist(),
        'source_only_transforms':True,'input_sha256':file_sha256(dest/'input.npz'),'grade_sha256':file_sha256(private/'grade.npz'),
        'metadata_sha256':file_sha256(private/'metadata.csv'),'genes_sha256':digest(list(map(str,genes)))}
    write_json(private/'lineage.json',lineage)
    return {'name':name,'input':str(dest/'input.npz'),'private':str(private),**lineage}


def prepare(parent_run,root,outer_context,seed=17):
    parent=Path(parent_run);verify(parent);lin=read(parent/'lineage.json');data=load_prepared(lin['data_dir'])
    meta=data['meta'];delta=data['delta'];genes=data['genes'];ll=lines(meta)
    if len(genes)!=1834:raise ValueError('Round22 requires frozen 1834-gene panel')
    if 'gene_order' in lin and not np.array_equal(genes,lin['gene_order']):raise ValueError('Parent gene order changed')
    if 'parent' in lin:
        old_inputs=read(Path(lin['parent'])/'inputs.json')
        for filename,field in [('dataset.npz','dataset_sha256'),('metadata.csv','metadata_sha256')]:
            if file_sha256(Path(lin['data_dir'])/filename)!=old_inputs[field]:raise ValueError('Parent prepared data changed: '+filename)
    old=next(p for p in lin['folds'] if p['outer_context']==outer_context)
    cfg=old['folds'][0]['configs']['expanded'];part=cfg['partition']
    allowed=np.unique(np.r_[part['fit'],part['calibration']]).astype(int)
    allowed=allowed[meta.iloc[allowed].stimulus.eq('IFNG').to_numpy()]
    if any(protected(str(s)) for s in ll[allowed]) or outer_context in set(meta.iloc[allowed].context):raise ValueError('Outer/reserved source leak')
    held=set(lin['held_targets']);allowed=allowed[~meta.iloc[allowed].perturbation.isin(held).to_numpy()]
    # Freeze the same three inner query cell lines as the original experiment.
    source_contexts=sorted(f['held'] for f in old['folds'][1:])
    source_contexts=[c for c in source_contexts if c.endswith('_IFNG') and not protected(c)]
    if len(source_contexts)!=3:raise ValueError('Expected three inherited source IFNG query contexts')
    targets=sorted(set(meta.iloc[allowed].perturbation))
    confirmation={t for t in targets if int(digest(['round22_confirmation',seed,t])[:8],16)%5==0}
    if len(confirmation)<3 or len(set(targets)-confirmation)<8:raise ValueError('Too few fixed confirmation/search targets')
    priors=np.load(cfg['relations'],allow_pickle=False)
    if not np.array_equal(priors['genes'],genes):raise ValueError('STRING gene order mismatch')
    lookup={str(t):v[1] for t,v in zip(priors['perturbations'],priors['relations'])}
    if not set(meta.perturbation)<=set(lookup):raise ValueError('Missing frozen STRING target prior')
    def features(train,query):
        a,b=projection(data['baseline'][train],data['baseline'][query])
        # Freeze projections on training target priors, not old response-trained embeddings.
        t,q=projection(np.array([lookup[s] for s in meta.iloc[train].perturbation]),np.array([lookup[s] for s in meta.iloc[query].perturbation]),32)
        return np.c_[a,t].astype(np.float32),np.c_[b,q].astype(np.float32)
    folds={'repeat':{'search':[],'confirmation':[],'outer':[]},'transfer':{'search':[],'confirmation':[],'outer':[]}}
    units=pairs(meta,allowed,delta[allowed]);plan=[]
    for context in source_contexts:
        block=[u for u in units if u['context']==context]
        search=[u for u in block if u['perturbation'] not in confirmation]
        for stage in ('search','confirmation'):
            count=3 if stage=='search' else 1
            for k in range(count):
                query=[u for u in block if (u['perturbation'] in confirmation if stage=='confirmation' else u['perturbation'] not in confirmation and int(digest(['round22_repeat',seed,u['perturbation']])[:8],16)%3==k)]
                train=[u for u in search if u['perturbation'] not in {q['perturbation'] for q in query}]
                if not query:continue
                if len(train)<4:raise ValueError('Too few other-target training pairs: '+context)
                ti=np.array([int(u['rep2_rows'][0]) for u in train]);qi=np.array([int(u['rep2_rows'][0]) for u in query]);tx,qx=features(ti,qi)
                a=np.array([u['rep1'] for u in train]);b=np.array([u['rep2'] for u in train]);q=np.array([u['rep1'] for u in query]);truth=np.array([u['rep2'] for u in query])
                sigma=np.sqrt(np.median(np.square(a-b)/2,axis=0));mask=np.zeros_like(q,bool)
                for j,v in enumerate(q):mask[j,top(v,100)]=True
                mask &= np.abs(q)>=np.maximum(.05,2*sigma)
                wi=[int(digest(['wrong',seed,u['perturbation']])[:8],16)%len(a) for u in query]
                wrong=a[wi]
                name=f'repeat_{stage}_{context}_{k}'
                record=pack(root,name,'repeat',ti,qi,b,truth,tx,qx,[u['perturbation'] for u in train],[u['perturbation'] for u in query],
                    meta.iloc[qi].row_id.tolist(),[context]*len(qi),genes,a,q,wrong,qx,mask,np.ones(len(qi),bool),np.ones(len(qi),bool),np.zeros(len(qi),bool),
                    extra={'control':(data['baseline'][ti],data['baseline'][qi]),'prior':(np.array([lookup[u['perturbation']] for u in train]),np.array([lookup[u['perturbation']] for u in query])),
                        'source_context':(meta.iloc[ti].context.to_numpy(dtype='U'),meta.iloc[qi].context.to_numpy(dtype='U'))},wrong_targets=[train[i]['perturbation'] for i in wi])
                record['train_rep1_rows']=[int(u['rep1_rows'][0]) for u in train];record['query_rep1_support_rows']=[int(u['rep1_rows'][0]) for u in query]
                folds['repeat'][stage].append(record)
    def transfer_fold(stage,context,train,query):
        if not len(query):return
        tx,qx=features(train,query);y=delta[train];mean=y.mean(0)
        # Donor use is computed only from current training labels and exact conditions.
        refs=[];wrong=[];wx=[];wt=[];av=[];common=[];mask=[]
        table,_,masks=signature_table(meta,train,y)
        from ..round21.signal import KEYS
        cond=('dataset','stimulus','time','modality','batch')
        tr=meta.iloc[train]
        for j,idx in enumerate(query):
            row=meta.iloc[idx];legal=np.array([all(str(r[c])==str(row[c]) for c in cond) for _,r in tr.iterrows()])
            matched=np.flatnonzero(legal & tr.perturbation.eq(row.perturbation).to_numpy())
            other=np.flatnonzero(legal & ~tr.perturbation.eq(row.perturbation).to_numpy())
            supported=bool(len(matched));paired=supported and bool(len(other))
            refs.append(y[matched].mean(0) if supported else np.zeros(len(genes)))
            oi=other[int(digest(['wrong',seed,str(row.perturbation)])[:8],16)%len(other)] if paired else None
            wrong.append(y[oi] if oi is not None else np.zeros(len(genes)));av.append(supported);common.append(paired)
            wx.append(tx[oi] if oi is not None else qx[j])
            wt.append(str(tr.iloc[oi].perturbation) if oi is not None else str(row.perturbation))
            strong=np.zeros(len(genes),bool)
            for pos in matched:
                key=tuple(tr.iloc[pos][c] for c in KEYS)
                if key in masks:strong |= masks[key]
            mask.append(strong)
        # Training support excludes each row's complete cell line.
        support=[]
        for idx in train:
            row=meta.iloc[idx];ok=(ll[train]!=ll[idx]) & tr.perturbation.eq(row.perturbation).to_numpy()
            for c in cond:ok &= tr[c].eq(row[c]).to_numpy()
            support.append(y[ok].mean(0) if ok.any() else np.zeros(len(genes)))
        # Wrong-target ablation preserves the true query control feature block.
        control_dim=min(16,len(train));wx=np.array(wx);wx[:,:control_dim]=qx[:,:control_dim]
        name=f'transfer_{stage}_{context}'
        record=pack(root,name,'transfer',train,query,y,delta[query],tx,qx,tr.perturbation.tolist(),meta.iloc[query].perturbation.tolist(),
            meta.iloc[query].row_id.tolist(),meta.iloc[query].context.tolist(),genes,np.array(support),np.array(refs),np.array(wrong),wx,np.array(mask),np.array(av),np.array(common),meta.iloc[query].perturbation.isin(held).to_numpy(),
            extra={'control':(data['baseline'][train],data['baseline'][query]),'prior':(np.array([lookup[s] for s in tr.perturbation]),np.array([lookup[s] for s in meta.iloc[query].perturbation])),
                'source_context':(tr.context.to_numpy(dtype='U'),meta.iloc[query].context.to_numpy(dtype='U')),
                'conditions':(tr[list(cond)].to_numpy(dtype='U'),meta.iloc[query][list(cond)].to_numpy(dtype='U'))},wrong_targets=wt)
        record['donor_policy']='all legal same-target source responses; reliability is available for diagnostics, not a mandatory zero gate'
        folds['transfer'][stage].append(record)
    for stage in ('search','confirmation'):
        for context in source_contexts:
            train=allowed[meta.iloc[allowed].context.ne(context).to_numpy() & ~meta.iloc[allowed].perturbation.isin(confirmation).to_numpy()]
            query=allowed[meta.iloc[allowed].context.eq(context).to_numpy() & (meta.iloc[allowed].perturbation.isin(confirmation).to_numpy()==(stage=='confirmation'))]
            if stage=='confirmation':train=allowed[meta.iloc[allowed].context.ne(context).to_numpy()]
            transfer_fold(stage,context,train,query)
        if len(folds['transfer'][stage])!=3:raise ValueError('Every inherited source context needs '+stage+' queries')
    query=np.asarray(part['outer'],int);query=query[meta.iloc[query].stimulus.eq('IFNG').to_numpy()]
    transfer_fold('outer',outer_context,allowed,query)
    plan={'folds':folds,'outer_context':outer_context,'seed':seed,'gene_order':list(map(str,genes)),
        'confirmation_targets':sorted(confirmation),'global_held_targets':sorted(held),'source_contexts':source_contexts,
        'source_rows':allowed.tolist(),'parent_sha256':file_sha256(parent/'COMPLETE.json'),'raw_controls_verified':read(parent/'summary.json')['raw_audit'].get('independent_repeat_controls_verified',False),
        'data_sha256':file_sha256(Path(lin['data_dir'])/'dataset.npz'),'relations_sha256':file_sha256(cfg['relations']),
        'scope':'IFNG-only rebuilt control/STRING features; no pretrained Qwen or DeepSeek numerical backbone; observed development folds',
        'teacher_policy':'No old response-trained embeddings or teacher scores reused across new confirmation split',
        'repeat_outer':'Not evaluated: supported repeat has no whole-line training source for an unseen line','test_evaluated':False}
    freeze(root/'data_plan.json',plan);return plan
