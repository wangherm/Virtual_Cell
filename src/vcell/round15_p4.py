"""Matched pre-Mixscale / Mixscale training on fixed historical development rows."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .background_training import fingerprint
from .data import load_prepared, save_prepared
from .fixed_training import response_basis
from .knowledge import cached_request, load_knowledge, resolve_cards
from .round11 import freeze, verify_files
from .round12 import design, encode_background, fit_ridge, projection, score, target_features
from .round13 import identity_indices, retrieval
from .round14 import response_diagnostics, target_representation
from .round15 import infer, reconstruct, seal, train_head
from .selection import mse, row_weights
from .specialization import reference_responses
from .train import source_fingerprint
from .utils import file_sha256, write_json

METHODS = ('string_ridge', 'factor_rank16', 'factor_rank32')
CONTEXTS = {'A549_IFNG','BXPC3_IFNG','HAP1_IFNG','K562_IFNG','MCF7_IFNG'}


def assert_mixscale_extension(reference, expanded, allow_target_aliases=False):
    n = len(reference['meta'])
    if not np.array_equal(reference['genes'], expanded['genes']):
        raise ValueError('Historical output panel changed')
    for field in ('baseline','delta'):
        if not np.array_equal(reference[field], expanded[field][:n]):
            raise ValueError('Historical expression changed')
    for field in ('row_id','context','perturbation','split','batch'):
        if not np.array_equal(reference['meta'][field], expanded['meta'][field].iloc[:n]):
            raise ValueError('Historical metadata changed: '+field)
    extra = expanded['meta'].iloc[n:]
    if extra.empty or not extra.context.isin(CONTEXTS).all() or not extra.split.eq('train').all():
        raise ValueError('Require only audited development IFNG training rows')
    if not extra.stimulus.eq('IFNG').all() or not extra.modality.eq('CRISPRi').all():
        raise ValueError('Mixscale stimulus/modality changed')
    if expanded['meta'].context.astype(str).str.upper().str.contains('HT29',regex=False).any():
        raise ValueError('Reserved HT29 found in prepared data')
    if not allow_target_aliases and expanded['audit'].get('historical_fingerprint') != reference['audit']['fingerprint']:
        raise ValueError('Mixscale reference lineage mismatch')


def canonical_id(card):
    if card.get('status') != 'mapped':
        return None
    if card.get('entrezgene') is not None:
        return 'entrez:'+str(card['entrezgene'])
    return 'symbol:'+str(card['symbol']) if card.get('symbol') else None


def target_aliases(reference, expanded, cards):
    """Keep historical labels; join new aliases by stable annotation identity.

    Unknown historical identities remain reported as unknown. Every added label
    must resolve uniquely. Historical alias collisions cannot be repaired without
    changing the old benchmark, so they block this matched experiment.
    """
    old = set(map(str,reference['perturbations'])); owners = {}; rows = []
    for label in sorted(old):
        identity = canonical_id(cards.get(label,{}))
        if identity:
            if identity in owners:
                raise ValueError('Historical target alias collision: '+owners[identity]+' / '+label)
            owners[identity] = label
    aliases = {}
    for label in sorted(set(map(str,expanded['perturbations']))-old):
        identity = canonical_id(cards.get(label,{}))
        if identity is None:
            raise ValueError('Added target has missing/ambiguous annotation: '+label)
        chosen = owners.setdefault(identity,label)
        if chosen != label:
            aliases[label] = chosen
    for label in map(str,expanded['perturbations']):
        rows.append({'target':label,'canonical_id':canonical_id(cards.get(label,{})),
                     'training_label':aliases.get(label,label),'historical':label in old,
                     'status':cards.get(label,{}).get('status','unavailable')})
    return aliases, pd.DataFrame(rows)


def prepare_relations(reference, expanded, parent, dest):
    """Preserve every historical STRING vector; query only added target metadata.

    This artifact contains STRING features only. It does not pretend to provide
    newly encoded Qwen semantic vectors. No responses are sent to any API.
    """
    parent, dest = Path(parent), Path(dest); dest.mkdir(parents=True,exist_ok=True)
    verify_files(parent)
    base = load_knowledge(parent/'knowledge.npz',reference)
    identity = {'parent_complete':file_sha256(parent/'COMPLETE.json'),
                'genes':expanded['genes'].tolist(),'targets':expanded['perturbations'].tolist(),
                'code':file_sha256(__file__),'string_version':'12.0','neighbor_limit':16}
    freeze(dest/'plan.json',identity)
    if (dest/'COMPLETE.json').exists():
        verify_files(dest)
        return json.loads((dest/'target_aliases.json').read_text())['aliases']
    cards = json.loads((parent/'cards.json').read_text())
    new = sorted(set(map(str,expanded['perturbations']))-set(map(str,reference['perturbations'])))
    missing = [label for label in new if label not in cards]
    print(f'P4 ANNOTATION: {len(new)} added labels; {len(missing)} require annotation lookup',flush=True)
    records = []
    for start in range(0,len(missing),100):
        records += cached_request(dest/'requests','https://mygene.info/v3/query',
            {'q':','.join(missing[start:start+100]),'scopes':'symbol,ensembl.gene','species':'9606',
             'fields':'symbol,name,summary,entrezgene,taxid','size':'5'})
    cards.update(resolve_cards(missing,records))
    aliases, audit = target_aliases(reference,expanded,cards)
    audit.to_csv(dest/'target_id_audit.csv',index=False)
    write_json(dest/'target_aliases.json',{'aliases':aliases})
    write_json(dest/'cards.json',cards)
    # Old labels and new aliases of old labels use the original frozen vector.
    old_index = {str(label):i for i,label in enumerate(reference['perturbations'])}
    genuinely_new = sorted(set(aliases.get(label,label) for label in new)-set(old_index))
    symbols = sorted({cards[label]['symbol'] for label in genuinely_new})
    edges = []
    for start in range(0,len(symbols),50):
        edges += cached_request(dest/'requests','https://version-12-0.string-db.org/api/json/interaction_partners',
            {'identifiers':'\r'.join(symbols[start:start+50]),'species':'9606','required_score':'700',
             'limit':'16','network_type':'functional','caller_identity':'Virtual_Cell_round15_p4'})
        print(f'P4 STRING: {min(start+50,len(symbols))}/{len(symbols)} new symbols',flush=True)
    panel_symbols = [cards.get(str(g),{}).get('symbol') for g in reference['genes']]
    vocabulary = sorted(set(aliases.get(str(label),str(label)) for label in expanded['perturbations']))
    relations = np.zeros((len(vocabulary),3,len(reference['genes'])),np.float32)
    for i,label in enumerate(vocabulary):
        if label in old_index:
            relations[i] = base['relations'][old_index[label]]
            continue
        symbol = cards[label]['symbol']; row = relations[i,2]
        for edge in edges:
            if edge.get('preferredName_A') != symbol: continue
            value = float(edge['score'])
            if not np.isfinite(value) or not 0 <= value <= 1: raise ValueError('Invalid STRING confidence')
            for j,other in enumerate(panel_symbols):
                if other and other != symbol and other == edge.get('preferredName_B'):
                    row[j] = max(row[j],value)
        keep = np.argsort(-row,kind='stable')[:16]
        mask = np.ones(len(row),bool);mask[keep]=False;row[mask]=0
        row /= max(float(row.sum()),1e-12)
    np.savez_compressed(dest/'relations.npz',genes=reference['genes'],perturbations=np.array(vocabulary,dtype='U'),relations=relations)
    pd.DataFrame({'target':vocabulary,'historical':[label in old_index for label in vocabulary],
                  'panel_neighbors':(relations[:,2]>0).sum(1)}).to_csv(dest/'coverage.csv',index=False)
    write_json(dest/'audit.json',{'old_features_preserved':True,'new_labels':new,'canonical_aliases':aliases,
        'features':'STRING v12 top16 functional associations; no semantic/GO features used',
        'missing_panel_neighbors':'retained with explicit unavailable flag; not zero-effect labels',
        'pretraining_overlap':'Public annotations may include held-out biology; exploratory development',
        'annotation_requests_only':True,'test_evaluated':False})
    seal(dest);return aliases


def harmonize(reference, expanded, aliases, dest):
    """Apply audited label joins only to appended rows, preserving old labels."""
    dest = Path(dest)
    if not aliases: return None
    if (dest/'COMPLETE.json').exists():
        verify_files(dest);return dest
    meta = expanded['meta'].copy();n=len(reference['meta'])
    meta.loc[meta.index[n:],'perturbation']=meta.iloc[n:].perturbation.replace(aliases).to_numpy()
    vocab = sorted(set(meta.perturbation));lookup={label:i for i,label in enumerate(vocab)}
    save_prepared(dest,expanded['baseline'],expanded['delta'],np.array([lookup[p] for p in meta.perturbation]),
        expanded['genes'],vocab,meta,{**expanded['audit'],'target_label_aliases':aliases,
                                    'adapter_fingerprint':expanded['audit']['fingerprint']})
    assert_mixscale_extension(reference,load_prepared(dest),allow_target_aliases=True)
    seal(dest);return dest


def load_relations(path,data):
    with np.load(path,allow_pickle=False) as f: values={key:f[key] for key in f.files}
    if not np.array_equal(values['genes'],data['genes']): raise ValueError('Relation panel mismatch')
    lookup={str(label):i for i,label in enumerate(values['perturbations'])}
    try: ix=[lookup[str(label)] for label in data['perturbations']]
    except KeyError as exc: raise ValueError('Relation target missing') from exc
    relations=values['relations'][ix]
    if relations.shape!=(len(ix),3,len(data['genes'])) or not np.isfinite(relations).all():
        raise ValueError('Invalid relation matrix')
    return {'relations':relations}


def worker(cfg,resume=False):
    torch.set_num_threads(2)
    dest=Path(cfg['output_dir']);dest.mkdir(parents=True,exist_ok=True)
    data=load_prepared(cfg['data_dir']);assets=load_relations(cfg['relations'],data)
    identity={'config':cfg,'source':source_fingerprint(),'data':data['audit']['fingerprint'],
              'metadata':file_sha256(Path(cfg['data_dir'])/'metadata.csv'),
              'relations':file_sha256(cfg['relations']),'encoder':file_sha256(cfg['encoder'])}
    stamp=fingerprint(identity)
    if (dest/'manifest.json').exists() and not resume: raise ValueError('Use --resume or new name')
    freeze(dest/'manifest.json',{'fingerprint':stamp,**identity})
    if (dest/'COMPLETE.json').exists():verify_files(dest);return dest
    from .round8 import validate_partition
    validate_partition(data,cfg['partition'])
    fit,cal,outer=[np.array(cfg['partition'][role],int) for role in ('fit','calibration','outer')]
    selected=data['meta'].iloc[np.r_[fit,cal,outer]]
    if selected.context.str.upper().str.contains('HT29|JURKAT',regex=True).any():raise ValueError('Reserved data selected')
    latent,norm=encode_background(data,cfg);bg,bt=projection(latent,fit,16)
    target,transform,present=target_representation(data,assets,fit,'local')
    meta=data['meta'].iloc[outer].reset_index(drop=True);truth=data['delta'][outer]
    ids,seen=identity_indices(data,fit);w=row_weights(data['meta'].iloc[fit]);cw=row_weights(data['meta'].iloc[cal])
    reference,_=reference_responses(data,fit)
    records=[];ranks=[];diagnostics=[]
    threshold=max(float(np.quantile(np.linalg.norm(data['delta'][fit],axis=1),.1)),1e-8)
    def evaluate(name,prediction,available=None):
        records.extend(score(meta,truth,prediction,name,available))
        ranks.extend(retrieval(meta,truth,prediction,name))
        diagnostics.extend(response_diagnostics(meta,truth,prediction,name,threshold))
    evaluate('zero',np.zeros_like(truth));evaluate('mean_transfer',reference[data['pert_idx'][outer]])
    basis_cache={}
    for name in cfg['methods']:
        folder=dest/name;folder.mkdir(exist_ok=True)
        if (folder/'COMPLETE.json').exists():
            verify_files(folder)
            if json.loads((folder/'COMPLETE.json').read_text())['fingerprint']!=stamp:raise ValueError('Head fingerprint mismatch')
            for file,collection in [('per_target',records),('retrieval',ranks),('response_diagnostics',diagnostics)]:
                collection.extend(pd.read_csv(folder/(file+'.csv')).to_dict('records'))
            continue
        rank=16 if name=='factor_rank16' else 32
        if rank not in basis_cache:basis_cache[rank]=response_basis(data,fit,rank)
        basis,ba=basis_cache[rank];write_json(folder/'basis_audit.json',ba)
        np.savez_compressed(folder/'background_transform.npz',**bt)
        print(f'P4 HEAD: {cfg["variant"]}/{cfg["protocol"]} seed={cfg["seed"]} {name}',flush=True)
        if name=='string_ridge':
            tf,tr,covered=target_features(data,assets,fit,'string',cfg['seed'],32)
            x=design(bg,tf[data['pert_idx']],True);mean=w@x[fit]
            scale=np.maximum(np.sqrt(w@np.square(x[fit]-mean)),.1);x=((x-mean)/scale).astype(np.float32)
            y=data['delta'][fit]@basis.T/norm['scale'];trials=[];models=[]
            for alpha in cfg['ridge_alphas']:
                coef,bias=fit_ridge(x[fit],y,w.copy(),alpha)
                pred=(x[cal]@coef+bias)@basis*norm['scale']
                trials.append({'alpha':alpha,'calibration_mse':mse(pred,data['delta'][cal],cw)})
                models.append((coef,bias))
            best=int(np.argmin([trial['calibration_mse'] for trial in trials]));coef,bias=models[best]
            coefficients=x[outer]@coef+bias
            np.savez_compressed(folder/'model.npz',coef=coef,intercept=bias)
            np.savez_compressed(folder/'transforms.npz',**tr,xmean=mean,xscale=scale)
            write_json(folder/'selection.json',{'trials':trials,'chosen':trials[best],'selection':'source calibration only'})
        else:
            offset=np.zeros_like(data['delta']);rx=np.zeros((len(data['meta']),1),np.float32)
            arm='factor_response_rank16' if rank==16 else 'local_factor_noid'
            model=train_head(target[data['pert_idx']],bg,ids,rx,offset,data,basis,norm['scale'],fit,cal,cfg,
                             arm,'mse_reference',folder,stamp)
            coefficients=infer(model,target[data['pert_idx'][outer]],bg[outer],ids[outer],rx[outer],cfg['device'])
            np.savez_compressed(folder/'transforms.npz',**transform,fit_target_indices=seen)
            covered=present;del model
            if cfg['device']=='cuda':torch.cuda.empty_cache()
        np.savez_compressed(folder/'coefficients.npz',coefficients=coefficients.astype(np.float32),basis=basis,
            scale=np.float32(norm['scale']),offset=np.zeros_like(truth),row_ids=meta.row_id.to_numpy(dtype='U'),genes=data['genes'])
        prediction=reconstruct(folder/'coefficients.npz',meta.row_id.to_numpy(dtype='U'),data['genes'])
        start=[len(records),len(ranks),len(diagnostics)]
        evaluate(name,prediction,covered[data['pert_idx'][outer]])
        for file,collection,first in zip(('per_target','retrieval','response_diagnostics'),(records,ranks,diagnostics),start):
            pd.DataFrame(collection[first:]).to_csv(folder/(file+'.csv'),index=False)
        seal(folder,fingerprint=stamp)
    for file,collection in [('per_target',records),('retrieval',ranks),('response_diagnostics',diagnostics)]:
        pd.DataFrame(collection).to_csv(dest/(file+'.csv'),index=False)
    seal(dest,fingerprint=stamp,test_evaluated=False)
    print('P4 WORKER COMPLETE:',dest,flush=True);return dest


def report(specs,root):
    root=Path(root);frames={key:[] for key in ('per_target','retrieval','response_diagnostics')}
    for cfg in specs.values():
        folder=Path(cfg['output_dir']);verify_files(folder)
        for key in frames:
            frame=pd.read_csv(folder/(key+'.csv'))
            for col in ('protocol','seed','variant'):frame[col]=cfg[col]
            frames[key].append(frame)
    frames={key:pd.concat(value,ignore_index=True) for key,value in frames.items()}
    for key,value in frames.items():value.to_csv(root/(key+'.csv'),index=False)
    keys=['protocol','variant','method','seed','context']
    context=frames['per_target'].groupby(keys).agg(mse=('mse','mean'),pearson=('pearson','mean'),targets=('target','size')).reset_index()
    rr=frames['retrieval'].groupby(keys).agg(top1=('top1_credit','mean'),top5=('top5_credit','mean'),specificity_gap=('specificity_gap','mean')).reset_index()
    context=context.merge(rr,on=keys,validate='one_to_one');context.to_csv(root/'context_scores.csv',index=False)
    per_seed=context.groupby(keys[:-1]).agg(mse=('mse','mean'),top1=('top1','mean'),top5=('top5','mean'),specificity_gap=('specificity_gap','mean')).reset_index()
    per_seed.to_csv(root/'comparison.csv',index=False)
    summary=per_seed.groupby(keys[:3]).agg(mse_mean=('mse','mean'),seed_sd=('mse','std'),seeds=('seed','nunique'),
        top1=('top1','mean'),specificity_gap=('specificity_gap','mean')).reset_index()
    summary.to_csv(root/'comparison_mean.csv',index=False)
    # Average optimizer seeds first; sample complete target identities, retaining
    # every context of that target. Seeds are not biological replications.
    averaged=frames['per_target'].groupby(['protocol','variant','method','context','target']).mse.mean()
    contrasts=[]
    for protocol in summary.protocol.unique():
        for method in summary.method.unique():
            left=averaged.loc[(protocol,'reference',method)];right=averaged.loc[(protocol,'expanded',method)]
            if not left.index.equals(right.index):raise ValueError('Paired evaluation membership changed')
            gain=left-right;labels=sorted(set(gain.index.get_level_values('target')));rng=np.random.default_rng(15154)
            draws=[]
            for _ in range(2000):
                counts=pd.Series(rng.choice(labels,len(labels),replace=True)).value_counts()
                weights=np.array([counts.get(target,0) for _,target in gain.index])
                sampled=pd.DataFrame({'gain':gain.to_numpy()*weights,'count':weights,'context':gain.index.get_level_values('context')}).groupby('context').sum()
                sampled=sampled.loc[sampled['count']>0]
                draws.append(float((sampled.gain/sampled['count']).mean()))
            contrasts.append({'protocol':protocol,'method':method,'reference_minus_expanded_mse':float(gain.groupby(level='context').mean().mean()),
                'ci_low':float(np.quantile(draws,.025)),'ci_high':float(np.quantile(draws,.975)),
                'targets':len(labels),'positive_means_expansion_better':True,'exploratory_uncorrected':True})
    pd.DataFrame(contrasts).to_csv(root/'paired_data_gain.csv',index=False)
    write_json(root/'report_contract.json',{'test_evaluated':False,'scope':'Historical development tasks; matched reference vs full eligible Mixscale expansion',
        'selection':'No automatic winner or external test release','weights':'Equal contexts, equal targets within context, equal rows per target',
        'interpretation':'Not a pure cell-count experiment: IFNG background/target coverage and context-weight allocation change',
        'new_IFNG_generalization':'Not evaluated by these historical outer folds',
        'uncertainty':'Exploratory paired target bootstrap after averaging optimizer seeds; no multiple-comparison correction'})
    html='<h1>Round15 P4 matched data expansion</h1><p>Historical development evaluation; HT29/Jurkat remain unscored. No automatic winner.</p>'
    html+=summary.to_html(index=False)+pd.DataFrame(contrasts).to_html(index=False)
    (root/'report.html').write_text(html,encoding='utf-8')
