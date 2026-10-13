"""Nested, response-label cross-fitted teacher features for a numeric fusion student."""
from pathlib import Path
import numpy as np
import torch
from ..fixed_training import response_basis
from ..round12 import projection,design,fit_ridge,encode_background
from ..round13 import FactorizedResponse
from ..round14 import target_representation
from ..round15_p4 import load_relations
from ..selection import row_weights
from ..model_agent.artifacts import arrays,read,seal,verify
from ..model_agent.contracts import digest,cell_line_key,NORMALIZATION,protected
from ..model_agent.experts import Registry
from ..round11 import freeze
from ..utils import seed_all,write_json,file_sha256
from .agents import TEACHERS


def source_lines(meta):
    return np.array([cell_line_key(r.context,r.get('cell_line','unknown')) for _,r in meta.iterrows()])


def crossfit_responses(data,control,assets,fit,steps,seed):
    """Shared fit covariates; each numerical teacher excludes the predicted line's labels.

    This is response-head cross-fitting, NOT a claim that every unsupervised
    covariate transform was refitted for each training-line subfold.
    """
    fit=np.asarray(fit,int);line=source_lines(data['meta']);unique=sorted(set(line[fit]))
    if len(unique)<2:raise ValueError('Cross-fitted teacher features require >=2 source-fit cell lines')
    result=np.zeros((len(fit),len(TEACHERS),len(data['genes'])),np.float32);audit=[]
    for held in unique:
        train=fit[line[fit]!=held];positions=np.flatnonzero(line[fit]==held);query=fit[positions]
        # New feature projections and response basis use response-training rows only.
        t,_,_=target_representation(data,assets,train,'local');t=t[data['pert_idx']]
        b,_=projection(control,train,16);x=design(b,t)
        w=row_weights(data['meta'].iloc[train]);mean=w@x[train];sd=np.maximum(np.sqrt(w@((x[train]-mean)**2)),.1)
        x=(x-mean)/sd
        for j,name in enumerate(TEACHERS):
            rank=16 if name=='factor_rank16' else 32;basis,_=response_basis(data,train,rank)
            scale=max(float(np.sqrt(w@np.square(data['delta'][train]).mean(1))),1e-4)
            y=data['delta'][train]@basis.T/scale
            if name=='string_ridge':
                coef,bias=fit_ridge(x[train],y,w.copy(),1.)
                pred=(x[query]@coef+bias)@basis*scale
            else:
                seed_all(seed+j);model=FactorizedResponse(t.shape[1],b.shape[1],1,len(basis),True,False,32)
                opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0005,weight_decay=.01)
                tx=torch.tensor(t);bx=torch.tensor(b);ty=torch.tensor(y)
                for step in range(steps):
                    pos=np.random.default_rng(seed+step).choice(len(train),min(128,len(train)),p=w)
                    ix=train[pos];opt.zero_grad(set_to_none=True)
                    loss=(model(tx[ix],bx[ix],torch.zeros(len(ix),dtype=torch.long))-ty[pos]).square().mean()
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite cross-fitted teacher')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                model.eval()
                with torch.no_grad():pred=model(tx[query],bx[query],torch.zeros(len(query),dtype=torch.long)).numpy()@basis*scale
            result[positions,j]=pred
        audit.append({'held_training_cell_line':held,'response_fit_cell_lines':sorted(set(line[train])),
            'response_fit_rows':train.tolist(),'predicted_rows':query.tolist(),
            'head_fit_steps':steps,'ridge_alpha':1.,'response_basis_fit_rows':train.tolist()})
        print(f'ROUND19 TEACHER OOF line={held} rows={len(query)}',flush=True)
    return result,audit


def deployed_predictions(expert,data,latent,rows):
    """Batch execution of the actual exported P4 weights, in their original basis."""
    e=expert;tr=arrays(e.path/'transforms.npz');bg=arrays(e.path/'background_transform.npz')
    b=((latent[rows]-bg['mean'])@bg['axes'].T/bg['scale']).astype(np.float32)
    lookup={str(p):i for i,p in enumerate(e.stats['targets'])}
    raw=e.stats['relations'][[lookup[str(data['meta'].iloc[i].perturbation)] for i in rows]]
    t=np.c_[((raw-tr['mean'])@tr['axes'].T/tr['scale']).astype(np.float32),(raw.sum(1)>0).astype(np.float32)]
    if e.card['method']=='string_ridge':
        saved=arrays(e.path/'model.npz');coeff=((design(b,t)-tr['xmean'])/tr['xscale'])@saved['coef']+saved['intercept']
    else:
        from ..round15_heads import ExperimentalHead
        saved=torch.load(e.path/'model.pt',map_location='cpu',weights_only=True)
        if saved['spec']['use_id'] or saved['spec']['reference']:raise ValueError('Unsupported teacher response head')
        model=ExperimentalHead(saved['target_dim'],saved['background_dim'],saved['n_ids'],saved['rank'],saved['spec'],saved['seed'])
        model.load_state_dict(saved['state']);model.eval()
        with torch.no_grad():coeff=model(torch.tensor(t),torch.tensor(b),torch.zeros(len(rows),dtype=torch.long),torch.zeros((len(rows),1))).numpy()
    basis=arrays(e.path/'response_basis.npz')
    return (coeff@basis['basis']*basis['scale']).astype(np.float32)


def build_cache(data,fold,pool,dest,teacher_steps=200,model_identity=None):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);cfg=fold['configs']['expanded']
    registry=Registry(pool);part=cfg['partition'];fit=np.array(part['fit'],int);cal=np.array(part['calibration'],int);query=np.array(fold['query_rows'],int)
    line=source_lines(data['meta']);used=np.r_[fit,cal]
    if set(line[query])&set(line[used]) or set(query)&set(used):raise ValueError('Student held background leaked into source')
    if data['audit'].get('normalization')!=NORMALIZATION or any(protected(v) for v in line[np.r_[used,query]]):raise ValueError('Unknown normalization or reserved cell line')
    identity={'data':data['audit']['fingerprint'],'genes':data['genes'].tolist(),'partition':part,
        'query_rows':query.tolist(),'registry':registry.fingerprint,'teacher_steps':teacher_steps,
        'encoder_sha256':file_sha256(cfg['encoder']),'relations_sha256':file_sha256(cfg['relations']),
        'model_identity':model_identity,'code':digest({p.name:file_sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})}
    freeze(dest/'manifest.json',identity)
    if (dest/'COMPLETE.json').exists():verify(dest);return dest
    background_manifest=read(Path(cfg['encoder']).parent/'manifest.json')
    if background_manifest.get('model_config')!=file_sha256(Path(cfg['model']['model_id'])/'config.json'):
        raise ValueError('Parent B2 model configuration changed')
    torch.set_num_threads(2);latent,norm=encode_background(data,cfg);b,bt=projection(latent,fit,16)
    assets=load_relations(cfg['relations'],data);t,tt,_=target_representation(data,assets,fit,'local');t=t[data['pert_idx']]
    basis,ba=response_basis(data,fit,32);scale=float(norm['scale'])
    train_predictions,audit=crossfit_responses(data,latent,assets,fit,teacher_steps,cfg['seed'])
    pred=np.zeros((len(data['meta']),len(TEACHERS),len(data['genes'])),np.float32);pred[fit]=train_predictions
    eval_rows=np.r_[cal,query]
    teacher_audit=[]
    for j,name in enumerate(TEACHERS):
        e=registry.experts['expanded_'+name]
        if set(e.provenance['fit_query_ids'])!=set(data['meta'].iloc[fit].row_id) or set(e.provenance['calibration_query_ids'])!=set(data['meta'].iloc[cal].row_id):
            raise ValueError('Teacher/student fit-cal scopes differ')
        if set(e.provenance['fit_cell_lines']+e.provenance['calibration_cell_lines'])&set(line[query]):raise ValueError('Teacher held-line leakage')
        pred[eval_rows,j]=deployed_predictions(e,data,latent,eval_rows)
        teacher_audit.append({'teacher':name,'bundle_sha256':file_sha256(e.path/'COMPLETE.json'),
            'inference_rows':eval_rows.tolist(),'provenance':e.provenance})
    texts=[]
    for _,r in data['meta'].iterrows():
        texts.append(f'Predict signed gene-expression changes. Human CRISPRi target {r.perturbation}; cell {r.context}; stimulus {r.get("stimulus","unknown")}.')
    contexts=np.unique(data['meta'].context);targets=np.unique(data['meta'].perturbation)
    # No held response labels are present in the worker cache.
    np.savez_compressed(dest/'features.npz',control=b,target=t,teachers=pred@basis.T/scale,
        basis=basis,response_scale=np.float32(scale),genes=data['genes'],row_ids=data['meta'].row_id.to_numpy(dtype='U'),
        query=query,context_codes=np.searchsorted(contexts,data['meta'].context),target_codes=np.searchsorted(targets,data['meta'].perturbation),
        prompt_texts=np.array(texts,dtype='U'),fit_weights=row_weights(data['meta'].iloc[fit]),calibration_weights=row_weights(data['meta'].iloc[cal]))
    np.savez_compressed(dest/'source_labels.npz',fit=fit,calibration=cal,fit_delta=data['delta'][fit]/scale,calibration_delta=data['delta'][cal]/scale)
    np.savez_compressed(dest/'teacher_query_predictions.npz',prediction=pred[query],genes=data['genes'],row_ids=data['meta'].iloc[query].row_id.to_numpy(dtype='U'))
    np.savez_compressed(dest/'transforms.npz',basis=basis,**{'background_'+k:v for k,v in bt.items()},**{'target_'+k:v for k,v in tt.items()})
    write_json(dest/'lineage.json',{'training_response_head_oof':audit,'basis_audit':ba,'held_teachers':teacher_audit,
        'shared_covariate_scope':'Parent B2 encoder fitted on student source only; nested teacher PCA/response heads exclude predicted training line',
        'external_native_teachers':{'state_st':'NOT_CONNECTED','txpert':'NOT_CONNECTED'},'test_evaluated':False})
    seal(dest);return dest
