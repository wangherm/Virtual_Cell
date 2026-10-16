"""Source construction only. Candidate modules are never imported by the controller."""
import ast
import hashlib
from pathlib import Path
from ..round22.evaluator import START,END

WINNER='e8b08448d8e7d92c8ac0c4c511ebec117c0f1e3f8b573a59582761d900568314'


def module_body(code):
    if len(code.encode())>100000:raise ValueError('Algorithm module exceeds 100 kB')
    # The entire submitted Python module is editable, including helper imports.
    # EVOLVE markers are comments for Shinka, not byte-for-byte protected shells.
    tree=ast.parse(code)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit_predict']
    if len(functions)!=1:raise ValueError('Exactly one synchronous fit_predict entrypoint required')
    return ''.join(line for line in code.splitlines(keepends=True) if line.strip() not in (START,END)).strip()+'\n'


ALIGN='''
def _r24_align(data):
    d=dict(data)
    d['source_context']=np.r_[d['train_source_context'],d['calibration_source_context']]
    d['query_context']=d['query_source_context']
    return d
'''


def winner_module(code,repair=False,disabled=(),no_correction=False,uniform=False):
    if hashlib.sha256(code.encode()).hexdigest()!=WINNER:raise ValueError('Unsupported winner version: ablation anchors are pinned to Round23')
    body=code.split(START)[1].split(END)[0]
    def replace(old,new):
        nonlocal body
        if body.count(old)!=1:raise ValueError('Frozen winner anchor mismatch: '+old[:60])
        body=body.replace(old,new)
    if repair:
        replace('def fit_predict(data, budget):','def fit_predict(data, budget):\n    data = _r24_align(data)')
        replace('gene_lower_to_idx = {g.lower(): i for i, g in enumerate(genes_list)}',
                "gene_lower_to_idx = {g.lower(): i for i, g in enumerate(genes_list)}\n    gene_lower_to_idx.update({str(s).lower():i for i,s in enumerate(data['gene_symbols']) if str(s)})")
        body=ALIGN+body
    if disabled:
        replace('    A_oof_full = np.column_stack([pred.ravel() for pred in oof_preds])',
                f"    _r24_active = [i for i,n in enumerate(base_names[:-1]) if n not in {tuple(disabled)!r}]\n    oof_preds = [oof_preds[i] for i in _r24_active]\n    base_names = [base_names[i] for i in _r24_active] + ['const']\n    A_oof_full = np.column_stack([pred.ravel() for pred in oof_preds])")
        replace('    pred = np.zeros((n_query, G), dtype=np.float64)',
                '    final_preds = [final_preds[i] for i in _r24_active] + [final_preds[-1]]\n    pred = np.zeros((n_query, G), dtype=np.float64)')
    if no_correction:replace('    valid_q = (query_tidx >= 0)','    valid_q = np.zeros(n_query, dtype=bool)')
    if uniform:
        body+='''
def _nnls(A,b,iters=2000):
    w=np.zeros(A.shape[1],np.float64)
    if len(w)>1:w[:-1]=1.0/(len(w)-1)
    return w
'''
    return START+'\n'+body.strip()+'\n'+END+'\n'


def simple_module(alpha_grid=(0.,.01,.03,.1,.2,.3,.5,.7,1.,1.5),ridge=False,penalty=10.):
    body=ALIGN+f'''
def fit_predict(data,budget):
    data=_r24_align(data)
    y=np.r_[data['train_y'],data['calibration_y']].astype(np.float64)
    s=np.r_[data['train_support'],data['calibration_support']].astype(np.float64)
    q=data['query_support'].astype(np.float64)
    ctx=data['source_context'];qctx=data['query_context']
    pred=np.zeros_like(q)
    for context in np.unique(qctx):
        fit=ctx==context
        if not fit.any():fit=np.ones(len(y),bool)
        yy=y[fit];ss=s[fit];mu=yy.mean(0);sm=ss.mean(0)
        alpha=min({tuple(alpha_grid)!r},key=lambda a:float(np.square(mu+a*(ss-sm)-yy).mean()))
        selected=qctx==context
        pred[selected]=mu+alpha*(q[selected]-sm)
    if {ridge!r}:
        # Shared low-rank calibration, fitted on fit labels and selected on calibration labels.
        tx=data['train_support'].astype(np.float64);ty=data['train_y'].astype(np.float64)
        cx=data['calibration_support'].astype(np.float64);cy=data['calibration_y'].astype(np.float64)
        mean=tx.mean(0);ym=ty.mean(0)
        _,_,vh=np.linalg.svd(tx-mean,full_matrices=False);basis=vh[:min(8,len(vh))]
        x=(tx-mean)@basis.T;coef=np.linalg.solve(x.T@x+{float(penalty)!r}*np.eye(x.shape[1]),x.T@(ty-ym))
        rcal=((cx-mean)@basis.T)@coef+ym
        basecal=np.tile(ym,(len(cx),1))
        beta=min((0.,.1,.3,.5,.7,1.),key=lambda b:float(np.square(b*rcal+(1-b)*basecal-cy).mean()))
        pred=(1-beta)*pred+beta*(((q-mean)@basis.T)@coef+ym)
    return pred.astype(np.float32)
'''
    return START+'\n'+body+END+'\n'


def methods(code):
    targets=('target_resid','target_self','target_svd','direct_target','target_support_ridge')
    return {'winner_original':winner_module(code), 'winner_repaired':winner_module(code,repair=True),
        'simple_support':simple_module(),'simple_shared_ridge':simple_module(ridge=True),
        'no_target_branches':winner_module(code,repair=True,disabled=targets),
        'no_low_rank_branches':winner_module(code,repair=True,disabled=('target_svd','xridge')),
        'uniform_ensemble':winner_module(code,repair=True,uniform=True),
        'no_final_correction':winner_module(code,repair=True,no_correction=True)}


TRACE='''
_r24_method=fit_predict
def fit_predict(data,budget):
    import sys,json
    audit={'calls':{}}
    def trace(frame,event,arg):
        name=frame.f_code.co_name
        if event=='call':
            if name.startswith('_fit_') or name=='_extract_aux_profiles':audit['calls'][name]=audit['calls'].get(name,0)+1
            return trace if name in ('fit_predict','_extract_aux_profiles') else None
        if event=='return':
            v=frame.f_locals
            if name=='fit_predict':
                for key in ('base_names','ref_contexts','query_contexts'):
                    if key in v:audit[key]=list(dict.fromkeys(map(str,v[key])))
                if 'target_gene_idx' in v:audit['target_output_index_matches']=int(np.sum(v['target_gene_idx']>=0))
                if 'w_global' in v:audit['weights_global']=np.asarray(v['w_global']).tolist()
                if 'w_context' in v:audit['weights_context']={str(k):np.asarray(w).tolist() for k,w in v['w_context'].items()}
                if 'final_preds' in v and 'support_query' in v:
                    audit['branch_difference_from_support']=[float(np.linalg.norm(p-v['support_query'])) for p in v['final_preds']]
            if name=='_extract_aux_profiles' and 'keep' in v:
                keep=v['keep'];matched=(v.get('aux_tidx',np.full(len(keep),-1))>=0)
                ds=np.asarray(data.get('aux_dataset',np.repeat('unknown',len(keep))))
                audit['aux_usage']={str(k):{'input':int(np.sum(ds==k)),'condition_kept':int(np.sum((ds==k)&keep)),
                    'target_matched':int(np.sum((ds==k)&keep&matched))} for k in np.unique(ds)}
        return trace
    sys.settrace(trace)
    try:prediction=_r24_method(data,budget)
    finally:sys.settrace(None)
    with open('output/audit.json','w') as f:json.dump(audit,f,allow_nan=False)
    return prediction
'''


def compose(module,audit=True):
    body=module_body(module)
    return 'import numpy as np\nimport torch\n'+body+(TRACE if audit else '')
