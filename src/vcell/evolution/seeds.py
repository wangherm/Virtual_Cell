"""Strong starting point and disclosed random-search grammar."""
from pathlib import Path
import numpy as np


def wrap(repo,body):
    code=(Path(repo)/'experiments/round22_evolve/initial.py').read_text(encoding='utf-8')
    before,rest=code.split('# EVOLVE-BLOCK-START');_,after=rest.split('# EVOLVE-BLOCK-END')
    return before+'# EVOLVE-BLOCK-START\n'+body+'\n# EVOLVE-BLOCK-END'+after


def seed(repo,baseline=None):
    return wrap(repo,'''def fit_predict(data,budget):
    chosen=BASELINE_NAME
    y=np.r_[data['train_y'],data['calibration_y']]
    a=np.r_[data['train_support'],data['calibration_support']]
    if chosen is None:chosen='demeaned_support' if str(data['task'])=='repeat' else 'zero'
    if chosen=='zero':return np.zeros((len(data['query_ids']),len(data['genes'])),np.float32)
    if chosen=='background_mean':return np.tile(y.mean(0),(len(data['query_ids']),1)).astype(np.float32)
    mu=y.mean(0);mu1=a.mean(0)
    grid=(0.,.01,.03,.1,.3,.7,1.)
    if chosen=='demeaned_support':
        alpha=min(grid,key=lambda v:float(np.square(mu+v*(a-mu1)-y).mean()))
        return (mu+alpha*(data['query_support']-mu1)).astype(np.float32)
    if chosen=='calibrated_support':
        alpha=min(grid,key=lambda v:float(np.square(v*a-y).mean()))
        return (alpha*data['query_support']).astype(np.float32)
    tx=data['train_x'];cx=data['calibration_x'];ty=data['train_y'];cy=data['calibration_y'];qx=data['query_x']
    design=np.c_[tx,np.ones(len(tx))];cal=np.c_[cx,np.ones(len(cx))];query=np.c_[qx,np.ones(len(qx))]
    if chosen=='raw_control_STRING_ridge':
        scores=[]
        for penalty in (.1,1.,10.,100.):
            reg=np.eye(design.shape[1])*penalty;reg[-1,-1]=0
            coef=np.linalg.solve(design.T@design+reg,design.T@ty)
            scores.append((float(np.square(cal@coef-cy).mean()),coef))
        return (query@min(scores,key=lambda v:v[0])[1]).astype(np.float32)
    rank=16 if chosen=='raw_control_STRING_factor16' else 32
    _,_,basis=np.linalg.svd(ty-ty.mean(0),full_matrices=False);b=basis[:min(rank,len(basis))];scores=[]
    for penalty in (.1,1.,10.,100.):
        reg=np.eye(design.shape[1])*penalty;reg[-1,-1]=0
        coef=np.linalg.solve(design.T@design+reg,design.T@((ty-ty.mean(0))@b.T))
        scores.append((float(np.square((cal@coef)@b+ty.mean(0)-cy).mean()),coef))
    return ((query@min(scores,key=lambda v:v[0])[1])@b+ty.mean(0)).astype(np.float32)
'''.replace('BASELINE_NAME',repr(baseline)))


def random_method(repo,number):
    rng=np.random.default_rng(number);rank=int(rng.choice([4,8,16,32]));penalty=float(rng.choice([.1,1.,10.,100.,1000.]));mix=float(rng.choice([0.,.03,.1,.3,.7,1.]));use_aux=bool(rng.integers(2))
    return wrap(repo,f'''def fit_predict(data,budget):
    y=data['train_y'].astype(np.float64);x=data['train_control'].astype(np.float64);q=data['query_control'].astype(np.float64)
    if {use_aux!r} and len(data.get('aux_delta',[])):
        y=np.r_[y,data['aux_delta']];x=np.r_[x,data['aux_baseline']]
    ym=y.mean(0);xm=x.mean(0);sd=np.maximum(x.std(0),.1)
    _,_,vh=np.linalg.svd((x-xm)/sd,full_matrices=False);b=vh[:min({rank},len(vh))]
    xx=np.c_[((x-xm)/sd)@b.T,np.ones(len(x))];qq=np.c_[((q-xm)/sd)@b.T,np.ones(len(q))]
    reg=np.eye(xx.shape[1])*{penalty};reg[-1,-1]=0
    pred=qq@np.linalg.solve(xx.T@xx+reg,xx.T@y)
    if str(data['task'])=='repeat':
        a=np.r_[data['train_support'],data['calibration_support']];ref=data['query_support']-a.mean(0)+np.r_[data['train_y'],data['calibration_y']].mean(0)
    else:ref=data['query_support']
    return ({mix}*pred+(1.-{mix})*ref).astype(np.float32)
''')
