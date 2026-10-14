"""Return original-unit, ordered gene mean-deltas. Only evolve the marked method."""
import numpy as np
import torch

# EVOLVE-BLOCK-START
def fit_predict(data, budget):
    """The whole method can evolve: features, background, loss, architecture, optimizer.

    train_y contains authorized source responses, never query truth. train_x and
    query_x are train-fit control/STRING projections. task repeat also supplies
    experimental Rep1 support. reference/wrong_reference are same legal source pool.
    step count is a maximum; closed-form algorithms need not use gradient steps.
    """
    y=data['train_y'];mu=y.mean(0);x=data['train_x'];qx=data['query_x']
    support=data['train_support'];qs=data['query_support']
    if str(data['task'])=='repeat':
        mu1=support.mean(0);support=support-mu1;qs=qs-mu1
    else:
        # Legal donors; unknown targets have explicit zero support.
        mu=data['background_reference'][0] if len(qx) else mu
    _,_,vh=np.linalg.svd(y-y.mean(0),full_matrices=False);basis=vh[:min(16,len(vh))]
    scale=max(float(np.sqrt(np.square(y).mean())),1e-4)
    xx=np.c_[x,support@basis.T/scale];qq=np.c_[qx,qs@basis.T/scale]
    mean=xx.mean(0);sd=np.maximum(xx.std(0),.1);xx=(xx-mean)/sd;qq=(qq-mean)/sd
    design=np.c_[xx,np.ones(len(xx))];query=np.c_[qq,np.ones(len(qq))]
    penalty=np.eye(design.shape[1]);penalty[-1,-1]=0
    coef=np.linalg.solve(design.T@design+10.*penalty,design.T@((y-y.mean(0))@basis.T))
    # This initial residual ridge is a starting point, not an assumed best method.
    prediction=(query@coef)@basis+y.mean(0)
    return prediction.astype(np.float32)
# EVOLVE-BLOCK-END
