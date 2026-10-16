"""Descriptive paired target-cluster sensitivity; no independent-test claim."""
import numpy as np
import pandas as pd


def cluster_gain(frame,baseline,seed=17,draws=2000):
    keys=['row_id','context','target']
    joined=frame.merge(baseline[keys+['mse']],on=keys,suffixes=('_method','_baseline'),validate='one_to_one')
    if len(joined)!=len(frame) or len(frame)!=len(baseline):raise ValueError('Paired statistical panels differ')
    grouped=joined.groupby(['context','target'])[['mse_method','mse_baseline']].mean()
    a=grouped.mse_method.unstack('target');b=grouped.mse_baseline.unstack('target')
    x=a.to_numpy();y=b.to_numpy();rng=np.random.default_rng(seed);samples=[]
    def gain(left,right):return 1-float(np.nanmean(left,axis=1).mean())/max(float(np.nanmean(right,axis=1).mean()),1e-12)
    for _ in range(draws):
        idx=rng.integers(x.shape[1],size=x.shape[1]);xx=x[:,idx];yy=y[:,idx]
        if np.any(np.all(np.isnan(xx),axis=1)):continue
        samples.append(gain(xx,yy))
    loo=[]
    for i,t in enumerate(a.columns):
        idx=np.arange(x.shape[1])!=i
        if idx.any() and not np.any(np.all(np.isnan(x[:,idx]),axis=1)):loo.append({'omitted_target':str(t),'gain':gain(x[:,idx],y[:,idx])})
    return {'gain':gain(x,y),'context_target_wins':int((grouped.mse_method<grouped.mse_baseline).sum()),'context_target_pairs':len(grouped),
        'distinct_targets':x.shape[1],'contexts':len(a),'bootstrap_valid_draws':len(samples),'exploratory_interval_95':np.quantile(samples,[.025,.975]).tolist() if samples else None,
        'leave_one_target_out':loo,'interpretation':'Development target-cluster sensitivity, preserving cross-context target dependence; not corrected for algorithm selection and not an independent-test confidence interval'}
