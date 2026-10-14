"""Numerical grading; no LLM judges or candidate-reported scores."""
import numpy as np
import pandas as pd


def macro(frame,column):
    return float(frame.groupby(['context','target'])[column].mean().groupby('context').mean().mean())


def score(prediction,truth,background,mask,metadata):
    f=metadata.copy();error=np.square(prediction-truth)
    f['mse']=error.mean(1);f['zero_mse']=np.square(truth).mean(1);f['background_mse']=np.square(background-truth).mean(1)
    f['prediction_norm']=np.linalg.norm(prediction,axis=1);f['truth_norm']=np.linalg.norm(truth,axis=1)
    counts=mask.sum(1);f['strong_mse']=np.divide((error*mask).sum(1),counts,out=np.full(len(f),np.nan),where=counts>0)
    f['strong_genes']=counts;f['residual_mse']=np.square((prediction-background)-(truth-background)).mean(1)
    f['residual_norm']=np.linalg.norm(prediction-background,axis=1);f['truth_residual_norm']=np.linalg.norm(truth-background,axis=1)
    residual=truth-background;pr=prediction-background;gap=[];direction=[]
    def cosine(a,b):return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
    for i in range(len(f)):
        own=cosine(pr[i],residual[i]);peers=np.flatnonzero(metadata.context.eq(metadata.iloc[i].context) & metadata.target.ne(metadata.iloc[i].target))
        gap.append(own-np.mean([cosine(pr[i],residual[j]) for j in peers]) if len(peers) else np.nan)
        direction.append(own)
    f['specificity_gap']=gap;f['residual_cosine']=direction
    return f


def fitness(frame,baseline):
    mse=macro(frame,'mse');b=macro(baseline,'mse');gain=1-mse/max(b,1e-12)
    amplitude=macro(frame,'prediction_norm')/max(macro(frame,'truth_norm'),1e-12)
    gap=macro(frame,'specificity_gap');strong=macro(frame,'strong_mse')
    wins=frame.groupby(['context','target']).mse.mean()<baseline.groupby(['context','target']).mse.mean()
    values={'mse':mse,'gain_over_fixed':gain,'target_win_fraction':float(wins.mean()),'amplitude_ratio':amplitude,
        'residual_amplitude_ratio':macro(frame,'residual_norm')/max(macro(frame,'truth_residual_norm'),1e-12),
        'specificity_gap':gap,'strong_mse':strong,'contexts':int(frame.context.nunique()),'targets':int(frame.target.nunique())}
    # Smooth exploration score; separate binary promotion prevents tiny zero-like wins.
    combined=100*gain+.1*(gap if np.isfinite(gap) else 0.)-.05*max(0.,.05-amplitude)
    values['biological_success']=bool(gain>=.01 and amplitude>=.05 and np.isfinite(gap) and gap>0 and np.isfinite(strong))
    return combined,{k:(float(v) if isinstance(v,(float,np.floating)) and np.isfinite(v) else None if isinstance(v,(float,np.floating)) else v) for k,v in values.items()}
