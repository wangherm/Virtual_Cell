"""Label-safe reference banks and target-fold out-of-fold Ridge predictions."""
import hashlib

import numpy as np

from .round12 import design, fit_ridge
from .round14 import target_representation
from .selection import row_weights


def reference_bank(data, fit, raw_profiles, mode):
    """All donors are fit rows. Training queries exclude their own biological unit."""
    meta = data['meta']; fit = np.asarray(fit, int)
    groups = meta.iloc[fit].reset_index(drop=True).groupby(['context','perturbation'], sort=True).indices
    units = [(context, target, fit[ix]) for (context,target),ix in groups.items()]
    values = np.stack([data['delta'][ix].mean(0) for _,_,ix in units])
    contexts = np.array([c for c,_,_ in units]); names = np.array([p for _,p,_ in units])
    vocab = {str(p):i for i,p in enumerate(data['perturbations'])}
    profiles = np.asarray(raw_profiles, np.float64)
    profiles = profiles / np.maximum(np.linalg.norm(profiles,axis=1,keepdims=True),1e-12)
    result = np.zeros_like(data['delta']); available = np.zeros(len(meta), bool); audit = []
    for (context,target), rows in meta.groupby(['context','perturbation'],sort=True).indices.items():
        if mode == 'reference':
            eligible = (names==target) & (contexts!=context)
        elif mode == 'neighbor':
            eligible = names!=target
        elif mode == 'shared':
            eligible = (names!=target) & (contexts==context)
        else: raise ValueError('Unknown reference mode')
        donor = np.flatnonzero(eligible); weights = None
        if mode == 'neighbor' and len(donor):
            candidate_names = sorted(set(names[donor]))
            sims = np.array([profiles[vocab[target]]@profiles[vocab[p]] for p in candidate_names])
            # Fixed top five, nonnegative similarity; deterministic stable ties.
            selected = np.argsort(-sims,kind='stable')[:5]
            positive = {candidate_names[j]:float(sims[j]) for j in selected if sims[j]>1e-8}
            donor = np.array([j for j in donor if names[j] in positive], int)
            if len(donor):
                weights = np.array([positive[names[j]] / np.sum(names[donor]==names[j]) for j in donor])
        found = bool(len(donor))
        if not found:
            # Never put a query target's own labels in its fallback, including fit queries.
            donor = np.flatnonzero(names!=target)
            if mode == 'reference': donor = donor[contexts[donor]!=context]
        if len(donor):
            if weights is None:
                # Equal contexts, then equal targets within context.
                weights = np.array([1./(len(set(contexts[donor]))*np.sum(contexts[donor]==contexts[j])) for j in donor])
            result[rows] = np.average(values[donor],axis=0,weights=weights)
        available[rows] = found
        donor_rows = np.concatenate([units[j][2] for j in donor]).tolist() if len(donor) else []
        if set(rows) & set(donor_rows): raise ValueError('Reference includes query response')
        if mode in ('neighbor','shared') and any(names[donor]==target): raise ValueError('Reference target leakage')
        audit.append({'context':context,'target':target,'mode':mode,'reference_available':found,
                      'donor_units':donor.tolist(),'donor_row_count':len(donor_rows)})
    lineage={'units':[{'context':c,'target':p,'rows':ix.tolist()} for c,p,ix in units],'queries':audit}
    return result.astype(np.float32), available, lineage


def target_folds(data, fit, count=3):
    names = data['meta'].iloc[fit].perturbation.to_numpy()
    ordered = sorted(set(names),key=lambda p:hashlib.sha256(('round15:oof:'+p).encode()).hexdigest())
    if len(ordered)<count: raise ValueError('Insufficient fit targets for OOF Ridge')
    assignment = {p:i%count for i,p in enumerate(ordered)}
    side = np.array([assignment[p] for p in names])
    return [(np.asarray(fit)[side!=j],np.asarray(fit)[side==j]) for j in range(count)]


def ridge_reference(data, assets, background, fit, query):
    """Full-gene Ridge avoids a response basis fitted on held-out inner labels.

    The frozen background representation uses controls, never perturbation responses.
    Target PCA and design scaling are refitted for each target fold. Alpha is fixed
    at .1 before outcomes are seen; outer/calibration labels never enter the bank.
    """
    output = np.zeros_like(data['delta']); audit = []
    for train, rows in target_folds(data,fit)+[(np.asarray(fit),np.asarray(query))]:
        targets,_,_ = target_representation(data,assets,train,'local')
        x = design(background,targets[data['pert_idx']],interactions=True)
        weights = row_weights(data['meta'].iloc[train])
        mean = weights@x[train]; scale = np.maximum(np.sqrt(weights@((x[train]-mean)**2)),.1)
        x = ((x-mean)/scale).astype(np.float32)
        coefficient,bias = fit_ridge(x[train],data['delta'][train],weights,.1)
        output[rows] = (x[rows]@coefficient+bias).astype(np.float32)
        if set(train)&set(rows): raise ValueError('Ridge OOF row overlap')
        audit.append({'fit':train.tolist(),'query':rows.tolist(),'alpha':.1,'response_basis':'none; full-gene outputs'})
    return output,audit
