"""Identifier-frozen cell-line folds. No response-driven case selection."""
from pathlib import Path
import numpy as np
import pandas as pd
from .contracts import NORMALIZATION, cell_line_key, digest, protected
from ..round8 import validate_partition

DEVELOPMENT=('A549_IFNG','BXPC3_IFNG','HAP1_IFNG','MCF7_IFNG')


def lines(meta):
    return np.array([cell_line_key(r.context,r.get('cell_line','unknown')) for _,r in meta.iterrows()])


def holdout_targets(meta):
    """Identifier-only target holdout, shared across all conditions and folds."""
    targets=sorted(set(meta.loc[meta.context.isin(DEVELOPMENT)&meta.split.eq('train'),'perturbation']))
    if len(targets)<4:raise ValueError('Need >=4 IFNG targets to include unseen-target queries')
    order=sorted(targets,key=lambda t:digest(['nested_target_holdout',str(t)]))
    return order[:max(1,len(order)//5)]


def cases(meta,context,targets,count):
    rows=meta.loc[meta.context.eq(context)&meta.split.eq('train')]
    groups=[sorted(set(rows.perturbation)&set(targets)),sorted(set(rows.perturbation)-set(targets))]
    groups=[sorted(g,key=lambda t:digest(['case',str(t)])) for g in groups]
    chosen=[]
    while any(groups) and len(chosen)<count:
        for group in groups:
            if group and len(chosen)<count:chosen.append(group.pop(0))
    if len(chosen)<min(4,count):raise ValueError('Too few distinct targets in '+context)
    # Preserve all batches/replicates of a selected target, not a response-chosen row.
    return rows.index[rows.perturbation.isin(chosen)].tolist()


def partition(data,historical_rows,excluded,query_rows,held_targets,variant,final=False):
    m=data['meta'];ll=lines(m);excluded={cell_line_key(c) for c in excluded}
    eligible=m.split.eq('train').to_numpy() & ~np.isin(ll,list(excluded))
    eligible &= ~m.perturbation.isin(held_targets).to_numpy()
    eligible &= ~np.array([protected(str(c)) for c in m.context]) & ~np.array([protected(c) for c in ll])
    historical=np.arange(len(m))<historical_rows
    # Complete context-target units (including batches) get a stable role.
    roles=np.array([int(digest(['source_split',str(c),str(t)])[:8],16)%20 for c,t in zip(m.context,m.perturbation)])
    cal=np.flatnonzero(eligible & historical & (roles<3))
    diagnostic=np.flatnonzero(eligible & historical & (roles==3))
    fit=np.flatnonzero(eligible & (roles>=4 if final else roles>=3) & (historical if variant=='reference' else True))
    outer=diagnostic if final else np.asarray(query_rows,int)
    part={k:v.tolist() for k,v in zip(('fit','calibration','outer'),(fit,cal,outer))}
    validate_partition(data,part)
    for role in ('fit','calibration'):
        if set(ll[part[role]])&excluded:raise ValueError('Whole-cell-line exclusion failed')
        if set(m.iloc[part[role]].perturbation)&set(held_targets):raise ValueError('Target holdout failed')
    return part


def audit_public_cache(cache,excluded):
    """Use primary Tabula training cells only; reject explicitly labelled lines."""
    from ..round11 import verify_files
    verify_files(cache)
    m=pd.read_csv(Path(cache)/'train_metadata.csv')
    if not m.dataset.astype(str).str.startswith('ts_').all():raise ValueError('Public fitting pool must be primary Tabula training cells')
    if 'cell_line' in m and {cell_line_key(c) for c in m.cell_line.dropna()}&{cell_line_key(c) for c in excluded}:
        raise ValueError('Held-out cell line found in public background cache')
    return {'train_cells':len(m),'sources':sorted(set(m.dataset)),
        'claim':'Supervised cell-line holdout with shared primary-cell public priors; no claim of biological-prior independence'}


def source_summary(frame):
    # Equal backgrounds, targets, then technical rows. Never optimizer seeds as samples.
    unit=frame.groupby(['expert_id','context','target']).mse.mean()
    fixed=unit.groupby(level=0).apply(lambda x:x.groupby(level=1).mean().mean())
    best=min(fixed.index,key=lambda e:(fixed[e],e))
    wide=frame.pivot(index=['query_id','context','target'],columns='expert_id',values='mse')
    oracle=wide.min(axis=1).groupby(level=['context','target']).mean().groupby(level='context').mean().mean()
    gain=float((fixed[best]-oracle)/max(fixed[best],1e-12))
    winners=wide.idxmin(axis=1).value_counts().to_dict()
    by_context=unit.groupby(level=['expert_id','context']).mean().unstack(0)
    gaps=[]
    for target in sorted(set(frame.target)):
        part=frame.loc[~frame.target.eq(target)]
        if part.empty:continue
        u=part.groupby(['expert_id','context','target']).mse.mean()
        fs=u.groupby(level=0).apply(lambda x:x.groupby(level=1).mean().mean()).min()
        ow=part.pivot(index=['query_id','context','target'],columns='expert_id',values='mse').min(axis=1)
        og=ow.groupby(level=['context','target']).mean().groupby(level='context').mean().mean()
        gaps.append(float((fs-og)/max(fs,1e-12)))
    return {'fixed_source_mse':{k:float(v) for k,v in fixed.items()},'best_fixed':best,
        'oracle_macro_mse':float(oracle),'oracle_relative_gap':gain,'winner_counts':winners,
        'best_fixed_per_context_mse':float(by_context.min(axis=1).mean()),
        'leave_one_target_out_gap_min':min(gaps) if gaps else None,
        'contexts':sorted(set(frame.context)),'distinct_targets':int(frame.target.nunique()),
        'note':'Inner OOF diagnostics only; oracle excluded from planner observations; technical rows and seeds are not biological replication'}
