"""Explicit control-only query and allowlisted decision contracts."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

import numpy as np

NORMALIZATION='mean(log1p(10000*counts/full_library))'
FEATURES=('control_distance','target_coverage','reference_count','fit_target_seen','context_seen')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def safe_name(value):
    if not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in value):
        raise ValueError('Simple identifier required: '+str(value))
    return value


def protected(context):
    return any(name in context.upper() for name in ('HT29','JURKAT'))


def context_key(value):return value.strip().upper()


@dataclass(frozen=True)
class Query:
    query_id: str
    context: str
    target: str
    stimulus: str
    genes: tuple
    control: tuple
    species: str='human'
    modality: str='CRISPRi'
    output: str='mean_delta'
    normalization: str=NORMALIZATION
    source_partition: str='unlabelled'
    source_fingerprint: str | None=None

    def __post_init__(self):
        if not isinstance(self.query_id,str) or not self.query_id or len(self.query_id)>256:raise ValueError('Explicit bounded query ID required')
        if any(not isinstance(v,str) or not v.strip() for v in (self.context,self.target,self.stimulus)):
            raise ValueError('Explicit context, target and stimulus required')
        if protected(self.context):raise ValueError('Reserved background remains closed')
        if self.source_partition not in ('train','val','unlabelled'):raise ValueError('Invalid query source partition')
        if self.source_fingerprint is not None and (len(self.source_fingerprint)!=64 or any(c not in '0123456789abcdef' for c in self.source_fingerprint)):
            raise ValueError('Invalid query source fingerprint')
        if self.species!='human' or self.modality!='CRISPRi' or self.output!='mean_delta' or self.normalization!=NORMALIZATION:
            raise ValueError('This stage supports human CRISPRi normalized mean-delta only')
        x=np.asarray(self.control,dtype=float)
        if not self.genes or len(set(self.genes))!=len(self.genes) or any(not isinstance(g,str) or not g for g in self.genes):
            raise ValueError('Unique ordered gene panel required')
        if x.shape!=(len(self.genes),) or not np.isfinite(x).all() or (x<0).any():
            raise ValueError('Finite nonnegative normalized control mean required')

    @property
    def fingerprint(self):return digest(asdict(self))

    @property
    def public(self):
        # The planner receives explicit task metadata; no paths, labels or matrix.
        return {key:getattr(self,key) for key in ('query_id','context','target','stimulus','species','modality','output')}


def load_query(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    # Unknown fields, including response/score fields, are rejected by dataclass.
    try:return Query(**value)
    except TypeError as exc:raise ValueError('Unexpected/missing query fields') from exc


@dataclass(frozen=True)
class Action:
    tool: str
    expert_id: str | None=None
    reason: str=''

    def __post_init__(self):
        if self.tool not in ('inspect','predict','finish','abstain'):raise ValueError('Tool is not allowlisted')
        if self.tool!='abstain':safe_name(self.expert_id)
        elif self.expert_id is not None:raise ValueError('Abstain has no expert')
        if not isinstance(self.reason,str) or len(self.reason)>2000:raise ValueError('Invalid decision reason')
