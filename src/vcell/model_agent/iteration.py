"""Bounded self-iteration from measured source feedback; no outer access."""
import numpy as np


class IterationBudget:
    def __init__(self,cycles,trials,patience):
        if not 1<=cycles<=3 or not 1<=trials<=2 or not 1<=patience<=3:raise ValueError('Bounded self-iteration budget required')
        self.cycles=cycles;self.trials=trials;self.patience=patience;self.records=[];self.stale=0;self.best=float('inf')
    def complete(self,cycle,results,baseline):
        values=[r['source_oof_mse'] for r in results if r.get('status')=='evaluated']
        new=min(values+[baseline]);old=min(self.best,baseline)
        improved=new<old*(1-.001)
        self.stale=0 if improved else self.stale+1;self.best=min(old,new)
        value={'cycle':cycle,'source_best_mse':self.best,'improved_by_at_least_0_1pct':bool(improved),
            'stale_cycles':self.stale,'next_cycle_allowed':cycle<self.cycles and self.stale<self.patience,
            'outer_used':False,'stop_reason':'budget exhausted' if cycle>=self.cycles else 'source stagnation' if self.stale>=self.patience else None}
        self.records.append(value);return value


def feedback(previous,cycle,total,best_candidate=None):
    failures=[{'trial':r['trial'],'status':r['status'],'error_type':r.get('error_type'),'reason':r.get('reason'),
        'context_relative_gain':r.get('context_relative_gain'),'relative_gain':r.get('relative_gain'),
        'accepted':r.get('accepted')} for r in previous[-4:]]
    return {'cycle':cycle,'max_cycles':total,'measured_source_feedback':failures,
        'best_source_candidate':best_candidate,
        'instructions':'Use measured source feedback to revise one testable hypothesis and head recipe. '
            'A small inner gain is not proof of transfer. Stop if useful changes lack evidence. '
            'Do not request outer results, unrestricted actions or extra budget.',
        'learning_scope':'Experiment-memory and candidate-head iteration; no fine-tuning of the DeepSeek model or controller code'}
