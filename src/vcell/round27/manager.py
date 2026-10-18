"""Persistent balanced task management and cooperative process control."""
from pathlib import Path
import os
import threading
import time
from ..utils import write_json
from ..model_agent.artifacts import read
from ..round11 import freeze


class PauseRequested(RuntimeError):pass


def task_queue(root,cfg,arm,counts,reason=None,active=None):
    path=Path(root)/'tasks'/f'{arm}.json'
    rows=[{'task':t,'target':cfg['task_targets'][t],'distinct_new':counts[t],
        'status':'complete' if counts[t]>=cfg['task_targets'][t] else 'running' if t==active else 'pending' if reason is None else 'blocked',
        'reason':None if counts[t]>=cfg['task_targets'][t] else reason} for t in cfg['tasks']]
    write_json(path,{'arm':arm,'tasks':rows,'coverage_complete':all(r['status']=='complete' for r in rows)})
    return rows


def choose_task(cfg,counts,seconds,blocked=()):
    available=[t for t in cfg['tasks'] if counts[t]<cfg['task_targets'][t] and t not in blocked]
    return min(available,key=lambda t:(counts[t]/cfg['task_targets'][t],seconds[t])) if available else None


def control_path(work,name):
    if not name or Path(name).name!=name or name in ('.','..'):raise ValueError('A simple campaign name is required')
    return Path(work)/'agent_control'/(name+'.json')


def checkpoint(work,name):
    p=control_path(work,name)
    if p.exists() and read(p).get('pause_requested'):raise PauseRequested('Pause requested; current completed work is retained')


class Heartbeat:
    def __init__(self,root,interval=10):self.root=Path(root);self.interval=interval;self.stop=threading.Event()
    def __enter__(self):
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start();return self
    def run(self):
        while not self.stop.is_set():
            write_json(self.root/'heartbeat.json',{'pid':os.getpid(),'updated_at':time.time(),'state':'running',
                'meaning':'Process liveness only; consult worker progress and task counts for experimental progress.'})
            self.stop.wait(self.interval)
    def __exit__(self,*args):
        if getattr(self,'closed',False):return
        self.closed=True
        self.stop.set();self.thread.join(timeout=self.interval+5)
        write_json(self.root/'heartbeat.json',{'pid':os.getpid(),'updated_at':time.time(),'state':'stopped'})


def revision_summary(root):
    from .memory import experience
    from ..evolution.journal import events
    rows=events(Path(root)/'memory/events.jsonl');chains=[]
    for r in rows:
        link=r.get('revision')
        if not link:continue
        before=next((p for p in rows if p['ticket']==link['predecessor_experiment_id']),None)
        first=(before or {}).get('prediction') or {};second=r.get('prediction') or {}
        chains.append({'experiment':r['ticket'],'predecessor':link['predecessor_experiment_id'],
            'arm':r['arm'],'task':r['task'],'failure_explanation':link['failure_explanation'],
            'changed_assumption':link['changed_assumption'],'measured':any(c.get('status')=='valid' for c in r.get('children',[])),
            'hypothesis_supported':second.get('status')=='supported',
            'mse_before':first.get('after') if first.get('metric')=='mse' else None,
            'mse_after':second.get('after') if second.get('metric')=='mse' else None,
            'causal_claim':False})
    return chains
