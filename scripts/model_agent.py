"""Model-selection architecture: demo, export, source-OOF profiles and controlled prediction."""
import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo',help='Synthetic architecture checks; no LLM or biological result')
    demo.add_argument('--output',required=True);demo.add_argument('--resume',action='store_true')
    export=sub.add_parser('export-p4',help='Export one completed P4 worker as an executable expert pool')
    export.add_argument('--worker',required=True);export.add_argument('--output',required=True)
    merge=sub.add_parser('combine',help='Join exported expert pools; preserve unique expert IDs')
    merge.add_argument('--registries',nargs='+',required=True);merge.add_argument('--output',required=True)
    query=sub.add_parser('query',help='Export a control-only query; no response member is decoded')
    query.add_argument('--data',required=True);query.add_argument('--row-id',required=True);query.add_argument('--output',required=True)
    package=sub.add_parser('package',help='Create a small architecture review archive')
    package.add_argument('--run',required=True)
    inspect=sub.add_parser('inspect',help='Inspect capability and provenance, without reading historical scores')
    inspect.add_argument('--registry',required=True);inspect.add_argument('--query')
    profile=sub.add_parser('fit-profiles',help='Consume supplied nested source-OOF bundles; no refitting is performed')
    profile.add_argument('--spec',required=True);profile.add_argument('--output',required=True)
    run=sub.add_parser('run',help='Run deterministic selection baselines through the agent tool controller')
    for key in ('registry','query','output'):run.add_argument('--'+key,required=True)
    run.add_argument('--profiles');run.add_argument('--policy',choices=['fixed','rule','risk','workflow'],default='rule')
    run.add_argument('--budget-units',type=float,default=3.);run.add_argument('--max-steps',type=int,default=12)
    run.add_argument('--resume',action='store_true')
    evaluation=sub.add_parser('evaluate',help='Post-decision private development evaluation, outside planner tools')
    for key in ('run','registry','query','truth','output'):evaluation.add_argument('--'+key,required=True)
    a=p.parse_args(argv)
    if a.command=='demo':
        from vcell.model_agent.demo import run_demo
        print(run_demo(a.output,a.resume))
    elif a.command=='export-p4':
        from vcell.model_agent.experts import export_p4
        print(export_p4(a.worker,a.output))
    elif a.command=='fit-profiles':
        from vcell.model_agent.evidence import fit_profiles
        print(fit_profiles(a.spec,a.output))
    elif a.command=='combine':
        from vcell.model_agent.workflows import combine
        print(combine(a.registries,a.output))
    elif a.command=='query':
        from vcell.model_agent.workflows import query_from_prepared
        print(query_from_prepared(a.data,a.row_id,a.output))
    elif a.command=='package':
        from vcell.model_agent.workflows import package_review
        print(package_review(a.run))
    elif a.command=='inspect':
        from vcell.model_agent.experts import Registry
        from vcell.model_agent.contracts import load_query
        registry=Registry(a.registry)
        print(json.dumps(registry.candidates(load_query(a.query)) if a.query else {name:e.card for name,e in registry.experts.items()},indent=2))
    elif a.command=='run':
        from vcell.model_agent.runtime import execute
        print(json.dumps(execute(a.registry,a.query,a.output,a.policy,a.profiles,a.budget_units,a.max_steps,a.resume),indent=2))
    else:
        from vcell.model_agent.runtime import evaluate
        from vcell.utils import write_json
        # Evaluations must stay outside sealed run folders.
        if Path(a.output).resolve().is_relative_to(Path(a.run).resolve()):p.error('Evaluation output must be outside the immutable run')
        value=evaluate(a.run,a.query,a.truth,a.registry);write_json(a.output,value);print(json.dumps(value,indent=2))


if __name__=='__main__':main()
