"""Source-only scientific memory, literature provenance, guarded routing and real refits."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from vcell.model_agent.artifacts import read,seal,verify
from vcell.model_agent.contracts import FEATURES,digest
from vcell.model_agent.scientific_memory import ScientificMemory,collect_literature,retrieve_papers,validate_citations
from vcell.model_agent.scientist import ScientistPlanner,control_proposal
from vcell.model_agent.reliability import fit_relative,estimate,choose,fit_reliability
from vcell.utils import write_json
REPO=Path(__file__).resolve().parents[1]


def paper():
    p={'id':'MED:12345','title':'Single-cell perturbation prediction with low-rank interactions','abstract':'Low-rank bilinear models support cross-context perturbation prediction and regularization.',
        'doi':'10.1234/example','source_url':'https://europepmc.org/article/MED/12345','publication_date':'2025-01-01',
        'version':'1','evidence_kind':'abstract','retrieved_utc':'2026-10-12T00:00:00Z','code_links':[],
        'method_claim':'Abstract inspiration only'}
    p['snapshot_sha256']=digest(p);return p


def test_memory(tmp_path):
    db=ScientificMemory(tmp_path/'memory.sqlite');p=paper();db.add_paper(p);db.add_paper({**p,'retrieved_utc':'later'})
    value={'evaluation_role':'source_inner_oof','outer_used':False,'used_cell_lines':['BXPC3','HAP1'],
        'data_fingerprint':'data','arm':'agent','status':'evaluated','source_mse':.2,'baseline_mse':.21,'reason':'source result','recipe_sha256':'code'}
    db.add_experiment('one',value);db.add_experiment('one',value)
    assert len(db.retrieve(['A549'],'data'))==1
    assert db.retrieve(['bxpc3_IFNG'],'data')==[] and db.retrieve(['A549'],'other_data')==[]
    for update in ({'outer_used':True},{'outer_mse':.1},{'evaluation_role':'outer'}):
        with pytest.raises(ValueError):db.add_experiment('bad',{**value,**update})
    with pytest.raises(ValueError,match='Immutable'):db.add_experiment('one',{**value,'source_mse':.1})


def test_literature(tmp_path):
    class Response:
        status_code=200
        def json(self):return {'resultList':{'result':[{'source':'MED','id':'12345','title':paper()['title'],
            'abstractText':'<p>'+paper()['abstract']+'</p>','firstPublicationDate':'2025-01-01','doi':'10.1234/example'},
            {'source':'MED','id':'future','abstractText':'Future work','firstPublicationDate':'2099-01-01'},
            {'source':'MED','id':'finance','title':'Financial sentiment models','abstractText':'Bilinear adversarial prediction improves financial modelling.','firstPublicationDate':'2025-01-01'}]}}
    class Client:
        def __init__(self):self.calls=[]
        def get(self,url,**kw):self.calls.append((url,kw));return Response()
    c=Client();papers=collect_literature(tmp_path/'lit','2026-10-12',client=c)
    assert len(papers)==1 and len(c.calls)==2 and c.calls[0][1]['params']['resultType']=='core'
    assert all('Authorization' not in str(kw) for _,kw in c.calls)
    assert collect_literature(tmp_path/'lit','2026-10-12',client=c)==papers and len(c.calls)==2
    selected=retrieve_papers(papers,'low-rank prediction');assert selected[0]['evidence_kind']=='abstract'
    validate_citations([{'paper_id':'MED:12345','quote':'Low-rank bilinear models'}],selected)
    for citations in ([],[{'paper_id':'fake','quote':'Low-rank bilinear models'}],[{'paper_id':'MED:12345','quote':'fabricated quote'}]):
        with pytest.raises(ValueError):validate_citations(citations,selected)


def test_scientist_and_controls(tmp_path,monkeypatch):
    from test_deepseek_agent import Client,reply
    from vcell.model_agent.candidate_code import check_head
    monkeypatch.setenv('DEEPSEEK_API_KEY','UNIT_SECRET')
    p=control_proposal('fixed',1,17,'BXPC3');value={**asdict(p),'hypothesis':'Bilinear factorization improves source OOF beyond zero.',
        'citations':[{'paper_id':'MED:12345','quote':'Low-rank bilinear models'}]}
    planner=ScientistPlanner(tmp_path/'api',client=Client([reply(content=json.dumps(value))]),max_api_calls=1)
    obs={'literature':[paper()],'source_inner_oof':{'best_fixed':'zero'}}
    result=planner.decide(obs);assert result.citations==value['citations'];assert planner.decide(obs)==result
    for i in range(1,3):
        for arm in ('fixed','random'):
            p=control_proposal(arm,i,17,'BXPC3');assert p==control_proposal(arm,i,17,'BXPC3');check_head(p.code,rank=p.rank)


def source_frame():
    rows=[]
    for ci,c in enumerate(['A','B','C']):
        for t in range(6):
            for name,ratio in [('zero',1.),('weak',1.1),('strong',.7)]:
                f=dict(zip(FEATURES,[float(ci+1),1.,0.,float(t%2),0.]))
                rows.append({'context':c,'target':'G'+str(t),'query_id':c+str(t),'expert_id':name,'mse':(1+t*.1)*ratio,**f})
    return pd.DataFrame(rows)


def test_reliability(tmp_path):
    f=source_frame();models=fit_relative(f);features={n:dict(zip(FEATURES,[2.,1.,0.,1.,0.])) for n in models}
    assert choose(models,features,'conservative','strong')=='strong'
    far={n:{**v,'control_distance':999.} for n,v in features.items()}
    assert choose(models,far,'conservative','strong')=='zero'
    assert estimate(models['weak'],far['weak'])[1]
    missing={n:{**v,'target_coverage':0.} for n,v in features.items()}
    assert choose(models,missing,'conservative','strong')=='zero'
    base=tmp_path/'base';write_json(base/'profile.json',{'models':{n:{'recipe_fingerprint':n} for n in models},'fixed_expert':'strong'})
    out=fit_reliability(f,base,tmp_path/'new');verify(out)
    diag=pd.read_csv(out/'private_source_meta_cv.csv');assert len(diag)==3*6*4
    assert read(out/'profile.json')['safe_routing']['models']['zero']['source_mse']>0
    with pytest.raises(ValueError,match='>=3'):fit_reliability(f.loc[f.context.ne('C')],base,tmp_path/'bad')


def test_acceptance():
    sys.path.insert(0,str(REPO/'scripts'));import run_round18 as runner
    f=source_frame();result=runner.candidate_acceptance(f,'strong','zero',.01)
    assert result['accepted'] and result['relative_gain']==pytest.approx(.3)
    # A gain confined to one source background does not promote a model.
    g=f.copy();g.loc[g.expert_id.eq('strong') & g.context.ne('A'),'mse']*=1.6
    assert not runner.candidate_acceptance(g,'strong','zero',.01)['accepted']


def test_iteration():
    from vcell.model_agent.iteration import IterationBudget,feedback
    budget=IterationBudget(3,1,1)
    assert budget.complete(1,[{'status':'evaluated','source_oof_mse':.9}],1.)['next_cycle_allowed']
    stop=budget.complete(2,[{'status':'evaluated','source_oof_mse':.95}],1.)
    assert stop['stop_reason']=='source stagnation' and stop['source_best_mse']==.9
    state=feedback([{'trial':1,'status':'evaluated','source_oof_mse':.9,'context_relative_gain':{'A':.1}}],2,3,{'proposal':{'code':'safe'}})
    assert state['best_source_candidate']['proposal']['code']=='safe' and 'outer_mse' not in json.dumps(state)


def test_search_cycles(tmp_path,monkeypatch):
    from types import SimpleNamespace
    sys.path.insert(0,str(REPO/'scripts'));import run_round18 as runner
    observations=[]
    class Planner:
        def __init__(self,*args,**kwargs):assert kwargs['max_api_calls']==2
        def decide(self,obs):observations.append(obs);return control_proposal('fixed',len(observations),17,'BXPC3')
        def summary(self):return {'attempted_requests':len(observations)}
    fake=SimpleNamespace(path=tmp_path,provenance={'fit_cell_lines':['A549','MCF7'],'calibration_cell_lines':['HAP1']})
    monkeypatch.setattr(runner,'ResearchPlanner',Planner)
    monkeypatch.setattr(runner,'Registry',lambda _:SimpleNamespace(experts={'expanded_factor_rank16':fake}))
    monkeypatch.setattr(runner.core,'admit',lambda *args:None);monkeypatch.setattr(runner.core,'dispatch',lambda *args:None)
    monkeypatch.setattr(runner,'export_candidate',lambda worker,dest,*args:dest)
    base=source_frame().loc[lambda f:f.expert_id.eq('zero')]
    def evidence(pl,data,root,pools,extra,suffix,include_zero):
        name=next(iter(next(iter(extra.values()))));trial=int(name.rsplit('_',1)[1]);new=base.copy()
        new.expert_id=name;new.mse*=1-.03*trial
        return tmp_path,pd.concat([base,new],ignore_index=True),{}
    monkeypatch.setattr(runner.core,'evidence_for',evidence)
    cfg={'output_dir':str(tmp_path/'base')};pl={'name':'BXPC3_seed17','outer_context':'BXPC3','seed':17,
        'folds':[{'name':'final','configs':{'expanded':cfg}}]+[{'name':'inner'+c,'held':c,'configs':{'expanded':cfg}} for c in ['A','B','C']]}
    a=SimpleNamespace(model='deepseek-v4-pro',research_effort='high',trials=1,cycles=2,patience_cycles=2,
        max_output_tokens=8192,api_timeout=180,parallel_jobs=1,minimum_gain=.01,max_steps=2,batch_size=8,inner_folds=3)
    summary={'fixed_source_mse':{'zero':1.25},'best_fixed':'zero','contexts':['A','B','C'],'distinct_targets':6,'note':'source'}
    memory=ScientificMemory(tmp_path/'memory.sqlite');root=tmp_path/'run';root.mkdir()
    pools={'inner'+c:tmp_path for c in ['A','B','C']}
    result=runner.search_arm(a,pl,{'audit':{'fingerprint':'data'}},root,pools,summary,'agent_plain',[],[],memory)
    assert len(result['cycles'])==2 and len(result['trials'])==2
    assert observations[1]['self_iteration']['cycle']==2 and observations[1]['self_iteration']['best_source_candidate']
    assert result['selected_candidate']['trial']==2
    assert runner.search_arm(a,pl,{'audit':{'fingerprint':'data'}},root,pools,summary,'agent_plain',[],[],memory)==result
    assert len(observations)==2  # completed self-iteration resumes without requests


def test_full(tmp_path,monkeypatch):
    sys.path.insert(0,str(REPO/'scripts'));import run_round18 as runner
    from test_agent_nested import fixture
    from vcell.model_agent.research import ResearchPlanner
    from test_deepseek_agent import Client,reply
    from vcell.model_agent.candidate_code import check_head
    cfg,data,parent,n=fixture(tmp_path);monkeypatch.setenv('DEEPSEEK_API_KEY','UNIT_SECRET')
    def dispatch(specs,root,parallel,label,stage):
        for cfg in specs.values():
            if stage=='background':runner.core.pretrain(cfg,True)
            elif stage=='candidate':runner.core.candidate_worker(cfg,True)
            else:runner.core.worker(cfg,True)
    def literature(dest,cutoff):
        dest=Path(dest);write_json(dest/'papers.json',[paper()]);seal(dest);return [paper()]
    monkeypatch.setattr(runner.core,'dispatch',dispatch);monkeypatch.setattr(runner,'collect_literature',literature)
    monkeypatch.setattr(runner.core,'admit',lambda code,rank,dest:write_json(dest/'admission.json',check_head(code.read_text(),rank=rank)))
    base=asdict(control_proposal('fixed',1,17,'BXPC3'))
    monkeypatch.setattr(runner,'ResearchPlanner',lambda *args,**kw:ResearchPlanner(*args,**kw,client=Client([reply(content=json.dumps(base))])))
    scientific={**base,'hypothesis':'Test a source-only low-rank interaction',
        'citations':[{'paper_id':'MED:12345','quote':'Low-rank bilinear models'}]}
    monkeypatch.setattr(runner,'ScientistPlanner',lambda *args,**kw:ScientistPlanner(*args,**kw,client=Client([reply(content=json.dumps(scientific))])))
    args=['--work-dir',str(tmp_path),'--p4-run',str(parent),'--name','r18','--outer-contexts','BXPC3_IFNG',
        '--seeds','17','--device','cpu','--max-steps','2','--pretrain-steps','2','--save-every','1',
        '--batch-size','8','--targets-per-context','4','--profile-targets','4','--trials','1','--cycles','1']
    root=runner.main(args);verify(root)
    profile=read(root/'evidence/BXPC3_IFNG_seed17/base/profiles/profile.json');assert len(profile['models'])==7 and 'zero' in profile['models']
    from vcell.model_agent.experts import Registry
    registry=Registry(root/'pools/BXPC3_IFNG_seed17_outer/registry_with_zero.json');assert registry.experts['zero'].card['kind']=='zero'
    scores=pd.read_csv(root/'comparison.csv');assert set(scores.policy)>={'zero','fixed','risk_shrunk','conservative','legacy_fixed'}
    assert scores.coverage.eq(1).all();assert read(root/'search_and_routing_sealed.json')['outer_scores_used_for_search'] is False
    records=read(root/'search_comparison.json');assert {r['arm'] for r in records}=={'fixed','random','agent_plain','agent_literature'}
    for arm in ('agent_plain','agent_literature'):
        selection=read(root/'search/BXPC3_IFNG_seed17'/arm/'selection.json')
        assert selection['provider_summary']['attempted_requests']==1 and selection['trials'][0]['status']=='evaluated'
    marker=(root/'COMPLETE.json').read_bytes();runner.main(args+['--resume']);assert marker==(root/'COMPLETE.json').read_bytes()
    # Changing outer labels cannot enter the scientific memory projection.
    db=ScientificMemory(tmp_path/'knowledge/round18/scientific_memory.sqlite')
    assert all('outer' not in str(item) for item in db.retrieve(['BXPC3'],data['audit']['fingerprint']))
    # Reuse exact completed fits without modifying the old experiment.
    a=runner.main.__globals__['argparse'].Namespace(**{**read(root/'plan.json'),'reuse_round17':str(root)})
    newer=tmp_path/'reuse';newer.mkdir();_,plans,bs,ws=runner.core.prepared_inputs(a,parent,newer,freeze_inputs=False)
    reused=runner.reuse_completed(a,plans,bs,ws);assert len(reused)==8 and not bs and not ws
    assert (root/'COMPLETE.json').read_bytes()==marker
