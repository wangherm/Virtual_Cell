"""Role/ticket-audited relay with persisted responses for exact interrupted requests."""
import hashlib
import json
from pathlib import Path
from ..round22.gateway import Gateway
from ..model_agent.artifacts import read
from ..model_agent.contracts import digest
from ..utils import write_json


class ResearchGateway(Gateway):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.scope='unset';self.role='mutation'

    def forward(self,body):
        canonical=dict(body)
        canonical.pop('temperature',None);canonical.pop('top_p',None)
        canonical.update(max_tokens=min(int(body.get('max_tokens',self.max_tokens)),self.max_tokens),
                         reasoning_effort=self.reasoning_effort,thinking={'type':'enabled'},n=1)
        key=hashlib.sha256(json.dumps({'ticket':self.scope,'role':self.role,'body':canonical},sort_keys=True).encode()).hexdigest()
        ticket_key=hashlib.sha256((self.scope+'|'+self.role).encode()).hexdigest()
        ticket=self.path.parent/'provider_tickets'/(ticket_key+'.json')
        if ticket.exists() and read(ticket)['request_key']!=key:
            raise ValueError('Provider ticket payload changed on resume; no duplicate request was sent')
        if not ticket.exists():write_json(ticket,{'ticket':self.scope,'role':self.role,'request_key':key})
        path=self.path.parent/'provider_responses'/(key+'.json')
        if path.exists():
            cached=read(path)
            if cached['state']=='complete':
                if digest({k:v for k,v in cached.items() if k!='sha256'})!=cached.get('sha256'):
                    raise ValueError('Saved provider response checksum changed')
                return cached['http_status'],cached['response']
            # A crash after sending a request has unknown billing/outcome. Never silently resend.
            raise RuntimeError('Unresolved provider ticket; inspect '+str(path)+' before any new campaign')
        write_json(path,{'state':'reserved','ticket':self.scope,'role':self.role,'request_sha256':key})
        before=len(self.records)
        try:
            status,response=super().forward(body)
        except BaseException:
            # Reservation is deliberately retained on an ambiguous interrupted request.
            raise
        if len(self.records)>before:
            self.records[-1].update(role=self.role,ticket=self.scope,response_cache=key)
            write_json(self.path,self.records)
        completed={'state':'complete','ticket':self.scope,'role':self.role,'http_status':status,'response':response}
        completed['sha256']=digest(completed);write_json(path,completed)
        return status,response


def critic_request(gateway,prompt,cap):
    status,body=gateway.forward({'model':gateway.model,'messages':[
        {'role':'system','content':'You are a source-only scientific critic. Return one strict JSON action, no code or markdown.'},
        {'role':'user','content':prompt}], 'max_tokens':cap,'stream':False})
    if status!=200: raise RuntimeError('Critic provider request failed; charged in API ledger')
    return body['choices'][0]['message']['content']
