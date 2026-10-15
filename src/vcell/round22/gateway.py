"""Bounded OpenAI-compatible relay; only this trusted process holds provider key."""
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import threading
import time
from pathlib import Path
import requests
from ..model_agent.artifacts import read
from ..utils import write_json


class Gateway:
    def __init__(self,path,key,model,token,requests_max=24,output_tokens_max=196608,port=18022,endpoint='https://api.deepseek.com/v1',max_tokens=32768,reasoning_effort='low',failure_limit=2):
        self.path=Path(path);self.key=key;self.model=model;self.token=token;self.requests_max=requests_max;self.output_tokens_max=output_tokens_max;self.endpoint=endpoint.rstrip('/');self.lock=threading.Lock()
        if reasoning_effort not in ('low','high','max') or not 1<=max_tokens<=65536 or failure_limit<1:raise ValueError('Invalid bounded provider policy')
        self.max_tokens=max_tokens;self.reasoning_effort=reasoning_effort;self.failure_limit=failure_limit
        self.records=read(path) if Path(path).exists() else [];gateway=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                status=400;response={'error':{'message':'Invalid bounded relay request','type':'invalid_request_error'}}
                if self.path=='/v1/chat/completions' and self.headers.get('Authorization')=='Bearer '+gateway.token:
                    try:
                        size=int(self.headers.get('Content-Length','0'))
                        if not 0<size<2000000:raise ValueError('Request size cap')
                        body=json.loads(self.rfile.read(size));status,response=gateway.forward(body)
                    except Exception as exc:response={'error':{'message':type(exc).__name__,'type':'invalid_request_error'}}
                payload=json.dumps(response).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        self.server=ThreadingHTTPServer(('127.0.0.1',port),Handler)

    def forward(self,body):
        if body.get('model')!=self.model or body.get('stream') or not isinstance(body.get('messages'),list):raise ValueError('Frozen model/nonstreaming messages required')
        cap=min(int(body.get('max_tokens',self.max_tokens)),self.max_tokens)
        if cap<1:raise ValueError('Positive output cap required')
        with self.lock:
            consecutive=0
            for previous in reversed(self.records):
                if previous['status']=='complete':break
                consecutive+=1
            if consecutive>=self.failure_limit:
                return 400,{'error':{'message':'Provider circuit open after consecutive failures; inspect API ledger before a new frozen run','type':'provider_circuit_open'}}
            if len(self.records)>=self.requests_max or sum(r.get('charged_output_tokens',0) for r in self.records)+cap>self.output_tokens_max:
                return 400,{'error':{'message':'Round22 API budget exhausted','type':'budget_exhausted'}}
            record={'attempt':len(self.records)+1,'started_at':time.time(),'status':'inflight','request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),
                'charged_output_tokens':cap,'requested_model':self.model,'max_tokens':cap,'reasoning_effort':self.reasoning_effort}
            self.records.append(record);write_json(self.path,self.records)
        started=time.monotonic()
        try:
            body=dict(body);body['max_tokens']=cap;body['reasoning_effort']=self.reasoning_effort;body['thinking']={'type':'enabled'};body['n']=1
            # Effort and total output (reasoning + final code) are frozen policy.
            body.pop('temperature',None);body.pop('top_p',None)
            r=requests.post(self.endpoint+'/chat/completions',json=body,headers={'Authorization':'Bearer '+self.key},timeout=(15,480));data=r.json()
            usage=data.get('usage',{});choice=(data.get('choices') or [{}])[0]
            message=choice.get('message') or {};content=message.get('content');valid_content=isinstance(content,str) and bool(content.strip())
            failure=None if r.ok and choice.get('finish_reason')=='stop' and valid_content else ('http_error' if not r.ok else 'incomplete_response' if choice.get('finish_reason')!='stop' else 'empty_response')
            record.update(status='complete' if failure is None else 'failed',http_status=r.status_code,failure_category=failure,
                usage=usage,finish_reason=choice.get('finish_reason'),response_id=data.get('id'),returned_model=data.get('model'),
                final_characters=len(content) if isinstance(content,str) else 0,
                charged_output_tokens=int(usage.get('completion_tokens',cap)),latency_seconds=time.monotonic()-started,
                currency_cost=None,currency_note='Unknown price; measured usage is authoritative')
            # Do not let truncated code become a valid evolution proposal.
            if r.ok and failure:return 400,{'error':{'message':'Provider response invalid; attempt charged; inspect API ledger','type':failure}}
            return r.status_code,data
        except requests.RequestException:
            record.update(status='transport_failed',failure_category='transport_failure',latency_seconds=time.monotonic()-started)
            return 400,{'error':{'message':'Provider transport failed; attempt charged','type':'transport_failure'}}
        finally:
            with self.lock:write_json(self.path,self.records)

    def __enter__(self):
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.server.shutdown();self.server.server_close();self.thread.join(timeout=5)
