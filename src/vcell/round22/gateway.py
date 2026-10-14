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
    def __init__(self,path,key,model,token,requests_max=24,output_tokens_max=196608,port=18022,endpoint='https://api.deepseek.com/v1'):
        self.path=Path(path);self.key=key;self.model=model;self.token=token;self.requests_max=requests_max;self.output_tokens_max=output_tokens_max;self.endpoint=endpoint.rstrip('/');self.lock=threading.Lock()
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
        cap=min(int(body.get('max_tokens',8192)),8192)
        if cap<1:raise ValueError('Positive output cap required')
        with self.lock:
            if len(self.records)>=self.requests_max or sum(r.get('charged_output_tokens',0) for r in self.records)+cap>self.output_tokens_max:
                return 400,{'error':{'message':'Round22 API budget exhausted','type':'budget_exhausted'}}
            record={'attempt':len(self.records)+1,'started_at':time.time(),'status':'inflight','request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),
                'charged_output_tokens':cap,'requested_model':self.model}
            self.records.append(record);write_json(self.path,self.records)
        started=time.monotonic()
        try:
            body['max_tokens']=cap;body['reasoning_effort']='high';body['thinking']={'type':'enabled'};body['n']=1
            r=requests.post(self.endpoint+'/chat/completions',json=body,headers={'Authorization':'Bearer '+self.key},timeout=(15,240));data=r.json()
            usage=data.get('usage',{});choice=(data.get('choices') or [{}])[0]
            record.update(status='complete' if r.ok and choice.get('finish_reason')=='stop' else 'failed',http_status=r.status_code,
                usage=usage,finish_reason=choice.get('finish_reason'),response_id=data.get('id'),returned_model=data.get('model'),
                charged_output_tokens=int(usage.get('completion_tokens',cap)),latency_seconds=time.monotonic()-started,
                currency_cost=None,currency_note='Unknown price; measured usage is authoritative')
            # Do not let truncated code become a valid evolution proposal.
            if r.ok and choice.get('finish_reason')!='stop':return 400,{'error':{'message':'Incomplete provider response; attempt charged','type':'incomplete_response'}}
            return r.status_code,data
        except requests.RequestException:
            record.update(status='transport_failed',latency_seconds=time.monotonic()-started)
            return 400,{'error':{'message':'Provider transport failed; attempt charged','type':'transport_failure'}}
        finally:
            with self.lock:write_json(self.path,self.records)

    def __enter__(self):
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.server.shutdown();self.server.server_close();self.thread.join(timeout=5)
