"""Offline installed-framework/interface check. No biological results or paid API."""
import asyncio
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile
import threading


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr,'reconfigure'):sys.stderr.reconfigure(encoding='utf-8')
    if version('shinka-evolve')!='0.0.7':raise RuntimeError('Wrong Shinka version')
    os.environ['SHINKA_PRICING_MODE']='offline';os.environ['ROUND22_GATEWAY_KEY']='offline-test'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert body['model']=='deepseek-v4-pro'
            payload=json.dumps({'id':'offline-mock','object':'chat.completion','created':1,'model':body['model'],
                'choices':[{'index':0,'message':{'role':'assistant','content':'ROUND22_READY'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':2,'completion_tokens':3,'total_tokens':5}}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    from shinka.llm import AsyncLLMClient
    model=f'local/deepseek-v4-pro@http://127.0.0.1:{server.server_port}/v1?api_key_env=ROUND22_GATEWAY_KEY'
    try:
        result=asyncio.run(AsyncLLMClient(model_names=[model],max_tokens=128,verbose=False).query(msg='probe',system_msg='offline interface test'))
        if not result or result.content!='ROUND22_READY':raise RuntimeError('Dynamic endpoint/response parsing failed')
        from shinka.core import ShinkaEvolveRunner,EvolutionConfig
        from shinka.database import DatabaseConfig
        from shinka.launch import LocalJobConfig
        with tempfile.TemporaryDirectory(prefix='round22-shinka-check-',ignore_cleanup_errors=True) as temp:
            root=Path(temp);initial=root/'initial.py';evaluator=root/'evaluate.py'
            initial.write_text('# EVOLVE-BLOCK-START\ndef value():\n    return 1\n# EVOLVE-BLOCK-END\n')
            evaluator.write_text('''import argparse, json, importlib.util
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--program_path');p.add_argument('--results_dir');a=p.parse_args()
spec=importlib.util.spec_from_file_location('toy',a.program_path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=Path(a.results_dir);out.mkdir(parents=True,exist_ok=True)
(out/'correct.json').write_text(json.dumps({'correct':m.value()==1,'error':None}))
(out/'metrics.json').write_text(json.dumps({'combined_score':1.,'public':{'interface_probe':1},'private':{},'text_feedback':'offline toy only'}))
''')
            runner=ShinkaEvolveRunner(evo_config=EvolutionConfig(num_generations=1,embedding_model=None,meta_rec_interval=None,
                llm_models=[model],init_program_path=str(initial),results_dir=str(root/'results')),
                db_config=DatabaseConfig(num_islands=2),job_config=LocalJobConfig(eval_program_path=str(evaluator),python_executable=sys.executable,time='00:00:30'),
                max_evaluation_jobs=1,max_proposal_jobs=1,max_db_workers=1)
            runner.run()
            codes=list((root/'results').glob('gen_*/main.py'))
            if not codes:raise RuntimeError('Installed framework candidate layout differs from adapter')
            if not any(json.loads((c.parent/'results/correct.json').read_text())['correct'] for c in codes):raise RuntimeError('Installed grader result interface failed')
            import logging
            logging.shutdown()
        print('ROUND22 SHINKA INTERFACE VERIFIED: actual runner seed + mocked OpenAI-compatible transport; no live API/GPU search')
    finally:server.shutdown();server.server_close();thread.join(timeout=5)


if __name__=='__main__':main()
