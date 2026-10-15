"""Actual Shinka failed-generation -> database resume -> repaired-generation test.

Uses an entirely synthetic toy and a local mock provider. No paid API or biology.
"""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    os.environ.update(ROUND22_GATEWAY_KEY='offline',SHINKA_PRICING_MODE='offline')
    calls=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']));calls.append(1)
            method="raise ValueError('shape example')" if len(calls)==1 else 'return 2.0'
            content=f'<NAME>repair_probe</NAME>\n<DESCRIPTION>Offline controlled failure/recovery.</DESCRIPTION>\n```python\n# EVOLVE-BLOCK-START\ndef value():\n    {method}\n# EVOLVE-BLOCK-END\n```'
            payload=json.dumps({'id':'mock','model':'deepseek-v4-pro','choices':[{'index':0,'message':{'role':'assistant','content':content},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        from shinka.core import ShinkaEvolveRunner,EvolutionConfig
        from shinka.database import DatabaseConfig
        from shinka.launch import LocalJobConfig
        with tempfile.TemporaryDirectory(prefix='round23-resume-',ignore_cleanup_errors=True) as temp:
            root=Path(temp);seed=root/'seed.py';seed.write_text('# EVOLVE-BLOCK-START\ndef value():\n    return 0.\n# EVOLVE-BLOCK-END\n')
            evaluator=root/'evaluate.py';evaluator.write_text('''import argparse,json,importlib.util
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--program_path');p.add_argument('--results_dir');a=p.parse_args()
out=Path(a.results_dir);out.mkdir(parents=True,exist_ok=True)
try:
 s=importlib.util.spec_from_file_location('controlled_toy',a.program_path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);v=m.value();error=None
except Exception as e:v=-1.;error=str(e)
(out/'correct.json').write_text(json.dumps({'correct':error is None,'error':error}))
(out/'metrics.json').write_text(json.dumps({'combined_score':v,'public':{},'private':{},'text_feedback':error or 'toy'}))
''')
            model=f'local/deepseek-v4-pro@http://127.0.0.1:{server.server_port}/v1?api_key_env=ROUND22_GATEWAY_KEY'
            def runner(generations):
                return ShinkaEvolveRunner(evo_config=EvolutionConfig(num_generations=generations,patch_types=['full'],patch_type_probs=[1.],max_patch_resamples=1,max_patch_attempts=1,
                    embedding_model=None,meta_rec_interval=None,llm_models=[model],init_program_path=str(seed),results_dir=str(root/'evolution'),use_text_feedback=True),
                    db_config=DatabaseConfig(num_islands=2),job_config=LocalJobConfig(eval_program_path=str(evaluator),python_executable=sys.executable,time='00:00:30'),max_evaluation_jobs=1,max_proposal_jobs=1,max_db_workers=1)
            runner(2).run()
            assert not json.loads((root/'evolution/gen_1/results/correct.json').read_text())['correct']
            runner(3).run()
            assert json.loads((root/'evolution/gen_2/results/correct.json').read_text())['correct']
            assert json.loads((root/'evolution/gen_2/results/metrics.json').read_text())['combined_score']==2.
            assert len(calls)==2
            import logging
            logging.shutdown()
        print('ROUND23 REAL SHINKA RESUME VERIFIED: failed generation retained, same database extended, repaired generation evaluated; offline toy only')
    finally:server.shutdown();server.server_close();thread.join(timeout=5)


if __name__=='__main__':main()
