"""Native Shinka's trusted grader; generated modules execute only in protected workers."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from vcell.round27.runtime import assess
from vcell.evolution.self_correction import loss_card
from vcell.evolution.grader import write_result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--program_path',required=True);p.add_argument('--results_dir',required=True);a=p.parse_args()
    result=assess(Path(os.environ['ROUND27_POOL']),Path(a.program_path).read_text(encoding='utf-8'))
    if result['status']=='valid':
        result['metrics']['text_feedback']=json.dumps({'scope':'source_search_only','metrics':result['metrics']['public'],
            'diagnosis':loss_card(result)},allow_nan=False)
    write_result(result,a.results_dir)
