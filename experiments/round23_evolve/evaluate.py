"""Pinned Shinka evaluator contract; generated code is never imported here."""
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from vcell.evolution.grader import assess,write_result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--program_path',required=True);p.add_argument('--results_dir',required=True);a=p.parse_args()
    pool=os.environ['ROUND23_POOL']
    record=assess(Path(a.program_path).read_text(encoding='utf-8'),pool)
    write_result(record,a.results_dir)
