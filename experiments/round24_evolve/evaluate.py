"""Shinka evaluates editable modules through the independent kernel grader."""
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from vcell.round24.runtime import assess
from vcell.evolution.grader import write_result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--program_path',required=True);p.add_argument('--results_dir',required=True);a=p.parse_args()
    result=assess(Path(os.environ['ROUND24_ROOT']),Path(a.program_path).read_text(encoding='utf-8'),scope='evolved',audit=False)
    write_result(result,a.results_dir)
