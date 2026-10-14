"""Shinka's trusted CLI contract; never import a generated module here."""
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from vcell.round22.evaluator import evaluate
from vcell.utils import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--program_path',required=True);p.add_argument('--results_dir',required=True)
    p.add_argument('--root',default=os.environ.get('ROUND22_ROOT'));p.add_argument('--task',default=os.environ.get('ROUND22_TASK'));a=p.parse_args()
    if not a.root or a.task not in ('repeat','transfer'):raise ValueError('Trusted root and task required')
    try:evaluate(Path(a.program_path).read_text(encoding='utf-8'),a.root,a.task,'search',a.results_dir)
    except Exception as exc:
        write_json(Path(a.results_dir)/'correct.json',{'correct':False,'error':str(exc)})
        write_json(Path(a.results_dir)/'metrics.json',{'combined_score':-1000000.,'public':{},'private':{},'text_feedback':str(exc),'extra_data':{}})


if __name__=='__main__':main()
