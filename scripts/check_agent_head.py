"""CPU admission worker; candidate Python is accepted only in the bounded AST language."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vcell.model_agent.candidate_code import check_head
from vcell.utils import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--code',required=True);p.add_argument('--output',required=True)
    p.add_argument('--rank',type=int,choices=[16,32],required=True);a=p.parse_args()
    import torch
    torch.set_num_threads(1)
    write_json(a.output,check_head(Path(a.code).read_text(encoding='utf-8'),rank=a.rank))

if __name__=='__main__':main()
