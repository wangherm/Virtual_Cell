"""CPU admission worker; candidate Python is accepted only in the bounded AST language."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vcell.model_agent.candidate_code import check_head
from vcell.utils import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--code',required=True);p.add_argument('--output',required=True)
    p.add_argument('--rank',type=int,choices=[16,32],required=True)
    p.add_argument('--target-dim',type=int,default=33);p.add_argument('--background-dim',type=int,default=16);a=p.parse_args()
    import torch
    torch.set_num_threads(1)
    write_json(a.output,check_head(Path(a.code).read_text(encoding='utf-8'),a.target_dim,a.background_dim,a.rank))

if __name__=='__main__':main()
