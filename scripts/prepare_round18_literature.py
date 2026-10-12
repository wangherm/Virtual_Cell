"""Warm a checksummed public abstract snapshot without a DeepSeek API request."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vcell.model_agent.scientific_memory import collect_literature
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--cutoff',default='2026-10-12')
a=p.parse_args();papers=collect_literature(a.output,a.cutoff)
print('LITERATURE READY:',a.output,'papers=',len(papers))
for paper in papers:print(paper['id'],paper['publication_date'],paper['title'])
