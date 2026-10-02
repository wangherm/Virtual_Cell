"""Clean inspected public backgrounds, freeze donor splits, and write audit reports."""
import argparse
import os
from pathlib import Path

from vcell.clean_background import run, run_lock


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',default=os.environ.get('VCELL_WORK','/root/autodl-tmp/vcell-work'))
    p.add_argument('--input',help='Completed public inspection directory (contains h5ad/ and inspect_report.json)')
    p.add_argument('--output',help='New preparation directory; matching completed stages are reused on restart')
    p.add_argument('--panel',help='Round8 plus_h1 dataset.npz or one-gene-per-line text file')
    p.add_argument('--expected-genes',type=int,default=1834)
    p.add_argument('--min-counts',type=int,default=500)
    p.add_argument('--min-genes',type=int,default=200)
    p.add_argument('--max-mito-pct',type=float,default=20)
    p.add_argument('--per-stratum',type=int,default=2000,help='Maximum cells per donor/tissue/cell-type/assay stratum')
    p.add_argument('--max-cells-per-dataset',type=int,default=60000)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--chunk-rows',type=int,default=512)
    p.add_argument('--shard-rows',type=int,default=8192)
    p.add_argument('--reserve-gib',type=float,default=10)
    a=p.parse_args(argv)
    work=Path(a.work_dir)
    a.input=a.input or str(work/'background/public_inspect_01')
    a.output=a.output or str(work/'background/human_background_v1')
    a.panel=a.panel or str(work/'runs/round8_01/prepared/plus_h1/dataset.npz')
    del a.work_dir
    with run_lock(Path(a.output)/'.prepare.lock'):
        run(a)


if __name__=='__main__':
    main()
