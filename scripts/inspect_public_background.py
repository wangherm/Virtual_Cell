"""Download and inspect public tissue atlases without cloud accounts."""
import argparse
from pathlib import Path
from vcell.public_background import DEFAULT_DATASETS, run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--manifest',default=str(Path(__file__).resolve().parents[1]/'configs/public_backgrounds.json'))
    p.add_argument('--datasets',nargs='+',default=DEFAULT_DATASETS)
    p.add_argument('--max-total-gib',type=float,default=20)
    p.add_argument('--sample-rows',type=int,default=128)
    p.add_argument('--panel',help='Optional exact output-gene IDs/symbols, one per line, no header')
    p.add_argument('--catalog-only',action='store_true',help='Generate the pinned catalogue report without network access')
    a=p.parse_args()
    if not 0<a.max_total_gib<=100 or not 1<=a.sample_rows<=1024:
        p.error('max-total-gib must be >0 and <=100; sample-rows must be 1..1024')
    run(a)


if __name__=='__main__':
    main()
