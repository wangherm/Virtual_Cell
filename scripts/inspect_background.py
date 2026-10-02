"""Download a bounded human background sample and write an inspection report."""
import argparse
from vcell.background_inspect import FEATURES, RELEASE, run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--billing-project', help='GCP project subscribed to Arc Virtual Cell Atlas')
    p.add_argument('--release', choices=[RELEASE], default=RELEASE)
    p.add_argument('--feature', choices=FEATURES, default='Gene')
    p.add_argument('--metadata', help='Existing sample_metadata.parquet or CSV (offline mode)')
    p.add_argument('--local-dir', help='Inspect existing h5ad files by basename, without expression downloads')
    p.add_argument('--metadata-only', action='store_true')
    p.add_argument('--panel', help='Optional exact gene IDs or symbols, one per line, no header')
    p.add_argument('--per-background', type=int, default=2)
    p.add_argument('--max-file-gib', type=float, default=5)
    p.add_argument('--max-total-gib', type=float, default=20)
    p.add_argument('--sample-rows', type=int, default=256)
    a = p.parse_args()
    if not 1 <= a.per_background <= 20 or not 1 <= a.sample_rows <= 1024:
        p.error('per-background must be 1..20 and sample-rows 1..1024')
    if not 0 < a.max_file_gib <= a.max_total_gib <= 100:
        p.error('Require 0 < max-file-gib <= max-total-gib <= 100')
    run(a)


if __name__ == '__main__':
    main()
