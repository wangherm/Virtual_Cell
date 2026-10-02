#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
work="${VCELL_WORK:-/root/autodl-tmp/vcell-work}"
mkdir -p "$work/logs" "$work/tmp"
export TMPDIR="$work/tmp" PYTHONUNBUFFERED=1
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
log="$work/logs/background_clean_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log") 2>&1
echo "LOG: $log"
cd "$repo"
exec "$repo/.venv/bin/python" -u scripts/prepare_public_background.py --work-dir "$work" "$@"
