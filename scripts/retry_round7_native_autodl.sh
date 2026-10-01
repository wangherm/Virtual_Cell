#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
work="${VCELL_WORK:-/root/autodl-tmp/vcell-work}"
mkdir -p "$work/logs" "$work/tmp"
export TMPDIR="$work/tmp" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
log="$work/logs/round7_native_retry_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log") 2>&1
echo "LOG: $log"
cd "$repo"
exec "$repo/.venv/bin/python" -u scripts/retry_round7_native.py --work-dir "$work" "$@"
