#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
work="${VCELL_WORK:-/root/autodl-tmp/vcell-work}"
mkdir -p "$work/logs" "$work/tmp" "$work/cache"
export TMPDIR="$work/tmp" PIP_CACHE_DIR="$work/cache/pip" PYTHONUNBUFFERED=1
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
log="$work/logs/public_background_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log") 2>&1
echo "LOG: $log"
cd "$repo"
exec "$repo/.venv/bin/python" -u scripts/inspect_public_background.py --output "$work/background/public_inspect_01" "$@"
