#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
work="${VCELL_WORK:-/root/autodl-tmp/vcell-work}"
mkdir -p "$work/logs" "$work/tmp"
export TMPDIR="$work/tmp" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
log="$work/logs/round15_D1_adapter_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log") 2>&1
echo "LOG: $log"
exec "$repo/.venv/bin/python" -u "$repo/scripts/adapt_round15_mixscale.py" --work-dir "$work" "$@"
