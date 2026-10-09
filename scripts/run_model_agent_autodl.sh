#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
work="${VCELL_WORK:-/root/autodl-tmp/vcell-work}"
mkdir -p "$work/logs" "$work/tmp"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 TMPDIR="$work/tmp"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log="$work/logs/model_agent_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log") 2>&1
echo "LOG: $log"
cd "$repo"
exec "$repo/.venv/bin/python" -u scripts/model_agent.py "$@"
