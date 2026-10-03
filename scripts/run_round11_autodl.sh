#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
work="${VCELL_WORK:-/root/autodl-tmp/vcell-work}"
mkdir -p "$work/logs" "$work/tmp" "$work/cache"
export TMPDIR="$work/tmp" HF_HOME="$work/models/huggingface" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
log="$work/logs/round11_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log") 2>&1
echo "LOG: $log"
cd "$repo"
exec "$repo/.venv/bin/python" -u scripts/run_round11.py --work-dir "$work" "$@"
