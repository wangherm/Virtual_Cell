#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=${VCELL_WORK_DIR:-/root/autodl-tmp/vcell-work}
mkdir -p "$work/logs"
log="$work/logs/round20_pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "LOG: $log"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HOME="$work/models/huggingface"
"$repo/.venv/bin/python" -u "$repo/scripts/run_round20.py" --work-dir "$work" "$@" 2>&1 | tee -a "$log"
