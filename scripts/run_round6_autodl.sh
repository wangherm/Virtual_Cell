#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
work="${VCELL_WORK:-/root/autodl-tmp/vcell-work}"
export HF_HOME="${HF_HOME:-$work/models/huggingface}"
export TMPDIR="${TMPDIR:-$work/tmp}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$work/logs" "$TMPDIR"
log="$work/logs/round6_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log") 2>&1
echo "LOG: $log"
cd "$repo"
py="$repo/.venv/bin/python"
if [[ ! -x "$py" ]]; then echo "Missing existing .venv" >&2; exit 1; fi
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
"$py" scripts/run_round6.py --work-dir "$work" "$@"
