#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
work="${VCELL_WORK:-/root/autodl-tmp/vcell-work}"
export HF_HOME="${HF_HOME:-$work/models/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export TMPDIR="${TMPDIR:-$work/tmp}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Downloads are verified before model loading switches to local files only.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p "$HF_HOME" "$TMPDIR" "$work/logs"
log="$work/logs/round5_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log") 2>&1
echo "LOG: $log"
cd "$repo"
py="$repo/.venv/bin/python"
if [[ ! -x "$py" ]]; then
  echo "Missing .venv: first complete the documented Qwen environment setup." >&2
  exit 1
fi
"$py" -c 'import torch, transformers, peft, huggingface_hub; print("PyTorch:",torch.__version__,"GPU:",torch.cuda.get_device_name(0)); print("CUDA probe:",torch.ones(8,device="cuda").sum().item())'
# Use this checkout directly; the Qwen smoke already installed the dependencies.
# No package-index access, legacy scgpt stack or replacement CUDA wheel is needed.
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
"$py" scripts/run_round5.py --work-dir "$work" "$@"
