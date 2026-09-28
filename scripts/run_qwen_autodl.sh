#!/usr/bin/env bash
# Run from a JupyterLab terminal or inside screen. Reuse working AutoDL PyTorch.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
work="${VCELL_WORK:-/root/autodl-tmp/vcell-work}"
export HF_HOME="${HF_HOME:-$work/models/huggingface}"
export TMPDIR="${TMPDIR:-$work/tmp}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TMPDIR" "$work/logs"
cd "$repo"
if [[ ! -x .venv/bin/python ]]; then
  python -m venv --system-site-packages .venv
fi
py="$repo/.venv/bin/python"
"$py" -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available()); assert torch.cuda.is_available(), "The current environment needs working CUDA PyTorch"; x=torch.ones(8,device="cuda"); print("GPU:",torch.cuda.get_device_name(0),"probe:",x.sum().item())'
"$py" -m pip install --no-cache-dir -e '.[foundation]'
"$py" -m vcell qwen-run --config configs/qwen_smoke.yaml \
  --data "$work/prepared/real_min10_01" --output "$work/runs/qwen_smoke_01" "$@" \
  2>&1 | tee -a "$work/logs/qwen_smoke_01.log"
