#!/usr/bin/env bash
# Install into a project-local virtual environment; do not modify system Python.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mode="${1:-cpu}"
if [[ "$#" -gt 1 || ! "$mode" =~ ^(cpu|gpu)$ ]]; then
  echo "Usage: bash scripts/setup.sh [cpu|gpu]" >&2
  exit 2
fi
if [[ "$mode" == gpu && -z "${TORCH_INDEX_URL:-}" ]]; then
  echo "GPU setup needs TORCH_INDEX_URL from https://pytorch.org/get-started/locally/" >&2
  echo "Select a CUDA wheel compatible with the server's driver, then rerun." >&2
  exit 2
fi

python_bin="${PYTHON:-python3}"
"$python_bin" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ is required"'
if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv
fi
python_bin="$ROOT/.venv/bin/python"
"$python_bin" -m pip install --upgrade pip
if [[ "$mode" == cpu ]]; then
  "$python_bin" -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
else
  "$python_bin" -m pip install 'torch>=2.2,<3' --index-url "$TORCH_INDEX_URL"
fi
"$python_bin" -m pip install -e '.[dev]'
if [[ "$mode" == gpu ]]; then
  "$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable: check the GPU allocation, driver and PyTorch build"; print(torch.cuda.get_device_name(0))'
fi
echo "Setup complete. Activate with: source .venv/bin/activate"
