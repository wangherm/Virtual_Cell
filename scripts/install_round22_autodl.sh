#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=${VCELL_WORK_DIR:-/root/autodl-tmp/vcell-work}
python3 -m venv "$work/envs/shinka"
"$work/envs/shinka/bin/python" -m pip install -r "$repo/experiments/round22_evolve/requirements.txt"
"$work/envs/shinka/bin/python" -c 'from importlib.metadata import version; from shinka.core import ShinkaEvolveRunner; print("SHINKA INSTALLED:", version("shinka-evolve"))'
"$work/envs/shinka/bin/python" -m pip freeze > "$work/envs/shinka/installed.txt"
