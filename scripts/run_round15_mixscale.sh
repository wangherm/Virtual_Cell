#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
output="${1:?output directory required}"
work="${2:?work directory required}"
mkdir -p "$output"
rprefix="$work/envs/round15-r"
if [ ! -x "$rprefix/bin/Rscript" ]; then
  conda_cmd="$(command -v conda || true)"
  if [ -z "$conda_cmd" ] && [ -x /root/miniconda3/bin/conda ]; then conda_cmd=/root/miniconda3/bin/conda; fi
  if [ -n "$conda_cmd" ] && [ ! -d "$rprefix" ]; then
    # A dedicated environment; never alter the training environment or replace an existing prefix.
    if ! "$conda_cmd" create -y -p "$rprefix" -c conda-forge 'r-base>=4.3,<4.6' 'r-seuratobject>=5,<6' r-matrix r-jsonlite; then
      echo "R environment setup failed; download and report runtime status anyway."
    fi
  fi
fi
rargs=()
if [ -x "$rprefix/bin/Rscript" ]; then
  rargs=(--rscript "$rprefix/bin/Rscript")
  "$rprefix/bin/Rscript" -e 'print(sessionInfo())' > "$output/r_environment.txt"
fi
exec "$repo/.venv/bin/python" -u "$repo/scripts/round15_data.py" --output "$output" --download "${rargs[@]}"
