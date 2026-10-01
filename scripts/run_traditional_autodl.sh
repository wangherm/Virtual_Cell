#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
family="$1"; inputs="$2"; output="$3"; work="$4"; targets="$5"
export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
# Native workers import no vcell modules. Avoid exposing its editable metadata to pip.
unset PYTHONPATH
export PYTHONNOUSERSITE=1
mkdir -p "$work/envs" "$work/cache/pip" "$work/cache/conda" "$work/cache/matplotlib" "$work/tmp"
export PIP_CACHE_DIR="$work/cache/pip" TMPDIR="$work/tmp"
export CONDA_PKGS_DIRS="$work/cache/conda" XDG_CACHE_HOME="$work/cache" MPLCONFIGDIR="$work/cache/matplotlib"
if [[ "$family" == celloracle ]]; then
    py="$work/envs/celloracle/bin/python"
    if [[ ! -x "$py" ]]; then
        conda_bin="${CONDA_EXE:-/root/miniconda3/bin/conda}"
        "$conda_bin" create -y -p "$work/envs/celloracle" python=3.10 pip
    fi
    if [[ ! -f "$work/envs/celloracle/.vcell-round7-ready" ]]; then
        "$py" -m pip install 'numpy==1.26.4' 'Cython<3' 'setuptools<81' wheel
        "$py" -m pip install --no-build-isolation -r "$repo/configs/celloracle_requirements.txt"
        "$py" -c 'import celloracle; assert celloracle.__version__ == "0.20.0"'
        "$py" -m pip freeze > "$work/envs/celloracle/installed.txt"
        touch "$work/envs/celloracle/.vcell-round7-ready"
    fi
else
    [[ "$family" == sctenifold ]] || exit 2
    py="$work/envs/sctenifold/bin/python"
    if [[ ! -x "$py" ]]; then
        "$repo/.venv/bin/python" -m venv "$work/envs/sctenifold"
    fi
    if [[ ! -f "$work/envs/sctenifold/.vcell-round7-ready" ]]; then
        "$py" -m pip install 'scTenifoldpy==0.4.0' 'numpy<3' 'pandas<3'
        "$py" -m pip freeze > "$work/envs/sctenifold/installed.txt"
        touch "$work/envs/sctenifold/.vcell-round7-ready"
    fi
fi
exec "$py" -u "$repo/scripts/run_traditional_teacher.py" --family "$family" --inputs "$inputs" --output "$output" --threads 4 --max-targets "$targets"
