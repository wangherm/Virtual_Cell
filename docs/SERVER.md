# Server installation and operations

These instructions target a Linux server accessed over SSH, including a GPU cloud
instance or AutoDL. The repository does not connect to or provision a server.
Run every **SERVER** command on your server; **LOCAL** commands run on your laptop.

## Install and check on CPU

SERVER — clone into a persistent location with space for the environment and runs:

```bash
git clone https://github.com/wangherm/Virtual_Cell.git
cd Virtual_Cell
bash scripts/setup.sh cpu
source .venv/bin/activate
python -m pip check
python -m vcell --help
python -m pytest -q
python scripts/smoke_test.py --work-dir runs/smoke_01 --device cpu
```

Use `PYTHON=python3.12 bash scripts/setup.sh cpu` to select a particular Python.
Python must provide `venv` and pip. The script installs in `.venv/`, not system
Python. It requires internet access to the Python package indexes. Notebook and
external-API dependencies are not part of the standard installation.

If a managed Conda or virtual environment is already active, install directly
instead of creating a nested environment:

```bash
python -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

## Install and check CUDA

SERVER — inspect the allocated GPU and driver with `nvidia-smi`. Choose a
compatible PyTorch CUDA wheel index using the [official installation selector](https://pytorch.org/get-started/locally/).

```bash
# Replace the placeholder with the full CUDA index URL from the selector.
TORCH_INDEX_URL='https://download.pytorch.org/whl/cuXXX' bash scripts/setup.sh gpu
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python scripts/smoke_test.py --work-dir runs/gpu_smoke_01 --device cuda
```

The `cuXXX` URL is a placeholder, not an installable command until edited. For a
managed GPU environment with working PyTorch, install `-e '.[dev]'` directly and
run the CUDA check. If switching an existing CPU-only environment to GPU, install
the selected CUDA build explicitly or create a fresh environment; the setup
script does not forcibly replace an already satisfied torch installation.

A successful smoke test prints finite training/validation losses and a report
path, with `best.pt`, `last.pt` and `history.csv` under `run/seed_0/`. Inspect the
prepared dimensions in `demo/prepared/data_audit.json` and GPU memory with
`nvidia-smi` before starting full training. Smoke success verifies execution, not
real-data scientific performance. Training uses one visible device; select a GPU
with `CUDA_VISIBLE_DEVICES=0` when needed.

## Store data and run long experiments

Use persistent storage for raw counts, prepared data, checkpoints and logs. On
AutoDL, a possible checkout is `/root/autodl-tmp/virtual_cell`; check your instance's
actual storage persistence and available space. Paths in this guide are examples.
Treat `data/raw/` as immutable. Re-preprocess into a new directory when settings change.

SERVER — enter a persistent terminal session (install `tmux` through your server's
normal package management if it is unavailable):

```bash
tmux new -s vcell
cd /absolute/path/to/Virtual_Cell
source .venv/bin/activate
mkdir -p logs
set -o pipefail
python -u -m vcell run --config configs/pilot.yaml \
  --data /absolute/path/to/prepared --output runs/pilot_01 --device cuda \
  2>&1 | tee logs/pilot_01.log
```

Detach with Ctrl-b, then d; reconnect with `tmux attach -t vcell`. Replace
`configs/pilot.yaml` with `configs/real.yaml` and use a **new** output/log name
for the full experiment. Copy presets into `configs/local/` before customizing
them. Preserve the local YAML with the results, even though Git ignores it.

## Resume without changing the experiment

Rerun the original command with identical paths, configuration and device, adding
`--resume`:

```bash
python -u -m vcell run --config configs/pilot.yaml \
  --data /absolute/path/to/prepared --output runs/pilot_01 --device cuda --resume
```

The manifest fingerprints source, effective configuration, prepared data and
external teacher caches. Any mismatch requires a new output directory. `last.pt`
stores optimizer state and epoch history; `best.pt` is the selected inference
bundle. Completed fits are reused. The smoke helper also requires `--resume`
explicitly and refuses to overwrite a nonempty work directory.

Preserve the prepared data path: subsequent evaluation loads it from the run
manifest. Copying only the HTML report is fine for viewing; copying only a run
directory is insufficient to reproduce evaluation on another machine.

## Pull updates

SERVER — let active jobs finish, or keep them on their original checkout. Record
the commit for an experiment before updating:

```bash
git rev-parse HEAD
git status --short
git pull --ff-only origin main
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/smoke_test.py --work-dir runs/smoke_after_update_01 --device cpu
```

Commit or preserve your source changes before pulling. `--ff-only` refuses a
diverged history instead of creating an unexpected merge. Use a new smoke path
on every update; avoid pulling into an active training checkout or resuming an
old run with changed source.

To reproduce a previous experiment, use the recorded commit in a separate
checkout with its corresponding environment and unchanged prepared data.

## View reports on your laptop

LOCAL — copy the validation report directory over SSH, then open `report.html`:

```bash
scp -r USER@HOST:/absolute/path/to/Virtual_Cell/runs/pilot_01/evaluation_validation ./pilot_01_report
```

For the smoke helper, the report is under `runs/smoke_01/run/evaluation_validation/`.
CSV files can be inspected directly over SSH. Back up complete runs, prepared
data and local configuration separately; Git does not store them.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `No module named vcell` | Activate the environment and install `python -m pip install -e .` from the checkout. |
| CUDA unavailable | GPU allocation, `nvidia-smi`, selected PyTorch build, and visible devices; use `--device cpu` for a CPU run. |
| Existing/nonempty output directory | Resume the exact run with `--resume`, or choose a new output directory. |
| Fingerprint mismatch | Source, config, data or teacher cache changed; keep the old run and start a new one. |
| Missing field / non-integer counts | Run `inspect` and correct the manifest using the [data guide](DATA.md). |
| Empty train/val/test partition | Check context assignments, label mapping and minimum group/control counts. |
| Out of memory | Reduce batch size or feature count for a new experiment; preprocessing may load a counts layer into memory. |
| No space left | Choose a larger persistent volume or free space for environments, package caches, data and runs. |
