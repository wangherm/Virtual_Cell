# VCell Teacher–Student Experiments

[![Tests](https://github.com/wangherm/Virtual_Cell/actions/workflows/tests.yml/badge.svg)](https://github.com/wangherm/Virtual_Cell/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A command-line research pipeline for predicting **mean gene-expression changes**
under perturbation across cellular contexts. Clone it onto a Linux server, install
the Python package, and run experiments from the terminal. No Colab, notebook,
Google Drive, or external model API is required.

Two workflows are available: the original reference-model experiments and a
pretrained **Qwen3 numerical student** with optional State/scGPT/scFoundation
teacher caches. Biological weights are downloaded separately. This repository
predicts mean responses, not single-cell distributions.

## New: pretrained Qwen server smoke test

On the existing AutoDL server, reuse the prepared real-data pilot:

```bash
cd /root/autodl-tmp/virtual_cell
git pull --ff-only
bash scripts/run_qwen_autodl.sh
```

The script preserves the installed CUDA PyTorch, installs the optional Qwen
dependencies, and downloads pinned `Qwen/Qwen3-0.6B-Base` weights into the data
disk. It trains on 128 prepared training groups for two epochs and validates on
64 groups. **This first check is supervised Qwen, not three-teacher distillation.**
No test scores are revealed. Watch `vcell-work/logs/qwen_smoke_01.log`; successful
runs print `QWEN RUN COMPLETE` and write `runs/qwen_smoke_01/summary.csv`.

Run inside `screen` for SSH disconnection protection. Resume an unchanged run
with `bash scripts/run_qwen_autodl.sh --resume`. Use a fresh output directory for
changed code/configuration. See **[the foundation-model guide](docs/FOUNDATION.md)**
for checkpoint reload, larger pilots and the three-teacher preparation pipeline.

After the Qwen smoke succeeds, connect the first real biological teacher:

```bash
bash scripts/run_scgpt_autodl.sh
```

This downloads and verifies the author-released scGPT-human checkpoint (about
207 MB), extracts frozen control features, trains a conditional response head,
and runs supervised versus scGPT-distilled Qwen. It reuses the working CUDA
environment and local Qwen files. See **[the scGPT run guide](docs/SCGPT_FIRST_RUN.md)**
for progress, outputs, larger experiments and the published-source audit limits.

There is one Qwen student architecture; each comparison arm trains a separate
copy from the same initialization. The three-teacher joint arm is still gated:
State needs a compatible checkpoint excluding both held-out contexts, and
scFoundation still needs real-weight verification. Missing teachers are never
replaced with random networks. scGPT/scFoundation use custom response heads;
State uses native State Transition predictions.

## Quick start on a server

The following is the original four-reference-teacher workflow, trained from
scratch. It remains available as a baseline.

Requires Git, Bash, Python **3.10+** with venv/pip, and enough disk space for PyTorch
and experiment outputs. Python 3.12 is used in CI. The synthetic check works on CPU
and downloads no biological dataset.

```bash
git clone https://github.com/wangherm/Virtual_Cell.git
cd Virtual_Cell

bash scripts/setup.sh cpu
source .venv/bin/activate
python -m pytest -q
python scripts/smoke_test.py --work-dir runs/smoke_01 --device cpu
```

This generates synthetic counts, preprocesses them, trains all four teachers and
all five four-student experiments, and writes a validation report:

```text
runs/smoke_01/
├── demo/prepared/                     # Prepared synthetic data and audit
└── run/
    ├── run_manifest.json             # Effective configuration and fingerprints
    ├── seed_0/                       # Checkpoints, predictions and training logs
    └── evaluation_validation/
        ├── report.html
        ├── summary.csv
        └── paired_comparisons.csv
```

Use a fresh work directory for a new run. To continue the same smoke test:

```bash
python scripts/smoke_test.py --work-dir runs/smoke_01 --device cpu --resume
```

Synthetic scores check the software workflow; they are not evidence of biological
generalization. For CUDA installation, persistent storage, tmux, updating with
Git, and downloading reports, see the **[server guide](docs/SERVER.md)**.

## Run a real-data experiment

All commands below run from the repository root with the environment activated.
You supply the .h5ad raw-count files; no real dataset is bundled.

```bash
mkdir -p configs/local
cp configs/data.example.yaml configs/local/data.yaml
python -m vcell inspect /absolute/path/to/k562.h5ad
```

Edit `configs/local/data.yaml`: set **absolute paths** for every dataset and
`output_dir`, verify the gene/perturbation/batch/count fields and control labels,
and review the train/validation/test contexts. Inspect each input, not only K562.
The [data guide](docs/DATA.md) explains the schema and preprocessing audit.

```bash
python -m vcell prepare --manifest configs/local/data.yaml

# Replace /absolute/path/to/prepared with the manifest's output_dir.
python -u -m vcell run --config configs/pilot.yaml \
  --data /absolute/path/to/prepared --output runs/pilot_01 --device cpu

# After validating the pilot, run the full budget. Use cuda on a tested GPU server.
python -u -m vcell run --config configs/real.yaml \
  --data /absolute/path/to/prepared --output runs/real_01 --device cpu
```

| Preset | Seeds | Teacher epochs | Student epochs | Patience |
| --- | --- | --- | --- | --- |
| `configs/smoke.yaml` | 0 | 5 | 5 | 5 |
| `configs/pilot.yaml` | 0 | 20 | 20 | 6 |
| `configs/real.yaml` | 0, 1, 2 | 100 | 80 | 15 |

Budgets are upper bounds; validation early stopping can shorten training. Full
real-data CPU training may be slow. See [configuration rules](configs/README.md)
before copying or moving YAML files.

## Evaluation and inference

`run` automatically writes **validation** metrics and an HTML report. Freeze the
experiment choice using validation results before revealing test scores:

```bash
python -m vcell evaluate --run runs/real_01 --include-test
```

The primary comparison is `contrastive/mean` versus `kd/mean`. In
`paired_comparisons.csv`, `delta_mse` is candidate minus reference; negative values
favor the candidate. Keep the no-change and mean-transfer baselines in the
comparison. See [experiment design](docs/EXPERIMENTS.md).

For inference in a new context, provide control cells and a CSV of training-known
perturbations. Select a mode using validation results first; `supervised` below
is an example, not a claim that it is best.

```bash
python -m vcell predict-controls \
  --checkpoint runs/real_01/seed_0/supervised/best.pt \
  --manifest configs/local/query.yaml \
  --targets /absolute/path/to/targets.csv \
  --output runs/new_context_01 --device cpu
```

See [control-only inference](docs/DATA.md#control-only-inference) for query and
target formats. Outputs are group means, not simulated individual cells.

## Project layout

```text
src/vcell/             Python package and the python -m vcell / vcell CLI
configs/               Shared smoke, pilot, full-training and data templates
scripts/               Server setup, terminal smoke test and source packaging
tests/                 Core and server workflow tests; no notebook dependency
docs/                  Server, data, experiment and reference documentation
docs/legacy/           Archived v0.2.1 Chinese Colab walkthrough
docs/qa/               Historical synthetic QA reports
examples/colab/        Archived optional notebook, outside the server workflow
optional_tests/        Notebook and external-API extension tests, opt-in only
```

`data/`, `runs/`, `logs/`, `.venv/` and `configs/local/` are ignored by Git. Keep
large inputs and results on persistent server storage; a Git pull only updates
source code. The default pipeline uses one device, not distributed training.

## Documentation

- [Server installation and operations](docs/SERVER.md)
- [Data preparation and inference](docs/DATA.md)
- [Models, losses, splits and experiment interpretation](docs/EXPERIMENTS.md)
- [Importing external teacher predictions](docs/EXTERNAL_TEACHERS.md)
- [Validation scope and historical evidence](docs/VALIDATION.md)
- [References](docs/REFERENCES.md)
- [Optional API-driven training controller](docs/advanced/AGENT.md)

Code is released under the [MIT license](LICENSE). External datasets, models and
weights retain their own licenses. Version 0.3.0 changes deployment and project
organization; it does not change the teacher–student model or loss design.
