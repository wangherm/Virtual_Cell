# Virtual Cell: Perturbation Prediction and Autonomous Research

[![tests](https://github.com/wangherm/Virtual_Cell/actions/workflows/tests.yml/badge.svg)](https://github.com/wangherm/Virtual_Cell/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)

A server-first research project for predicting **group-mean gene-expression
changes after perturbation**, and for studying whether an agent can develop
better prediction algorithms from controlled experimental feedback.

The project evolved from teacher–student distillation into a ShinkaEvolve +
DeepSeek research system with isolated numerical evaluation, code repair,
scientific criticism, source-only memory and persistent task management.
It runs from the terminal; Colab and notebooks are optional historical examples.

## Project status

**Research is paused as of 19 October 2026. Round27 is unfinished, according to
the project owner.** Its implementation is available; it is not a completed
experiment or a validated final model. Preserve its full server directory and
parent campaigns for possible resumption.

Round26 demonstrated an executable scientific-critic loop. It did **not**
establish a reliable predictive improvement over the strongest fixed methods.
Earlier supported-repeat experiments produced promising development results,
but reliable cross-cell-line transfer and new independent external validation
remain unresolved. Agent engineering success and biological prediction success
are separate claims. See the [research history](docs/PROJECT_STATUS.md).

This repository contains code, configurations and documented protocols.
**A Git pull does not download model binaries or private experiment artifacts.**
The reviewed [research model release](https://github.com/wangherm/Virtual_Cell/releases/tag/research-models-2026-10-19)
distributes the historical Qwen Student A adapter and seven frozen Round26
algorithms separately from Git history. Round27 is excluded.

## Released model: configuration, dependencies and inference

The Qwen adapter predicts **2,000 ordered gene deltas for 196 saved perturbation
targets**. It uses Qwen3-0.6B-Base at revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, BF16, eight control tokens and LoRA
rank 8 / alpha 16 on `q_proj` and `v_proj`. The 54.75 MB checkpoint stores 13,645,776
trainable parameters, prompts and normalization, with best epoch 7; the full
base is downloaded separately. It is a custom regression adapter, not a
standalone text-generation model. Its State-derived non-commercial terms are
included with the release.

The seven Round26 methods instead use a **1,809-gene** source/query contract and
refit from legal source arrays. Their panel and inputs cannot be substituted
for the Qwen adapter's. They include research/diagnostic candidates, not seven
validated best models.

The [published catalog](assets/research_release_20261019/catalog.json) records
individual hashes and the release archive checksum; [adapter configuration](assets/research_release_20261019/qwen_student_a/config.json),
ordered vocabularies and exact algorithm source files are also readable in Git.

Use Python >=3.10 and the repository dependencies. Qwen additionally requires
the `foundation` extra, including transformers==4.57.3 and peft==0.18.0:

```bash
# In an existing compatible environment; see the server guide for a new GPU setup.
python -m pip install -e '.[foundation]'
```

After downloading and checksum-verifying the release archive and preparing the
pinned base, predict from a query NPZ containing `baseline`, `genes`, `targets`
and unique `row_ids`:

```bash
python scripts/predict_release.py qwen \
  --release-dir /absolute/path/to/virtual-cell-research-models \
  --model-dir /absolute/path/to/Qwen3-0.6B-Base \
  --input /absolute/path/to/queries.npz \
  --output /absolute/path/to/new_predictions.npz \
  --device cuda --batch-size 16
```

`baseline` must be the original-preprocessing matched-control mean expression,
in exact saved gene order; raw counts, another panel and unseen targets are
rejected or unsupported. The helper applies saved standardization and returns
signed expression deltas. It supports relocating a base snapshot only when
its content hashes match. CPU inference uses float32 by default. No DeepSeek
API key is needed.

See the [complete model guide](docs/MODEL_RELEASE_20261019.md) for download and
base preparation commands, the query schema/example, exact configuration,
licensing, software checks and protected Round26 method inference. Original
biological prediction accuracy has not been independently rerun for this release.

## Main research components

| Component | Role |
| --- | --- |
| Numerical baselines | Zero/no-change, mean transfer, Ridge, STRING features and factorized response heads |
| Teacher–student experiments | Pretrained biological encoders with fitted response heads; numerical Qwen students and supervised controls |
| Supported Repeat | Predict Rep2 using the same perturbation's measured Rep1 response; not zero-shot prediction |
| Transfer | Predict in a held-out cell line using legal source information; source OOF is distinct from independent external testing |
| ShinkaEvolve | Generate and retain candidate Python prediction methods |
| DeepSeek / Scientific Critic | Propose methods, diagnose failures and revise source-supported research hypotheses |
| Trusted evaluator | Fix partitions, metrics, resource caps and controls; isolate generated programs on supported Linux systems |
| Round27 agent | Domain catalog, provenance-linked memory and continued task management; execution unfinished |

Teacher encoders include scGPT, scFoundation, Geneformer, State SE and UCE in
historical experiments. State SE is not native State Transition inference.
Encoder embeddings plus a fitted response head are not automatically the native
perturbation model. Pretraining overlap and external model terms must be checked
for each experiment. See [foundation models](docs/FOUNDATION.md) and
[third-party notices](THIRD_PARTY_NOTICES.md).

The later evolved methods are numerical Python algorithms, not automatically
pretrained Qwen language-model inference. DeepSeek develops or selects methods;
that role is distinct from the numerical expression predictor.

## Close out an existing server workspace

The inventory command needs **Python 3.10+ only**, no GPU, API key or new package
installation. It reads existing runs and writes a new snapshot outside them:

```bash
cd /root/autodl-tmp/virtual_cell
python3 -u scripts/project_closeout.py \
  --work-dir /root/autodl-tmp/vcell-work \
  --name project_closeout_01 \
  --unfinished-run round27_agent_01
```

Use the actual Round27 run name if different; repeat `--unfinished-run` for
additional unfinished campaigns. Each snapshot name must be new. Outputs:

```text
/root/autodl-tmp/vcell-work/closeout/project_closeout_01/
  REPORT.md                 English progress summary
  runs.json / runs.csv       Completion declarations and reported scientific flags
  models.csv                Checkpoint and evolved-method inventory
  reported_metrics.csv      Existing aggregate scores, not recomputed results
  model_inventory.json      Model paths, sizes and recorded provenance
  MODEL_CARD.md             Research use, dependencies and limitations
  selected_models.json      Explicit exports only; empty in inventory-only mode
  ARCHIVE_MANIFEST.json     Payload sizes and SHA-256 checksums
  project_closeout.tar.gz   Lightweight review package, unless models were requested
  SHA256SUMS                Archive checksum
  progress.json             Closeout operation status
```

Completion markers are inventoried, not silently treated as full integrity
verification. Older runs without markers remain unknown. The tool does not
select a best model, rerun evaluation, delete files or change experiment seals.
It excludes matrices, environment files, provider responses and logs. A light
snapshot is **not a resumable backup**.

For explicit model export, progress commands, private backup guidance and
GitHub publication, follow [project closeout](docs/PROJECT_CLOSEOUT.md).

## Installation and a small reproducible run

For a new checkout, install the CPU environment and exercise the original
synthetic workflow before using real data:

```bash
bash scripts/setup.sh cpu
.venv/bin/python scripts/smoke_test.py --work-dir /tmp/vcell-smoke --device cpu
```

GPU installation requires a compatible PyTorch wheel; follow the
[server guide](docs/SERVER.md). Do not reinstall an existing working server
environment just to collect a closeout report.

Real inputs are supplied separately. The [data guide](docs/DATA.md) documents
inspection, count processing, context labels, control matching and preparation.
Historical 1,834-gene panels and newer 1,809-gene panels are not interchangeable.
Methods predict group means, not trajectories of individual simulated cells.

## Agent operations and inference

For an existing Round27 campaign:

```bash
.venv/bin/python scripts/round27_status.py --name round27_agent_01
.venv/bin/python scripts/round27_agent.py memory --name round27_agent_01
```

Round27 prediction uses a completed registry, an explicitly selected method hash
and a compatible public source/query template. An evolved algorithm usually
refits from legal source arrays; its `.py` file alone is not a standalone trained
checkpoint. A Qwen adapter or response head likewise needs the original base,
architecture, gene order and preprocessing. See the
[Round27 interface](docs/ROUND27_VIRTUAL_CELL_AGENT.md) and the relevant
historical inference guide before distributing or running a model.

Generated code executes through the protected evaluator. Do not bypass the
Linux isolation contract by importing arbitrary generated programs directly.
Paid DeepSeek calls are required for live evolution, but not for inspection or
closeout. Credentials belong in the server environment and must never be
committed or included in a release.

## Repository layout

```text
src/vcell/          Numerical pipelines, encoders, evaluators and agent modules
scripts/            Setup, experiments, status tools, closeout and packaging
configs/            Shared experiment settings; private overrides in configs/local/
experiments/        Protected evolution experiment entry points
tests/              Unit, contract and workflow regression checks
docs/               Protocols, historical rounds and operating instructions
examples/colab/     Optional archived notebook workflow
```

Server data, checkpoints, model caches, environments and run directories stay
outside version control. New closeout code uses English comments and reports.
Historical documents are retained in their original languages where applicable.

## Documentation

- [Current status, experiment phases and terminology](docs/PROJECT_STATUS.md)
- [Server closeout, model export and GitHub release instructions](docs/PROJECT_CLOSEOUT.md)
- [Round27 persistent Virtual Cell agent](docs/ROUND27_VIRTUAL_CELL_AGENT.md)
- [Round26 scientific critic and matched evolution](docs/ROUND26_SCIENTIFIC_CRITIC.md)
- [Round25 self-correcting framework](docs/ROUND25_SELF_CORRECTING_EVOLUTION.md)
- [Round24 mechanism analysis](docs/ROUND24_MECHANISM_EVOLUTION.md)
- [Round23 persistent evolution and external sources](docs/ROUND23_EVOLUTION_FRAMEWORK.md)
- [Round22 ShinkaEvolve integration](docs/ROUND22_SHINKA_EVOLVE.md)
- [Data, models and evaluation](docs/EXPERIMENTS.md)
- [References](docs/REFERENCES.md)
- [Legacy workflows and earlier round documentation](docs/LEGACY_WORKFLOWS.md)

## License and evidence

Repository code is [MIT licensed](LICENSE). External data, annotations, base
models and weights retain their own terms; this code license does not relicense
them. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Do not interpret a completed campaign, a valid generated program, or a lower
development MSE as independent evidence of biological generalization.
