# Validation scope

## Server workflow (v0.3.0)

The [tests workflow](../.github/workflows/tests.yml) validates the documented Linux
server path on CPU:

1. Install using `bash scripts/setup.sh cpu` in a fresh Python 3.12 environment.
2. Run dependency checks, CLI help, core tests and overwrite-protection tests.
3. Run `scripts/smoke_test.py` from outside the repository checkout.
4. Resume the same smoke run and regenerate its validation report.

Default test dependencies exclude notebook tooling and external model APIs.
Notebook schema validation is opt-in via `.[dev,notebook]`; API-controller tests
are opt-in via `.[dev,agent]`. See the [latest CI runs](https://github.com/wangherm/Virtual_Cell/actions/workflows/tests.yml)
for the result on each specific commit. A configured test is not a passed test.

The server refactor changes documentation, entry scripts, optional dependencies
and project organization. Model, loss, preprocessing and evaluation algorithms
are preserved. The version and CUDA diagnostic text change source fingerprints,
so old v0.2.1 runs should be resumed from their original checkout.

## Historical v0.2.1 evidence

The supplied release documented 25 passing core tests (including one notebook
schema test) and a synthetic workflow with 96 genes, 20 targets, four contexts,
four teachers and five four-student modes, each with up to five epochs.

The original import also passed [GitHub CPU CI](https://github.com/wangherm/Virtual_Cell/actions/runs/36340750612).
The archived CSV/HTML files in [qa](qa/) came from the supplied v0.2.1 synthetic
validation, not a new real-data or server-GPU experiment. The historical
[Chinese guide](legacy/STRATEGY_AND_HOWTO_ZH.md) describes that old notebook workflow.

Those short synthetic experiments did not establish a benefit from the
contrastive objective; the mean-transfer baseline remained stronger. Treat
synthetic runs as execution checks, not biological evidence.

## Not established by CPU CI

- CUDA installation, memory requirements and training on your specific server.
- Complete real-data preprocessing/training on the Replogle/Nadig releases.
- Biological generalization or superiority of any teacher/student objective.
- Online calls from the optional API controller.

Run a CUDA smoke check on the allocated server and inspect its audit, losses and
GPU memory before full training. Record the Git commit, environment and effective
configuration for every real experiment.
