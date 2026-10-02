# Round 10: background granularity and transfer to additional contexts

Round10 narrows the Round9 matrix to background attribution. The main candidates
are Qwen B1, Qwen B2 and a new Qwen B3 grouped-background condition, with MLP B0
as the simple reference. Main candidates use no module loss. A small, separately
queued module-warmup comparison tests the early-gradient issue from Round9.

## Launch on AutoDL

```bash
cd /root/autodl-tmp/virtual_cell && \
git -c http.version=HTTP/1.1 pull --ff-only && \
screen -dmS vcell10 bash scripts/run_round10_autodl.sh --name round10_01 --resume
```

Defaults are **36 background pretraining jobs and 54 students**, with two GPU
processes at a time. The 48 core students run before the six warmup diagnostics
in the student queue. All students are freshly fitted to keep this round's
comparisons paired; old Round9 runs are neither overwritten nor automatically
combined into the new scores. No new dataset/model downloads or APIs are needed.

Inputs are the same as Round9: `runs/round8_01/prepared/plus_h1`,
`background/human_background_v1`, `knowledge/round8/knowledge.npz`, and
`models/Qwen3-0.6B-Base`, relative to `/root/autodl-tmp/vcell-work`.
Overrides are `--data`, `--background-dir`, `--knowledge-npz`, and `--model-dir`.
The full plan requires at least **37.4 GiB free reserve**; no old artifacts are
deleted. The existing cleaned H5AD files are verified and cached on the data disk.

```bash
/root/autodl-tmp/virtual_cell/.venv/bin/python \
  /root/autodl-tmp/virtual_cell/scripts/round10_status.py

tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round10_pipeline_*.log | head -n 1)"
```

Rerun the same command with `--resume` after interruption. Weights, optimizer,
update count and the warmup schedule resume together. Completed artifacts require
matching hashes. Code, data or scientific-plan changes require a new run name;
only `--parallel-students` can be reduced without changing the plan. This round
must use a new directory, not resume a completed Round9 directory after updating
code. No server job is started merely by pulling this repository.

After `ROUND10 COMPLETE`, download:

```text
/root/autodl-tmp/vcell-work/runs/round10_01/round10_review_light.tar.gz
```

The light package contains summaries, audits, per-target results, calibration,
gradient histories and log tails. Checkpoints, predictions and expression arrays
remain on the server. A root `COMPLETE.json` requires every requested job and
report to finish; an empty screen listing is not proof of success.

## Main experiment

| Arm | Background pretraining | Module supervision |
|---|---|---|
| `mlp_b0_none` | None in this experiment | None |
| `qwen_b1_none` | Matched control pseudobulks only | None |
| `qwen_b2_none` | Matched controls + Tabula single cells, 1:1 | None |
| `qwen_b3_none` | Matched controls + Tabula group means, 1:1 | None |

Each runs on four protocols with paired seeds 17, 29 and 43: 4 x 4 x 3 = 48
students. These use the same 1,834-gene panel, model families, rank-32 fit-only
response basis, fit-only normalization, 2,000 downstream updates, effective
batch 16, learning rate 0.0002 and fixed endpoint as Round9. Every allowed
downstream training row remains in the sampling pool. No outer-set stopping or
hyperparameter selection occurs; calibration uses a separate calibration pool.

### B3 isolates the public-background input granularity

Only the existing **74,678 Tabula training cells** enter public background
pretraining. A group never crosses dataset, donor, tissue, cell type, assay or
available study/sample identifiers. Groups also share the same measured-feature
mask. Each group is an average of the already normalized **log expression**:
`mean(log1p(10000 * counts / full-source library size))`. This matches the
processing order used by the prepared matched control means. It is not
`log1p` of pooled counts and does not pool across donors or cell types.

B2 and B3 use **exactly the same original cell-index schedule**. In B3, each
sampled cell is replaced by its group's mean. A 100-cell group therefore retains
100 times the exposure of a singleton under uniform cell sampling. This avoids
silently adding group reweighting to the granularity comparison. Small groups,
including singletons, are retained and their size distribution is audited.
The grouping does not create additional independent biological samples.

Background budgets stay fixed at 5,840 updates, batch 256 for the reviewed
reference (10 public-cell passes rounded to full batches). B1 repeats its smaller
matched-control pool at the same update budget. B2/B3 retain the same 1:1 matched
versus public exposure. The original missing-feature masks remain active in the
input and reconstruction loss. Pretrained control projectors transfer into the
existing students and continue to train on actual perturbation responses.

`background_aggregates/audit.json` and `groups.csv` record grouping and source
cell counts. `cell_to_group.npy` records exact membership on the server. Training
aggregates never include validation or external cells.

Background diagnostics evaluate all B1/B2/B3 models on **both** the same held-out
single cells and a separately built view of held-out group means. Grouped results
include group-equal and cell-count-weighted reconstruction error plus per-group
donor/type/sample metadata. They are background reconstruction diagnostics, not
perturbation-prediction scores. The original Tabula donor partition stays fixed;
this round adds context holdouts, not donor cross-validation. HLCA and immune
references still provide only unverified-overlap single-cell diagnostics.

## Four development protocols

- `context`: the existing full HepG2 outer panel; fitting and calibration use
  disjoint training-pool batches, as in Round9.
- `target`: the same held-target and separate calibration-target split as Round9.
- `context_rpe1`: remove **all RPE1 rows** from fitting and calibration, then
  evaluate on those RPE1 rows. Other original training contexts supply fit and
  calibration rows. HepG2 and Jurkat responses are not imported into this fit.
- `context_h1`: similarly remove **all H1_hESC rows** from fitting and calibration.
  K562/RPE1 provide fitting and calibration; H1 becomes the outer context.

New context protocols also fit their own background normalization, matched
controls, response basis, pretraining and calibration without their held context.
Every protocol writes split membership and target coverage. Report results
separately for outer targets seen and unseen in that protocol's fit pool: a new
context with many unseen targets is harder than context transfer alone.

RPE1 and H1 were previously used during development; reassigning them creates
new **development splits**, not independent unseen datasets. Batch labels may
represent pseudo-batches. Jurkat is never fitted, scored, or packaged.

## Limited module warmup diagnostic

Only on the HepG2 `context` protocol, Qwen B2 adds true-module and matched-random
auxiliary losses with a fixed **500-update linear warmup from 0 to 0.1**. Those
two arms x three seeds add six students. The existing Qwen B2 no-module arm is
their paired reference. Initial weights, module library, random permutation,
training rows and all other settings remain fixed. The schedule is not tuned
against the outer panel. It does not also introduce a smaller final coefficient,
PCGrad, GradNorm, a new module library or different head initialization.

The matched random matrix still uses one fixed gene-column permutation, seed
818. Three optimizer seeds are not three random-module replicates. Warmup
diagnostics report the actual loss weight and control-projector gradient norms
and cosine. Modules remain off in the main background comparison regardless of
these results. Round9 cold-start auxiliary results remain historical evidence,
not silently pooled into this fresh paired comparison.

The unchanged semantic/STRING probes and rank oracles are **not rerun**. Existing
knowledge supplies only the fixed module matrix for the small warmup test.

## Reporting and interpretation

`comparison.csv`, `seed_summary.csv`, `per_context.csv` and `per_target.csv`
separate raw, shrunk, calibrated, mean-transfer and zero-change results.
`target_coverage_scores.csv` separates fit-seen versus fit-unseen targets.
`paired_comparisons.csv` contains each seed's primary-metric improvements;
`paired_diagnostics.csv` reports seed-averaged paired improvements, unit win rates,
median improvements and removal of the largest-contributing target.

Primary MSE weights contexts equally, targets equally within context, then rows
equally within target. Sensitivity calculations preserve this rule instead of
flattening all targets into one mean. Bootstrap resamples entire targets across
contexts after averaging paired seeds, accepting draws that represent every
context. Counts of accepted draws and the interval definition are recorded.
If deleting the largest-contributing target would remove a whole context, that
sensitivity result is unavailable instead of silently changing the evaluation
population. Intervals are exploratory and not adjusted for repeated development
or selection among many candidates. No automatic winner/test release occurs.

## Optional infrastructure check

Use a separate name to avoid mixing this with the full run:

```bash
bash scripts/run_round10_autodl.sh --name round10_smoke_01 --resume \
  --protocols context --seeds 17 --arms qwen_b2_none qwen_b3_none \
  --max-steps 10 --pretrain-steps 10 --save-every 5
```

`--prepare-only` verifies data, creates background caches/aggregates and split
audits, and writes jobs without fitting them. Resume the same plan without that
flag to start training. To omit warmup diagnostics entirely, pass the four main
arm names with `--arms`.
