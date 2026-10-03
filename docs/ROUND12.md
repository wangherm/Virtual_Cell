# Round 12: target representation before model complexity

Round11 supports retaining the B2 public-atlas background procedure. It does not
establish universal calibration transfer or reliable unseen-target prediction.
Round12 isolates target representations with Ridge and small MLP heads, then
compares the original training pool with uncapped K562/RPE1 training targets.
This is a development experiment, not a new untouched-test release.

## Start on AutoDL

Keep the existing data, local Qwen snapshot, cleaned backgrounds, original raw
K562/RPE1 H5AD files, and complete Round10 run at their recorded paths. The light
review archives are insufficient. No package upgrade is required beyond the
existing working Round10 environment.

```bash
cd /root/autodl-tmp/virtual_cell && \
git -c http.version=HTTP/1.1 pull --ff-only && \
screen -dmS vcell12 bash scripts/run_round12_autodl.sh --name round12_01 --resume
```

`&&` prevents launch if Git fails. The script first hashes and streams the raw
files, creates new prepared data, builds annotation-only gene knowledge, then
trains and reports. During raw preparation the GPU can be idle. Existing data
and completed experiments are not overwritten or deleted. A new run requires
15 GiB free reserve; the expanded pseudobulks are small compared with the raw
files. This reserve is a preflight floor, not a precise storage forecast.

MyGene and STRING public APIs need network access for the new target annotations.
Their requests are cached under `knowledge/round12/requests`; rerunning resumes
cached work. Hugging Face stays offline and gene text uses the existing local
Qwen weights. No new raw dataset or large model is downloaded. On API failure,
inspect the log and resume; no synthetic annotation fallback is used.

Defaults: three paired optimizer seeds (17, 29, 43), two data conditions, three
protocols, eight representations, two predictors. This means 18 fresh background
jobs and 18 benchmark jobs containing 288 small heads. Backgrounds copy the
fixed Round10 B2 update budget (5,840 updates in the reviewed run). MLP heads
use 2,000 fixed updates, batch 128, hidden width 128. Two workers run concurrently,
with two CPU math threads each. Ridge runs a five-alpha calibration grid; it is
not an iterative GPU workload. This round does not fine-tune the Qwen backbone.

Use `--parallel-jobs 1` to reduce concurrency. `--prepare-only` stops after data,
knowledge and job configuration; rerun without it and with `--resume` to train.
`--pretrain-steps` and `--max-steps` are explicit budget overrides, recorded in the
plan; changing them requires a new run name. Do not interpret a shortened pilot
as the full-budget result.

## Progress, recovery and review

```bash
/root/autodl-tmp/virtual_cell/.venv/bin/python \
  /root/autodl-tmp/virtual_cell/scripts/round12_status.py

tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round12_pipeline_*.log | head -n 1)"
```

Each worker prints the current representation, predictor and update. Raw sources
are cached after a whole source finishes; interruption mid-source restarts that
source. Background and MLP optimizers checkpoint every 100 updates. Ridge
restarts only the unfinished head. Completed heads are checksum-verified.
Use the same launch command with `--resume`; code/input/plan changes require a
new name. Keep `last.pt` until the entire experiment completes.

The pipeline creates:

`/root/autodl-tmp/vcell-work/runs/round12_01/round12_review_light.tar.gz`

It contains reports, metadata and short log tails, excluding model weights,
prediction arrays, raw counts and background matrices. Download this archive
for review. Main files:

- `comparison_mean.csv` and `comparison.csv`: mean and individual-seed scores.
- `paired_comparisons.csv`: paired representation/expansion differences, target
  bootstrap intervals, wins and removal of the largest-gain target.
- `target_specificity.csv`: own/other correlations, tie-aware retrieval rank.
- `diagnostics/half_specificity.csv`: raw-cell technical repeatability.
- `diagnostics/expansion_qc.csv`: eligible, rejected and added groups.
- `annotation_coverage.csv`: actual functional-summary availability and STRING
  panel-neighbor counts, including unmapped and ambiguous genes.
- `splits/*/membership.csv` and `audit.json`: complete membership and scope.
- `benchmarks/*/*/coverage.json`: missing annotation coverage and model capacity.
- `report.html`: compact overview; `evaluation_scope.json`: limitations.

## Fixed comparisons

| Protocol | Outer evaluation | Required exclusion |
|---|---|---|
| target | Historical source contexts, held-out targets | Outer targets excluded from fitting in every context |
| context | HepG2, targets seen in the fit pool | All HepG2 response labels excluded from fitting/calibration |
| double | H1, globally held-out targets | All H1 responses and all H1 outer target labels excluded from source fitting |

Partitions use identifiers only, with seed-817 hash ordering. Calibration targets
are disjoint from fit targets and drawn from historical source rows. Expansion
does not add calibration or evaluation rows. Jurkat is neither scored nor used
for selection. HepG2/H1 and the historical panel have been inspected in earlier
rounds: do not describe this as an untouched benchmark.

Both data conditions use the exact same per-gene knowledge snapshot and gene
panel. New data only append targets absent from the corresponding original
K562/RPE1 context; no extra rows for old context/target pairs are added. Raw
counts must be finite, nonnegative and integer, with positive full-library size.
Existing row filters and minimum perturbation/control counts are retained;
there is no new mitochondrial-percent filtering. Compound labels are rejected.
If study/intervention/time varies within a declared batch, preparation stops
instead of silently pooling incompatible controls. Additional candidate counts
are not automatically interpreted as QC-certified biological samples.

Each fold/seed/data condition trains a fresh B2 control projector on its fit
controls and the same public background training cells, then freezes it. Old
supervised endpoints are not reused. Projector outputs are pooled and reduced
to 16 dimensions using fit rows only. The output basis is fit-only rank 32.
Thus the same procedures/budgets are frozen, not a potentially contaminated
historical final checkpoint. This small-head diagnostic differs from the full
Qwen sequence model and must not be labeled a new Qwen student result.

Representations:

- `none`: background only; essential identity-free control.
- `id`: fit-target one-hot; unseen targets are all-zero. This is an additive
  control with no background interaction; its capacity is not dimension-matched.
- `random`: fixed gene-keyed random vectors, unchanged when vocabulary expands.
- `text`: existing frozen Qwen gene-function-card embeddings, fit-only reduction.
- `string`: STRING v12 functional-neighbor profiles on the measured panel,
  retaining the existing top-16 neighbor limit; **not full-network embeddings**.
- `shuffled_string_1/2/3`: three label-free gene-to-profile permutations.

Except the additive ID control, heads receive background, target features and
their outer-product interactions. Structured features use fit-only rank-32
reduction and a coverage indicator; missing genes remain in the evaluation.
Text vectors can exist for a gene with missing functional annotation; inspect
the annotation coverage report rather than assuming vector presence means
functional evidence. STRING functional associations are not signed causal edges.

Ridge chooses alpha only on calibration MSE. MLP uses the fixed endpoint, no
outer-based early stopping. Training samples balance context/target/row units.
The full-delta metric equally weights contexts, then targets, then rows. Targets
are matched before comparisons; bootstrap averages optimizer seeds first and
resamples target identities jointly across contexts. These intervals are
exploratory, uncorrected for multiple comparisons, not biological replicates.

## Diagnostics and decisions

`basis_oracle` projects outer truth into the **fit-only** response basis. It is
a descriptive representation ceiling, never a usable predictor or a training
signal. Split-half diagnostics assign raw cells by a stable cell-ID hash and
use disjoint control halves; they assess technical repeatability, not independent
biological reproducibility. H1 raw cells are unavailable, so no H1 split-half
claim is produced.

Keep a representation only after checking MSE, identity-free/random/shuffled
controls, coverage, target retrieval and paired uncertainty together. Compare
original vs expanded within each method to distinguish data and representation
effects. The pipeline selects no automatic winner.

Contrastive loss, confidence routing, ESM and Qwen fusion are deferred until this
benchmark establishes a usable signal. New-dataset metadata inspection is a
separate follow-up: predeclare the untouched context before inspecting model
performance, check source overlap and intervention units, and do not mix new
labels into this round implicitly.
