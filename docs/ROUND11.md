# Round 11: calibration transfer and a bounded H1 adaptation diagnostic

Round10 supported modest B2 gains on HepG2 and RPE1, but H1 remained worse than
zero change. This round freezes the background/response architecture and tests
where to learn calibration. It does not add teachers, atlas cells, modules,
semantic features, graphs or a larger backbone.

## Launch

The default command **includes the predeclared Jurkat release** after freezing
the rules. It reuses completed Round10 endpoints, fits only three missing
K562-out B2 students and their three background pretraining jobs, and performs
H1 adaptation without any backbone training.

```bash
cd /root/autodl-tmp/virtual_cell && \
git -c http.version=HTTP/1.1 pull --ff-only && \
screen -dmS vcell11 bash scripts/run_round11_autodl.sh --name round11_01 --resume
```

Inputs default to `/root/autodl-tmp/vcell-work/runs/round10_01`. All original
checkpoints, prediction arrays, background caches, the local Qwen snapshot,
knowledge file and prepared data must still exist at their recorded paths.
The light review archive alone is insufficient. Use `--round10-run` to select
another complete run; this does not relocate its absolute dependency paths.

Two training processes run concurrently, with two CPU math threads each. The
pipeline requires 8 GiB free reserve for a new run; it deletes nothing and makes
no network downloads. New K562 jobs inherit the audited Round10 settings
(2,000 student updates and 5,840 pretraining updates in the reviewed run).
The gene panel, rank-32 fit-only basis procedure and optimization budget stay
fixed. Each fold fits its own basis, normalization and control encoder.

```bash
/root/autodl-tmp/virtual_cell/.venv/bin/python \
  /root/autodl-tmp/virtual_cell/scripts/round11_status.py

tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round11_pipeline_*.log | head -n 1)"
```

The initial cache audit hashes saved arrays, recomputes fit-only basis and
normalization, and reads raw observation annotations. This is CPU/disk work;
GPU inactivity during this stage is expected. Stage messages distinguish this
from background pretraining, student training and final inference.

Use the same command with `--resume` after interruption. New training jobs
retain exact optimizer/update checkpoints. Historical Round10 jobs are never
sent through the trainer or overwritten. Completed artifacts are hash-checked.
Code, input or plan changes require a new name. A completed Round11 run is
verified and returned without opening the test again.

Optional modes:

- `--prepare-only`: audit caches and metadata, freeze support membership, write
  the three training jobs, and stop before training. Resume without this flag.
- `--development-only`: run training and development diagnostics, freeze the
  release manifest, then stop before scoring Jurkat. Resume without this flag
  to execute the same frozen release. This is a stage switch, not a new model.
- `--parallel-students 1`: reduce concurrent GPU jobs without changing the plan.

## Evidence and definitions

The existing target is the difference between perturbed and matched-control
means of `log1p(10000 * counts / full-source library size)`. Training scales this
delta by fit-only RMS. Inference multiplies by that scale; it does not exponentiate
the response or claim to reconstruct raw counts.

Qwen represents targets through the same gene-name text prompt for seen and
unseen targets. The MLP uses target-specific embeddings, which lack supervised
examples for targets absent from the fold. `reference_responses` uses the
equal-context target mean for seen targets and the fit-only global
equal-context/equal-target mean delta for unseen targets.

The audit verifies exact split membership, prepared-data fingerprint, gene and
target order, saved artifact hashes, historical source identity, background
partition/normalization and response-basis subspace. It recomputes reference,
old coefficients and raw/calibrated MSE from saved arrays. Reused folds are
historical evidence, not additional independent trials.

Each evidence CSV keeps row IDs, prepared row index, source split, context,
target, available batch/control counts, fit-target coverage, checkpoint hash
and metric weight. It exports `t=mean(y*y)`, `q=mean(p*p)`, `r=mean(ref*ref)`,
`c=mean(p*y)`, `h_ref=mean(ref*y)`, `g_cross=mean(p*ref)` and vector means.
Candidate order is `[raw, reference, zero]`; these values reconstruct its Gram
matrix and cross-products. Any fixed mixture has row error
`a*a*q + b*b*r + 2*a*b*g_cross - 2*a*c - 2*b*h_ref + t`.
Centered Pearson and cosine are reported; zero vectors have undefined
correlations, not artificial perfect agreement.

These statistics support quadratic/scale diagnostics; they do not recover
gene ranks, DEG overlap or distributions. Full prediction arrays stay on the
server. Original control-cell membership is not in the prepared data: shared
control pools and pseudo-batches cannot be treated as independent replicates.

## Source-only out-of-context calibration

For each optimizer seed (17, 29, 43), calibration uses three complete source
context folds:

| Fitted contexts | Calibration prediction context | Origin |
|---|---|---|
| K562 + H1 | RPE1 | Verified Round10 cache |
| K562 + RPE1 | H1 | Verified Round10 cache |
| RPE1 + H1 | K562 | Three new B2 jobs |

Background control fitting, normalization and response basis exclude each
fold's held context. Public Tabula data follow the original fixed donor split.
Public pretraining overlap remains unverified; these are strict pipeline splits,
not a claim that the frozen output-gene panel or public backbone is independent
of all historical development choices.

The same convex family as old calibration is fitted:
`a * Qwen + b * reference`, `a,b >= 0`, `a+b <= 1`, with remaining zero-change
mass and the existing `1e-6` simplex penalty. A separate alpha-only `[0,1]`
calibration tests scale correction. Coefficients are global within each paired
optimizer seed. There is no cell-line lookup, target-specific gate or selected
best seed. Coverage-specific coefficient pairs are deferred; seen/unseen scores
are always reported.

These source OOF caches are eligible for HepG2/Jurkat deployment only. They are
not reused to claim an H1-unseen or RPE1-unseen recalibration evaluation: that
would need additional nested folds entirely inside each outer source pool.
`source_oof_fit/` reports calibration fit error and is explicitly not outer
performance. HepG2 transfer is a previously inspected development result;
Jurkat is the predeclared held-out application.

## H1 few-target adaptation

The three H1-out B2 endpoints are frozen. SHA256 ordering of target identifiers
defines 100 query targets and a disjoint 40-target support pool before response
values influence any adaptation. Three fixed hash orderings (101/202/303) give
nested 0/4/8/16-target support subsets. Every row of a target stays together.
At zero support, alpha is 1 (unchanged model). At positive support sizes, only
one alpha in `[0,1]` is fitted with equal-target weights on support responses.

All methods use the same query panel. Zero, reference and unchanged predictions
are recomputed on that panel; the old 140-target full-panel MSE is not used as a
baseline. Query labels cannot determine support membership, alpha, checkpoint
or regularization. Seed 43 remains included. Draws are sensitivity repeats,
not extra biological replicates. Shared H1 controls remain allowed model inputs.

Alpha near zero may be an appropriate reliability limit, but does not establish
useful perturbation prediction. Only nonzero query improvements with useful
response direction support a later adapter experiment. No adapter is trained here.

`target_specificity.csv` additionally compares query-target mean predictions to
their true and hash-shuffled targets, and ranks the correct target among the
100 query response profiles using gene-centered Pearson correlation. Ties receive
midranks and fractional top-1 credit. This is a fixed raw-model diagnostic, not
a tuned retrieval objective: positive alpha preserves direction and alpha zero
has none. Target-mean MSE is labeled separately from row-level primary MSE.

## Perturbation coverage audit

Before training, the pipeline reads K562/RPE1 raw `obs` annotations through the
original manifest, ignoring the old target cap. It counts cells and matching
controls within available batch/study/intervention/time strata. Original
labels, unknown metadata and absent files are reported explicitly.

H1 was streamed in earlier work; when the raw file is absent, its verified
pre-filter `groups.npz` supplies target/batch counts. This is labeled an
aggregate-count audit, not a raw annotation audit. It cannot recover unavailable
intervention/time fields. Candidates with enough cells and controls are only
count-eligible; expression QC and intervention comparability are still required.
No new training rows are added and no Jurkat raw file is opened for this audit.

## Frozen Jurkat release

After source coefficients are fitted, `release_manifest.json` freezes the data
fingerprint, source provenance, checkpoints, gene order, test row IDs, inference
definition, old/new coefficients, primary metric and candidate list. It is
written before applying the new rule to development/query responses or scoring
Jurkat. There is no score-dependent release gate.

The fixed panel is zero, reference, MLP B0, Qwen B1, B2 raw, B2 old calibration,
B2 OOF mixture and B2 OOF alpha, each with all three seeds. No Jurkat labels fit
any component. `TEST_OPENED.json` marks release; if these results guide later
model changes, a new untouched final evaluation is needed.

Jurkat is held out within this pipeline, not proof of absent external pretraining
overlap or independent biological-study provenance. The full prepared file is
read and fingerprinted for integrity before release; label sealing is enforced
by computation/selection boundaries, not by separate encrypted storage.

## Results

Download after `ROUND11 COMPLETE`:

```text
/root/autodl-tmp/vcell-work/runs/round11_01/round11_review_light.tar.gz
```

The review includes `provenance/`, `evidence/`, `coverage/`,
`coefficients.json`, `h1_support_plan.json`, `release_manifest.json`,
`development/`, `h1_few_shot/`, `source_oof_fit/` and `jurkat/` plus log tails.
Weights and full prediction arrays are excluded. Primary MSE weights contexts
equally, then targets, then rows. Target-cluster paired bootstrap averages
optimizer seeds first; seed variability is also reported separately. Intervals
are exploratory, not adjusted for previous development or multiple comparisons.
Never pool absolute MSE across different evaluation panels or crown an automatic
winner. The default module branch remains closed.
