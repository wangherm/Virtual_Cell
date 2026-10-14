# Round21: reproducible signatures, supported repeats and cell-line transfer

This round diagnoses measurable perturbation information before another LLM search.
It reads a complete Round20 run and preserves its data, gene order, global unknown
targets, whole-cell-line source folds and held development queries. Round20 files
are not overwritten. HT29 and Jurkat remain unscored. These are previously observed
development backgrounds, not a new independent external test.

## AutoDL start

Keep the full Round20 run and its inherited cache paths, the completed Mixscale
adapter and original RDS. The light review archive alone cannot run this workflow.
No new data, teacher or large language-model weights are downloaded or trained.

```bash
cd /root/autodl-tmp/virtual_cell && \
git -c http.version=HTTP/1.1 pull --ff-only && \
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY && \
export DEEPSEEK_API_KEY && \
screen -dmS vcell21 bash scripts/run_round21_autodl.sh \
  --name round21_signal_01 \
  --round20-run /root/autodl-tmp/vcell-work/runs/round20_signal_01 \
  --parallel-jobs 3
```

Stop if the pull fails. Upload the supplied full-history bundle instead:

```bash
cd /root/autodl-tmp/virtual_cell && \
git fetch /root/autodl-tmp/vcell-work/virtual-cell-round21-HASH.bundle main && \
git merge --ff-only FETCH_HEAD
```

The default raw inputs are
`/root/autodl-tmp/vcell-work/raw/round15_D1/IFNG` and adapter
`/root/autodl-tmp/vcell-work/prepared/round15_mixscale_ifng_04`.
Use `--raw-dir`, `--adapter-dir` or `--rscript` only if your actual paths differ.
Rscript is discovered from the existing `R_SETUP.json`; it needs SeuratObject,
Matrix and jsonlite, already used by the Mixscale adapter. Only one RDS audit
process runs before the parallel numerical workers. The sparse RDS is deserialized
in memory, with a memory-reserve check; reserved expression columns are never selected.

`--prepare-only` freezes the plan and lineage without raw inspection, training or
API calls. To continue, keep the original options, omit `--prepare-only` and add
`--resume`. Resume a failed run with the full original command plus `--resume`.
Changing budgets, code or data requires a new name. Existing paid rejected or
truncated responses consume their slots and are not retried automatically.

`--skip-raw-audit` is an explicit partial diagnostic mode. It does not verify
independent controls and cannot enable Agent search. `--no-agent` disables paid
search while keeping all numerical diagnostics. Neither flag can be changed on
the same frozen run. An API key is needed only if source evidence passes the gate.

## Progress and packaging

```bash
.venv/bin/python scripts/round21_status.py --name round21_signal_01
tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round21_pipeline_*.log | head -n 1)"
```

The status command shows raw-cell progress, active numerical steps, source gates,
paid attempts and the last failure. `INCOMPLETE.json` can remain during a resumed
run until successful completion. A complete queue is not completion of the whole
experiment. `BLOCKED_NO_RELIABLE_TRAIN_REFERENCE` means a correction could not be
trained; its explicitly marked fallback is calibrated reference only.

The final artifact is
`/root/autodl-tmp/vcell-work/runs/round21_signal_01/round21_review_light.tar.gz`.
It contains reports, lineage, candidate code and log tails. Per-cell membership,
expression arrays and checkpoints stay on the server. Retain full artifacts for
reproduction and resume.

## P0: observed raw and prepared-data facts

The raw audit measures full-RNA library size and detected genes for all development
NT controls and the four HAP1 diagnostic perturbations BATF2, IFNGR2, MAFF and PRDM1.
It checks exact inherited matched-control strata, positive-library NT membership,
repeat labels and actual control-cell reuse across repetitions. HAP1 guide/cell
retention and control-pool counts are joined to library summaries.

The audit separates observed measurements, possible depth/control/guide/cell-state
mechanisms, and unresolved explanations. It does not change normalization or delete
HAP1 outliers. Rep1/Rep2 prepared response norms and mean squared effects for those
four known development perturbations are computed only after all choices freeze.
These response diagnostics are never supplied to search or used to select queries.
Control memberships can establish cell-pool disjointness, not independent biological
replication or absence of batch confounding.

## P1: source-only target-specific reliability

Each cell context, target, study, stimulus, time and intervention has a separate
technical signature. Reports include own-target Rep1/Rep2 Pearson/cosine, other-target
correlations under the same conditions, mismatch percentile, own-minus-mismatch
gap, top50/100 response-gene intersection, sign agreement, response norms and
technical discrepancy. A leave-own-target-out common response is subtracted for an
additional descriptive correlation diagnostic.
Targets missing either repeat remain in the CSV with an explicit MISSING status;
they are not silently removed from coverage accounting or labelled reliable.

The predeclared reliable-library policy requires at least three other target pairs,
own Pearson at least0.1, own-minus-mismatch median at least0.05, mismatch percentile
at least0.8 and a supported strong gene. Per-target genes have consistent signs and
replicate mean absolute effect at least `max(0.05, 2*abs(Rep1-Rep2)/sqrt(2))`, capped
at the top100 response genes. The absolute floor is in original log-normalized
response units, not student-normalized units. These rules are exploratory reliability
screens, not p-values or formal differential-expression testing.

Only each source fold's declared fit labels define its library and masks. No held
cell-line response, calibration response or globally unknown target is a donor.
Strong masks for cross-cell queries come from their legal same-target source
signatures; unavailable targets have an undefined strong-gene score. The original
pooled MSE remains and all identifier-frozen queries are retained.

## P2: two separate prediction tasks

**Supported technical-repeat prediction.** Within each allowed source cell context,
identifier-hashed target folds train on other targets. Query Rep1 is an explicit
experimental input; query Rep2 is sealed from the worker. The worker derives response
basis, control-difference transforms, amplitude calibration, noise thresholds and
model weights solely from other training-target pairs. Query strong-gene masks may
use its permitted Rep1 input, never its Rep2 outcome. Source reliability reports do
not filter evaluation targets.

Controls are Zero, direct Rep1 copy, calibrated Rep1, target-agnostic Rep2 training
mean, de-meaned/calibrated Rep1, wrong-target Rep1, Ridge residual correction and a
bounded small MLP correction. Calibration on other training targets is deliberately
simple; target-fold outputs provide the evaluation. Rep2 matched NT controls are
required for an independent-repeat interpretation. Unverified-control mode is
explicitly flagged and cannot enable search. This supported task is not unknown-target
prediction or unseen-cell-line generalization.

**Whole-cell-line transfer.** Source-only reliability selects allowed donors from
different entire cell lines. Study, stimulus, time, intervention and prepared batch
must match. Globally held targets have no same-target responses anywhere in the
library. Methods are Zero, target-agnostic source response mean, calibrated same-target
source mean, nearest-background same-target response, inherited source-OOF STRING
Ridge and bounded MLP correction. The inherited Ridge and background features retain
their exact Round20 provenance; they are not newly refitted large backbones.

Wrong/shuffled references choose a different target in that query's same permitted,
reliable donor pool. They share the correct-reference amplitude coefficient; they
are not individually tuned toward Zero. Comparison reports include the identical
common support set, reference availability, global unknown targets and all queries.
Insufficient alternate donors are reported as lack of common support. Rep1/Rep2 is
the prepared Mixscale batch label, not a reconstruction of every original sample
stratum; raw-control audit preserves the original strata separately.

A correction receives target STRING features, source reference-basis coefficients,
query background, query-minus-donor background and availability. Its external tanh
bound is0.2 times the source-fit typical response norm. Correction training requires
at least four source-fit rows with reliable own signatures and independent-line
same-target support. Otherwise it is BLOCKED and returns labelled reference-only
fallback. Checkpoints use a fixed final step, avoiding selection on historical-only
calibration. Every trial starts with fresh seeded weights.

## P3: evidence-gated autonomous code search

The gate uses only current inner source-held cell lines, not the two outer results.
It requires supported source-repeat prediction to beat wrong-target input, and at
least two source-held contexts to show a1% gain over Zero/background-mean/Ridge,
nonzero amplitude (at least5% truth norm), defined strong-gene error and positive
target specificity. Each qualifying context additionally needs at least four common
supported targets on which correct reference beats shuffled reference in MSE and
specificity. Raw NT pools must have independently verified repeat memberships.
Every source correction fold must also have enough legal training references; a
known untrainable module does not trigger paid code search.
This is a screening gate, not a statistical-significance claim.

When enabled, fixed/random/single/dual arms share source tensors, folds, optimizer
steps and candidate counts. Agents may redesign a CandidateHead and choose rank,
learning rate, weight decay and MSE/Huber within the existing bounded AST language.
No imports, I/O, loops, repository edits, arbitrary operations, changed reference
libraries or evaluation changes are allowed. Isolated CPU shape/gradient admission
precedes training; head capacity is at most200,000 parameters, width128.

At most12 attempted candidate slots are allocated: two per arm in the first outer
experiment and one per arm in the second. Duplicates, rejections and training failures
consume slots. There are at most nine low-effort4096-output-token API requests:
three single proposals, three dual proposals and three independent reviews. Dual
reviews inform the next proposal. Actual API usage differs between arms and is
reported, not claimed equal in currency cost. Inner MSE must improve at least1%
over the strongest proper fixed method without degrading strong-gene error or
specificity; amplitude cannot collapse. Only promoted candidates are outer-refitted.

All selections across both outer experiments are sealed before private development
scoring. If only BXPC3 has reliable signatures, leaving it out can remove the strongest
source evidence; a negative transfer result is informative and does not trigger
unbounded extra API search or reopening HT29.

## Read these reports

- `summary.json`, `comparison.csv`, `run_plan.json`, `lineage.json`, `README.md`.
- `replicate_qc_by_target.csv`, `signal_vs_mismatch.csv`, `batch_audit.csv`.
- `within_line_repeat_scores.csv`, `source_cross_cell_transfer_scores.csv`,
  `cross_cell_transfer_scores.csv`; keep task and subset columns separate.
- `raw_audit/`, `selections/`, `search_frozen.json`, `candidate_scores.csv`,
  `api_ledger.json` and individual paid-request audits.

MSE averages technical rows within targets, then targets within cell contexts. Seeds
are optimizer repetitions, not biological samples. Specificity compares the correct
target with other targets under identical measured conditions. Top response genes
are a descriptive ranking; calling them statistically significant DEGs would require
a separate inferential analysis. Original matrices and predictions are excluded
from the light package, so gene-level reproduction needs the full server run.

Software verification uses tiny offline numeric data and mocked provider responses.
The raw sparse RDS pass requires the server's existing R environment; it is not
replaced by a simulated success or an automatic package download.
