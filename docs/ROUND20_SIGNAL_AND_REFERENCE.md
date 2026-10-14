# Round20: signal repair and autonomous, bounded head search

Round19 completed 100 proposal and 100 review requests, with 61 evaluated candidates.
Its selected trial27 predicted approximately 0.35% of the observed response norm on
BXPC3 development queries. The tiny MSE gain over zero did not survive a target-level
uncertainty check. This motivates signal diagnostics and informative experimental
reference transfer before spending another large API budget.

Round20 is a two-background, one-seed development experiment. Default outer folds
are BXPC3_IFNG and HAP1_IFNG, seed17. Each has three source cell-line holdouts. HT29
and Jurkat remain unscored. Historical development exposure means these scores are
not independent external-test results.

## Start on AutoDL

The previous run must be complete, not just have a light review archive. Keep the
original full run, P4 parent, prepared matrices, source experts and pinned local
DeepSeek snapshot. No new dataset or model download is performed by this command.

```bash
cd /root/autodl-tmp/virtual_cell
git -c http.version=HTTP/1.1 pull --ff-only
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY; echo
export DEEPSEEK_API_KEY
screen -dmS vcell20 bash scripts/run_round20_autodl.sh \
  --name round20_signal_01 \
  --round19-run /root/autodl-tmp/vcell-work/runs/round19_dual_100_01 \
  --parallel-jobs 3
```

The key is needed only if a source gate permits live search. Numerical workers
receive stripped environments and fit/calibration labels, not query response arrays.
Use the actual completed Round19 path if yours has a different name.

Inspect P0 without starting new training or paying for API requests:

```bash
bash scripts/run_round20_autodl.sh --name round20_signal_01 --prepare-only
```

Continue that same preparation with `--resume` and identical options, omitting
`--prepare-only`. Resume a stopped complete workflow with its original full command
plus `--resume`. Changed folds, code, steps or options require a new run name; API
audits and checkpoints must be retained. Paid rejected/truncated responses consume
their slots and are not requested again.

If GitHub is unavailable, upload the supplied full-history bundle and run:

```bash
git fetch /path/to/virtual-cell-round20-HASH.bundle main
git merge --ff-only FETCH_HEAD
test -f scripts/run_round20.py
```

## Progress and review

```bash
.venv/bin/python scripts/round20_status.py --name round20_signal_01
tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round20_pipeline_*.log | head -n 1)"
```

`COMPLETE.json` confirms the declared workflow completed. A `NOT_RUN` search or
transformer branch is a deliberate diagnostic gate, not evidence those methods
succeeded. `INCOMPLETE.json` may retain the last failure while resuming; the stage
and active worker queue show current activity. The final small package is:

`/root/autodl-tmp/vcell-work/runs/round20_signal_01/round20_review_light.tar.gz`

It includes reports, decisions, code proposals, lineage and log tails; matrices
and model weights remain on the server. Do not delete full artifacts needed for
resume or deployment after downloading the light review.

## P0: prepared-data technical signal audit

Parent completion, dataset and metadata hashes, gene order, global unknown targets,
teacher/model revisions and whole-cell-line source folds are frozen. Existing
Round19 caches are used read-only only when data, partition, query rows and panel
match. Missing source expert folds are trained with their parent source budgets.

Rep1/Rep2 effects are paired within dataset, cell context, target, stimulus, time and
intervention. Reports include vector Pearson, cosine, sign agreement, replicate MSE
and mean-effect norm. Per-gene technical sigma is the square root of the median
`(Rep1-Rep2)^2/2` across source-fit pairs. A strong gene has mean absolute replicate
average effect at least `max(0.05, 3*sigma)`. Source-derived gene masks and target
lists remain frozen for held development scoring.

The predeclared exploratory gate requires at least eight pairs in at least two
source contexts, median Pearson at least0.1, positive cosine in at least60% of
pairs, and at least one strong gene. These are screening rules, not significance
tests. Technical replicates are not biological replicates. Noise comes from the
existing matched, log-normalized pseudobulk matrices; within-cell variance and
new raw-cell control re-estimation are **not** performed. The inherited adapter
audit and metadata retain study, stimulation, timing and batch provenance.

## P1: paired simple controls

Ridge, factor16, factor32, teacher mean and reference-only predictions are scaled
by `{0, .01, .03, .05, .1, .2, .4, .7, 1}`. Alpha is selected exclusively from
source OOF macro MSE; ties choose the smaller coefficient. It is frozen for outer
refitting and evaluation. Macro scores average technical rows within targets,
targets within contexts, then contexts.

The background-only direct MLP has no teachers or STRING input. An additional
MLP with STRING receives target information rather than silently losing all gene
identity; its paired DeepSeek+STRING control has the same numeric inputs and
also retains target text. The pretrained direct DeepSeek
control uses Round19 trial27's exact recipe, including MSE and specificity0.1.
All new fits use the same fixed50:50 historical/IFNG sampling mass when both
strata exist, balanced by context/target inside each stratum. Therefore this is
a paired Round20 comparison, not an exact re-execution of Round19's old exposure.

When source replicate evidence passes, additional paired controls replace only
the pretrained backbone with a frozen random initialization (same prediction-head
initialization), shuffle target names
while preserving cell/stimulus text, or change the correction scale from0.05 to
0.2,1.0 or a train-only sigmoid-bounded learnable coefficient. Fit and calibration
amplitudes are logged. No LoRA change is mixed into these ablations. The base
transformer's frozen weights are not copied into each trained head checkpoint.

## P2: same-target experimental reference transfer

Donors must be declared fit rows from a different whole cell line, with the same
canonical target, dataset, stimulus, time, intervention and batch. Unknown or
missing conditions cannot generate a reference. Globally held targets are absent
from every source label library. Calibration and query labels are never donors.
For Mixscale, the prepared batch label is Rep1/Rep2. It is not a reconstruction of
every original `Batch_info`/sample stratum; those strata belong to the inherited
matched-control adapter. These reports cannot establish absence of batch confounding.

Reference-only alpha is calibrated on source OOF. The correction receives target
STRING features, source-fit response-basis reference coefficients, query B2 control
features, query-minus-donor control features, and availability. A controller-applied
`tanh` bounds its L2 norm to0.2 times the source-fit typical response norm. The
response basis and normalization are learned from fit only.

Controls are matched reference plus correction, correction only, wrong-target
reference, shuffled reference and no reference. Wrong/shuffled controls also cannot
retrieve responses for globally unknown targets. Without a legal reference the
reference-based methods explicitly return zero; the separately labelled
correction-only control can predict without reference content. Coverage, known
targets, global unknown targets and actual reference availability are reported
separately; compare them rather than relying solely on a pooled score.

Reference corrections use their fixed final optimizer step. The parent calibration
set is historical and often lacks IFNG references, so it cannot select a meaningful
IFNG checkpoint. Source OOF selects designs; historical fit/calibration scores are
still logged. Each candidate starts from independently seeded fresh weights.

## P3: bounded code freedom and matched candidate slots

Live search requires both reproducible source signal and a control with at least1%
source OOF MSE improvement over the strongest simple baseline, non-degraded
strong-gene MSE and specificity, and amplitude at least5% of truth norm. A
`zero-MSE minus2e-7` outcome does not pass. Otherwise the package records a negative
diagnostic and does not buy further API calls.

Fixed, random, single-agent and dual-agent arms share source folds, tensors, steps,
batch, correction radius and the bounded `CandidateHead` code language. Agents
may design a complete small prediction head, choose rank16/32, learning rate,
weight decay and MSE/Huber. Layers/operations use the existing AST allowlist;
maximum200,000 head parameters and hidden width128. A separate CPU process checks
syntax, real input dimensions, finite outputs and gradients before training.
This is a bounded code language, not arbitrary repository editing or unrestricted
Python execution. The controller owns reference retrieval and evaluation.

The total ceiling is12 candidate slots: two per arm in the first outer fold, one
per arm in the second. This asymmetric allocation supports one real iteration
while keeping the same per-background candidate count across all four arms.
Every candidate is evaluated on three source holdouts (at most36 candidate fits).
Duplicate, invalid and failed proposals consume slots. Each arm has its own
history; outer folds do not exchange search feedback. The dual review proposes
an actionable next hypothesis, which its training agent sees on the next slot.
There are at most nine API requests: three single proposals, three dual proposals,
three independent reviews, each with4096 output-token cap and low reasoning
effort. Candidate budgets match; actual API costs do not, and dual review cost is
explicitly reported. Fixed/random controls are samplers within the same permitted
head language, not exhaustive searches of every possible code program.

Only source-promoted candidates are refitted for development evaluation. All
selections across both backgrounds are sealed before any held development scores
are read. No expanded four-background/three-seed experiment starts automatically.

## Files to inspect first

- `summary.json`, `comparison.csv`, `source_scores.csv`, `outer_scores.csv`.
- `qc/*/source_signal.json`, `genes.csv`, `replicates.csv` and held development
  replicate reports written after selection is sealed.
- `selections/*.json`, `search_frozen.json`, `candidate_scores.csv`.
- `run_conditions.json`, `inputs.json`, `cache_lineage.json`, worker lineage and
  `api_ledger.json`; individual API audits retain all truncated/invalid usage.

OOF means predictions for rows excluded from the response model's fitting line.
Alpha is a multiplicative amplitude calibration. A response basis compresses the
gene vector using source-fit variation. B2 is the project's source-trained numeric
background encoder; it is not language-model reasoning. Specificity compares the
correct target with other targets measured under the same conditions. None of
these development summaries alone establishes statistical or biological benefit.
