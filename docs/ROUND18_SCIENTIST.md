# Round18: scoped Virtual Cell Scientist

Round17's uploaded archive verified 1,175 payload hashes. On its single A549
development holdout, the original fixed expert scored MSE 0.00796103, the selected
agent head 0.00799667, and zero 0.00791567. Raw risk routing picked reference rank16
on all 19 queries and scored 0.00839799. Both DeepSeek head proposals executed,
but the 0.60% inner gain did not transfer. `live_routing=false`: these were
deterministic routing results, not LLM routing. No gain is claimed from that run.

Round18 keeps the existing controller, numerical experts and bounded head language.
It adds source-scoped scientific memory, traceable public literature, conservative
relative-to-zero routing, and equal-maximum-budget search controls. These are
research hypotheses and runnable tests; they are not demonstrated improvements.

## Start a small complete run

Update Git first, or fetch/merge the supplied offline Git bundle if GitHub is
unreachable. Set the DeepSeek key in the server environment; it is never required
for public literature retrieval. All data/model paths remain on the data disk.

```bash
cd /root/autodl-tmp/virtual_cell
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY; echo
export DEEPSEEK_API_KEY
screen -dmS vcell18 bash scripts/run_round18_autodl.sh \
  --name round18_pilot_01 --outer-contexts BXPC3_IFNG --seeds 17 \
  --parallel-jobs 3 --reuse-round17 /root/autodl-tmp/vcell-work/runs/round17_oof_01 \
  --cycles 2 --trials 2 --live-routing --live-targets 3
```

This is a complete small pilot, with three inner background folds, **all available
source IFNG targets** for capability evidence, four search arms, two candidates per
arm per cycle, two self-iteration cycles, and a matched three-target outer routing comparison. Every technical row of
those three identifier-chosen targets is retained. All policies see the same query
subset in this live comparison. It is not a complete multi-background confirmation.

The default larger experiment uses BXPC3, HAP1 and MCF7, seeds17/29/43. A549 remains
an observed development source, not a fresh independent test. Without reuse, that
larger experiment requires 72 background fits and 216 numerical heads, plus at
most 432 candidate inner-head fits and 36 accepted final-head refits. Each candidate
uses the same steps and batch size as baseline heads. Actual counts and bounded
API requests are written to `budget.json`. Optimizer seeds are not biological
replicates. No final external HT29/Jurkat scoring is opened here.

Run with `--prepare-only` first to read the frozen budget without training, network
access or paid API calls. Continue with identical options plus `--resume`. The
default larger non-live run keeps ten distinct outer targets per background; when
`--live-routing` is enabled, every comparator uses the predeclared live subset.
Use a new run name and `--live-targets 10` to compare live policies on all ten.

```bash
.venv/bin/python scripts/round18_status.py --name round18_pilot_01
tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round18_pipeline_*.log | head -n 1)"
```

`Current queue` describes the last process queue, not the whole experiment. The
final signals are stage `complete`, root `COMPLETE.json`, and `ROUND18 COMPLETE`.
Background/head checkpoints and accepted provider decisions resume; changed code,
data, configuration or knowledge snapshots require a new run name.

## Literature and scientific memory

The bounded literature tool uses the official [Europe PMC REST service](https://europepmc.org/RestfulWebService)
with `resultType=core`. It performs two fixed public method queries, at most six
records each, with a fixed publication cutoff. Bio-domain filtering removes
unrelated records. There is no per-query prediction-time browsing, arbitrary URL
fetching, paper-code execution or automatic model download.

Records retain publication IDs, DOI where supplied, publication date, reported
version, source URL, retrieval timestamp and checksummed abstract snapshots.
The first implementation reads **abstracts**, not complete papers. A design must
include a literal quote from a retrieved abstract and a testable hypothesis.
Local validation rejects invented IDs and quotes. A real quote makes the source
traceable; it does not establish that the paper supports every architectural
choice. Candidates are explicitly *literature-inspired*, not claimed reproductions
of full-text methods. Source texts are untrusted data, never tool instructions.

```bash
.venv/bin/python scripts/prepare_round18_literature.py \
  --output /root/autodl-tmp/vcell-work/knowledge/round18/literature_2026-10-12
```

If AutoDL cannot reach Europe PMC, run the same preparer on a network-enabled
workstation and transfer its completed directory into
`/root/autodl-tmp/vcell-work/knowledge/round18`. Then use `--literature-cache`
pointing to that completed directory. Its
plan/checksums are verified; no silent fallback to invented papers occurs.

SQLite stores immutable paper and source-experiment records by content identity.
Experiments require an explicit source-inner-OOF role, `outer_used=false`, data
fingerprint and all cell lines used for fitting/calibration/scoring. Retrieval
rejects any record involving the current outer line. Unknown/outer score fields
are rejected. Outer reports are never imported into the memory database.

For fair search controls, allowed memory snapshots are frozen for **all** folds,
seeds and arms before this run writes new trial results. Both LLM arms receive
the same initial memory and aggregate source scores. The literature arm alone
also receives a bounded lexical retrieval of public abstracts. Each arm can use
its own preceding trials. Memory can inform later *new* experiments without
silently changing a resumed run. It does not fine-tune the DeepSeek weights.

## Search comparison

| Arm | Proposed designs | Paid research requests |
|---|---|---|
| fixed | Predeclared width64/32, tanh/GELU, LayerNorm and dropout0/0.1 factorial sequence | 0 |
| random | Deterministic sampled widths, activation, normalization, dropout, rank, learning rate and weight decay | 0 |
| agent_plain | DeepSeek source scores + identical permitted experiment memory | At most 4 per outer fold/seed by default |
| agent_literature | Same inputs plus traceable abstracts, hypothesis and checked citations | At most 4 per outer fold/seed by default |

The fixed control deliberately covers the Round17 designs. Random sampling uses
a frozen seed and legal head-language space. All arms have the same maximum
candidate count, training steps, batches and source folds. Early stops and failed
admissions are reported with actual used budget; they are not repaired by extra
attempts or scored as successful experiments. Only source-fit/calibration labels
train parameters. The language has no imports, shell, network, file access,
arbitrary repository editing, loops or unbounded dimensions.

Promotion is stricter than Round17: beat the strongest fixed expert **including
zero** by at least 1% macro-source MSE, improve at least two source backgrounds,
allow no source background to worsen by more than 1%, and retain positive gain
after removing any single target. These are predeclared screening rules, not a
confidence interval or significance test. Adaptive inner reuse still risks
overfitting; only the untouched outer evaluates the complete search procedure.

An accepted candidate is refit on outer-allowed sources and added to a separate
arm-specific pool. The baseline pool is retained. If no candidate is accepted,
the baseline remains deployed, while raw rejected candidate scores still appear
in `search_comparison.json`. No automatic code merge changes the repository.

## Autonomous self-iteration

The loop borrows feedback/refinement and episodic-memory ideas from
[Self-Refine](https://arxiv.org/abs/2303.17651) and
[Reflexion](https://arxiv.org/abs/2303.11366). It is an implementation for this
experiment, not a reproduction of their benchmarks. The feedback here is measured
source-OOF performance, admission failures and background-specific regressions.

Each cycle exposes the current best *observed source candidate's actual code*,
its hypothesis and acceptance status, plus recent measured feedback. The agent
can revise the head/hyperparameters and hypothesis; literature retrieval can
choose a new subset of the same frozen abstract cache. Improvements are not
inferred from the wording of its explanation. Explicit short hypotheses and
source feedback are stored; private provider reasoning is not logged.

Default limits are two cycles × two candidates per arm. `--cycles 3` permits a
third cycle in a new run, still with the same source folds and fixed maximum
budget. A cycle with less than 0.1% improvement in best source MSE counts as
stagnant. `--patience-cycles 2` stops after two stagnant cycles. This continuation
rule is weaker than the 1% promotion rule: a promising small source gain can
justify one more experiment without being promoted to the outer pool. The agent
can also explicitly stop. All stop reasons and used/max budgets are recorded.

Fixed and random controls receive the same cycle/candidate ceilings and stopping
rule. Both LLM arms receive the same measured feedback and memory; only literature
access differs. This is iteration of experiment hypotheses and prediction-head
design, with source training for each candidate. DeepSeek's base weights and the
controller's execution permissions are not automatically altered. Only after
every arm/fold finishes or stops are routing decisions sealed and outer responses
scored. Completed cycles replay offline from immutable observations/proposals.

## Reliability and routing

Zero is an executable analytical expert with genuine OOF error evidence and
source-only control statistics. It returns a zero vector in the unchanged output
gene order and response units. It is distinct from `abstain`, which returns no
prediction. Its declared tool cost is small and unmeasured; it is not a dollar
estimate. The cheapest-tool rule can therefore choose zero.

Raw Round17 log-MSE risk is retained as a comparator. New profiles fit paired
log-error ratios against zero, shrink slopes toward global source behavior,
clip feature extrapolation and report covariates outside their source support.
The conservative router selects a nonzero expert only when its penalized expected
gain clears the margin and inputs remain in support; otherwise it chooses zero.
Residual penalties and support flags are heuristics, **not calibrated predictive
confidence intervals** or guarantees of improvement.

Source-context meta cross-validation reports selection MSE/regret and capability
cards split by source background and target seen/unseen. It diagnoses transfer of
the risk estimator. The underlying OOF expert fits can share source labels across
those meta folds, so this diagnostic is not a fully independent third nested
evaluation. The outer complete-line holdout remains the transfer assessment.

Policies include zero, best fixed including zero, fixed excluding zero, cheapest
rule, original risk, original workflow, softmax gate, shrunk risk and conservative
routing. Optional one-shot/adaptive DeepSeek policies see the same shrunk source
capability evidence and query subset. Model pools and source data remain matched;
different estimators are explicit baseline alternatives. One-shot uses one API
call. Adaptive calls are capped at four by default; low effort is the initial
default. Add `--routing-efforts low max` in a new run for a matched effort study.

Round18 allows this small live benchmark even when source oracle gap is small,
so a no-gain result is still an informative cost/accuracy comparison. Oracle
diagnostics remain private and are never exposed to research/routing planners.
Request caps, input/output/reasoning tokens, API latency and tool execution time
are reported. Output tokens already include reasoning tokens. No current currency
price is assumed. Full live routing across every pool can be costly; inspect the
printed maximum request budget before starting a large run.

## Artifacts, reuse and what constitutes a result

`--reuse-round17` verifies a completed experiment and reuses only exact source
fit/calibration, seed, feature cache, encoder and training-budget matches. Expanding
source query count is permitted as an inference-only operation; old diagnostic
partitions remain bound to old encoders. Old weights/results are never modified.
Other seeds and changed source scopes refit normally. This option does not import
Round17's A549 outer scores into memory or capability estimates.

The small review archive contains:

- `plan.json`, `inputs.json`, `budget.json`: scopes, reused artifact hashes and budgets.
- `literature_snapshot.json`, `memory_snapshots/`: exactly what evidence was available.
- `search/<fold>/<arm>/`: proposals, citations, code, admission and source trial results.
- `evidence/`: paired real source-OOF scores and lineage, including zero.
- `reliability/`: private meta diagnostics, public capability cards and shrunk profiles.
- `episodes/`: sealed decisions and provider accounting, without response truth.
- `evaluation/`, `comparison.csv`, `comparison_macro.csv`, `search_comparison.json`,
  `report.html`: post-decision private evaluation, including paired fixed contrasts.

`round18_review_light.tar.gz` excludes weights, raw expressions and query controls.
Numerical predictions remain checksummed on the server. Summaries alone cannot
independently reproduce gene-level scores. Compare search arms to study model
discovery, and policies **within identical pools** to study selection. Different
pools must not be used to attribute a prediction-head gain to routing alone.

The next milestone is reproducible held-background benefit over zero, the strongest
fixed expert and ordinary learned routing. A functioning research loop or plausible
paper-grounded explanation alone does not meet that milestone.

## Rejected proposals and legacy recovery

A successful HTTP response can still contain an invalid research proposal. JSON,
recipe, bounded Python syntax and citation checks remain mandatory. Rejections are
now failed trial slots: no head is executed, no model is fitted, and the next
proposal receives source-only failure feedback. The paid attempt remains in the
original ledger. API authentication, transport failures and unknown outcomes still
stop the run; inference-time routing validation is unchanged.

New research audits include `api/validation_NNNN.json` with an allowlisted local
validation rule. Raw responses, hidden reasoning and secrets are not saved. Old
audits cannot reconstruct a discarded response; their rejected trial explicitly
records that the specific rule was not retained.

For an incomplete **80692cd** experiment stopped on an invalid proposal, run the
original command with `--resume --repair-invalid-proposals`. The migration accepts
only that known launcher, unchanged model-agent/core code, identical data, seeds,
prompts and budgets, and no existing outer routing/evaluation. The original plan
and old/new plan digests are retained under `recovery/`. Existing completed source
fits, search controls and accepted trials are reused; failed paid responses count
toward the original budget. Subsequent resumes only need `--resume`.

Do not increase an existing experiment's budget during recovery. A changed budget
requires a new name, so the original comparison remains reproducible.

## Larger evidence-guided self-iteration

New runs support 1–8 cycles, 1–4 trials per cycle, and patience of 1–8 cycles.
The defaults remain 2 × 2. The larger preset is:

```bash
screen -dmS vcell18iterate bash scripts/run_round18_iterate_autodl.sh \
  --reuse-round17 /root/autodl-tmp/vcell-work/runs/round18_pilot_01
```

Use that reuse path only after the recovered pilot completes; otherwise reuse a
completed Round17 run. The preset uses one BXPC3 development fold and seed 17,
6 cycles × 3 trials per arm, patience 3, and the original 3-target live benchmark.
It compares the same four arms. Upper bounds are **36 research API requests**,
**216 candidate inner-fold fits**, and **4 final candidate fits** for this one
fold/seed. Live routing has its own unchanged maximum, printed in `budget.json`.
Accepted and rejected proposals both use the trial/request budget; there are no
unbounded retries or budget resets.

Every cycle carries the best measured source candidate, per-background gains and
recent failure feedback. Larger runs ask for one testable change at a time and
skip identical code/recipe training. Fixed controls expand to distinct predeclared
recipes; random controls use the same maximum budget. Early stopping still applies:
three consecutive cycles without at least 0.1% source improvement stop the preset.
Promotion still requires >1% gain over the strongest fixed expert including zero,
gains in at least two source backgrounds and the original robustness checks.
Outer scores cannot steer subsequent trials. More trials increase search capacity;
they do not establish transfer or improve the DeepSeek model weights.
