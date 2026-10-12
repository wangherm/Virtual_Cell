# Round17: source-OOF routing and bounded agent head experiments

The real P4 pilot established that DeepSeek can execute the deployed experts.
It did not establish improved prediction: the four ATF3 queries were expanded-fit
rows and there were no source-OOF profiles. This runner replaces that deployment
smoke with newly trained, cell-line-held development folds. It also implements the
user-authorized extension: DeepSeek may propose and test isolated prediction-head
code, using inner OOF evidence only. No improvement is assumed before execution.

## Run the first fold on AutoDL

Update the repository first. If GitHub is unavailable, transfer the supplied Git
bundle to the server, then `git fetch /path/to/the.bundle main` followed by
`git merge --ff-only FETCH_HEAD`. Existing data and model downloads are reused.

```bash
cd /root/autodl-tmp/virtual_cell
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY; echo
export DEEPSEEK_API_KEY
screen -dmS vcell17 bash scripts/run_round17_autodl.sh \
  --name round17_oof_01 --parallel-jobs 3 \
  --outer-contexts A549_IFNG --seeds 17 --inner-folds 2 \
  --targets-per-context 10 --auto-improve --trials 2 --live-routing
```

This runs training, source-OOF evidence, at most two code proposals, deterministic
routing and optionally live routing, then private outer evaluation. Research
effort defaults to `high`. Routing compares `low` and `max` with otherwise matched
inputs and limits. The existing explicit model ID is `deepseek-v4-pro`; `--model`
can override it with an ID returned by the authenticated provider model listing.
No model availability or currency pricing is inferred from its name.

Live routing is gated by source expert complementarity: relative oracle gap must
reach 1% and remain positive when any single target is removed. The gate is fixed
before outer scoring. A skipped live comparison is reported explicitly; it is
not an API failure. Use `--oracle-gap` to predeclare a different threshold in a
**new** run. The oracle values are never included in a planner observation.

For training and deterministic evaluation without any API requests, omit both
`--auto-improve` and `--live-routing`. To inspect frozen inputs before training,
add `--prepare-only`; continue the identical command with `--resume`.

```bash
.venv/bin/python scripts/round17_status.py --name round17_oof_01
tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round17_pipeline_*.log | head -n 1)"
```

Status prints the current stage, job counts, running/failed worker log paths and
any `INCOMPLETE.json`. Do not restart training just because the parent log pauses:
independent worker progress is in `runs/round17_oof_01/logs/`. Three processes share
the single GPU; each uses two CPU threads. This is parallel independent fitting,
not distributed training of one model. Reduce `--parallel-jobs` in a new run if
concurrent GPU memory or I/O becomes a bottleneck.

Resume with the same options, adding `--resume`. Saved checkpoints preserve
optimizer state, deterministic per-step sampling and best source-calibration
weights. Completed artifacts are checksum-verified. Interrupted HTTP attempts
remain counted; valid accepted decisions are cached. Failed trials are reported
and retained as failures; they do not become fictitious scores or trigger
unbounded retries. Source/code/configuration changes require a new run name.

## Fixed experimental boundaries

- The default parent is `runs/round15_p4_02`, `context` protocol, seed17.
  It supplies the frozen harmonized expanded dataset, full STRING annotation
  vocabulary, existing public background cache and local projector dimensions.
  Its trained prediction weights and historical performance rankings are not
  reused as OOF evidence.
- The output gene order and response units are frozen from the parent: normally
  1,834 genes and mean log1p(10,000 × counts/full library) differences. Metadata,
  prepared data, annotations, parent and public-cache hashes are recorded.
- Outer backgrounds are A549_IFNG, BXPC3_IFNG, HAP1_IFNG and MCF7_IFNG. Exclusion
  uses the cell line, across studies/stimuli, rather than only a context string.
  Thus holding out K562 would also exclude its old unstimulated supervised rows.
  The first run uses A549; two other complete lines supply inner OOF queries.
- Reference and expanded conditions use the same expanded inference dataset.
  Reference fitting is restricted to its historical prefix. Expanded fitting
  can use eligible appended IFNG rows. Both use the same source-only historical
  calibration IDs, frozen by context-target identifier hash. Every batch of a
  context-target unit receives the same role. Outer lines never enter source
  fitting, calibration, PCA, response basis, reference banks or matched-control
  pretraining. Final workers score a separate source diagnostic subset; actual
  outer responses are scored only after all search and routing decisions seal.
- An identifier-hash fifth of the IFNG target vocabulary is excluded from every
  fit/calibration, yielding truly untrained target queries. Cases alternate this
  group and the remaining targets, up to ten distinct targets per background.
  All technical rows of selected targets are retained; no response magnitude or
  model error determines query inclusion. Target annotations are public priors,
  not perturbation-response labels.
- Public pretraining uses the existing primary Tabula training pool and rejects
  explicit held-line metadata. Public validation/external pools do not fit
  weights. This is a **supervised cell-line holdout with shared public priors**,
  not proof that all prior biological knowledge excluded those cell types.
- HT29 and Jurkat stay closed for fitting, query construction and evaluation.
  These are development holdouts, not the final independent external test.

## What trains, and what the agent can change

Each fold trains six numerical experts: reference/expanded × STRING Ridge,
factor rank16 and factor rank32. Each condition gets a newly source-trained B2
background encoder. The default one outer plus two inner folds requires six
background fits and 18 numerical heads, before optional candidate heads. The
B2 encoder is a **Qwen-sized expression projector**, not a pretrained Qwen LLM
thinking about genes. These are the currently deployed P4 numerical experts.

The two fit-only reference tools are also exported. Donors require a matching
known study, stimulus, intervention, time and batch, and a distinct cell line.
Unknown or incompatible conditions mean unavailable. Different batches are
never pooled and represented as a condition-matched donor. References without
sufficient matched OOF evidence do not receive fabricated risk profiles.

DeepSeek's research observation contains aggregate **inner** expert MSE, source
contexts, target counts and previous source-trial results. It sees no outer
truth, outer predictions, rankings, or oracle values. It returns a typed JSON
proposal with a complete `CandidateHead`, rank16/32, learning rate, weight decay
and MSE/Huber loss, or stops.

Allowed code is a bounded Python neural-head language: a single `nn.Module`
class, small approved layers, tensor arithmetic and selected activations. It
cannot import modules, read files, call shells/networks, recurse, use loops,
inspect private attributes or modify the repository. Hidden widths are at most
128 and head parameters at most 200,000. CPU output/finite-gradient admission
runs in a separate process with a 60-second timeout and credentials removed.
This is **not** a general-purpose OS sandbox for arbitrary Python; arbitrary
Python is never admitted. GPU training workers also remove credential variables.

Patches live under `research/<fold>/trial_N/head.py`. They reuse each expanded
fold's source-trained encoder but refit their target/background transforms,
response basis and head from the same allowed source split. Source calibration
chooses the head checkpoint. At most three proposals can be configured; the
default is two, using the same training steps and batch budget as baseline heads.

The lowest macro-inner-OOF candidate is admitted only if it strictly beats the
strongest fixed baseline expert. Ties and worse scores retain the baseline.
Adaptive reuse of inner evidence can overfit it; the untouched outer fold
evaluates the complete proposal/search process. If selected, the candidate is
refit on outer-allowed sources and added as a seventh numerical expert to an
**additional** improved pool. The original six-expert pool is always retained.
There are no automatic merges or changes to preprocessing/evaluation.

## Routing comparison and interpretation

All policies receive the same query and complete supported-model pool. Formal
observations expose the same five source features and predicted risks; extra
inspection is marked as providing no new numerical feature information.

| Policy | Selection evidence / behavior |
|---|---|
| fixed | Best global expert on macro-inner-OOF MSE |
| rule | Predeclared cost/capability rule: Ridge, then rank16, then other supported tools; no accuracy claim from names |
| risk | Ridge prediction of log MSE fitted on actual inner OOF |
| softmax_gate | Regularized soft-label classifier on the same OOF features/errors; fixed optimizer settings |
| workflow | Execute the top two source-risk experts, finish using frozen source risk |
| one_shot_low/max | One actual API request, one expert execution and deterministic finish |
| adaptive_low/max | Bounded inspect/predict/finish decisions, at most five requests |

Reference availability can vary; it is recorded, never filled with a zero response.
Rule/LLM strategies can use supported references, while empirical risk strategies
require a fitted profile. This difference is explicit in tool capabilities and
must be considered when interpreting selected-model counts. The six numerical
expert pool is matched for every OOF and outer query.

Reports contain per-query selections, availability/abstention, macro-context and
target MSE, zero comparison, reference availability, response norms and retrieval
specificity. First-observed/repeat CPU inference timings reload tensors each
time; OS cache state is uncontrolled, so these are not strict cold-cache timings.
API latency/tokens come from the durable provider ledger; tool units are declared
charges, not actual dollars. Output tokens already include reasoning tokens.
Session wall time after resume is not a reconstructed uninterrupted runtime.

Compare **policies within an identical pool** for routing effects. Compare the
baseline and improved pools for predictor/search effects. Optimizer seeds are
not biological replicates. A single successful fold is an engineering and
development result, not statistical significance or external generalization.

## Review files and expansion

Under `runs/round17_oof_01`:

- `inputs.json`, `budget.json`: frozen folds, labels allowed for each role and workload.
- `evidence/<fold>/base/profiles`: real source OOF scores, paired numerical
  predictions and lineage; `source_summary.json` is a private complementarity
  diagnostic, not an agent input.
- `research/<fold>/selection.json`: every proposal, admission/training failure,
  inner score and selected or rejected patch.
- `episodes/...`: sealed decision traces and routed predictions, without truth.
- `evaluation/.../per_query.csv`, `comparison.csv`, `specificity.csv`,
  `private_expert_scores.csv`: private evaluator outputs after decisions seal.
- Root `comparison.csv`, `summary.json`, `COMPLETE.json`, and
  `round17_review_light.tar.gz`: final small review package. Weights, expression
  matrices and raw query controls stay on the server; packaged bytes have hashes.

After validating the first fold, use a new name and expand the unchanged protocol:

```bash
screen -dmS vcell17full bash scripts/run_round17_autodl.sh \
  --name round17_oof_full_01 --parallel-jobs 3 \
  --outer-contexts A549_IFNG BXPC3_IFNG HAP1_IFNG MCF7_IFNG \
  --seeds 17 29 43 --inner-folds 3 --targets-per-context 10 \
  --auto-improve --trials 2 --live-routing
```

This expanded command needs 96 background fits and 288 baseline numerical heads
(four outer folds × three seeds × four folds × two data conditions). Review its
prepared budget before allocating the larger training/API workload. Do not use
outer scores to edit a fold's recipe and then label the same fold independent.
