# Round25: self-correcting source research

Round25 compares ordinary Shinka mutation, mutation with bounded runtime repair,
and mutation with repair plus a source-evidence scientific critic. These are
actual native `shinka-evolve==0.0.7` campaigns, not a simulated conversation or a
replacement population database. Numerical models remain unrestricted full
`fit_predict(data, budget)` modules within the existing worker contract.

## Tasks and evidence boundary

The full completed `round24_mechanism_01` server directory is required. A light
review archive cannot supply the matrices or original worker predictions.
Existing raw data and models are reused; this launcher downloads no new datasets
or foundation models and does not install another agent framework.

- **Supported repeat:** the query perturbation's measured Rep1 response is an
  allowed input, and Rep2 is predicted. This is not unknown-perturbation inference.
- **Whole-cell-line transfer:** each query cell line is excluded from the fold's
  fit/calibration labels. Legal source signatures may be provided when available.
  Report all queries, available/unavailable-reference queries, matched common
  support and globally unseen targets separately. An empty subset stays undefined.

The inherited 1,809-gene order, normalization, source folds and condition matching
are immutable. At least two source folds per task are required. The controller
checks the full parent manifest, frozen cell-line exclusion and available Round24
predictions before paid search. Historical files are not edited.

**Round23/24 confirmation has already been viewed.** Its reuse here is explicitly
`historically_viewed_development`, never fresh independent confirmation. Reusing
these data cannot establish an external generalization claim. HT29 and whole-line
outer tests stay closed. Changing the configuration to call inherited confirmation
independent is rejected. A future clean protocol needs new, audited labels/replicates
not used in earlier selection; simply reshuffling old target IDs is insufficient.

## Three matched arms

| Arm | Mutation | Bounded code repair | Typed scientific critic |
| --- | --- | --- | --- |
| `shinka` | Native Shinka | No extra repair policy | No |
| `repair` | Native Shinka | At most two corrections per failed lineage | No |
| `meta` | Native Shinka | Same repair policy | Yes |

Each arm gets separate repeat/transfer archives, the same initial methods, public
arrays, granular source diagnostics, worker limits and total request/token caps.
No arm can read another arm's evolving methods or private confirmation scores.
Each archive has two native Shinka islands; the method pool persists across turns
and resume. The two tasks have separate source fitness and finalist selection.

Native Shinka alone already uses textual feedback. The comparison specifically
adds a repair policy and then a separate structured research decision; it is not
a comparison of a feedback-free baseline with an information-rich proposed model.
Only one evaluation/proposal runs at a time on the single GPU.

Fixed numerical comparators include zero, background mean, calibrated support,
demeaned support for repeat, STRING ridge/factor16/factor32, uniform ensemble,
simple support/shared ridge, and the Round24 **source-selected** finalist. The
best fixed method is selected by source macro MSE separately for each task. Its
predictions are frozen before any agent runs. All arms start from that method.
Fixed source scoring and replay are shared to avoid charging the same setup to
each arm. No historical confirmation score chooses the initial method.

## Executable self-correction

Each new proposal first passes the real two-row, five-step deterministic contract
twice. The existing Landlock/seccomp worker then trains and evaluates it on legal
source folds, with the same full method under separate support-only, identity-only
and joint wrong/shuffled-input controls. Candidate code never runs in the trusted
controller and never receives query truth, provider credentials or an unrestricted
shell/network. The trusted numerical evaluator decides the score.

For a submitted failing numerical module, `repair` and `meta` select the actual
incorrect program in the **native archive** and enable native Shinka fix mode.
The generated correction becomes a new generation with its own code, evaluation
and parent link. A failed repair can receive one more correction; the original
failed lineage is limited to two total repairs. Valid but inaccurate methods are
not treated as runtime errors. A proposal rejected before an executable program
reaches the archive remains a counted generation/provider failure; it is not
misreported as a successful repair.

Every five distinct valid source behaviors, or following source stagnation, the
`meta` arm requests a typed `ResearchAction`. Source cards include three best and
three worst context/target gains, reference coverage, strong-response error,
response magnitude and independent input controls. The critic sees only source
records, its own arm's measured action history and the actual allowed parent IDs.

| Action | Controller effect |
| --- | --- |
| `repair_candidate` | Select an incorrect archive parent, within its two-repair cap |
| `refine_parent` | Use the chosen measured valid archive parent |
| `crossover_methods` | Select the first parent and supply the second as native archive inspiration |
| `explore_new_mechanism` | Supply the new hypothesis, optionally selecting a measured parent |
| `run_source_probe` | Execute an allowlisted source-table diagnostic and pass its measured result into mutation |
| `stop_unproductive_branch` | Exclude that parent from subsequent sampling; stop the task if none remain |

The native archive sampling hook implements these selections. It does not modify
the fitness evaluator or create another population. Every scientific action must
name evidence and predict a metric/direction/minimum change. The next measured
child is compared to the specified parent, and the hypothesis is recorded as
supported, unsupported or undefined. This is a source observation, not causal
proof that the critic itself improved the method. Code diffs, original failures,
repair lineage and outcomes are retained.
For a new-mechanism action with no declared parent, the fixed source-selected seed
is frozen as its prediction reference before the experiment. The controller does
not choose a convenient reference after seeing the child's score.

The critic may name a mechanism such as shared/target residual decomposition,
reference reliability, conditional transfer, low-rank/sparse responses, graph
features or learned blending. None is mandatory. Mechanism names are hypotheses;
different output signatures alone do not establish different biological mechanisms.

The installed 0.0.7 API is checked before starting. Native meta fields exist, but
native free-text meta recommendations are disabled here to avoid duplicate,
unaccounted critics; the typed critic uses the same audited provider relay.

## Frozen default budget

| Item | Default scope |
| --- | --- |
| Provider | DeepSeek `deepseek-v4-pro`, low reasoning effort |
| API requests | 40 per arm, **120 total**, including mutations, repairs and critic |
| Charged output tokens | 1.3 million per arm; mutation cap 32,768, critic cap 4,096 |
| Distinct behaviors | First 10 per arm; evidence-gated extension to at most 20, **60 total** |
| Behavior definition | Correct source predictions quantized at `1e-5`, excluding seed behavior |
| Source extension evidence | MSE gain >=0.1%, or strong-MSE gain >=1% with specificity gap gain >0.005 |
| Native islands | Two per task/arm, persistent archives |
| Worker compute | Shared baselines 1.5 hours + three arms ×3.5 hours = **12 aggregate worker hours** |
| Search reserve | 80% of each arm; remaining 20% reserved for frozen-finalist retest |
| Task allocation | Initially repeat 60% / transfer 40%; critic may request repeat 40–70% |
| Per-task source ceiling | At most 70% of the arm's source-worker allowance |
| Each worker | 120 seconds, 500 optimizer steps, 12 GiB RSS cap |
| Preflight contract | Two query rows, five optimizer steps, two deterministic executions |
| Disk reserve | At least 10 GiB before launch; identical public input files deduplicated where supported |

These are ceilings and desired candidate counts, not elapsed-time estimates or
guaranteed yield. Every failed paid attempt is counted. Unknown interrupted
provider outcomes retain their reserved quota. Repeated provider failures open a
circuit; caps never increase automatically. Allocation is a target for scheduling
one complete candidate at a time, not an exact fractional GPU occupancy guarantee.

The first tranche is extended only on source evidence. Stagnant tasks stop after
bounded opportunities to repair/revise them; reaching the candidate count alone
does not justify more requests. Unused budget and early stops are reported, so
the efficiency comparison uses actual requests and charged worker time as well
as the common caps. One seed is a comparative pilot, not a robust framework-wide
superiority claim. The fixed family is an ordinary comparator; no new random arm
is claimed in this release.

Source fitness is smooth; promotion is separate. An overall candidate needs >=1%
macro-MSE gain over the strongest fixed method. The target-response tier also
requires no strong-gene degradation, non-degraded positive specificity, useful
correct-vs-wrong/shuffled support and no near-zero collapse. Explicit gene-name
benefit is reported separately: a supported-repeat method may legitimately use
Rep1 without using a gene name. Transfer must beat zero and the strongest fixed
method on both all-query and matched common-support metrics. Undefined strong
metrics cannot pass a response tier.

At most two distinct finalists per task/arm are frozen using source metrics, one
for overall error and one for target response. **All six source selections freeze
before any evolved development-confirmation evaluation.** Confirmation does not
feed another mutation or critic in this campaign. Target-cluster paired bootstrap
and leave-contributor-out sensitivity are exploratory development analyses.
If no new method reaches a source promotion tier, the best new source method can
still receive a frozen `diagnostic_only_no_promotion` retest. This allows the three
arms' failures to be compared without misreporting them as promoted methods.

## Server use

Keep the existing main `.venv` and `/root/autodl-tmp/vcell-work/envs/shinka`.
No changes to raw data or older run directories are required.

```bash
cd /root/autodl-tmp/virtual_cell &&
git -c http.version=HTTP/1.1 pull --ff-only &&
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY &&
echo && export DEEPSEEK_API_KEY &&
screen -dmS vcell25 bash scripts/run_round25_autodl.sh \
  --name round25_self_correct_01 \
  --round24-run /root/autodl-tmp/vcell-work/runs/round24_mechanism_01
```

Optional numeric-only preflight uses the **same name/configuration**:

```bash
bash scripts/run_round25_autodl.sh --name round25_self_correct_01 --mode preflight
# Then start the paid workflow with --resume; export the provider key first.
```

Progress and logs:

```bash
.venv/bin/python scripts/round25_status.py --name round25_self_correct_01
tail -n 50 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round25_pipeline_*.log | head -n 1)"
screen -ls
```

Resume after interruption, exporting the key again in a new terminal:

```bash
cd /root/autodl-tmp/virtual_cell
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY
screen -dmS vcell25resume bash scripts/run_round25_autodl.sh \
  --name round25_self_correct_01 --resume
```

Do not start a second process while the campaign/GPU lock is held. Identical
assessments and saved critic decisions are reused. Completed native generations
remain in SQLite; interrupted turns reuse their frozen generation target.
Exact provider responses are cached by role, turn and request, preventing another
paid request for an already saved response. A crash after sending a request but
before its response is saved is ambiguous: the controller stops rather than
silently resending. A changed dataset, code, budget or environment requires a new
named campaign. Completed runs verify their manifests before returning.
Each turn also freezes its source-feedback snapshot. A changed request under the
same role/turn ticket is rejected before billing, rather than silently regenerating
a different proposal during recovery.

For an offline update, upload the supplied bundle to the data disk and run:

```bash
cd /root/autodl-tmp/virtual_cell &&
git bundle verify /root/autodl-tmp/virtual-cell-round25-<commit>.bundle &&
git fetch /root/autodl-tmp/virtual-cell-round25-<commit>.bundle main &&
git switch main && git merge --ff-only FETCH_HEAD
```

Then run the same launch/key command, omitting `git pull`.

## Outputs and interpretation

### Recovery from the initial baseline path error

The initial release incorrectly required baseline artifacts to be inside
`baseline_budget/`, although production places them in `baselines/{task}/`.
The corrected frozen pool plan has an explicit `budget.worker_root` pointing to
that pool's `assessments/` directory. `budget.root` remains the shared accounting
and input-cache directory. Sibling pools cannot write into each other's worker
directories, and both baseline tasks still share one evaluation-time cap.

If a campaign stopped at the first baseline with `Worker outside assigned budget
root`, update the code and start a new name, such as `round25_self_correct_02`.
Keep the failed directory unchanged: the correction changes the frozen code
fingerprint, and the old run also contains a cached invalid assessment. Do not
edit hashes or remove cache records to force a resume. At this failure point no
agent API request or numerical baseline training has started; only the kernel
preflight has consumed evaluation time. The new campaign repeats preflight and
baseline preparation. Subsequent interruptions of the corrected campaign can
use `--resume` with its new name.

The new directory is `/root/autodl-tmp/vcell-work/runs/round25_self_correct_01`.

- `campaign_plan.json`, `lineage.json`, `shinka_capabilities.json`,
  `sandbox_verified.json`, `baseline_replay.json`: frozen identities and preflight.
- `arms/{shinka,repair,meta}/{repeat,transfer}/evolution/programs.sqlite`:
  persistent native archives, kept on the server.
- `archive_export.json`, `attempt_log_export.json`: native parent/island records
  and failed-proposal diagnostics included in light review.
- `research_decisions.jsonl`: hash-chained actions, source probes and measured outcomes.
- `arms/*/turns/*/plan.json`, `done.json`, `code.diff`, `shinka.log`: exact turn recovery and modifications.
- `source_frontier.json`: compact source views, not another evolutionary database.
- `source_memory_library.json`: source-tested code and fold identities for future
  audited seed import. Automatic cross-campaign import is not enabled; this avoids
  carrying previously viewed validation labels into a different source scope.
- `evolution_trace.csv`, `repair_audit.csv`, `efficiency.json`: validity, novelty,
  correction success, measured hypothesis outcomes, tokens and compute.
- `source_scores.csv`, `confirmation_scores.csv`, `comparison.csv`,
  `paired_sensitivity.json`: per-query scores and development retest evidence.
- `summary.json`, `COMPLETE.json`, `round25_review_light.tar.gz`: bounded campaign
  completion and review package; completion does not imply predictive success.

Generated method runtime/peak CUDA allocation/optimizer-step measurements are
retained in each assessment's `candidate_runtime.csv`; actual RSS peaks are not
claimed, although RSS limits remain enforced. Currency cost remains unknown when
verified prices are unavailable. Saved provider response bodies and matrices are
excluded from the light archive; API usage, source decisions and code diffs are
included. Keep full runs for resume and exact numerical reproduction.
