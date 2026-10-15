# Round22: bounded autonomous method discovery

This is a **small IFNG method-discovery pilot**, using ShinkaEvolve 0.0.7 and
DeepSeek to mutate an entire prediction method. It does not update the DeepSeek
API model's weights. It does not reuse a pretrained Qwen numerical backbone.
The numerical inputs are source-only raw matched controls, frozen STRING priors,
fresh source-fit projections and legal experimental donor signatures.

## Server installation and first run

Keep the complete Round21 run and its parent prepared data, Round15 relations and
earlier lineage artifacts. A light review archive alone cannot train this experiment.
The default parent is `/root/autodl-tmp/vcell-work/runs/round21_signal_01`.
The default outer development line is HAP1, so reproducible BXPC3 signatures remain
available as source evidence. Nothing writes to completed Round21 data.

After pulling main or applying the supplied Git bundle:

```bash
cd /root/autodl-tmp/virtual_cell
bash scripts/install_round22_autodl.sh

# Real kernel protection and two independent deterministic numerical evaluations.
# No API key or paid search is needed for this stage.
bash scripts/run_round22_autodl.sh --name round22_evolve_01 --mode smoke
```

Only after that command succeeds, start the live pilot:

```bash
cd /root/autodl-tmp/virtual_cell
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY
screen -dmS vcell22 bash scripts/run_round22_autodl.sh \
  --name round22_evolve_01 --resume
```

Same command with `--resume` continues the same frozen experiment. Shinka restores
its database and population. Completed identical workers are verified and reused.
Once `search_frozen.json` exists, resume bypasses the search and API altogether;
confirmation or outer feedback cannot reopen evolution.
Changed code, folds, model or budgets require a new run name. Attempts interrupted
before a usage/result record keep their reserved budget; they are never billed as
zero work. Failed worker retries create additional charged attempts.

```bash
.venv/bin/python scripts/round22_status.py --name round22_evolve_01
tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round22_pipeline_*.log | head -n 1)"
tail -n 40 -f /root/autodl-tmp/vcell-work/runs/round22_evolve_01/logs/shinka_repeat.log
```

The full default run uses two separate task searches, six framework generations
each, including initial seeds: **12 total generation slots**, not 100. Each task
has its own archive and comparable fitness; islands within a task explore method
diversity. A generation is not necessarily one successful candidate or one API
request. At most 24 forwarded provider requests and 196,608 completion tokens are
allowed across tasks, including failed/truncated calls. Provider reasoning tokens
are included in completion usage. Input usage is recorded; request bodies have a
2 MB size ceiling. Unknown provider prices remain `null`, never claimed free.

The default provider policy is `--reasoning-effort low --api-max-tokens 32768
--api-failure-limit 2`. The per-call output ceiling includes reasoning and final
code together. The original high-effort 8,192-token policy repeatedly truncated
before any final code on AutoDL; it must not be retried unchanged. The total
196,608-token and 24-request ceilings are unchanged. A request reserves its full
ceiling before forwarding, so the remaining total budget may stop another call
even when a few tokens remain. Two consecutive failed provider responses open a
persistent circuit: SDK/framework retries receive a local error without another
paid request. Success resets the consecutive-failure count. No automatic switch
of model, effort or budgets occurs mid-experiment.

Before research generation, the live provider smoke now requires a tiny final
Python function, parsed without executing it. A successful greeting alone does not
verify code output. This probe still cannot guarantee a full research mutation.
`round22_status.py` shows finish reasons, reasoning-token usage and final-answer
character counts. Framework `$0.0000` is an unknown local-model pricing placeholder,
not a measured zero-cost provider bill; use the relay's actual token usage.

For the truncated `round22_evolve_02` run, update and start `round22_evolve_03`:

```bash
bash scripts/run_round22_autodl.sh --name round22_evolve_03 --mode smoke --resume
# After numeric smoke succeeds, with DEEPSEEK_API_KEY exported:
screen -dmS vcell22fix bash scripts/run_round22_autodl.sh --name round22_evolve_03 --resume
```

One GPU evaluation at a time; at most four worker-affinity CPUs, two numerical
threads, 120 seconds per worker and a monitored 12 GiB host RSS ceiling. The shared
worker wall-time budget is two hours, including controls and confirmations. It is
an upper bound on occupied GPU evaluation time, not measured GPU-active time.
Standard Torch optimizers are instrumented to cap and report 500 total steps per
worker. Custom manual updates/closed-form methods remain bounded by elapsed time;
do not claim this instrumentation is a universal FLOP counter.
Torch starts with a 25% device allocator limit and reports allocated VRAM. This is
not hardware GPU partitioning and cannot enforce all allocations from arbitrary
native libraries. The supplied gradient-step budget is 500; arbitrary closed-form
and custom methods are compared under the hard time/resource ceilings rather than
claimed identical FLOPs or identical optimization updates.

An exhausted budget produces an incomplete scoped result, not a false COMPLETE.
Extend budgets only in a new frozen experiment. `--tasks repeat` or `--tasks transfer`
can run one task independently under a new name. `--outer-context BXPC3_IFNG` also
requires a new name; it has weaker legal source reference coverage.

## Scientific tasks and partition boundary

### Recovery of the empty outer fold in release 39019e5

The final-fit `partition.outer` field inherited from Round21 is a source diagnostic
partition, not the whole held-out cell line. The frozen held-line IDs are in that
fold's `query_rows`. Updated preparation validates these IDs and creates the outer
fold before allowing paid search. The original unit test incorrectly equated the
two fields; the regression fixture now mirrors the real parent schema.

If `round22_evolve_03` stopped with `No legal folds for transfer/outer` after both
freezes, retain the source run and finish in a separate recovery artifact:

```bash
cd /root/autodl-tmp/virtual_cell
.venv/bin/python -u scripts/recover_round22_outer.py \
  --source-name round22_evolve_03 --name round22_evolve_03_outerfix
.venv/bin/python scripts/round22_status.py --name round22_evolve_03_outerfix
```

No API key is needed. Recovery is restricted to the affected published namespace
and exact empty-outer failure. It verifies the original parent, candidate code,
search/confirmation metrics, winners and all public/private source fold hashes.
Rebuilding must reproduce every search and confirmation input, label, transform
and row ID exactly. It then grades only already-frozen transfer finalists on the
correct inherited held-line queries. It neither retrains source candidates nor
reopens search. The original run stays untouched; its elapsed evaluation charges
carry forward into the same total cap. Recovery has its own frozen code identity
and completion marker, and supports rerunning the same command after interruption.
Its review archive contains inherited source metrics, candidate code and API usage
alongside new outer scores; it is not a new independent research experiment.

Output: `/root/autodl-tmp/vcell-work/runs/round22_evolve_03_outerfix/round22_review_light.tar.gz`.

**Repeat:** permitted query Rep1 response predicts sealed Rep2. Whole target hashes
split other source targets into search and source confirmation. Training, response
projections and noise masks never see query Rep2. This is supported prediction of
technical repeats, not unsupported perturbation prediction or biological replication.
There is no whole-cell-line outer score for this task: that would need an independently
justified adaptation task, not pretending source training from the same line exists.

**Transfer:** the original three source IFNG query cell lines are held out in turn;
only other lines supply fitting responses. Confirmation target groups are withheld
from all search fitting and query feedback. After code freezes, confirmation fits
may use same-target responses in other permitted source lines, but never the held
query line's responses. Unknown-target outer queries never acquire same-target
responses from a forbidden source. The original whole-line development query rows
are preserved. HT29 remains reserved; historically opened Jurkat is not an untouched
test. HAP1/BXPC3 outer results are observed development diagnostics.
The new confirmation groups are isolated from this search, but their underlying
data were analyzed in earlier project rounds. This is algorithmically held-out
development confirmation, not a newly collected independent biological dataset.

All response units remain inherited `mean(log1p(10000 * counts / full_library))`
matched-control mean differences in the original ordered 1,834 genes. The raw data
and pseudobulk normalization are not repaired or recomputed in this round.

No old response-trained encoders or teacher scores are reused through the new
confirmation boundary. Fixed comparators are rebuilt on exactly these permitted
source inputs: Zero, background mean, calibrated support, demeaned supported repeat,
full-output raw-control/STRING Ridge and low-rank output Ridge ranks 16/32. These
are **not the original pretrained-Qwen P4 factor neural models**. Old P4/teacher
results are historical context, not matched baselines for this new partition.

Transfer makes all legal same-target donor responses available, with exact
dataset/stimulus/time/modality/batch matching, rather than forcing a hard reliability
filter to return zero before an algorithm can investigate weighting/noise handling.
Strong-gene masks still come only from source technical evidence. Reference coverage,
common support and known/global unknown subsets must be read alongside scores.

## Evolution and independent grading

Only `experiments/round22_evolve/initial.py`'s EVOLVE-BLOCK may change. Its stable
entrypoint is `fit_predict(data, budget) -> [n_query, 1834]`. Agent code may use
source training/calibration labels, raw control/prior arrays, fresh projections and
support responses. It can change features, losses, optimizers and architectures.
The trusted evaluator owns query truth and all scoring.

Each candidate is evaluated on correct, wrong and shuffled target inputs with the
same source labels, seed and worker limits. Query identifiers are opaque; true
target names cannot be recovered from a supposedly scrambled barcode. Wrong-target
features/priors/support change together; control state and response grading stay fixed.
Transfer specificity comparisons use the same common reference support.

Context-target macro original-delta MSE, strong-gene error, norm ratio, target win
fraction, background-subtracted direction/specificity and same-method ablation gains
are reported. Smooth source search fitness allows small improvements to inform the
next mutation; `signal_screen_passed` requires >=1% gain over the strongest frozen
source-selected baseline, noncollapsed amplitude, defined strong masks, positive
specificity and correct-input advantage against both wrong and shuffled controls.
Passing this numerical screen is not a biological-success declaration. A scalar search score is not evidence of
independent biological success.

The top two distinct evolved codes per task freeze before source confirmation.
One random-mutation finalist and one fixed seed accompany them. Confirmation selects
the evolved finalist before any outer score. Do not start another adaptive search
using confirmation/outer results and then call those same records independent.

Random control uses equal generation-slot counts and worker limits but a disclosed
rank/regularization/centering mutation grammar, narrower than unrestricted LLM method
code. It does not establish a perfectly scope-matched advantage of autonomous coding.
Round19's restricted head search is not claimed a fair unrestricted-method comparator.

## Candidate isolation

Shinka's default local-process evaluator is **not a filesystem sandbox**. We use its
documented `metrics.json` / `correct.json` contract, but never import generated code
inside the trusted grader or stock unrestricted `run_shinka_eval` loader.

Before loading any generated module, a standalone child applies Linux Landlock ABI
>=1 and libseccomp. Older ABIs have additional syscall guards for truncate/open
semantics; metadata-changing calls and filesystem mutation ioctls are denied in all
versions. Allowed reads are runtime libraries, its code and public input
pack; allowed writes are its own output and standard random/null devices. Network,
process creation, execution, ptrace/process-memory access, privilege changes and
namespace/mount operations are denied. Native numerical threads are allowed. A
parent-death SIGKILL prevents an evaluator timeout from orphaning its candidate.
The worker starts with a credential allowlist and no provider key or user PYTHONPATH.

The same production worker runs actual forbidden-read/write/socket/subprocess probes
before any API search. No unsafe fallback is available. If the AutoDL kernel blocks
Landlock or libseccomp is missing, inspect `preflight/worker.log`; install libseccomp2
if that is the missing dependency. A kernel restriction needs a host/container
supporting the boundary; a prompt or AST check is not a replacement.

The trusted orchestrator may run on the network-enabled AutoDL instance. It alone
holds the DeepSeek key; a localhost relay forwards bounded calls, audits usage and
passes only a temporary relay credential to Shinka. Candidate workers cannot use
the network or either credential. Large matrices remain on the server.

## Outputs and verification limits

The fixed-method smoke compares scientific metrics, per-context feedback and code
identity exactly. Execution time, peak GPU allocation and optimizer-step telemetry
remain in the reports but are excluded from this prediction reproducibility check.
If a run stopped with `Fixed-method metrics did not reproduce exactly` on the
initial release, update the code and use a new run name (for example
`round22_evolve_02`); the old run's frozen code hash must remain intact. Existing
datasets, the installed Shinka environment and the failed run are retained.

Read `summary.json`, `comparison.csv`, `search_frozen.json`,
`confirmation_frozen.json`, `api_ledger.json`, `sandbox_verified.json`, and
`evaluation_budget.json`. The light review archive is:

`/root/autodl-tmp/vcell-work/runs/round22_evolve_01/round22_review_light.tar.gz`

Keep the full server inputs/private grading matrices, candidate worker artifacts,
Shinka SQLite database and package lock to reproduce the run. The light archive
omits those matrices and any provider credentials.

Local Windows tests exercise numerical grading/contracts with synthetic fixtures.
Linux CI must run actual protection probes. Real AutoDL CUDA isolation and live
DeepSeek compatibility are checked by server smoke/search, not claimed tested by
offline unit tests. Published code alone is not an 8–12-generation search trace;
the server run produces that trace and may legitimately report no predictive gain.

Official interfaces: https://github.com/SakanaAI/ShinkaEvolve and
https://sakanaai.github.io/ShinkaEvolve/support_local_models/ .
Provider thinking/output semantics: https://api-docs.deepseek.com/guides/thinking_mode/
and https://api-docs.deepseek.com/api/create-chat-completion/ .
