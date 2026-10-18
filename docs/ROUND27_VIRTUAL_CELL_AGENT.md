# Round27: persistent Virtual Cell research agent

Round27 develops the existing ShinkaEvolve 0.0.7 + DeepSeek system into a
domain-aware, persistent research controller. It retains the numerical evaluator,
Linux Landlock/seccomp workers, native Shinka archive, source folds, ordered
1,809-gene panel, fixed comparators and seven donor/identity controls.

The agent can inspect the source catalog, execute numerical diagnostics, propose
complete prediction methods, repair code, revise a failed hypothesis, retain
complementary methods, pause/resume a campaign, and export a frozen research
method for prediction. It cannot edit the grader, silently change data partitions,
access HT29, or claim a mechanism from a method name. There is no new pretrained
Virtual Cell language model or automatic literature/database acquisition in this
release. Domain expertise consists of explicit task knowledge, checksum-verified
local GO/Reactome annotations, measured source evidence and provenance-linked
research experience.

## What changes after Round26

* Separate Repeat and Transfer goals: five distinct new numerical behaviors per
  task and per arm. Reaching ten in one task does not satisfy the other task.
  Resource exhaustion produces an explicit coverage-incomplete result.
* A fixed comparator equalling itself is not a scientific failure. Branch stops
  need actual exploration and cannot terminate the whole task. Stagnation asks
  for reconsideration rather than silently closing a task.
* Persistent hypothesis memory survives routine native mutations. A failed
  hypothesis or diagnostic remains pending until the Critic explicitly names its
  experiment ID, explains the failure and states the changed assumption. The
  subsequent measured experiment records that link. Memory does not prove that
  the Critic caused an improvement.
* `run_source_probe` executes the requested source computation. Coverage, context
  loss, target loss and strong-response diagnostics have frozen input hashes and
  outputs. A diagnostic is not counted as a new trained algorithm.
* The domain catalog records actual contexts, auxiliary source metadata, gene
  order, task definitions and controls. Unspecified metadata remain unknown.
  Gene symbols and supplied priors retain their existing input semantics; no new
  gene-function knowledge source is advertised as having been retrieved.

## Data and memory boundaries

The first campaign uses the full server Round26 run, normally
`/root/autodl-tmp/vcell-work/runs/round26_critic_02`. Its recorded Round25 parent
must remain present, including source inputs and numerical artifacts. A light
review archive is not sufficient. Parent checksums, folds, panel, fixed methods
and source prediction replay are verified before any paid call.

Common starts are selected from historical SOURCE records and replayed equally
for all arms. Confirmation winners are not used to select parents. For a later
tranche, pass a completed Round27 directory as `--parent-run`; its source-only
memory and methods become the common historical starting information. All three
arms receive the same historical information. Current-arm experiments remain
separate, so one arm cannot learn from another arm's ongoing search.

`memory/events.jsonl` is hash chained and idempotent by experiment ID.
`memory_library.json` exports source cards and typed hypothesis chains.
`/root/autodl-tmp/vcell-work/agent_memory/<name>/source_library.json` indexes the
immutable campaign export. Imported hypotheses remain hypotheses. Historical
confirmation and outer scores are excluded from planning memory.

Repeat is supported Rep1-to-Rep2 prediction, not unseen-target zero-shot
prediction. Transfer source folds exclude the query cell line from source labels.
The inherited confirmation sets are historically viewed development data. This
campaign opens neither a new whole-line outer test nor HT29. New independent
validation requires a separately audited, frozen protocol; a new random split of
previously inspected data is not automatically independent.

## Initial resource caps

| Item | Cap or goal |
| --- | ---: |
| Arms | Shinka / Repair / Repair + Meta |
| New behaviors per arm | Repeat 5 + Transfer 5 |
| API requests per arm, including Critic/schema repair | 30 |
| Output tokens per arm | 900,000 |
| Worker hours per arm | 3 |
| Shared historical replay | 2 worker hours |
| Live capability gate | 2.25 worker hours, 6 requests |
| Total new worker cap | 13.25 hours |
| Per numerical worker | 120 seconds, 500 optimizer steps, 12 GiB RSS |
| Concurrent GPU candidates | 1 |

These are resource ceilings, not runtime predictions or promises that every goal
will fit. Actual requests, Critic versus mutation roles, tokens, worker time,
cache reuse, task coverage and unused slots are reported separately. Currency
cost is not estimated. No automatic budget extension is enabled.

The live gate first tests the structured Critic, one actual native candidate and
post-result review. Main searches start only after numerical validity. There is
no local paid provider call in the offline test suite.

## Start on AutoDL

```bash
cd /root/autodl-tmp/virtual_cell &&
git -c http.version=HTTP/1.1 pull --ff-only &&
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY &&
echo &&
export DEEPSEEK_API_KEY &&
screen -dmS vcell27 bash scripts/run_round27_autodl.sh \
  --name round27_agent_01 \
  --parent-run /root/autodl-tmp/vcell-work/runs/round26_critic_02
```

The existing `.venv` and `vcell-work/envs/shinka/bin/python` are reused. No new
model or dataset download is required. The global evolution GPU lock prevents
overlap with an older running campaign. `--mode preflight` runs lineage/numerical
checks without opening the live provider; rerun identical options with `--resume`
to continue. `--mode live-smoke` stops after the paid capability gate.

For an offline bundle, upload the delivered file to `/root/autodl-tmp/` and use:

```bash
cd /root/autodl-tmp/virtual_cell &&
bundle=/root/autodl-tmp/virtual-cell-round27-REPLACE_WITH_DELIVERED_SHA.bundle &&
git bundle verify "$bundle" &&
git fetch "$bundle" main &&
git switch main &&
git merge --ff-only FETCH_HEAD
```

Then execute the key export and screen launch above. Keep the uploaded bundle out
of the trash directory; use its explicit actual path.

## Status, pause, resume and continued campaigns

```bash
cd /root/autodl-tmp/virtual_cell
.venv/bin/python scripts/round27_status.py --name round27_agent_01
tail -n 50 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round27_pipeline_*.log | head -n 1)"

.venv/bin/python scripts/round27_agent.py catalog --name round27_agent_01
.venv/bin/python scripts/round27_agent.py memory --name round27_agent_01
.venv/bin/python scripts/round27_agent.py models --name round27_agent_01
.venv/bin/python scripts/round27_agent.py knowledge --name round27_agent_01 --gene JAK1 --gene ATF3
```

Status distinguishes heartbeat freshness, experiment stage, worker progress,
per-task counts, pause requests, older errors and completion. A stage file alone
does not prove process liveness. The heartbeat stops before a completed run is
sealed, and subsequent status/control operations do not mutate the sealed run.

Gene queries use the repository's versioned GO Biological Process 2023 and
Reactome 2022 snapshots. `gene_knowledge.json` records source versions, alias
mapping, annotation terms and shared small-pathway associations. The Critic sees
bounded gene cards for its own source-error targets. Shared pathway membership
does not establish direct interaction, regulatory direction or a perturbation
effect. Missing annotations are explicit. No network download is performed.

```bash
.venv/bin/python scripts/round27_agent.py pause --name round27_agent_01
# Wait for the current committed algorithm experiment to finish and stage=paused.
.venv/bin/python scripts/round27_agent.py clear-pause --name round27_agent_01
screen -dmS vcell27resume bash scripts/run_round27_autodl.sh \
  --name round27_agent_01 \
  --parent-run /root/autodl-tmp/vcell-work/runs/round26_critic_02 \
  --resume
```

Export the API key in the current terminal again if it is a new SSH session.
`clear-pause` clears an external control flag; it does not launch a process.
An exact resume reuses committed decisions, provider responses, native
generations and numerical results. Unresolved paid-request tickets stop the run
rather than blindly resending a potentially charged request.

After completion, an explicitly started new tranche can use:

```bash
screen -dmS vcell27next bash scripts/run_round27_autodl.sh \
  --name round27_agent_02 \
  --parent-run /root/autodl-tmp/vcell-work/runs/round27_agent_01
```

That is a new bounded campaign with separately counted usage, not an unlimited
background loop. Code/config changes require a new campaign name.

## Frozen research prediction interface

After completion, `model_registry.json` lists method hashes and their public
source input templates. The prediction tool accepts that exact public NPZ schema
and ordered genes, holds all source training/calibration/auxiliary blocks fixed,
and allows new query blocks. It runs the registered method in a protected worker.
Query support provenance is caller supplied; this tool does not certify a donor
construction or perform an independent evaluation.

```bash
.venv/bin/python scripts/round27_agent.py predict --name round27_agent_01 \
  --method FULL_REGISTERED_MODULE_SHA256 --task repeat \
  --input /root/autodl-tmp/vcell-work/prepared/public_query_contract.npz \
  --output-name round27_prediction_01
```

Outputs go to `vcell-work/predictions/<output-name>/`. A new output name is
required. There is no raw-H5AD automatic query adapter or production deployment
claim. The source catalog/registry make the required preparation explicit.

## Review and acceptance

`round27_review_light.tar.gz` contains checksummed source diagnostics, typed
decisions, revision links, domain catalog, queues, model registry, numerical
comparisons and role-separated usage. It excludes private matrices, weights,
worker inputs and provider credentials.

* `COMPLETE.json`: bounded campaign execution ended and artifacts are sealed.
* `task_coverage_complete`: each arm met both task goals; exhausted caps do not
  silently count as coverage success.
* `framework_pass`: all arms met task goals and Meta completed at least one
  measured revision chain in each task. This tests workflow capability.
* `scientific_pass`: source and historical development criteria both passed for
  a selected method, with balanced coverage. It is not external validation.
* `agent_self_improvement_proven`: remains false. Measured chains and predictions
  alone do not prove a causal benefit over matched ordinary evolution.

Review `revision_chains.json` for changed assumptions and measured outcomes,
`comparison.csv` for algorithm performance, and `efficiency.json` for actual
Critic/mutation/repair costs. Numeric behavior diversity is not a count of new
biological mechanisms.
