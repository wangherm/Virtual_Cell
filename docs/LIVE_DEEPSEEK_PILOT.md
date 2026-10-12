# Live DeepSeek model-selection pilots

The API planner implements the existing `decide(observation)` interface. The
controller validates and executes every expert tool. An API failure never
silently switches to a deterministic policy.

Defaults, verified against official documentation on 2026-10-12:

| Setting | Default |
|---|---|
| Endpoint | `https://api.deepseek.com` |
| Model | `deepseek-v4-pro` |
| Thinking / reasoning effort | enabled / `max` |
| Protocol | Chat Completions JSON mode, strict **local** Action validation |
| Output tokens per request | 8,192, including reasoning |
| API attempts / controller steps per episode | 8 / 8 |
| Episodes | 3, sequential |
| Expert-tool budget per episode | 3.1 declared units, separate from API tokens |

Sources: [Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/),
[Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/),
[JSON Output](https://api-docs.deepseek.com/guides/json_mode/).
Discover account-visible models first. An unavailable Pro model causes failure,
not a downgrade. Returned model IDs are recorded; API aliases do not pin weights.

## Set up the server

Update with Git or the offline bundle. `requests` is already a core dependency;
no SDK install, new model download or GPU training is needed.

Enter the key interactively in the AutoDL/JupyterLab terminal:

```bash
cd /root/autodl-tmp/virtual_cell
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY
.venv/bin/python scripts/model_agent.py deepseek-models
```

The key is read from the environment and passed only in an Authorization header.
It is absent from prompts, configs, fingerprints, traces and archives. The new
screen process inherits the exported variable. After reconnecting, set it again
to resume unfinished API decisions.

## Pilot 1: three real API episodes with synthetic experts

```bash
screen -dmS deepseekpilot bash scripts/run_deepseek_autodl.sh \
  --name deepseek_pilot_01 --mode synthetic --rounds 3
```

This prepares synthetic tools/profiles and calls DeepSeek to choose each action.
Episodes repeat the same synthetic query with independent audit sessions; they
are not biological replicates or optimizer seeds. Synthetic truth is read only
by the separate evaluator after sealed decisions. This checks API/tool execution,
not biological improvement.

## Pilot 2: completed P4 models and real controls

```bash
screen -dmS deepseekreal bash scripts/run_deepseek_autodl.sh \
  --name deepseek_p4_pilot_01 --mode p4 --rounds 3 \
  --p4-run /root/autodl-tmp/vcell-work/runs/round15_p4_02
```

After synthetic connectivity succeeds, use four episodes to cover all four
development backgrounds with the current exporter:

```bash
screen -dmS deepseekdata bash scripts/run_deepseek_autodl.sh \
  --name deepseek_real_data_01 --mode p4 --rounds 4 \
  --p4-run /root/autodl-tmp/vcell-work/runs/round15_p4_02
```

Both data conditions retain the entire shared frozen target-annotation cache,
including targets absent from the reference training dataset. Their original
encoder, PCA, response basis and model weights remain unchanged. This avoids
artificially disabling the reference no-ID heads on added genes such as ATF3.
Fit exposure and same-target reference responses are not invented: a reference
tool without a fit-only donor remains unavailable. Annotation hash and gene panel
are part of the model recipe identity, so incompatible older risk profiles fail.

Before any API request, `real_data_readiness.json` checks identical annotations
and input support for all six numerical predictors on each selected query.
It records capabilities of all eight tools and is included in the review package.
Use a new run/export name for this exporter update; previous bundles are immutable.

P4 must be complete and checksummed. The script exports reference and expanded
seed17/context workers, each with Ridge, rank16, rank32 and fit-only reference.
It uses the worker's canonical/harmonized target vocabulary. Queries are the
identifier-sorted first training row in each available A549_IFNG, BXPC3_IFNG,
HAP1_IFNG, MCF7_IFNG background. Three episodes use the first three backgrounds;
additional episodes cycle through the available cases. Responses and historical
error tables do not determine case selection.

This is a real model-execution pilot, potentially in sample. Expanded P4 models
were fitted on these development backgrounds. Real biological MSE or agent
quality rankings are not produced here; HT29/Jurkat stay closed. Without genuine
source-OOF profiles, estimated accuracy remains null. Descriptions, prediction
norm and disagreement cannot establish true quality. `--profiles DIRECTORY`
accepts compatible genuine profiles; synthetic profiles are not used with P4.

`--prepare-only` exports inputs without API requests. Start decisions later with
the same name/options plus `--resume`. `--protocol target/double`, `--seed`,
`--model`, `--base-url`, `--reasoning-effort high/max` and token limits allow
explicit alternatives. New frozen settings or source code require a new name.

## Progress and resume

```bash
.venv/bin/python scripts/deepseek_pilot_status.py --name deepseek_pilot_01
tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/deepseek_pilot_*.log | head -n 1)"
```

`DEEPSEEK REQUEST` prints before the API wait. A `started` attempt is outstanding
or interrupted with unknown outcome. `DEEPSEEK ACTION` marks a complete valid
answer. `DEEPSEEK ROUND COMPLETE` / `DEEPSEEK PILOT COMPLETE` mark completion.
Pro/max can take time. `--api-timeout` is a read-inactivity timeout, not a total
wall deadline; provider keepalive traffic can extend total waiting time.

```bash
screen -dmS deepseekresume bash scripts/run_deepseek_autodl.sh \
  --name deepseek_pilot_01 --mode synthetic --rounds 3 --resume
```

For P4 retain the same parent/protocol/seed/profile options. Completed episodes
are verified without another API request. Accepted decisions are cached by
observation and provider-config identity before tool execution. Pending tools
resume without a new decision. A crash before cache writing can require another
request; previous attempts still count against the cap. Resume never resets caps.
Preserve the full `api/` directory when moving results. Missing audit directories
or deleted attempt records are rejected on resume rather than resetting spending.

Review artifacts in `runs/NAME/`:

- `summary.json`: selected expert, task, tool units and API usage per episode.
- `episodes/round_*/trace.json`: observations, decisions and provider metadata.
- `episodes/round_*/selected.npz`: ordered numeric gene-expression delta.
- `api/round_*/attempt_*.json`: each attempt's status, usage, latency and IDs.
- `api/round_*/calls.jsonl`: append-only live API events.
- `INCOMPLETE.json` / episode `PLANNER_ERROR.json`: explicit failure records.
- **`model_agent_review_light.tar.gz`**: hashed review package without arrays,
  model weights, credentials, prompts or provider reasoning text.

HTTP errors, empty answers, truncated output, extra/duplicate fields, unsupported
experts, premature finish and over-budget actions stop explicitly. No automatic
retry, JSON repair, model downgrade or rule fallback occurs. Local tool failures
are recorded for the controller's next observation.

## HTTP 200 followed by JSONDecodeError

This means the API returned a normally ended answer, but its final `content`
could not be parsed as the action JSON. It is distinct from authentication,
network, quota or token truncation. The first reported failure's final text was
not retained, so its exact formatting cannot be established retrospectively.

The parser now removes only a complete outer Markdown JSON code fence or leading
BOM. It does not extract JSON from arbitrary prose, repair syntax, add fields,
choose a substitute action or make an extra provider request. Duplicate/extra
keys and unsupported/over-budget actions still fail. The prompt also explicitly
requires the final answer to begin/end with braces.

`final_format` records a format class, length and digest; `json_error` records
parse position/line/column. Raw final text, reasoning and credentials are not
logged. The status command prints these fields on failure. After this code/prompt
update, use a fresh `--name deepseek_pilot_02 --rounds 3`; `_01` retains its
existing attempt and token audit. Its frozen code/prompt identity is not resumed.

## Costs and failed attempts

Every POST attempt is reserved durably before transmission, including failed or
interrupted requests. Usage is logged before local action validation, so paid
invalid answers remain visible. Missing usage is unknown, not free. Output token
counts include reasoning; reasoning tokens are a subset and are not added twice.
Currency cost follows provider billing/cache/peak status and is not estimated
from missing data.

Defaults cap 3 × 8 = 24 attempted decision requests and 196,608 requested output
tokens; input counts vary. Expert `budget-units` excludes API tokens. Equal-cost
scientific comparison still needs measured local inference and planning costs.

For truncation, use a new name with `--max-output-tokens 16384` or
`--reasoning-effort high`. Request-cap exhaustion requires a new name for further
requests. HTTP status is recorded for 401/402/403/429; inspect account access,
balance, network and rate limits rather than assuming a model problem.

## A custom live query

```bash
bash scripts/run_model_agent_autodl.sh run \
  --registry /PATH/registry.json --query /PATH/query.json \
  --policy llm --provider deepseek --model deepseek-v4-pro \
  --reasoning-effort max --max-output-tokens 8192 --max-api-calls 8 \
  --max-steps 8 --budget-units 3.1 --output /PATH/live_query_01
```

The audit lives in sibling `live_query_01_api/`, outside sealed controller results.
Fresh observations are sent each turn without native API `tools` messages, so
provider `reasoning_content` is neither saved nor required in subsequent requests.
Only task/candidate metadata, source-only risk estimates and allowed feature/
prediction summaries are sent; control/response arrays and private labels stay local.

## Validation and next stage

Local tests use mocked HTTP, never a personal key or paid request. They cover
payload shape, failed/invalid accounting, redaction, decision caching, completed
offline resume, three-episode orchestration, review hashes and actual numerical
P4 export/inference. Account access and AutoDL connectivity are tested by your run.

Next: nested IFNG source-OOF refits, measured costs and matched fixed/rule/static/
one-shot/fixed-workflow/adaptive comparisons. Connectivity and repeated pilot
episodes do not establish predictive benefit or release closed backgrounds.
