# Round26: evidence-linked scientific self-correction

Round25 completed native evolution and two successful code repairs, but its three
Scientific Critic reviews were rejected. None of its new methods beat the fixed
source comparator. The supplied light archive contains 1,170 payloads; all sizes
and SHA-256 values were verified locally before implementing this round.

| Historical development confirmation | Fixed | Shinka | Repair | Meta |
| --- | ---: | ---: | ---: | ---: |
| Supported Rep1 -> Rep2 MSE | .004426115 | .004514674 | .004522489 | .004464435 |
| Held-cell-line transfer MSE | .004789582 | .004789787 | .004790065 | .004789787 |

Round25 used 13, 12 and 27 provider requests and obtained 10, 10 and 20 new
behaviors, respectively. These were unequal searches. The nonzero transfer
source candidates remain hypotheses: better strong-gene error and specificity
came with worse total MSE. They are shared starting material in Round26, not
confirmed transfer improvements.

## Execution order

1. Verify the full completed Round25 parent on the server. Freeze its exact
   source folds, gene order, strong masks, seed, optimizer steps and comparators.
   No new data are downloaded, and HT29/outer labels remain unavailable.
2. Replay the fixed source method and at most two nonzero source candidates per
   task. Fail before paid calls if ordered predictions do not reproduce within
   `rtol=1e-5, atol=1e-7`. All matched arms receive identical source starts.
3. Run a bounded **authenticated live gate**: an accepted non-fallback scientific
   action, one actual native Shinka generation, a valid numerical result and
   measured hypothesis outcome, then an accepted review explicitly citing that
   new result. Failure stops before the main search. Local mocked tests do not
   substitute for this gate; it runs with the server's exported key.
4. Run the three matched arms sequentially on one GPU. Each uses the same source
   folds, starts, Pareto sampling policy, 20-request ceiling, 10-new-behavior
   target, 700,000-output-token ceiling and 2.5 worker-hour allowance. Critic,
   schema correction and repair calls all consume that arm's request/token cap.
   Equal caps do not imply equal actual usage or attainment of all targets.
5. Freeze all matched source finalists. If predeclared with `--extend-meta`, a
   separate extension starts from the matched Meta source frontier, with its own
   native archive and 12 requests / 6 new behaviors / 1.5 worker hours. It runs
   only if source evidence passes the declared MSE or strong-response gain gate.
   It is never reported as an equal-budget fourth competitor.
6. Freeze all extension selections before retesting any evolved method on the
   historical development confirmation. Record every selected method; do not
   choose a new winner using those confirmation scores.

The default aggregate numerical-worker allowance is 9.5 hours: shared preparation
1.5, live gate .5, and three arms at 2.5. With extension it is 11 hours. The live
gate has at most six requests, so default total requests are at most 66, or 78
with extension. Provider wall time is separate; these are caps, not elapsed-time
predictions. Each candidate worker retains a 120-second, 12-GiB RSS and 500-step
cap. Source search reserves 20% of each arm's worker budget for confirmation.

## Scientific Critic and provenance

There is one `ResearchAction` type. The shorter contract supplies an actual valid
parent/evidence SHA, nonempty evidence IDs, available probes, fractional
`repeat_share` within `.4.. .7`, and a positive measurable prediction threshold.
Only invalid parents with remaining repair attempts can be repaired. Stopped
parents and undefined prediction metrics are rejected locally.

A locally rejected JSON/schema response gets **one explicit correction request**
containing the exact error. Both calls appear in the ledger. Transport errors,
HTTP errors and truncated responses are not silently retried. An ambiguous
in-flight ticket stops the campaign to avoid duplicate billing. Exact saved
responses, decisions, generation targets and source-feedback snapshots are reused
on resume. Saved decisions and response bodies are checksummed.
Numerical and provider cache-hit journals record reuses separately from paid
requests and new evaluations.

`critic`, `fallback`, `automatic_repair`, and `native_pareto` are separate action
attributions. A fallback never counts as an accepted critic action; an accepted
action without a measured child does not count as implemented. Two consecutive
failed scientific reviews stop the Meta arm. Runtime repair is limited to two
attempts per actual failed lineage.

The native Shinka 0.0.7 database remains the only evolutionary archive. Shared
source methods are registered there, and ordinary parent selection rotates
through the measured source Pareto view, including signal-bearing parents that
lose slightly on total MSE. An accepted explicit parent/fix/crossover request
takes precedence. Actual native parent IDs, islands and inspirations are saved.
No second population engine or LLM voting is introduced.
The pinned 0.0.7 row-based completion counter would count extra imported
generation-zero roots as completed trials. A narrow runner adapter counts its
native persisted generation IDs instead. Its archive, actual generation IDs and
controller request/worker caps remain authoritative. Sampling stays read-only;
shared roots are registered through the native writable API before the run.

Algorithms may change the whole `fit_predict(data,budget)` method. Source-fit and
calibration labels can determine mixtures, scaling, thresholds or training
parameters; source query labels, confirmation and outer labels cannot. The
unchanged independent donor-only and identity-only wrong/shuffled controls test
information usage. Whole-line transfer excludes the query cell line from fit and
calibration, and globally unknown targets cannot appear in source response labels.

The source-MSE finalist and an eligible signal finalist are retained separately.
The latter may be a diagnostic even when MSE loses. Scientific PASS requires the
existing meaningful MSE improvement and task-specific response/transfer checks;
nonzero amplitude by itself does not pass. Source-only Pareto coverage and a
historical confirmation score are not independent generalization evidence.

## Server commands

Use the **full** completed parent run, not the light archive. The default assumes
the repaired Round25 run was named `round25_self_correct_02`; change that argument
if your completed directory has a different name. Existing model environments
and datasets are reused. Round25's old frozen code identity cannot be resumed
under the changed evaluator adapter; keep completed historical runs intact.

```bash
cd /root/autodl-tmp/virtual_cell &&
git -c http.version=HTTP/1.1 pull --ff-only &&
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY &&
echo && export DEEPSEEK_API_KEY &&
screen -dmS vcell26 bash scripts/run_round26_autodl.sh \
  --name round26_critic_01 \
  --round25-run /root/autodl-tmp/vcell-work/runs/round25_self_correct_02
```

Default `--mode all` performs the live gate first and proceeds only if it passes.
For a smaller first execution, add `--mode live-smoke`. After successful smoke,
run the same name with `--mode all --resume`. `--mode preflight` makes no paid
calls. `--extend-meta` must be present from the initial launch if desired and must
be repeated on resume; it cannot be enabled after looking at confirmation scores.

```bash
.venv/bin/python scripts/round26_status.py --name round26_critic_01
tail -n 50 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round26_pipeline_*.log | head -n 1)"
ps -eo pid,etime,%cpu,args | grep -E '[r]un_round26.py|[r]un_shinka_round26.py'
```

Resume after exporting the key in a new shell:

```bash
screen -dmS vcell26resume bash scripts/run_round26_autodl.sh \
  --name round26_critic_01 \
  --round25-run /root/autodl-tmp/vcell-work/runs/round25_self_correct_02 \
  --resume
```

Upload a supplied offline bundle directly to `/root/autodl-tmp/` if GitHub fails:

```bash
cd /root/autodl-tmp/virtual_cell &&
git bundle verify /root/autodl-tmp/virtual-cell-round26-<commit>.bundle &&
git fetch /root/autodl-tmp/virtual-cell-round26-<commit>.bundle main &&
git switch main && git merge --ff-only FETCH_HEAD
```

Then run the key/start portion above. Do not overwrite frozen plans, edit old
hashes or remove failed assessments to force continuation with different code.

## Review outputs

`round26_review_light.tar.gz` includes its own `ARCHIVE_MANIFEST.json`, all action
and outcome traces, accepted/rejected critic responses, generated modules, native
archive/attempt exports, per-query scores, baseline comparisons, paired target
sensitivity and measured token/worker usage. Prediction matrices, model weights,
private grading data and provider credentials stay on the server; numerical
assessments retain independent checksums there.

`LIVE_SMOKE_COMPLETE.json` proves the bounded live capability gate, not accuracy.
`matched_search_frozen.json` fixes the equal-cap comparison before any extension.
`source_frontier.json` and `source_memory_library.json` retain source-only ideas.
`efficiency.json` separates critic validity, actual implementation, code repair,
fallbacks and unused slots. Method-family names are hypotheses, not automatically
counted biological discoveries. `COMPLETE.json` means the bounded search finished;
`summary.json.scientific_pass` separately states whether the selected methods met
the declared historical-development performance checks. Currency cost remains
unknown unless independently verified pricing is supplied.
