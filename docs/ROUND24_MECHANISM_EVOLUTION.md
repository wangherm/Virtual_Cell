# Round24: mechanism verification and bounded module evolution

Round23 produced a supported-repeat method with 2.315% lower development
confirmation MSE than demeaned support. Strong-response MSE was 1.82% worse and
specificity declined. This release freezes the exact winner
`e8b08448d8e7d92c8ac0c4c511ebec117c0f1e3f8b573a59582761d900568314`,
keeps its 1,809-gene panel, and explains the effect before extending the search.
The full server Round23 run is required; its light archive lacks worker inputs.

## What runs

1. Verify the completed parent and freeze source search/confirmation folds. No
   downloads, data reaggregation, changed gene panel, outer folds or HT29 access.
   Full parent verification may take several minutes on a large data disk.
2. Reexecute the original winner under Landlock/seccomp and compare every search
   and confirmation prediction with its saved Round23 counterpart, including row
   IDs and gene order (rtol 1e-5, atol 1e-7). A mismatch stops before paid search.
3. Evaluate the original and repaired winner under four auxiliary-data arms:
   none, historical only, new Papalexi/Adamson only, and all. The primary source
   training labels, queries, controls and panel remain identical. "New only"
   replaces auxiliary inputs; it does not remove the primary training data.
4. Evaluate simple demeaned support and shared low-rank calibration, disable
   target branches, disable the target-SVD/xridge branches, replace NNLS with a
   uniform nonconstant ensemble, and disable the final target-gene correction.
   Remaining base learners and stacking are refitted in each ablation. Uniform
   mixing is a defined alternative, not an assertion that it is equally flexible.
5. For original, repaired and simple support, evaluate the seven separate input
   conditions: correct; support-only wrong/shuffled; identity-only wrong/shuffled;
   both wrong/shuffled. Only identity changes alter target labels/STRING priors;
   true control covariates are preserved. Scores use the original query truth.
6. Separately evaluate original/repaired/simple methods on transfer source and
   development confirmation folds. No repeat gain is called transfer gain. No
   whole-line outer or sealed external evaluation is performed in this command.
7. Select an evolution seed using source MSE only: the shortest predefined module
   within 0.2% of the best fixed source MSE. Freeze the actual best fixed method
   as the comparator for new methods, including its source-selected confirmation
   predictions. A new candidate must beat this comparator, not merely Round23's
   older, potentially weaker demeaned-support baseline.
8. Run a small persistent Shinka module search and a disclosed narrow random
   alpha/shared-ridge comparator, then freeze source-selected finalists before
   any new-candidate confirmation. No confirmation-based winner selection follows.

The original winner is retained without semantic repair. The repaired version
joins native `train_source_context`, `calibration_source_context` and
`query_source_context` fields into the names used by the old code. It also adds
unambiguous HGNC-symbol-to-output-Ensembl joins. Original and repaired methods
have different source identities and are compared separately.

Kernel-only tracing records executed helper calls, global/context stacking
weights, matched target/output indices, auxiliary rows admitted by condition and
target, and branch differences from support. These are runtime diagnostics, not
candidate-reported scientific scores. Audits live in `branch_audit.json`.

## Search and acceptance rules

The complete algorithm module is editable, including imports and helper methods.
Shinka's EVOLVE comments are patch markers, not an immutable byte shell. Trusted
code adds numerical imports and executes only inside the existing restricted
worker; AST entrypoint validation is not presented as a security sandbox.
Candidate code cannot edit the grader, data splits, budget or parent artifacts.

Defaults in `configs/round24.json`:

| Budget / rule | Default |
| --- | --- |
| Provider | DeepSeek V4 Pro, low effort, 32,768 max output per call |
| Hard research limit | 60 requests, 1.8 million output tokens including reasoning |
| Pilot technical gate | At most 10 requests; at least 3 new valid behaviors and 40% yield |
| New valid behavior goal | 20, excluding seed clones and identical predictions |
| Search cycles | At most 12 five-slot cycles; stop after 2 zero-yield cycles |
| Charged worker budget | 8 cumulative hours; excludes preprocessing and API wait |
| Allocation cutoffs | Diagnostics 45%, source search 80%, random 90%, confirmation 100% |
| Each worker | 120 seconds, 12 GiB monitored RSS, at most 500 Torch optimizer steps |
| API circuit | Stop after 2 consecutive provider failures; no automatic reset |

The pilot checks engineering validity, not predictive success. A budget stop is
reported; 20 candidates are a goal, not a guarantee. Currency cost is unknown and
actual usage is recorded. One evaluation runs at a time on the single GPU; many
closed-form methods primarily use CPU. Deterministic methods are not rerun just
to attach different seed labels.

The smooth source objective rewards MSE gain and penalizes worsened strong-gene
MSE and specificity. Two frozen candidate tiers are reported:

* Overall-error candidate: at least 1% lower MSE than the strongest fixed method.
* Target-response candidate: also no increase in strong-gene MSE, positive
  specificity at least as large as the comparator, positive correct-vs-support
  and joint wrong/shuffled advantages, and no degradation when the correct
  identity is supplied. A separate `explicit_identity_benefit` flag requires an
  actual identity-only advantage. A method can use informative Rep1 without
  using explicit target-name metadata; that distinction is reported.

Amplitude is diagnostic and is never increased just to satisfy a minimum norm.
The non-worsening thresholds are numerical screening rules, not statistical
noninferiority tests. Source-selected response and overall finalists (at most
one per tier) are frozen before confirmation; their subsequent pass/fail results
are reported individually. The random comparator uses a narrower grammar, so
candidate-count matching does not make the search spaces equivalent.

Old Round23 confirmation targets and the winner have already been inspected.
Round24 confirmation therefore remains development evidence. Paired target-cluster
bootstrap summaries preserve cross-context dependence, but are exploratory,
uncorrected for algorithm selection, and not independent-test confidence intervals.
New independent guides/biological replicates or an untouched external experiment
would be needed for a final generalization claim. HT29 stays sealed.

## Server commands

Existing `.venv` and `work/envs/shinka` are reused. The default launcher finds the
latest completed Round23 run containing the exact frozen winner and prints its
path. To specify it, add `--round23-run /root/autodl-tmp/vcell-work/runs/round23_evolve_02`.

```bash
cd /root/autodl-tmp/virtual_cell &&
git -c http.version=HTTP/1.1 pull --ff-only &&
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY &&
echo && export DEEPSEEK_API_KEY &&
screen -dmS vcell24 bash scripts/run_round24_autodl.sh --name round24_mechanism_01
```

Numeric-only diagnostics do not need an API key:

```bash
bash scripts/run_round24_autodl.sh --name round24_mechanism_01 --mode diagnostics --resume
```

Progress, log and resume:

```bash
.venv/bin/python scripts/round24_status.py --name round24_mechanism_01
tail -n 50 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round24_pipeline_*.log | head -n 1)"
# Export the key again first if this is a new terminal.
screen -dmS vcell24resume bash scripts/run_round24_autodl.sh --name round24_mechanism_01 --resume
```

Do not launch a second process while the first holds the campaign/GPU lock.
Identical completed assessments are checksummed and reused. After source freeze,
resume skips paid search. Code/config/dependency changes require a new run name.
The full workflow keeps a 10 GiB minimum free-disk reserve. Identical public
worker inputs are hardlinked into an input cache where supported; private grader
truths are never part of that cache, and links are detached before a legacy
writer retries an interrupted worker.

If using the offline bundle, upload it, replace the path below with the actual
upload location, then launch using the commands above after the merge:

```bash
git bundle verify /root/autodl-tmp/virtual-cell-round24-<commit>.bundle &&
git fetch /root/autodl-tmp/virtual-cell-round24-<commit>.bundle main &&
git switch main && git merge --ff-only FETCH_HEAD
```

## Outputs

`runs/round24_mechanism_01/` contains `replay.json`, `diagnostics_frozen.json`,
`seed_selection.json`, `search_baseline.json`, `pilot.json`, `search_frozen.json`,
`evolved_confirmation.json`, `comparison.csv`, `paired_sensitivity.json`,
`report.md`, source-only feedback, API usage and failed proposal exports.
`summary.json` distinguishes overall error and target-response confirmation.
`COMPLETE.json` means the bounded workflow completed, not that a candidate passed.

The review file is `round24_review_light.tar.gz`. It includes source modules,
audits, per-record scores, paired sensitivity, stage/config and log tails. Keep
the full parent and new run directories on the server for exact replay; matrices,
worker inputs and weights are excluded from the light package.
