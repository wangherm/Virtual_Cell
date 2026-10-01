# Round 8: gene semantics, typed associations and pathway supervision

This is a **numerical perturbation-response experiment**, not a conversational
agent or a validated causal simulator. Existing Round 1–7 artifacts remain
unchanged. Jurkat test responses are never fitted, scored or packaged.

## Start on AutoDL

Reuse the current `.venv`, local Qwen3-0.6B-Base snapshot and prepared data.
No new foundation-model download, external LLM API or paid service is needed.
The initial knowledge preparation needs access to **mygene.info** and
**version-12-0.string-db.org**. Hugging Face remains offline.

```bash
cd /root/autodl-tmp/virtual_cell
git -c http.version=HTTP/1.1 pull --ff-only && \
screen -dmS vcell8 bash scripts/run_round8_autodl.sh --name round8_01 --resume
```

Do not launch a second screen job with the same run directory. Defaults are
**9 arms × 3 protocols × 3 seeds = 81 jobs**, 2 concurrent student processes on
one GPU, 2,000 optimizer updates per student, effective batch 16. A separate
teacher-preparation stage precedes the student jobs. This is a full development
matrix, not an 81-model ensemble or a preselected final winner.

For a short infrastructure check first, use a different run name:

```bash
bash scripts/run_round8_autodl.sh --name round8_smoke_01 \
  --protocols context --seeds 17 --arms qwen_base mlp_semantic qwen_modules \
  --max-steps 20 --save-every 10 --resume
```

The smoke run still builds the real knowledge snapshot; it does not replace
missing biology with random features. Later full runs reuse that snapshot.

## Progress and recovery

```bash
.venv/bin/python scripts/round8_status.py
tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round8_pipeline_*.log | head -n 1)"
```

Student logs live under `runs/round8_01/logs/`. `progress.json` lists running,
pending, complete and failed jobs. `INCOMPLETE.json` records a failed attempt;
`COMPLETE.json` is written only after every requested job and reporting finish.
An empty screen listing alone does not prove success.

After a process exits or the instance restarts, rerun the **same command** with
`--resume`. In-progress students restore model weights, AdamW optimizer state
and optimizer step. Each update's sampling/random seed is deterministic;
at most the unsaved updates since the last 100-step checkpoint are repeated.
Completed students verify checkpoint and prediction hashes before being reused.
Knowledge API requests and per-teacher OOF fold predictions are also cached.
Changing code, model weights, knowledge, dataset, split, seed or training budget
requires a new run name. You may reduce `--parallel-students 2` to `1` on resume
after an OOM; this does not change the scientific plan.

Use `--prepare-only` to finish the data, knowledge and teacher preparations
without launching students; continue with the same options plus `--resume`
and without `--prepare-only`.

## Inputs and disk use

Default inputs, relative to `/root/autodl-tmp/vcell-work`:

| Input | Location |
|---|---|
| Original prepared data | `prepared/real_min10_01` |
| H1 prepared expansion | `prepared/vcc2025_c_01` |
| Local Qwen | `models/Qwen3-0.6B-Base` |
| Four frozen teacher feature files | `teachers/four_teacher_01/{scgpt,state,scfoundation,geneformer}/features.npz` |
| UCE frozen feature file | `runs/round5_01/uce/features.npz` |
| Reusable knowledge snapshot | `knowledge/round8` |

The five feature paths can be overridden with `--teacher-features-json`, a JSON
object mapping those five family names to absolute paths. These must be frozen
control-feature files from the original preparation, with their audited JSON
sidecars. Old response predictions are deliberately not reused for new splits.

The full default run checks for about 49 GiB free output reserve. All new data,
caches, temporary files and logs go to the data disk. No old results are deleted.

If the public knowledge services are unreachable, requests retry and fail with
an explicit error; rerunning resumes cached requests. To prepare elsewhere,
use the same prepared panel and exact local Qwen snapshot, then transfer the
entire `knowledge/round8` folder. `--knowledge-npz /path/knowledge.npz` can use
an already prepared aligned snapshot; archive its accompanying provenance too.
This override does not certify the origin or biological correctness of supplied
features. Never construct knowledge from held-out response labels.

## Knowledge representation

* MyGene.info human gene names and summaries are resolved with explicit
  symbol/Ensembl-ID queries. Ambiguous mappings are unavailable, not guessed.
  At least half the perturbation targets must have summaries.
* The existing pinned GO Biological Process 2023 and Reactome 2022 libraries
  provide additional text and typed shared-annotation links.
* STRING 12.0 functional associations use confidence >=0.7. Only supported
  edges into the measured panel are used. These are **not signed causal edges**.
* Text is encoded once by the frozen local Qwen backbone (last real token,
  maximum 192 tokens, normalized vectors). The same target vectors enter the
  MLP and Qwen semantic arms; no external embedding service is called.
* Three relation channels (shared GO, shared Reactome, STRING association) each
  keep up to 16 measured neighbors. Their gene semantic vectors are pooled with
  weights conditioned on the normalized control state. Missing channels have an
  explicit availability feature. This first implementation uses **one hop**.
* Up to 64 Reactome modules, selected by fixed lexical order and measured-gene
  coverage (5–150 members), define mean signed expression-response targets.
  They are not inferred causal pathway-activation scores. Selection never uses
  response effect sizes. Gene descriptions, edges, raw requests, hashes and
  coverage are retained for audit.

## Fixed experimental matrix

| Arm | Input / loss change |
|---|---|
| `qwen_base` | Qwen with control tokens and target-name prompt |
| `mlp_base` | Control expression + learned target ID, small network |
| `mlp_semantic` | Same MLP family, fixed target semantic representation |
| `qwen_semantic` | Qwen + the same fixed target semantic representation |
| `qwen_graph` | Semantic Qwen + three typed, control-conditioned neighbor tokens |
| `qwen_modules` | Graph Qwen + true training-response module supervision |
| `qwen_kd` | Module-supervised Qwen + fit-pool OOF teacher distillation |
| `shuffled_semantic` | Semantic Qwen with target/vector correspondence shuffled |
| `random_graph` | Graph Qwen with a fixed permutation of destination gene labels |

Random graph controls preserve each relation row's size/weights and the global
destination-degree distribution (not each named gene's degree). Shuffling does
not remove target-name knowledge already present in the pretrained Qwen.
All arms have a module head feeding the numeric prediction head; only the module
and KD arms supervise it. Thus D→E changes the auxiliary loss without adding
parameters. The numeric head predicts a rank-32 response basis fitted solely on
that protocol's fit rows. This basis is not called a biological pathway basis.

Fixed budgets, shared seeds 17/29/43, constant learning rate 0.0002, row-uniform
normalized-delta MSE and the same calibration procedure apply to all arms.
The module loss coefficient is 0.1 and KD coefficient 0.5 times the fitted
reliability gate. No outer-set early stopping or checkpoint selection occurs.
These are controls for sample/update exposure, not equal FLOPs or parameter count.

## Splits and calibration

All arms use the same intersection gene panel and the same expanded training
pool. Each protocol creates disjoint fit, calibration and outer row sets:

* **context:** fit/calibration use separate training-pool batches; the complete
  original HepG2 validation panel is outer evaluation. Any naturally unseen
  targets remain present and their reference-coverage count is reported.
* **target:** 20% of training-pool targets are outer, another 20% calibration;
  all remaining targets fit. The same held target never appears in fitting.
* **double:** held targets are chosen among identifiers shared by the training
  pool and HepG2. Their HepG2 rows form outer evaluation and all their training
  responses are excluded. A different held-target set supplies calibration in
  training contexts. No target/context-response labels determine membership.

The context protocol retains batch separation inherited from preparation;
batch names are not automatically independent biological replicates. Target
and double protocols hold out targets across all batches. Existing development
sets have been examined previously; these are exploratory comparisons.

Every student fits its own scalar shrinkage and convex
`student + fit-only mean_transfer + zero` mixture on its calibration rows only.
Calibration is applied to the **same fitted model**, not transferred to a model
refitted on more labels. Report raw, shrunk and mixed scores separately.
`zero` means predicted expression delta = 0; mean transfer averages available
training-context responses for the target, falling back to a training-only
global response for an unseen target.

## Teacher use

Five frozen encoders are reused, but new VCell task heads are cross-fitted
leaving out a training context, using only the current protocol's fit rows.
Known fold targets with original-source features receive KD; H1 and unsupported
targets are masked. Reliability weights are tuned using these OOF predictions
relative to calibrated mean transfer, with a permitted zero KD gate. They are
**not proof of improvement over calibrated Qwen**; the independent F-vs-E outer
comparison tests that claim. Pretrained overlap remains unverified. State here
is SE plus a VCell head, not native State ST. CellOracle/scTenifold are not rerun
or promoted to global expression labels in this experiment.

## Outputs and interpretation

* `comparison.csv`: raw, shrinkage and calibrated MSE for every arm/seed/protocol.
* `seed_summary.csv`: mean, standard deviation and seed count.
* `paired_comparisons.csv`: within-seed adjacent-arm and shuffled-control gains.
* `per_target.csv`: context-target errors and row counts; inspect concentration.
* `splits/*/membership.csv`: exact membership, including excluded rows.
* `students/*/calibration.json`: coefficients, coverage and module-prediction error.
* `students/*/endpoint.pt`: final numerical student adapter/head checkpoint.
* `teachers/*/audit.json`: head-fit membership, feature provenance and KD support.
* `round8_review.tar.gz`: result tables, manifests, predictions, calibration/outer
  truth on development panels and log tails. Backbone files and raw data excluded.

The main metric remains equal-context/equal-target MSE. Equal context-target-unit
MSE is a sensitivity measure. The 81 jobs are a predeclared matrix; conclusions
must consider paired seeds, per-target support and the multiple comparisons.
The launcher deliberately does not automatically declare a winning model or
unseal Jurkat. Module outputs are numerical hypotheses; fluent mechanistic
explanations and biological causal validity are not claimed.

The `vcell.round8.load_checkpoint` helper restores an endpoint using its frozen
backbone and hashed knowledge file. Keep those dependencies when moving runs.
New inference datasets currently require the same ordered gene/target vocabulary;
dynamic arbitrary-gene onboarding is outside this round.
