# Round 14: local biology and controlled extrapolation tests

Round13 reduced double-holdout error relative to Ridge, but its bounded additive
network already matched its factorized network. Interaction is therefore an
unresolved hypothesis. This round tests local STRING features with a factorial
design and additional information, extrapolation, capacity and optimization controls.

## Launch on AutoDL

Keep the full `round12_01` and `round13_01` runs, their prepared data, knowledge,
background encoders and local model dependencies. A review archive is insufficient:
it intentionally omits predictions and checkpoints. No new downloads are needed.

```bash
cd /root/autodl-tmp/virtual_cell && \
git -c http.version=HTTP/1.1 pull --ff-only && \
screen -dmS vcell14 bash scripts/run_round14_autodl.sh --name round14_01 --resume
```

Default: 21 heads × 3 protocols × 3 optimizer seeds = **189 heads**, with
**414,000 optimizer steps** total. Nineteen heads use 2,000 steps; two extended
training controls use 4,000. Batch size is 128. Three independent worker processes
run concurrently, with two CPU math threads each. On the 96 GB RTX PRO 6000 this is
a conservative concurrency starting point, not a measured performance guarantee.
These small heads may be CPU/launch-bound; GPU percentage alone is not throughput.
`--parallel-jobs 1`, `2`, or `4` changes concurrency without changing experiment identity.
Keep at least 5 GiB free; this is a launch reserve, not a precise storage forecast.

Parent checksums are audited before training; this can involve substantial disk I/O.
An idle GPU during this audit is expected. Do not pull code updates while a run is active.

```bash
/root/autodl-tmp/virtual_cell/.venv/bin/python \
  /root/autodl-tmp/virtual_cell/scripts/round14_status.py

tail -n 30 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round14_pipeline_*.log | head -n 1)"
```

The same launch command resumes partial work. Completed heads are checksum-verified;
unfinished heads restore optimizer, current state, best calibration state and step.
Sampling is deterministic by optimizer step. Changed scientific settings require a
new run name. `--prepare-only` audits and writes the plan without starting workers;
run again with `--resume` and without that option to train.

For a small server smoke test, use a separate name:

```bash
bash scripts/run_round14_autodl.sh --name round14_smoke \
  --seeds 17 --protocols target --arms local_additive_id local_factor_id \
  --max-steps 100 --parallel-jobs 1 --resume
```

## Fixed inputs and evaluation

- Reuse Round12 expanded data and its exact fit/calibration/outer partitions.
- Reuse audited B2 background encoders, background projection and rank32 response basis.
- No new Qwen training, atlas expansion, teacher training, gate, or posthoc output shrinkage.
- Select checkpoints solely on source calibration coefficient MSE. Calibration targets
  are absent from fitting. Outer labels are read only for descriptive evaluation.
- Report target holdout, context holdout and double holdout separately. These are
  repeatedly inspected development cohorts, not new independent test datasets.
- Jurkat is not scored. A previously opened cohort must not be relabeled untouched.

## The 21 heads

| Heads | Count | Question |
|---|---:|---|
| `local_additive`, `local_factor`, `local_additive_id`, `local_factor_id` | 4 | Separate ID and interaction effects in a 2×2 design |
| `background_only`, `target_only`, `id_background` | 3 | Does the model need biological target features and background? |
| `shuffled_local_1/2/3_factor_id` | 3 | Does correct gene/profile alignment help? |
| `random_factor_id` | 1 | Is a random target representation sufficient? |
| `clipped_additive_id`, `clipped_factor_id` | 2 | Does controlling extreme background norms help? |
| `unit_additive_id`, `unit_factor_id` | 2 | Is background direction more transferable than magnitude? |
| `linear_additive_id`, `linear_factor_id` | 2 | Does removing bounded nonlinear activations hurt extrapolation? |
| `factor_id_width16`, `factor_id_width64` | 2 | Is width32 under/over-capacity? |
| `long_additive_id`, `long_factor_id` | 2 | Does twice the optimization budget help? |

All main heads use the same sample sequence, AdamW settings, checkpoint schedule
and source calibration selection. The long controls have explicitly different
budgets. The additive network is the same two-branch architecture with its cross
output disabled; Ridge is a historical reference, not the additive architecture control.
Widths change capacity as well as factor rank and are diagnostics rather than
strict parameter-matched comparisons.

The model is `Wt hp + Wb hb + bias`, optionally adding `Wx (hp * hb)`.
By default, `hp = tanh(Ut target + ID)` and `hb = tanh(Ub background + bias)`.
All output layers start at zero; gradient clipping is shared across arms.
The linear controls remove both tanh activations, so they test bounded nonlinearity
as a package, not boundedness independently of every other property.
The response decoder is fixed at 32 dimensions. ID embeddings exist only for fit
targets; all unseen targets have zero ID. Their gains cannot be attributed to
memorizing unseen targets. `target_only` still has a learned constant intercept.

Local STRING is the existing top-16 functional-neighbor representation on the
historical measured panel, projected to 32 dimensions using fit targets only,
plus an annotation-availability flag. Shuffles permute raw profiles within
coverage strata, before fitting PCA; they preserve each gene's coverage status.
The random control uses fixed gene-keyed Gaussian features, fit-only PCA and the
same real coverage flag. These are mapping/representation controls, not isolated
tests of graph topology or signed causal regulation.

Background clipping uses the **95th percentile of fit-row norms**, and radial
normalization uses the **median fit-row norm** as a common radius. Both are fixed
without outer covariates or labels. Zero vectors remain zero. All transformations
apply consistently during fitting, calibration and inference. They do not learn
an H1-specific adjustment from observed H1 results.

## Reports and interpretation

Download only:

```text
/root/autodl-tmp/vcell-work/runs/round14_01/round14_review_light.tar.gz
```

The archive excludes numerical prediction arrays and weights, which stay on the server.

- `comparison_mean.csv`, `comparison.csv`: macro MSE and retrieval, with optimizer seeds.
- `paired_comparisons.csv`: prespecified MSE contrasts and largest-target removal.
- `paired_specificity.csv`: matching contrasts for own-target minus other-target correlation.
- `per_target.csv`, `retrieval.csv`: target/context detail, tie-aware retrieval credits.
- `response_diagnostics.csv`: predicted/true response RMS, strong-effect MSE and sign accuracy.
  Strong effects use the 90th percentile of absolute fit responses, not an outer-tuned threshold.
  Masks based on outer truth are descriptive metrics only. Undefined metrics remain missing.
- `sensitivity.csv`: zero biological target features, zero background, zero ID, and
  shuffled target features at inference for the four main heads. These counterfactuals
  report changes in predictions, not candidate model performance; some inputs are
  outside the training distribution. Sensitivity alone does not prove biological correctness.
- `geometry.csv`: background norms before/after each head's transformation by partition/context.
- `coverage_scores.csv`: actual local-annotation coverage subgroups for new heads;
  historical baselines are omitted from this table. Main scores retain missing annotations.
- `oracle_sweep.csv`, `background_shift.csv`: reused Round13 diagnostics, not new oracle selection.
- `splits/`, manifests, checksums, histories and log tails: reproducibility evidence.

MSE is averaged within target, then equally across contexts, then across seeds.
Bootstrap intervals use 2,000 target resamples after averaging optimizer seeds,
jointly across contexts. They are exploratory and uncorrected for multiple comparisons.
Three optimizer seeds on one split are not three independent biological replications.
Read positive intervals cautiously with this many arms; do not select a final model
on outer results and report the same results as independent confirmation.

Keep the raw model table separate from any future calibrated system. A useful
advance must beat relevant raw baselines, show target specificity, and survive
matched biological nulls and background-only controls. Lower MSE from conservative
amplitude alone is valuable for stability but is not proof of perturbation recognition.
