# Experiment design

## Prediction task

Each sample represents a dataset/batch/perturbation group. Inputs are matched
control mean expression and a training-known perturbation ID. The output is the
mean expression change on the selected gene panel. The experiment tests transfer
to held-out cellular contexts, not generalization to unseen perturbations.

The default split uses K562/RPE1 for training, HepG2 for validation and Jurkat for
test. Actual context assignments are explicit in the data manifest. Training
labels update parameters, validation labels select checkpoints/ensemble weights,
and test labels are used only when final test evaluation is explicitly requested.

## Models and training

The four reference architectures are residual, module, MLP and bilinear models.
Each teacher is trained independently from scratch, then reused for all student
experiments within that seed. Four students are optimized together in each mode
on a single device. They do not require four GPUs or separate workers.

| Mode | Supervised | Distillation | Mutual | Contrastive |
| --- | --- | --- | --- | --- |
| `supervised` | Yes | No | No | No |
| `kd` | Yes | Yes | No | No |
| `mutual` | Yes | Yes | Yes | No |
| `contrastive` | Yes | Yes | No | Yes |
| `mutual_contrastive` | Yes | Yes | Yes | Yes |

All modes use true supervised targets. Distillation matches the teacher-ensemble
delta; mutual learning matches other students' predictions with stop-gradient
targets. The contrastive objective pairs two students' representations of the
same input and masks other rows sharing the perturbation ID from the negatives.
Different perturbations are only approximate negatives, not necessarily distinct
biological mechanisms.

The total loss is supervised MSE + weighted distillation MSE + ramped mutual and
contrastive losses. The latter two are disabled during warmup and ramp up over
five epochs. Four-student pair losses are averaged over the six pairs.

For a mode, validation MSE of the equal-weight student ensemble selects one shared
checkpoint epoch; it does not independently select a different epoch for each
student. Validation then fits the `valmix` ensemble weights. A zero ensemble
weight does not imply that a student was removed or retrained during learning.

## Budgets and comparisons

Start with `smoke.yaml` (synthetic, 5 epochs, one seed), then `pilot.yaml` (real
data, 20 epochs, one seed), and finally `real.yaml` (100/80 epochs, three seeds).
Pilot and full presets share the model/loss settings. Keep model counts, splits
and budgets fixed while comparing objectives.

Predeclare the primary comparison: **`contrastive/mean` versus `kd/mean`**.
Secondary comparisons include `kd/mean` vs `supervised/mean`, `mutual/mean` vs
`kd/mean`, and `mutual_contrastive/mean` vs `mutual/mean`. Compare individual
students and the `no_change` / `mean_transfer` baselines as well. Four teachers or
students are not guaranteed to improve performance over fewer models.

## Read outputs

- `run_manifest.json`: effective config, source/data/cache fingerprints and runtime versions.
- `seed_*/teachers/NAME/`: independently trained reference teacher fits.
- `seed_*/MODE/history.csv`: per-epoch losses, normalized validation MSE and timing.
- `seed_*/MODE/last.pt`: resumable optimizer/model state and history.
- `seed_*/MODE/best.pt`: inference bundle, normalization, vocabulary and ensemble weights.
- `evaluation_validation/summary.csv`: metrics aggregated across seeds.
- `evaluation_validation/paired_comparisons.csv`: paired candidate-minus-reference comparisons.
- `evaluation_validation/diversity.csv`: model diversity summaries.
- `evaluation_validation/report.html`: report for inspection in a browser.

`history.csv` losses use the training-derived RMS scale; reported `mse_delta_mean`
uses the original delta scale. Do not subtract scores across these two scales.
Negative `delta_mse` favors the candidate in the paired comparisons. Interpret
uncertainty with respect to the available contexts and perturbations, not as
proof of universal cross-context performance.

The runner computes prediction arrays for all prepared rows, but default reports
only evaluate validation. Freeze model/hyperparameter choices before running:

```bash
python -m vcell evaluate --run runs/real_01 --include-test
```

This writes `evaluation_with_test/`. Do not use test rankings to continue tuning.
Synthetic examples in `docs/qa/` demonstrate software outputs and are not new
real-data results. Full real-data and CUDA execution require validation on the
target server.

## Optional extensions

Import genuinely external teacher predictions using the [teacher interface](EXTERNAL_TEACHERS.md),
recording provenance and held-out context exclusions. The default reference models
are not pretrained State or GEARS models. The [API controller](advanced/AGENT.md)
is a separate optional extension and is unnecessary for any standard workflow.
