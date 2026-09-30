# Round 6: controlled background and teacher-specialization experiments

This round tests possible improvements; it does not assume any route will work.
It reuses all five real frozen encoder feature files from the completed 421 and
round-5 runs. Only conditional response heads and Qwen adapters are trained.
There are no model downloads, synthetic fallbacks, or final-test scores.
Pretraining overlap remains unverified. HepG2 is reused development validation.

## Run on the existing AutoDL server

```bash
cd /root/autodl-tmp/virtual_cell
git pull --ff-only && screen -dmS vcell6 bash scripts/run_round6_autodl.sh \
  --name round6_01 \
  --backgrounds-json configs/round6_backgrounds.example.json \
  --challenge-data /root/autodl-tmp/vcell-work/prepared/vcc2025_c_01
```

The default runs six arms with seeds 17 and 29, then two H1 expansion arms with
the same seeds if `--challenge-data` is supplied. Omit that argument to run only
the original full-panel experiment. No earlier checkpoint is overwritten.
Existing Qwen weights and the existing `.venv` are required. The default disk
reserve is 5.8 GiB without expansion or 7.4 GiB with expansion; already written
same-plan outputs count toward this reserve on resume. This is an estimate,
not a guarantee of maximum disk use. Nothing is deleted automatically.

For a quicker first screen use `--seeds 17` and a distinct run name. To repeat
with more seeds later, start a new run name; never mutate an existing plan.

Before GPU work, an optional CPU coverage audit is available:

```bash
bash scripts/run_round6_autodl.sh --name round6_coverage --audit-only
```

To inspect teacher evidence before starting the students, add
`--diagnostics-only` to the full command. Then rerun the identical command with
`--resume` and without `--diagnostics-only`. Teacher predictions are reused.
Do not change seeds, modules, source code, epoch counts or data when resuming.
An interrupted teacher fit restarts that one small head; completed head fits
are checksum-validated and reused. Student training resumes at epoch boundaries.

## Follow progress and collect results

```bash
cat /root/autodl-tmp/vcell-work/runs/round6_01/progress.json
log=$(ls -t /root/autodl-tmp/vcell-work/logs/round6_pipeline_*.log | head -n 1)
tail -n 40 -f "$log"
```

Stages are coverage/modules, nested teacher fitting, Ridge, each student/seed,
optional H1 expansion, and packaging. Teacher fits can take longer than a Qwen
epoch because calibration involves multiple fits. Do not infer a runtime from
the previous smoke test; use the first completed stages/epoch timings. Success
prints `ROUND6 COMPLETE` and creates `COMPLETE.json`. The initial coverage-only
run goes under `reports/`, not `runs/`.

Download `runs/round6_01/round6_review.tar.gz`. It automatically includes:

- `coverage/sample_counts.csv`, `coverage/coverage.csv` and the metadata audit;
- `modules.json`, `teacher_specialties.csv`, `teacher_diagnostics.json`;
- `comparison.csv`, `seed_summary.csv`, `by_perturbation.csv`, `by_module.csv`;
- `paired_comparisons.csv`, plans, model configurations and training histories;
- the separate `expansion/` comparison when requested.

Weights, large prediction arrays, raw data and intermediate fitting caches are
excluded. Completed diagnostic-only runs also create this review package.

## Controlled student comparisons

| Arm | Student background model | Teacher supervision |
| --- | --- | --- |
| A_supervised | Existing Qwen regression | None |
| B_background | Reference response + Qwen + explicit control/target interaction | None |
| C_equal | Same as B | Equal five-teacher OOF targets |
| D_modules | Same as B | Module-specific weights and abstention |
| E_specialists | Same as B | Specialty-weighted response heads, module routing |
| F_gated_equal | Same as B | Equal teachers, exactly D's gate strengths |

A->B tests the architectural package, including the reference offset; it does
not isolate each component of that package. B->C tests ordinary distillation.
C->D tests routing plus strength. **F->D isolates teacher selection from reducing
KD strength.** D->E tests the specialized teacher procedure. Same Qwen backbone,
LoRA settings, row weighting, seeds, validation panel and optimizer settings are
used. Maximum 20 epochs, patience 5, batch 4, accumulation 4, KD coefficient 0.5.
If the completed 421 run is present on the same dataset, historical students A/B
are added as references after verifying their checkpoint hashes and predictions.
They used a different teacher-fitting protocol and are not matched ablations.
Early stopping may produce different actual step counts. Loss weighting is
uniform in every main student arm, avoiding a repeat of the round-5 confound.

For B-F, the reference response is estimated only from training outcomes, with
equal averaging across training contexts. An additional small branch computes
an interaction between normalized control expression and the target embedding.
No learned cell-line ID is needed at inference. The frozen reference table is
stored inside the checkpoint, and checkpoint reload reconstructs the branch.
This models background dependence, not an identified causal decomposition of
cell type, lineage, study, or species effects. Targets must be seen in training.

## Modules and teacher calibration

By default, at most eight deterministic clusters of **training control profiles**
define output modules. No perturbation outcomes define the modules. Duplicate
control groups are counted once. These are expression modules, not named GO
pathways or a causal regulatory network. You can instead supply `--modules-json`
with `{ "module_name": ["ENSG...", ...] }`, using exact prepared gene IDs.
Overlapping memberships sum to one for each gene; uncovered genes remain in an
explicit unannotated module. Module definitions are fixed across comparisons.

Outer folds leave one training context out. Within each outer fit partition,
inner folds estimate teacher error and module weights. Inner folds leave out
contexts when at least two remain; otherwise they use disjoint groups of
dataset/context/batch. **With two original training contexts, these inner folds
are batch holdouts, not additional independent background-transfer evidence.**
Outer context predictions are the transfer diagnostic. No outer outcome is
used to fit its teacher head, selection weights or specialty loss weights.

Each module compares teachers against both zero change and fold-fitted mean
transfer. Positive gains set a small, shrunk mixture; poor or undersupported
modules abstain. Default minimum: ten distinct supported perturbations.
Shrinkage toward equal weights has strength `n_targets / (n_targets + 20)`.
Only perturbations observed in the corresponding fit fold receive KD. Genuine
supervised loss remains active on every training row and output gene.

For E, base inner-fold evidence gives each teacher a gene loss weight between
1 and 3 before mean normalization. The same head architecture is refitted for
the same number of epochs. Specialists retain a nonzero general objective.
Their inner calibration remains tuning; only outer transfer diagnostics assess
the full procedure. We do not force every teacher to have a specialization.
If gates close, D/E can legitimately match B; this is a result, not a failure.

`teacher_diagnostics.json` includes pairwise error correlations and a
label-aware per-row/module oracle. **The oracle is a diagnostic best-single-
teacher reference, not an executable model or a valid leaderboard score.** It
is not a mathematical bound on convex mixtures, which can cancel teacher errors.
Routed-teacher
diagnostics use mean-transfer fallback where a gate closes; students instead
keep their real-label loss. Their numbers must not be conflated.

## Background metadata and data expansion

`--backgrounds-json` accepts dataset identifiers mapped to species, tissue,
cell_type, cell_line, donor, study, platform, intervention, timepoint, dose and
culture. Unknown fields remain `unknown`; no annotations are invented. This
repository includes `configs/round6_backgrounds.example.json` for the four
original human cell lines; intervention details are intentionally left unknown
until verified from source metadata. Use a different mapping for new datasets.
This
metadata is for coverage/matching audit, not automatically supplied as model
inputs. The model currently uses matched controls and perturbation identity.
The audit reports groups and retained perturbation-cell counts separately.
Controls reused across targets are never summed as independent observations.

The optional H1 test reuses the **already prepared** student-C dataset. Its
reference fingerprint and normalization must match. Only H1 training rows are
added. Both controls are newly trained on the same common measured gene panel,
the same vocabulary, and **all original validation rows**. This avoids the old
53-row/two-target C-only comparison. No absent gene is filled with zero.
Both use B's architecture without KD. The expanded pool is sampled each epoch
at the original training-row budget, giving equal maximum optimizer steps.
Report this common-panel result separately from the original 2,000-gene panel.

New same-species prepared datasets can use `--data` plus `--features-json`
(all five families mapped to newly aligned frozen feature files). Old caches
are rejected for changed data. This runner does not download new datasets or
create a cross-species benchmark. Cross-species supervision still requires an
audited ortholog panel, comparable interventions and matched controls; ordinary
atlases cannot replace perturbation labels. Unknown-target and double-holdout
benchmarks also require a different target-generalization interface.

## Interpreting improvements

Use A/B/C/D/E/F comparisons within the same seed and panel, then examine seed
means, per-target effects and module effects. Bootstrap intervals resample
perturbation clusters across contexts, not individual batch rows. They are
descriptive intervals on reused validation, with no multiple-comparison
correction; two seeds are screening evidence, not a significance claim.
No automatic winner replaces student A. Jurkat outcomes are not scored or used
for fitting/selection anywhere in this experiment. Test metadata are counted
for the coverage audit only. Existing prepared data contain test labels, so
"sealed" means excluded from fitting and evaluation, not absent from disk.
