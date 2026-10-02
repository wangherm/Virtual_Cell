# Round 9: fixed backgrounds and module attribution

Round9 connects the cleaned public background reference to the **existing Qwen
and MLP control-expression input projectors**. It runs the background and module
comparisons, diagnostics, calibration and reporting in one resumable launch.
There are no new downloads or paid APIs. All paths default to the data disk.

## Full run on AutoDL

```bash
cd /root/autodl-tmp/virtual_cell && \
git -c http.version=HTTP/1.1 pull --ff-only && \
screen -dmS vcell9 bash scripts/run_round9_autodl.sh --name round9_01 --resume
```

This starts **24 background pretraining jobs**, followed by **132 downstream
student jobs**, with at most two independent GPU processes on the single GPU.
Each process owns its optimizer and random state. This is a comparison matrix,
not a 132-member prediction ensemble. No winner is selected automatically.

Default inputs, relative to `/root/autodl-tmp/vcell-work`:

| Input | Path |
|---|---|
| Existing expanded perturbation data | `runs/round8_01/prepared/plus_h1` |
| Cleaned background reference | `background/human_background_v1` |
| Existing aligned module/knowledge snapshot | `knowledge/round8/knowledge.npz` |
| Existing pretrained Qwen | `models/Qwen3-0.6B-Base` |
| New results | `runs/round9_01` |

Use `--data`, `--background-dir`, `--knowledge-npz`, or `--model-dir` for custom
locations. Gene order must match exactly; no implicit intersection or replacement
data are created. Input H5AD checksums, membership checksums and donor isolation
are verified before any training starts. The cache needs approximately 2 GiB for
this reference. The full default plan requires at least **84.2 GiB free reserve**
before launch and never deletes previous results.

```bash
.venv/bin/python scripts/round9_status.py
tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round9_pipeline_*.log | head -n 1)"
```

Per-job logs live in `runs/round9_01/logs`. `stage.json` distinguishes background
pretraining from student fitting; `progress.json` counts the jobs in the current
stage. GPU utilization alone is not a progress indicator. Initial verification,
cache creation and ridge diagnostics use CPU/disk and can precede GPU activity.

Rerun the exact full command with `--resume` after interruption. Background and
student workers restore weights, AdamW optimizer state and update count; sampling
and corruption are deterministic by update. Completed artifacts are verified
before reuse. You can reduce `--parallel-students 2` to `1` on resume after an OOM.
Changing a scientific parameter, code, data, panel, knowledge snapshot or model
requires a new run name. Resume does not quietly substitute a new protocol.

The root `COMPLETE.json` is written only when every requested job and report
finishes. `INCOMPLETE.json` records a failed attempt. After successful recovery it
is removed. Download **`runs/round9_01/round9_review_light.tar.gz`** for review; this
package excludes checkpoints, prediction arrays and expression matrices, which
remain on the server. It contains report tables, audits, configurations, training
history, calibration weights, gradient diagnostics and log tails.

## Background conditions and full-data exposure

| Condition | Control-projector initialization |
|---|---|
| B0 | Random initialization; learned only during perturbation-response fitting |
| B1 | Denoising reconstruction on unique matched control pseudobulks from that protocol's fit rows |
| B2 | Same objective and update budget, mixing matched controls and all Tabula training cells 1:1 |

The reviewed reference contributes **74,678 Tabula training cells**. Its **40,692
validation cells** and **120,000 external-candidate cells** are not fitted. Every
selected Tabula training cell enters the deterministic shuffled sampling pool;
there is no additional cell cap here. Defaults allocate 10 full public-cell
passes to B2 (5,840 updates at batch 256 for this reference). B1 uses exactly the
same number of updates and total sample exposures, repeating its smaller pool.
B0 has no background-pretraining budget. `exposure.json` records those differences.

The pretraining task randomly replaces 20% of standardized observed entries with
zero (the fit-control mean) and reconstructs observed expression. Missing genes
are set to the standardized mean and excluded from reconstruction loss using
the exported `var.available` mask. Normalization is fitted only on the respective
perturbation fit pool and shared by all background conditions. The reconstruction
decoder is discarded; the trained **input projector** transfers into the original
student and remains trainable during downstream fitting. Qwen's pretrained text
weights are not used for the background-only autoencoder.

Matched controls are existing **pseudobulks**, while public references are single
cells. B1 versus B2 tests this practical expansion; it does not isolate source,
sample count and aggregation level separately. Public normal tissues are not
relabeled as matched K562/RPE1/H1 controls and do not supply perturbation labels.
HLCA and immune references remain unverified-overlap diagnostics, not independent
test sets or additional training cells.

## Predeclared comparison matrix

For each of MLP and Qwen, every B0/B1/B2 condition has:

1. No module supervision.
2. True-module auxiliary-head supervision.
3. Structure-matched random-module auxiliary-head supervision.

That is 18 conditions. B2 adds four conditions: each backbone with true or random
**final-expression projection loss**. These 22 conditions run on **context and
target holdout**, with paired seeds **17, 29, 43**, giving 132 students.
The optional `--protocols double` uses the old small double-holdout protocol; it
is not included by default and does not expand its limited biological coverage.

The Reactome matrix from Round8 is fixed. A single gene-column permutation (seed
818) preserves module sizes, their overlap/row Gram matrix and singular values.
It changes gene identities. Three optimizer seeds are not three independent
module permutations. Module library redesign and further random permutations
are conditional follow-ups, not silently mixed into this attribution experiment.

All conditions keep the same free module head and numeric response head, so
adding/removing auxiliary supervision does not add parameters. Auxiliary loss
supervises that head. Projection loss instead computes modules from the final
predicted expression: `MSE(delta_prediction, delta_truth) + 0.1 *
MSE((delta_prediction-delta_truth) @ module_matrix.T, 0)` in normalized-delta
units. It therefore constrains final expression even if the auxiliary head is
inconsistent. These module summaries are not causal pathway activation scores.

Semantics, graph inputs and teacher KD are disabled in this factorial. Existing
Round8 results remain available as historical references; they are not claimed
as paired controls for this new initialization/pretraining procedure. All B0
controls are freshly trained. Qwen and MLP retain their prior respective model
families; shared module parameters and seeds are matched within each family.

Every downstream student uses all protocol-eligible expanded fit rows, 2,000
updates, effective batch 16, learning rate 0.0002, and a rank-32 response basis
fitted only on fit responses. Background pretraining is fitted separately for
each protocol, family, seed and B1/B2 condition, then reused across module arms.
No outer early stopping occurs. Calibration fits scalar shrinkage and the same
student/fit-only-mean-transfer/zero convex mixture on calibration rows only.
Jurkat test responses are never fitted, scored or packaged.

## Diagnostics included in this run

- **Background reconstruction and latent variance:** fixed corruption at endpoint,
  grouped by dataset, tissue, donor, cell type and assay. Cell-type coverage means
  presence in the Tabula training pool, not proof that B1 encountered that type.
  These errors are not perturbation-response MSE. The few validation donors and
  unequal tissue proportions must remain visible in interpretation.
- **Module consistency and optimization:** true-module error derived from final
  expression, auxiliary-head error, auxiliary/expression disagreement, and
  supervised versus weighted-module gradient norms/cosine at fixed steps. Gradient
  diagnostics concern the shared control projector, not every LoRA parameter.
  PCGrad and GradNorm are not enabled without evidence of a relevant problem.
- **Response-space bottleneck:** train-only bases at ranks 16/32/64, and the error
  of oracle projection on fit/calibration/outer rows. The oracle uses answers
  for diagnosis and is not a deployable model. Outputs are signed deltas with a
  scalar normalization and no centering; projection is evaluated in raw-delta
  units. Actual student training stays rank 32 for attribution.
- **Cheap representation probes:** identical ridge mapping into rank-32 responses
  from constant, existing text embeddings, STRING measured-neighbor weights,
  and shuffled text/STRING vectors. Alpha is selected only on calibration rows.
  This is an adapted probe, not a native SLIM reproduction or a new graph encoder.
- **Prediction evaluation:** raw, shrunk and calibrated results alongside
  mean-transfer and zero-change; per-context and per-target tables; paired win
  fractions, median changes, target-cluster bootstrap intervals and removal of
  the largest-contributing target as sensitivity analysis. Seeds are averaged
  before resampling whole targets across contexts. These are exploratory
  development comparisons, not multiplicity-adjusted significance claims.

## Small software check (optional, separate run name)

```bash
bash scripts/run_round9_autodl.sh --name round9_smoke_01 --resume \
  --protocols context --seeds 17 --arms mlp_b0_none qwen_b2_projection_true \
  --max-steps 10 --pretrain-steps 10 --save-every 5
```

This still verifies the real reference and computes its diagnostic reports. It
checks infrastructure, not full-training effectiveness. `--prepare-only` verifies
inputs, creates caches/splits, computes probes and writes all job configurations
without starting pretraining or students. Remove that flag and add `--resume` to
continue the identical plan.
