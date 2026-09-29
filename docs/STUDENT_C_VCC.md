# Student C: supervised challenge-data training

Train a fresh Qwen3-0.6B-Base numerical student using real perturbation labels,
then compare it with the completed four-teacher students A and B. Student C
does not load teacher predictions or either student's adapter. The default
training source is **only the official 2025 challenge training split**.

## Which challenge data are usable?

Checked on September 29, 2026:

| Release | Available data | Use here |
| --- | --- | --- |
| 2025 | Public H1 hESC training, validation and test splits | Training split only: 221,273 cells, 18,080 genes, 150 perturbation targets plus controls |
| 2026 | Control cells and target identities; perturbation outcomes withheld; no challenge-specific labeled training set | Not used as supervised labels |

Sources: [Arc's official dataset repository](https://github.com/ArcInstitute/arc-virtual-cell-atlas/tree/main/virtual-cell-challenge),
[2026 announcement](https://arcinstitute.org/news/virtual-cell-challenge-2026), and
[official challenge CLI documentation](https://vcc-cli-wiki.virtualcellchallenge.org/).
The 2025 validation/test files are deliberately excluded from training even
though they are now public. The 2026 challenge requires full-panel, per-cell
count predictions; this project's mean log-expression delta output is **not an
official challenge submission**. No leaderboard score is claimed.

## Start on the existing AutoDL instance

Prerequisites: completed `runs/qwen_421_four_teacher_01`, original
`prepared/real_min10_01`, the existing `.venv`, cached Qwen base model, CUDA with
bfloat16 support, and at least 3 GiB free on the data disk. Keep all original
checkpoints and prediction files. Updating the checkout does not retrain A/B.

```bash
cd /root/autodl-tmp/virtual_cell
git pull --ff-only && screen -dmS vcell_student_c \
  bash scripts/run_student_c_autodl.sh --name vcc2025_c_01
```

The public Google Cloud Storage source is approximately 15.5 GB. The loader
uses HTTPS byte ranges and a bounded memory cache; it does **not** save the full
source file. Expect roughly one source-file download's worth of network
traffic, potentially more after retries. Aggregation checkpoints are saved
every 8,192 cells; an interruption loses at most the work after the last saved
checkpoint. Throughput determines preparation time. Server access to this
Google endpoint must work; local endpoint verification does not guarantee
AutoDL connectivity. Do not delete existing data to make space automatically.

Watch the latest log:

```bash
log=$(ls -t /root/autodl-tmp/vcell-work/logs/student_c_pipeline_*.log 2>/dev/null | head -n 1)
if [ -n "$log" ]; then tail -n 40 -f "$log"; else screen -ls; fi
```

During aggregation, inspect this compact progress file:

```bash
cat /root/autodl-tmp/vcell-work/prepared/vcc2025_c_01_aggregation/progress.json
```

Expected stages: `STUDENT C PREFLIGHT`, `VCC AGGREGATE`, `STUDENT C DATA READY`,
`supervised epoch=...`, `THREE STUDENT COMPARISON COMPLETE`, `STUDENT C COMPLETE`.
If a run fails, inspect the log and restart the **same unchanged experiment**:

```bash
screen -dmS vcell_student_c bash scripts/run_student_c_autodl.sh \
  --name vcc2025_c_01 --resume
```

Do not start duplicate sessions while an earlier process is still active.
Code, data, source or configuration changes require a new run name.
`--prepare-only` stops after data preparation; subsequently use `--resume` to
train. `--h5ad /path/to/adata_Training.h5ad` uses an already available source
file instead of the remote reader. Its SHA256 is recorded; matching the official
file's byte identity is not asserted for a manually supplied file.

To train on both existing training data and the challenge training split, use
a **separate** experiment:

```bash
bash scripts/run_student_c_autodl.sh --name vcc2025_c_combined_01 \
  --training-source combined
```

## Alignment and interpretation

- Normalize each challenge cell using its full measured library:
  `log1p(10000 * count / library_size)`, then average by batch and target.
  Subtract matched non-targeting control means. Require at least 10 perturbed
  and 30 control cells per group; skip zero-library cells.
- Use the intersection of challenge Ensembl IDs and the existing frozen gene
  panel. Missing genes are never filled with invented labels. Save exact gene
  coverage and skipped groups in the preparation audit.
- Challenge-only C trains on H1. Existing HepG2 validation and Jurkat test rows
  are retained only for perturbations represented in C's training vocabulary.
  No validation/test labels are used for gradient updates. The test split is
  prepared for future use but **never scored** by this pipeline.
- Reuse A/B's saved full validation predictions and project them onto C's
  common rows and genes. Verify row order, dataset fingerprints, original
  reported MSE and A/B checkpoint hashes. Recompute all comparison scores with
  identical labels and context/perturbation/batch weights.
- C inherits the parent's training hyperparameters, defaults to seed 17, uses
  all eligible training rows and zero distillation loss. A/B remain frozen.
  A/B's checkpoints were selected on their original full validation set; C's
  early stopping uses the common subset. This selection asymmetry is disclosed.
- Differences reflect **training data, output/input gene coverage and training
  objective together**, not a controlled estimate of distillation benefit.
  Source measurement panels/platforms differ even with the same normalization
  formula. Parent teacher pretraining overlap remains unverified. One C seed
  does not establish statistical significance. No new overall winner replaces
  the previous selected checkpoint.

The range reader pins the GCS object generation and validates every returned
byte range. It does not claim to recompute the whole object's CRC32C. Aggregated
artifacts are independently hashed and verified on reuse.

## Files to review

Under `/root/autodl-tmp/vcell-work/runs/vcc2025_c_01`:

- `COMPLETE.json`: successful C training and comparison marker.
- `comparison/comparison_common.csv`: A, B, C and reference/C mean-transfer and
  no-change baselines, all evaluated on the same subset; lower MSE is better.
- `comparison/comparison_audit.json`: actual coverage, identities and limitations.
- `comparison/by_perturbation.csv`: per-target comparison.
- `student_c/supervised/history.csv`: training and validation history.
- `student_c/supervised/best.pt`: C's adapter/head checkpoint, requiring the
  pinned base model and matching prepared data/configuration to reload.
- `plan.json` and `student_c_config.json`: reproducible experiment inputs.

Also retain `prepared/vcc2025_c_01/data_audit.json`. The old full-panel A/B
scores and the new common-subset scores are different evaluations: use the
new comparison table to compare the three students.
