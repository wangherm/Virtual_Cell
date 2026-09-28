# Four teachers, two independent students, one selected model

This is an **exploratory 4 → 2 → 1 experiment**. Pretraining row-level exclusion
has not been verified. The launcher explicitly enables `allow_unverified_teachers`;
all four sidecars report unknown overlap. Strict loading elsewhere still rejects
these caches. Internal validation supports model selection, not a claim of clean
generalization to unseen data.

## Models and training

| Teacher | Locked pretrained backbone | Adaptation |
| --- | --- | --- |
| scGPT | `wanglab/scGPT-human` | Frozen encoder + VCell response head |
| State | `arcinstitute/SE-100M` | Frozen **State Embedding** encoder + VCell response head |
| scFoundation | `genbio-ai/scFoundation`, cell checkpoint | Frozen encoder + VCell response head |
| Geneformer | `ctheodoris/Geneformer`, **V1-10M** | Frozen BERT encoder + VCell response head |

**State SE is not State Transition.** This does not reproduce the published ST
perturbation benchmark. Geneformer uses final-layer mean gene embeddings, not
its official in-silico deletion method. The scFoundation checkpoint is a pinned
GenBio redistribution; original-author byte identity has not been established.

Each encoder receives eight genuine control cells per dataset/batch, sampled
without replacement with seed 0. Features are averaged within the batch. Each
head conditions on those features and the requested perturbation, and predicts
the same prepared numerical expression deltas. Heads train on train labels only
for at most 30 epochs, with validation patience 5.

Teacher predictions have fixed weights of 0.25. Two independent Qwen3-0.6B-Base
students use seeds 17 and 29 and the same full prepared train/validation rows.
They run concurrently by default, with LoRA, continuous control tokens and a
signed numerical regression head. Base Qwen weights remain frozen. The existing
loss is supervised MSE plus `0.5 * teacher-MSE`, with train-only response scaling.
Students train for at most 20 epochs, with validation patience 5. They do not
teach each other. The lower validation MSE wins; exact ties choose student A.
Test labels never enter fitting or selection.

For `real_min10_01`, **full data means 3,882 training groups**, 325 validation
groups, the existing 2,000-gene panel and 196 perturbations. It does not mean
training on every raw cell or adding validation/test labels to training.
Another compatible prepared dataset can be supplied with `--data`.

## Start on AutoDL

Use the same `.venv` that passed Qwen/scGPT smoke tests. No legacy teacher
package, GitHub source clone, FlashAttention build or replacement CUDA wheel is
required. In a JupyterLab or VS Code remote terminal:

```bash
cd /root/autodl-tmp/virtual_cell
screen -S vcell421
bash scripts/run_421_autodl.sh --name four_teacher_01
```

Press **Ctrl+A, then D** to detach; `screen -r vcell421` reconnects.
The launcher uses `https://hf-mirror.com`, validates pinned revisions and SHA-256
hashes, and reuses verified Qwen/scGPT files. Transient download retries retain
Hub partial files. New State, scFoundation and Geneformer downloads total about
**2.8 GB** decimal. Preflight also requires 5 GiB remaining after missing model
downloads for training and temporary files. Previous data/runs are preserved.

To continue after an interruption, first ensure the old process has stopped:

```bash
cd /root/autodl-tmp/virtual_cell
bash scripts/run_421_autodl.sh --name four_teacher_01 --resume
```

Use identical settings and source code for resume. Completed stages are verified
and reused. Qwen resumes at epoch checkpoints. An interrupted short head fit is
preserved in `head_interrupted_*` and restarted from epoch 1. An incomplete
feature-file write requires a new run name. Never launch two pipelines with the
same name at once.

Optional settings, fixed from the start: `--parallel-students 1` for sequential
students; `--epochs 10` for a different maximum. `--teacher-only` prepares all
four teachers and the student config without launching students.

## Progress and outputs

In another terminal:

```bash
log=$(ls -t /root/autodl-tmp/vcell-work/logs/421_pipeline_*.log 2>/dev/null | head -n 1)
if [ -n "$log" ]; then tail -n 40 -f "$log"; else echo 'No 421 log yet'; fi
```

Stages print `421 TEACHER 1/4 START`, batch extraction, head epochs, teacher
completion, then both student starts. Once students have started:

```bash
run=/root/autodl-tmp/vcell-work/runs/qwen_421_four_teacher_01
tail -n 20 -f "$run/logs/student_a.log" "$run/logs/student_b.log"
```

`all epoch=... step=...` shows progress; completed epochs report seconds and
validation MSE. Use those observed times for an ETA. Ctrl+C exits `tail` without
stopping training. Success requires **`421 COMPLETE`** and `COMPLETE.json`.

```text
/root/autodl-tmp/vcell-work/runs/qwen_421_four_teacher_01/
  comparison.csv
  run_manifest.json
  progress.json
  student_a/all/{best.pt,last.pt,history.csv,validation_predictions.npz}
  student_b/all/{best.pt,last.pt,history.csv,validation_predictions.npz}
  final/{student.pt,selection.json,backbone.json,README.md,licenses/}
  COMPLETE.json
```

The comparison includes all teachers, their average, both students, no-change
and mean-transfer baselines. The report states whether the winner beats mean
transfer. A winner alone does not establish usefulness or statistical significance.
`final/student.pt` contains one adapter plus numerical layers, not an ensemble
or weight average. Keep the shared Qwen backbone: its files are required to load
the adapter, and their hashes are recorded.

## Transformations and verification

- Duplicate symbols are summed **only for encoder input** before normalization.
  Raw files and prepared Ensembl-indexed targets remain unchanged. Batch audits
  record source columns/IDs, duplicate groups and vocabulary overlap.
- scGPT uses full-library normalization, binning and a seeded 1,200-gene cap.
- scFoundation maps to 19,264 official symbols, normalizes the mapped library to
  10,000 and log1p, and appends two resolution tokens. Pooling concatenates the
  last two tokens and max/mean gene representations.
- State uses protein embeddings, log1p raw-count ranking, 2,048 tokens, relative
  log-count weights, CLS and dataset tokens, with seeded tie-breaking. Attention
  dropout is disabled for inference. The pinned upstream CLS count convention
  is retained.
- Geneformer uses V1 dictionaries, author Ensembl aliases, full-library
  normalization and nonzero median-scaled ranks, capped at 2,048 tokens without
  V2 special tokens.

All encoding-path parameters load strictly from hash-verified checkpoints;
missing parameters cannot silently remain random. Unused pretrained task
decoders are excluded. Native State Transition and the older source-checkout
scFoundation adapter remain separate options.

Before release, real locked State, Geneformer and scFoundation checkpoints were
loaded on CPU and produced finite 1,024-, 256- and 3,072-dimensional embeddings
on synthetic count inputs. This establishes compatibility, not biological
quality. scGPT already passed real-checkpoint and AutoDL smoke checks.
Automated tests cover four-cache averaging, strict/exploratory separation, two
child training processes using a **tiny random Qwen test fixture**, complete
train/validation rows, final export/resume and mismatched-row rejection.
Full four-teacher CUDA training on the user's server remains to run.

## External evaluation and model terms

Keep unpublished evaluation labels outside training and selection. Cross-species
evaluation needs an explicit ortholog mapping and compatible gene/perturbation
vocabularies. Current human-panel artifacts do not directly support arbitrary
species or new perturbation IDs.

State's non-commercial terms explicitly include distilled derivatives. See
[third-party notices](../THIRD_PARTY_NOTICES.md); terms accompany the selected
model and are not replaced by the repository's MIT license.
