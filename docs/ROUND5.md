# Round 5: UCE, Ridge, reliability, functional priors and weighted loss

This round adds UCE-33 as a fifth frozen cell encoder, then trains seven Qwen
ablation arms on the **same original prepared dataset**. It reuses the original
K562/RPE1 training split and full HepG2 validation panel. It does not restrict
evaluation to the two perturbations shared with the 2025 challenge-only student.
Jurkat test labels remain unscored. Prior A/B/C outputs are preserved.

## Start

Requires the completed 421 run and its frozen features, the original raw h5ad
files, cached Qwen base model and existing CUDA `.venv`. No replacement PyTorch,
legacy UCE environment, Colab or online model API is needed.

```bash
cd /root/autodl-tmp/virtual_cell
git pull --ff-only && screen -dmS vcell_round5 \
  bash scripts/run_round5_autodl.sh --name round5_01
```

Default paths under `/root/autodl-tmp/vcell-work`:

- Data: `prepared/real_min10_01`.
- Previous run: `runs/qwen_421_four_teacher_01`.
- Existing frozen features: `teachers/four_teacher_01/{family}/features.npz`.
- New models: `models/UCE-33`, `models/UCE-aux`.
- New outputs: `runs/round5_01`.

Override inputs with `--data`, `--parent-run`, `--feature-root`, or `--work-dir`.
Data must match the completed parent run and all four frozen feature files.

UCE requires approximately **6.1 GB of new downloads**, mostly the 33-layer
checkpoint including its protein token table. The launcher checks missing-file
size plus a **3 GiB output reserve** for the default one-seed run. Additional
seeds reserve an additional 2 GiB each. It never deletes old data to make room.
Only the human auxiliary embeddings are downloaded; the multi-species archive
and duplicate standalone token table are unnecessary.

The default seed is 17. This is an initial ablation, not multi-seed statistical
evidence. Use a new name with `--seeds 17 29 43` for a larger experiment after
checking space/time. Seven students run sequentially, each with up to 20 epochs
and patience 5. UCE extraction, teacher cross-fitting and seven student fits
make this a substantially longer run than one student C fit.

```bash
log=$(ls -t /root/autodl-tmp/vcell-work/logs/round5_pipeline_*.log 2>/dev/null | head -n 1)
if [ -n "$log" ]; then tail -n 40 -f "$log"; else screen -ls; fi
```

Inspect `runs/round5_01/progress.json` for the active stage. Success prints
`ROUND5 COMPLETE` and writes the root `COMPLETE.json`. Resume an unchanged,
interrupted run after confirming no earlier process remains active:

```bash
screen -dmS vcell_round5 bash scripts/run_round5_autodl.sh --name round5_01 --resume
```

The Hub retains partial downloads. UCE saves extracted groups; a partially fit
short response-head stage restarts, while completed heads are reused. Qwen fits
resume at epoch boundaries. `--teacher-only` stops after frozen features,
cross-fitted heads and reliability estimation; resume without this flag to run
the remaining stages. Changes to source, data or configuration require a new name.

## What each comparison measures

| Arm | Teachers | Loss weights | Functional token |
| --- | --- | --- | --- |
| `supervised_uniform` | None | Equal prepared rows | No |
| `supervised_weighted` | None | Context/perturbation/batch balanced | No |
| `supervised_function` | None | Balanced | Yes |
| `kd4_equal` | Original four, equal mixture | Balanced | No |
| `kd5_equal` | Add UCE, equal mixture | Balanced | No |
| `kd5_reliable` | Five, train-only reliability mixture and KD strength | Balanced | No |
| `kd5_reliable_function` | Same reliable five | Balanced | Yes |

All Qwen arms use the same base model, eligible rows, panel, seed and inherited
421 optimization settings. Only the explicitly listed factors differ. The
functional arm adds one learned projection token and thus a small number of
parameters. Historical A/B scores are shown separately; their response heads
and teacher target fitting protocol differ from this round.

### Fifth teacher: real UCE-33 features

The [UCE paper](https://www.nature.com/articles/s41586-026-10689-z) provides a
cell representation, not a native mean-expression perturbation decoder. Here,
the frozen encoder embeds up to eight genuine control cells per dataset/batch;
their embeddings are averaged and passed to a VCell conditional response head.
The encoder uses the 33-layer checkpoint, pretrained protein tokens, log1p-count
gene sampling, chromosome ordering and token normalization from the
[upstream implementation](https://github.com/snap-stanford/UCE/tree/9c416007be15ad6753dc84af4468c1dc10421ab9).
Duplicate symbols are summed in raw-count space. Fewer than 100 mapped genes,
nonfinite outputs, wrong parameter names or shapes cause a hard error.

Pinned HF redistributions enable `hf-mirror.com` downloads. File hashes are
verified before loading; source-original byte equivalence is not established.
See `THIRD_PARTY_NOTICES.md` for attribution and conflicting weight-license
notices. Pretraining overlap for all teachers remains unverified, so these are
exploratory experiments.

The release checks the real remote checkpoint's 434 parameter names/shapes,
auxiliary mapping schema, and upstream tokenization on small fixtures. Full
UCE numerical inference requires the downloaded checkpoint and is checked on
the server before teacher fitting. Software tests do not establish biological
accuracy or successful GPU execution of the real checkpoint.

### Train-only teacher reliability

For each teacher, fit separate fixed-epoch response heads leaving out each
training context in turn. Produce out-of-context (OOF) predictions for training
rows, and fit another head on all training rows to predict validation contexts.
Head epochs are fixed at 20; the external validation labels do not choose head
epochs, mixture weights or KD strength. Frozen pretrained features are reused;
original validation-selected heads are not reused.

Fit nonnegative mixture weights summing to one against the eligible OOF training
rows. Scale the KD loss by `clip(1 - OOF_mixture_MSE / OOF_zero_MSE, 0, 1)`.
If that mixture does not beat zero change on these diagnostic rows, reliable KD
is disabled rather than forcing unreliable targets onto the student. Inspect
`kd_strength`; a zero value means no effective distillation in reliable arms.
The OOF score is used for tuning and is not an independent final evaluation.

Training rows with targets absent from a fold's fitting contexts are excluded
from KD and reliability estimation; they still receive supervised training.
The same eligible mask applies to equal-mixture KD comparisons. Held-out/test
rows cannot enter the reliability mask. All head fitting weights and student
weighted losses use equal contexts, equal perturbations per context, and equal
prepared batch groups per perturbation; weights are normalized globally, not
separately inside every minibatch.

### Gene function and relation features

Bundled, hashed Enrichr GO Biological Process 2023 and Reactome 2022 snapshots
encode perturbation-target membership in annotated functions/pathways. A sparse
membership representation is projected deterministically to 128 dimensions.
A cosine nearest-neighbor graph links targets sharing annotations, and one
diffusion step incorporates neighboring functions. A coverage indicator marks
missing annotations. These relations are **co-annotation similarities, not
experimentally established causal regulatory edges**. No validation/test
expression labels determine annotations, graph edges or projections.

The resulting fixed vector is projected into an extra Qwen input token. Vectors
are embedded in the checkpoint for reload; the artifact checksum and target
order are checked during training. Coverage and missing symbols are reported.
This round does not claim support for arbitrary unseen targets or cross-species
prediction; the prepared target vocabulary and output panel remain fixed.

### Ridge baseline

Weighted Ridge predicts a rank-32 expression-response basis from control-state
components, target identity and control-target interaction features. All means,
SVD bases and coefficients are fitted on training rows only. Select alpha from
1, 10 and 100 using leave-one-training-context-out folds; refit on all training
rows and evaluate on the same full validation panel as Qwen. Zero-change and
mean-transfer baselines are also retained.

## Review outputs

Under `runs/round5_01`:

- `comparison.csv`: all completed arms and historical A/B on the identical panel.
- `by_perturbation.csv`: per-target scores and group counts.
- `reliability.json`: weights, eligible rows, OOF errors and effective KD strength.
- `functional/features.json`: annotation coverage and graph provenance.
- `ridge/audit.json`: train-only alpha selection and validation score.
- `students/*/{supervised,all}/history.csv`: learning curves.
- `comparison_audit.json`, `plan.json`, `COMPLETE.json`: identities and limitations.
- `round5_review.tar.gz`: automatically packaged small review files, without weights.

Lower MSE is better. Review per-target coverage and the zero baseline before
choosing a model. The reused validation set is for exploratory selection;
do not report it as untouched test performance. No automatic winner overwrites
the previous final checkpoint. Lock the method before any future final test.
