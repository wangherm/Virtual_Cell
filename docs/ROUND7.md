# Round 7: expanded supervised learning and native mechanism reviewers

Round 6 did not test a plain Qwen student with H1. Round 7 fills that gap on
one shared gene panel, measures a training-only response basis, and checks
calibration before considering further distillation. All results are exploratory.

## Launch on the expanded AutoDL disk

The existing `.venv`, prepared datasets and local Qwen snapshot are reused.
The system-disk Round 6 folder is not needed to train Round 7; retain it for
historical comparison. Before starting training, use
`bash scripts/migrate_round6_to_data.sh` to move `/root/vcell-round6-work`
to the data disk, verify contents and preserve old paths via a symlink.
The script refuses migration while training is active. It only removes the
old system-disk copy after both content checks pass.

```bash
cd /root/autodl-tmp/virtual_cell &&
git pull --ff-only &&
screen -dmS vcell7 bash scripts/run_round7_autodl.sh --name round7_01
```

The default data paths are `vcell-work/prepared/real_min10_01` and
`vcell-work/prepared/vcc2025_c_01`. No raw dataset or foundation model download
is required. The traditional tools install additional packages in **separate
data-disk environments**; CellOracle also downloads a human promoter network.
Those steps need network access. Installation failures do not alter the Qwen
environment and are reported as failed jobs, never replacement predictions.

CellOracle uses Python 3.10, CellOracle 0.20.0 and a pinned compatible numeric
stack. Its native dependencies may require a compiler. scTenifoldKnk uses the
scTenifoldpy 0.4.0 implementation. Versions and installed packages are recorded.
Native tool inputs are genuine control cells from K562, RPE1 and HepG2, never
Jurkat. H1 has prepared aggregates but no configured local raw-cell file, so
no H1 native network is claimed.

## Workload and concurrency

The default job set contains **16 Qwen fits and 2 native tool workers**:

| Fits | Data | Architecture | Seeds |
|---|---|---|---|
| 3 | Original | Plain full output | 17, 29, 43 |
| 3 | Original + H1 | Plain full output | 17, 29, 43 |
| 3 | Original | Existing background/reference branch | 17, 29, 43 |
| 3 | Original + H1 | Existing background/reference branch | 17, 29, 43 |
| 3 | Original + H1, predeclared | Rank-32 signed response output | 17, 29, 43 |
| 1 | Training-internal fit/calibration/outer batches | Plain full output | 17 |

The main jobs sample full shuffled passes through their entire allowed data
pools, with no 3,882-row epoch truncation. Every job performs exactly 4,860
optimizer updates of effective batch 16: **77,760 examples drawn**. Repeated
draws are not new biological samples. Each full batch is filled across pass
boundaries. Learning rate is constant per update; evaluation occurs every 243
updates. The primary checkpoint is the last predeclared update, not the best
validation checkpoint. No early stopping is used. Normalization is refitted
using each job's allowed training rows, so adding H1 also changes statistics.

Two independent student **processes** share the single GPU. Each has its own
RNG and optimizer; two CPU tool workers run alongside them. The default micro
batch is 16 with accumulation 1 and gradient checkpointing disabled. This is
a throughput candidate, not a measured speedup guarantee. CPU thread counts
are capped. Peak allocated GPU memory is recorded per student. On OOM, use
`--resume --parallel-students 1`; changing batch size or scientific parameters
requires a fresh run name. Never edit code during a running experiment.

```bash
tail -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round7_pipeline_*.log | head -n 1)"
cat /root/autodl-tmp/vcell-work/runs/round7_01/progress.json
tail -f /root/autodl-tmp/vcell-work/runs/round7_01/logs/plain_plus_h1_seed17.log
nvidia-smi
```

Completed jobs are reused by an identical-plan `--resume`. Interrupted jobs
resume from their last evaluation checkpoint. An interrupted partial interval
is replayed. Full checksum/source guards remain in place; this update does
not bypass Round 6's resume guards.

## Scientific boundaries

- Same 1,834-gene intersection and vocabulary for all main fits; all 325 HepG2
  validation groups remain. Unsupported genes are not filled with zero.
- The existing background arm deliberately retains its reference coefficient
  of one. Its extra interaction output starts at zero; the entire model does
  not initially reduce to the plain architecture. Paired seeds are used and
  shared initial parameter hashes are checked, not merely assumed identical.
- Training is uniform-row normalized-delta MSE. Reporting is context/target
  macro MSE. Both objectives are stated; the loss is not silently replaced.
- The rank-32 uncentered SVD fits context-target means from allowed training
  responses only. It is a statistical signed response space, not a causal
  pathway model. The frozen basis is included in the adapter checkpoint.
- The additional calibration student has disjoint training-internal batches
  for fitting, calibration and outer scoring. This is **batch transfer inside
  known training contexts**, not new-cell-line generalization. It exports
  genuine held-out predictions, row identities, labels and fit membership.
  A global scalar in [0,1] and a regularized convex Qwen/reference/zero mixture
  are fit on calibration rows and scored on separate outer rows. They are
  diagnostics, not automatically transferred to a differently fitted model.
- Native tool outputs are independent reviewers. CellOracle uses its actual
  promoter-prior, GRN fitting and signal propagation. It outputs signed shifts
  from imputed, full-library-normalized controls. KO=0 does not estimate each
  CRISPRi experiment's efficiency. scTenifoldKnk outputs unsigned regulation
  distances; its FC is not expression fold change. No unsigned score is used
  as a signed expression teacher or mixed into the regression loss.
- Native networks use at most 512 sampled controls per context. CellOracle
  uses measured panel genes, perturbation genes and up to 500 other variable
  uniquely mapped genes. scTenifold uses a 512-gene network (plus required
  selected targets if needed). Up to 20 targets per context are evaluated per
  tool, selected without effect labels; CellOracle first requires a TF prior.
  This is a bounded coverage/quality pilot, not a transcriptome-wide benchmark.
- Native comparison reports both tool and student scores on the **same
  supported subset**, separately from full-panel primary scores. CellOracle
  amplitude shrinkage uses original training labels only before HepG2 review.
  Missing tools, targets and directions are reported explicitly.
- On jointly supported training-internal rows and measured genes, a separate
  convex Qwen/CellOracle/reference/zero ensemble is calibrated and then scored
  on the outer batches. Native networks use pooled control cells, including
  controls from these batches; held-out perturbation responses are excluded
  from fitting. This is a transductive control-access diagnostic, not a claim
  of transfer to completely unseen batches. Its weights do not depend on
  outer, HepG2 or Jurkat perturbation labels. It does not automatically launch
  another distillation fit.
- The prior five adapted teachers are not retrained. Their preprocessing is
  not newly certified by a flag. No final model is automatically promoted;
  Jurkat stays unscored until the finite candidate set is frozen.

## Results

`COMPLETE.json` is written only when every requested worker exits successfully.
`INCOMPLETE.json` lists failed jobs and the review package still includes
successful results. Tool coverage limitations remain explicit even on success.

- `implementation_audit.json`, `native_inputs/audit.json`: targeted checks,
  control identity, mapping exclusions and reconstruction of one real aggregate
  per non-test source (delta sign and full-library normalization).
- `comparison.csv`, `by_perturbation.csv`, `paired_comparisons.csv`: fixed-update
  primary comparison and descriptive target-cluster intervals.
- `students/*/supervised/history.csv`: actual steps, drawn examples, exposure
  by background, loss, LR and validation curves.
- `students/*/supervised/endpoint.pt`: deployable fixed-update adapter plus
  normalization/basis. `best.pt` is supplementary; `last.pt` supports resume.
- `traditional_review.csv`, `celloracle_calibration.json`: native reviewer
  coverage, matched-panel metrics and CellOracle shrinkage diagnostic.
- `calibration/`: training-internal predictions and independent outer scores;
  `native_ensemble.json` records CellOracle mixture weights, coverage and scores.
- `round7_review.tar.gz`: compact CSV/JSON review, excluding weights and arrays.

The review archive is not a migration backup. Keep raw data, model snapshots,
prepared data, native priors, environments, checkpoints and prediction arrays.

Sources: [CellOracle](https://github.com/morris-lab/CellOracle),
[scTenifoldKnk](https://github.com/cailab-tamu/scTenifoldKnk),
[scTenifoldpy](https://github.com/qwerty239qwe/scTenifoldpy).
