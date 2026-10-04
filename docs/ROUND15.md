# Round15: architecture and training screen, with public-data inspection

This command executes the **first, independently runnable stage** of the expanded
Round15 plan: the available P0/P1/P2 experiments and parallel D1 acquisition and
metadata inspection. It does **not** execute the complete 675-fit upper-bound plan.
P3 confirmation, P4 data expansion, P5 native models, P6 distillation, few-shot
adaptation and the additional closed-form benchmark suite need later implementation
and prerequisite evidence. Their status is recorded in `stage_readiness.json`.
The user-supplied [expanded proposal](round15_plan/Round15_Expanded_Plan_zh.md)
and its registries are preserved verbatim with provenance hashes. They describe
the intended broader program, not a claim that every branch is implemented.

## What Round14 established

The supplied light archive contained 189 completed heads across nine workers.
All 1,667 included non-log files listed in the server completion manifest matched
their hashes. The light archive omitted model/prediction binaries by design;
its nine truncated logs could not match the full-log hashes. Round15 gives the
actual packaged bytes a separate `ARCHIVE_MANIFEST.json`.

| Method | New context, seen targets | Seen contexts, new targets | H1 double holdout |
|---|---:|---:|---:|
| Zero change | 0.02839229 | 0.01858697 | **0.00495881** |
| Round12 ID Ridge | **0.02594700** | 0.01788747 | 0.00760438 |
| Round12 STRING Ridge | 0.02799216 | 0.01772756 | 0.00966127 |
| Local factor, no ID | 0.02713059 | **0.01762856** | 0.00613637 |
| Local factor + ID | 0.02622777 | 0.01769770 | 0.00580817 |

The 0.56% mean improvement over STRING Ridge for new targets is a candidate signal,
not an independently confirmed win; the exploratory paired interval crosses zero.
H1 double holdout remains unsolved. Three optimizer seeds use the same biological
split. Context evaluation has 316 aggregate rows / 27 targets; target evaluation
has 1,635 rows / 66 targets / 76 context-target pairs; double evaluation has
5,740 rows / 140 targets. These are aggregate records, not single-cell counts.

The old `id_background` control had a dead branch: both ID embeddings and its
output projection were zero with zero target inputs. `id_background_fixed` uses
small nonzero initialization for seen IDs with an independent random generator.
Unknown ID row zero stays zero; historical heads and files remain unchanged.

## Run on AutoDL

Requirements: the full local Round14 run, its Round13/12 ancestors and prepared
data/knowledge/background assets at their recorded paths. A review archive alone
cannot supply these dependencies. The training environment must already be set up.

```bash
cd /root/autodl-tmp/virtual_cell && \
git -c http.version=HTTP/1.1 pull --ff-only && \
screen -dmS vcell15 bash scripts/run_round15_autodl.sh --name round15_01 --resume
```

`&&` prevents starting stale code if `git pull` fails. Do not pull during a run.
The default queue uses **three concurrent small-head workers**, two CPU math
threads each, and a separate CPU data-inspection worker. Three is a starting
setting, **not a measured throughput optimum**. Use `--parallel-jobs 1`, `2`, or
`4` as appropriate. Scientific batch size, steps and seeds do not change with
concurrency. These experiments train small numerical heads over frozen B2
background features; they do not fine-tune the Qwen backbone or native teachers.

```bash
# Inspect current stage, job outcomes and head counts.
.venv/bin/python scripts/round15_status.py \
  /root/autodl-tmp/vcell-work/runs/round15_01

# Follow orchestration; individual worker logs are under the run's logs/ directory.
tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round15_pipeline_*.log | head -n 1)"

# Verify inputs and create the frozen queue without training or downloading.
bash scripts/run_round15_autodl.sh --name round15_01 --resume --prepare-only
```

Use the same scientific options and `--resume` after interruption. Completed
heads are hash-verified; partially trained heads restore optimizer, step, best
checkpoint and deterministic sampling. Changing source, inputs, seeds, steps,
batch size, or other scientific configuration requires a new run name. Concurrency
can change without changing scientific identity. `--skip-data` disables D1 for
an explicitly separate run; it is part of the frozen configuration.

## Implemented experiments and honest accounting

| Stage | Slots on 3 protocols × 3 seeds | Implemented behavior |
|---|---:|---|
| P0 | 9, included in P1 | Repair and rerun ID-only control first in each worker |
| P1 | 216 | 24 arm definitions; 23 available, control moments blocked |
| P2 | 108 | 12 definitions on one predeclared parent; 9 available, 3 blocked |
| D1 | 1 processed object initially | Pinned IFNG Seurat RDS download + metadata/count-layer QC |

With the exact supplied Round14 source/data/training identity, this means **225
new fits, 63 reused results, 36 blocked slots**, rather than 324 new fits. P0 is
not charged twice. If historical equivalence fails, available heads are retrained
and actual counts change. `execution_status.csv` is the authoritative accounting.
Reused results point to their checked parent prediction path and checksum.

P1 includes background-only, repaired ID-only, additive/factor heads with and
without IDs, three matched no-ID STRING shuffles, Gaussian and coverage-only
controls, matched target-only controls, Ridge residuals, same-target references,
neighbor-response retrieval, small FiLM, shared-response residuals, removal of
the context-only output, amplitude/direction factorization, control PCA, and
response ranks 16/64. The default response rank and hidden width are 32.

P2 uses the **same predeclared `local_factor_noid` parent** for every variant:
MSE reference, Huber, MSE + direction cosine, MSE + within-condition response
contrast, robust gene-scale weighting, learning rates 1e-4/1e-3, and weight decays
1e-3/1e-1. Defaults are 2,000 steps, batch 128, learning rate 5e-4, weight decay
0.01; checkpoint selection uses source calibration MSE. Gene-scale weighting
uses fit-only standard deviations with bounded weights; it is not a measurement
precision estimate. Contrastive negatives are restricted to the available
context/study/batch/stimulus/dose/time/modality fields; highly similar responses
are neutral. Eligible row counts are recorded, including zero eligibility.

Control moments, precision weighting, replicate consistency and biological
replicate balancing remain blocked: the current aggregate data do not establish
the required control-cell variance, cell-resampling lineage or biological
replicate identity. Cell count is not silently used as precision. A blocked head
gets `BLOCKED.json`, never a successful `COMPLETE.json` or fabricated score.

## Label boundaries and evaluation

- All historical split membership and lineage are checked before execution.
- Ridge references use target-level three-fold OOF predictions for fitting rows;
  target PCA and design scaling are refitted inside each fold. The full-gene
  Ridge reference uses fixed alpha 0.1 and no label-derived response basis.
- Same-target references exclude the query context. Neighbor/shared-response
  references exclude the entire query target. Donors are fitting rows only;
  audits identify every donor unit. No query's own response enters its bank.
- The frozen B2 control encoder has no perturbation-response training loss.
  The common response basis is fitted only on the outer fitting partition.
- Outer labels enter evaluation only. They never choose the checkpoint or loss
  weights. Jurkat is not scored and cannot be called untouched after prior rounds.
- Scores average targets within context, then contexts equally, then optimizer
  seeds. Paired intervals resample targets after averaging seeds (2,000 draws).
  These are exploratory, without multiplicity correction. Largest-target
  sensitivity uses the actual macro-score contribution, not unweighted gains.
- This first screen reports **raw predictions only**, plus amplitude, strong-gene
  and specificity diagnostics. Source-only output calibration is not yet added
  to this screen; `evaluation_scope.json` makes that limitation explicit.
- No automatic winner or independent-confirmation claim is produced. Candidates
  for P3 must be frozen after this development screen before new split-specific
  encoders/bases are built. Later phases cannot inherit label-contaminated bases.

## D1: Mixscale inspection, not automatic training ingestion

The initial public source is the author's [Zenodo record 14518762](https://zenodo.org/records/14518762),
linked to [GSE225775](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE225775).
The registry pins filenames, sizes and author-provided MD5 checksums. IFNG is
2,915,636,149 bytes (~2.72 GiB); the optional TGFB object is 2,642,041,433 bytes.
The command downloads IFNG only, not FASTQ, DE tables or pathway outputs.

Before acquisition, the workflow reserves **all HT29 sources and stimuli** as a
prospective test-v2 candidate in `test_v2_reservation.json`. Inspection is limited
to metadata, identifiers and sparse count-layer QC. HT29 responses are not used
for fitting, basis/normalization, retrieval, adaptation, scoring or selection.
No fallback cell line is opened automatically. This reservation is not proof of
zero overlap with older model pretraining or that HT29 will meet QC criteria.

The data worker creates a dedicated `vcell-work/envs/round15-r` environment with
R, SeuratObject, Matrix and jsonlite if Conda is available. It does not alter the
Python training environment. Network/runtime failure is recorded separately.
Sparse matrices remain sparse; the inspector refuses a dense counts layer.
The input is a multi-GB serialized R object and must be read into RAM: a conservative
memory check runs first, but actual peak usage depends on the object. No valid
raw count layer, missing runtime or inadequate memory cannot produce a training-ready marker.

Outputs live in `vcell-work/raw/round15_D1/IFNG/`: checksum manifest, reservation,
`INSPECT_STATUS.json`, `rds_inspect.json`, compressed metadata and gene identifiers.
Explicit field mapping and checks of CRISPR modality, guide identity, condition,
study/replicate and matched controls are required before P4. The RDS schema and
real server R environment cannot be validated by local synthetic Python tests.

## Files to return

`runs/round15_01/round15_review_light.tar.gz` contains the core report, comparison
tables, per-target metrics, paired diagnostics, completion/reuse/block accounting,
hardware/resource records, reference audits, D1 status/QC, and log tails.
`ARCHIVE_MANIFEST.json` hashes the bytes actually included. Full-server completion
manifests separately cover model/prediction files that the light package omits.

Keep the original run on the server. New predictions are stored losslessly as
float32 coefficients + basis + scale + offset with row/gene identifiers; the same
reconstruction is scored. Checkpoints and large matrices are excluded from the
review archive. `CORE_COMPLETE.json` means the available core screen finished;
`full_round15_complete` stays false. `stage_readiness.json` lists the remaining work.

No training-time or performance improvement is promised before the measured run.
`resources.json` records each head's wall time, examples, parameter count and peak
allocated GPU memory. Concurrent worker wall times must not be summed as GPU hours.
