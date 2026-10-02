# Prepare a fixed public background reference

This stage uses the already downloaded public H5AD files. It performs full-cell
QC, gene alignment, deterministic sampling, donor partitioning and reporting.
It does not download data, fit a model, read perturbation responses, or evaluate
the sealed test set. Original source files remain unchanged.

## AutoDL run

```bash
cd /root/autodl-tmp/virtual_cell && git pull --ff-only && \
screen -dmS vcellbg bash scripts/prepare_public_background_autodl.sh
```

Defaults:

- Input: `/root/autodl-tmp/vcell-work/background/public_inspect_01`.
- Gene panel: `runs/round8_01/prepared/plus_h1/dataset.npz`, relative to the work
  directory. Only its `genes` array is read; exactly 1,834 unique genes are required.
- Output: `/root/autodl-tmp/vcell-work/background/human_background_v1`.
- Log: `logs/background_clean_TIMESTAMP.log` in the work directory.

If Round8 used a different run name, pass `--panel /absolute/path/to/dataset.npz`.
Do not substitute the earlier `real_min10_01` panel without an explicit protocol
change. All input paths are checked; no fallback panel is generated.

```bash
tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/background_clean_*.log | head -n 1)"
```

Progress messages show source verification, `QC dataset: processed/total`, each
exported shard, and `BACKGROUND CLEAN COMPLETE`. File hashing can take time before
the first QC message. Read `report.json` for stage/error details and check the root
`COMPLETE.json` for successful preparation. Completion does **not** mean training
has run or that external references have passed overlap review.

## Frozen preparation policy

Every stored entry of `raw/X` is checked for finite, nonnegative, integer counts,
in streaming row blocks. This policy relies on the pinned source's count-layer
provenance; the integer check alone is not proof of biological provenance. QC uses
the full raw gene axis, including genes outside our model panel. `.X` from the
source is not used, because it is already processed. CSC inputs or missing count
layers stop the run instead of silently changing the policy.

Default cell filters retain normal-annotated 10x cells with known donor, tissue,
and cell type, at least 500 raw counts and 200 detected genes, and at most 20%
mitochondrial counts. Mitochondrial genes are symbols starting with `MT-`.
Annotated doublets, tumor-adjacent lung cells, repeated observation IDs within a
file, and HepG2/Jurkat annotations are excluded. Unknown unannotated doublets are
not inferred. `is_primary_data=False` is retained: that field does not itself
indicate poor cell quality. Thresholds are pilot defaults; inspect attrition by
cell type and donor before treating them as final biological QC standards.

Gene alignment uses exact source IDs first, otherwise a unique exact gene symbol
on `raw/var`. No fuzzy aliases, version stripping or duplicate-symbol aggregation
are applied. Missing, ambiguous or explicitly filtered source features get an
`available=False` mask. Zero placeholders for those features are **not measured
zero expression**. Future training code must consume this mask; these shards are
not automatically substituted into the existing perturbation trainer.

To prevent abundant populations from dominating, each donor/tissue/cell-type/assay
stratum is capped at 2,000 cells. Strata are sampled in deterministic round-robin
order up to 60,000 cells per dataset (at most 300,000 across these five sources).
Every input cell still receives QC. Both caps are CLI options; change them in a
new output directory if the report supports a larger reference. A smaller selected
count than QC-pass count therefore need not indicate poor quality.

Tabula blood, marrow and lung share one donor namespace. Approximately 20% of
eligible donors are held out globally for validation. Selection uses metadata
only, preferring donor coverage in every tissue dataset. The same donor cannot
appear in training and validation, even across tissues. Small donor counts can
prevent balanced tissue coverage; inspect `selected_coverage.csv` before training.
HLCA and the immune atlas are exported as `external_candidate` only: cross-atlas
source/donor overlap is not yet verified. They are neither training data nor an
independent test benchmark at this stage. Normal tissues also do not replace
matched K562/RPE1/H1 cell-line controls.

## Outputs and restart behavior

- `report.html`, `report.json`, `summary.csv`: cell retention, gene coverage, QC
  quantiles, overlapping exclusion counts, donor splits and limitations.
- `run_config.json`, `panel.txt`, `donor_split.json`: frozen parameters, input and
  implementation hashes, panel order and global Tabula donor assignments.
- Each dataset: `qc_cells.csv.gz` and `membership.csv.gz` track every input row,
  its QC, exclusion reasons and final assignment. `qc_by_*.csv` and
  `selected_coverage.csv` summarize attrition and selected strata.
- Each dataset: `gene_mapping.csv`, `COMPLETE.json`, and separate
  `train_part_*.h5ad`, `val_part_*.h5ad`, or `external_candidate_part_*.h5ad` shards.
  Each shard has raw panel counts in `layers['counts']`, the feature mask in
  `var['available']`, and `.X = log1p(10000 * counts / full-source library size)`.
  Normalization uses the full library, **not** the panel sum. Shards retain source
  row IDs, donor, tissue, cell type, assay, split and source metadata.
- `background_clean_review.tar.gz`: small report package, excluding H5AD matrices
  and full per-cell membership tables. Download this file for review.

Rerun the same command to resume. Source SHA256 values are verified each time;
completed per-dataset QC and exports are reused only after checksum verification.
An interrupted dataset's unfinished QC/export stage restarts; it does not resume
at an individual row. No duplicate writers are allowed. Configuration/code changes
require a new output directory rather than silently mixing results. At least
10 GiB free disk reserve is required by default; export also checks remaining
space. Runtime is primarily disk/CPU work; no GPU or new model downloads are needed.

After reviewing retention and panel masks, freeze this reference for the planned
background-encoder and module-supervision ablations. Do not infer an improvement
in perturbation prediction from preparation completion alone.
