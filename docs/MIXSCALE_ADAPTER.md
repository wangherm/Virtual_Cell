# Mixscale IFNG: matched controls, adaptation report, and training handoff

This stage consumes the successfully inspected local IFNG RDS. It downloads no
data or models and runs no training. It processes **all eligible development
cells**, writes an HTML/CSV/JSON QC report, and creates prepared training data
with the historical gene panel and evaluation rows preserved.

## Run on AutoDL

```bash
cd /root/autodl-tmp/virtual_cell && \
git -c http.version=HTTP/1.1 pull --ff-only && \
screen -dmS vcell15adapt bash scripts/adapt_round15_autodl.sh
```

```bash
tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round15_D1_adapter_*.log | head -n 1)"
```

Default output: `/root/autodl-tmp/vcell-work/prepared/round15_mixscale_ifng_01`.
Return **`round15_D1_adapter_review.tar.gz`** in that directory. Open `report.html`
for a readable summary. The review archive excludes the raw RDS, per-cell mapping,
expression matrices, model weights, and the merged historical metadata.

The command follows the completed Round15 parent chain to find the actual
reference data, and Round10's plan to find the original split anchor. Explicit
`--reference-data` and `--original-data` can override those paths. It uses the
validated R prefix recorded by D1 inspection. `--rscript` overrides that executable.
No environment installation is attempted. `VCELL_WORK` overrides the work root.

## Fixed adaptation contract

- Source: pinned `Seurat_object_IFNG_Perturb_seq.rds`, Zenodo 14518762 / GSE225775.
  The RDS size, MD5 and inspected SHA256 are rechecked locally.
- Source metadata fields must be complete. Each `guide` must agree with `gene`.
  `BXCP3_IFNG` is mapped to `BXPC3_IFNG` only when the actual `cell_type` agrees;
  original labels remain in the audit. Unexpected labels fail explicitly.
- All HT29 rows remain metadata-only. They never enter the R expression manifest.
  R independently compares the manifest against the original Seurat metadata.
  Loading the serialized RDS into memory is necessary, but its HT29 expression
  columns are not selected, normalized, aggregated, exported, or scored.
- Context is `cell_line_IFNG`, distinct from the historical unstimulated line.
  Study, replicate, CRISPRi, IFNG and 24h are recorded. Dose is not inferred.
  The published protocol is described in
  [Jiang et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12083445/).
- Exact control strata: `cell_type × pathway × Batch_info × orig.ident × sample_ID`.
  These technical IDs are preserved for matching, not declared independent
  biological replicates. No fallback to another library, replicate or line occurs.
- Counts must be finite, nonnegative integers. Positive full-library cells are
  normalized as `log1p(10000 * counts / full_library)`, then the frozen output
  panel is selected and means are calculated. No new mitochondrial/doublet
  threshold or response-based efficacy filter is introduced.
- A technical stratum needs at least **10 positive-library NT cells**. A guide
  needs at least **5 matched cells**. A line/replicate/target unit needs at least
  **2 eligible guides and 20 matched cells**. These are predeclared QC thresholds,
  not empirically optimized cutoffs. CLI overrides require a new output name.
- Every target cell is compared with its own stratum's NT mean. Those baselines
  and normalized target cells are averaged with identical cell weights within
  guide, then across eligible guides. Consequently control-pool composition
  follows the perturbation cells' technical-stratum proportions.
- A target absent from the measured genes is reported and can retain its target
  identity; its own expression knockdown cannot be checked. Missing **output**
  genes block preparation: no zero filling or silent panel reduction.
- Output identifiers need not use the same namespace as the new matrix. Exact
  identifiers are preferred. Otherwise, the adapter reads ID/symbol annotation
  pairs from the historical K562/RPE1 H5AD `var` tables and existing
  `knowledge/round12/cards.json`. No expression/obs or online lookup is used.
  Ensembl version suffixes are normalized for joining only. A unique measured
  annotated symbol is required, and two panel IDs cannot share one source row.
  `mapping/gene_mapping.csv` records evidence, candidates, selected source and
  unresolved/ambiguous status. Original output IDs and order remain unchanged;
  only the source row lookup is translated. An explicit `--gene-map mapping.csv`
  (columns `panel_gene,source_gene`) overrides automatic annotation collection.
- `mixscale_score`, published DE genes and normalized Seurat layers are never
  predictor features, weights, target selection criteria or supervision inputs.

## Reports and boundaries

`mapping/` includes panel and target coverage, verified sample aliases, cell counts
by line/replicate/target/guide, and technical-stratum counts. `aggregation/` contains
positive NT counts per exact stratum and input/positive/matched counts per guide.
`prepared/target_qc.csv` records each retained or excluded target unit. Zero eligible
units produce a nonzero process exit and a `BLOCKED` report, not a success marker.
Do not loosen matching simply to obtain more rows; inspect its scientific meaning.

On success, `prepared/` holds `dataset.npz`, `metadata.csv`, `data_audit.json`, and
three partition manifests (`context`, `target`, `double`). Historical baseline,
response and row membership are retained. Calibration targets, and outer targets
for target/double protocols, are excluded globally from new fitting rows. Jurkat
and HT29 are not scored. No evaluation metrics or automatic winner are produced.

`ADAPTER_READY` / `training_data_ready: true` means a checksummed data handoff,
**not completed training or a complete P4 experiment**. `training_handoff.json`
specifies the remaining steps: annotation-only knowledge for the expanded target
vocabulary, partition-specific background/response transforms, and matched
reference-versus-expanded training. The older Round12 launcher deliberately
rejects new cell-line expansion; do not feed this into it or remove its guard.

R aggregation uses two chunked passes over development columns. The serialized
Seurat object still needs RAM; a conservative memory reserve and 5 GiB free-disk
check run first. The process prints control/target cell progress to the log and
`aggregation/progress.json`. No runtime estimate is promised before measurement.

Rerun the same command after an interruption. Input/code/configuration identity
is frozen. A complete checksummed aggregation is reused; an interrupted R pass
is rebuilt. A completed output is verified without rewriting it. Changed inputs
or thresholds require `--name round15_mixscale_ifng_02`. No old artifacts are deleted.

## Validation

Python tests cover field/guide mapping, sample aliases, missing panels, reserved
cells, identical historical rows, global target holdouts, strict-count exclusions,
report checksums, and immutable completion/resume. A separate real Seurat sparse
integration test checks full-library normalization, exact-stratum control weights,
zero-library exclusion, unmatched cells, and HT29 exclusion. Run it with Rscript
on PATH and SeuratObject/Matrix/jsonlite installed:

```bash
VCELL_REQUIRE_R_TEST=1 .venv/bin/python -m pytest tests/test_round15_adapter.py -q
```

Without that R runtime the integration test is explicitly skipped; Python tests
alone do not validate the server's real RDS or its matching-stratum coverage.

## Recovery from all 1,834 output genes reported missing

The initial adapter compared frozen Ensembl identifiers directly to Mixscale gene
symbols. That namespace mismatch stopped before expression aggregation; it is
not evidence that all 1,834 biological genes are absent. Pull the mapping repair
and use a **new** name because code and annotation identity are frozen:

```bash
cd /root/autodl-tmp/virtual_cell && \
git -c http.version=HTTP/1.1 pull --ff-only && \
screen -dmS vcell15adapt2 bash scripts/adapt_round15_autodl.sh --name round15_mixscale_ifng_02
```

The existing RDS and R environment are reused. Successful alignment prints
`MIXSCALE PANEL VERIFIED: 1834/1834; original IDs/order preserved`. If annotation
is missing, mapping is ambiguous, or genes are genuinely absent, preparation still
stops and packages the detailed mapping report. Do not remove the coverage check.
Return `prepared/round15_mixscale_ifng_02/round15_D1_adapter_review.tar.gz` in that
case; no re-download or training restart is required to inspect the remaining issues.

## Recovery from 70 missing genes with numeric candidate symbols

Candidates such as `["6291", "6419", "SARS1"]` exposed a second annotation
reading problem: the historical H5AD files use legacy categorical encoding.
The integer values refer to labels in `var/__categories/gene_name`; they are
not gene symbols. The adapter now decodes that table, as well as modern
`codes/categories` encoding, and records each encoding in
`gene_annotation_sources.json`. Invalid codes or numeric columns without labels
fail explicitly. This repair only reads annotation tables and preserves the
frozen output panel, coverage checks, and holdouts.

Pull this repair and start a new name, since `_02` froze the earlier decoding:

```bash
cd /root/autodl-tmp/virtual_cell && \
git -c http.version=HTTP/1.1 pull --ff-only && \
screen -dmS vcell15adapt3 bash scripts/adapt_round15_autodl.sh --name round15_mixscale_ifng_03
```

Decoded historical labels can match measured older symbols while the cached
gene cards use newer symbols. No alias is guessed: exactly one annotated,
measured symbol is still required. The server must confirm `1834/1834` before
aggregation proceeds. Any remaining absent or ambiguous genes still block and
are included in `_03/round15_D1_adapter_review.tar.gz`. Local synthetic tests
cover legacy/modern decoding, renamed symbols, missing values, invalid codes,
and ambiguity; they cannot establish the real dataset's final coverage.

## Recovery from the remaining 21 current-name-only matches

After categorical decoding, 1,813/1,834 output IDs mapped. The remaining 21 had
only newer annotated symbols (e.g. `TLE5`, `CERT1`, `SEPTIN7`). Their official
HGNC records contain previous approved symbols such as `AES`, `COL4A3BP`, and
`SEPT7`. The fixed `configs/round15_hgnc_previous_symbols.json` snapshot contains
only those 21 Ensembl IDs, with HGNC identifiers, report URLs, source SHA256,
retrieval time and symbol-ownership evidence. It supplies exact annotation
evidence alongside the existing H5AD annotations; no server lookup is needed.

Only `symbol` and `prev_symbol` are used. Informal `alias_symbol` is excluded:
for example, `SEPT2` is a previous approved name of `SEPTIN2` but also an alias
of `SEPTIN6`. Names shared by multiple Approved/previous records anywhere in
the complete HGNC table are excluded; this rejects `DIFF6`. Multiple measured
names for one output ID or shared source rows still block preparation. The
snapshot and provenance are frozen into the plan and included in the review.

```bash
cd /root/autodl-tmp/virtual_cell && \
git -c http.version=HTTP/1.1 pull --ff-only && \
screen -dmS vcell15adapt4 bash scripts/adapt_round15_autodl.sh --name round15_mixscale_ifng_04
```

The new name is required because `_03` froze earlier annotation evidence. The
RDS and R installation are reused. Local tests cover all 21 ID/previous-name
pairs and reject absence, ambiguity, informal aliases, cross-gene name reuse,
and changed snapshot identity. Actual source-row presence must still pass the
server's `MIXSCALE PANEL VERIFIED: 1834/1834` check before aggregation. A naming
repair cannot supply expression measurements for genuinely absent genes.

Snapshot provenance comes from the [HGNC complete set](https://hgnc.genenames.org/download/)
and its [field definitions](https://hgnc.genenames.org/help/statistics-and-downloads/).
`scripts/build_round15_hgnc_snapshot.py --hgnc-tsv FILE --genes ID... --output FILE`
rebuilds a subset from a local full TSV, with ownership checked against every
Approved record in that TSV. Changing the snapshot requires a new run name.
