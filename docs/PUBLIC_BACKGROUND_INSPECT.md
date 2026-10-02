# Public background atlas download and inspection

No Google Cloud subscription, AWS account, API key or token is needed. The default
workflow anonymously downloads five **processed H5AD matrices** from pinned
CELLxGENE URLs and produces an inspection report. It does not download FASTQ,
train a model, normalize counts, filter cells or replace experiment controls.

## Sources and fixed selection

Source metadata and anonymous HEAD/range access were checked on 2026-10-02.
`configs/public_backgrounds.json` pins collection and dataset version IDs, exact
sizes, assay/tissue descriptions, source links and publication DOIs. URLs point
to dataset versions rather than dynamically following the latest collection.

| Key | Source | File GB (decimal) | Initial role |
| --- | --- | ---: | --- |
| `ts_blood` | Tabula Sapiens blood | 2.57 | General background candidate |
| `ts_bone_marrow` | Tabula Sapiens bone marrow | 1.03 | General background candidate |
| `ts_lung` | Tabula Sapiens lung | 3.20 | General background candidate |
| `hlca_core` | Healthy Human Lung Cell Atlas core | 5.87 | External validation candidate; overlap unresolved |
| `immune_global` | Cross-tissue Immune Cell Atlas, global | 3.04 | External validation candidate; overlap unresolved |

Total: about **15.7 GB / 14.63 GiB**. Additional optional keys are `ts_liver`
(0.83 GB) and `ts_skin` (0.51 GB). Do not download full atlases and overlapping
subsets as independent training samples. No independence between the selected
atlases has yet been established. Published cell counts are catalogue values,
not verified counts of unique, eligible cells obtained by this project.

Author/source references and usage terms:

- [Tabula Sapiens processed-data access and citation/redistribution policy](https://tabula-sapiens.sf.czbiohub.org/whereisthedata)
- [Tabula Sapiens collection](https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5)
- [HLCA source](https://github.com/LungCellAtlas/HLCA)
- [HLCA collection](https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293)
- [Cross-tissue Immune Cell Atlas author site](https://www.tissueimmunecellatlas.org/)
- [Immune atlas collection](https://cellxgene.cziscience.com/collections/62ef75e4-cbea-454e-a0ce-998ec40223d3)

Keep source citations and applicable data policies with any derived datasets.
The small review archive contains reports and source metadata, not expression
matrices or a redistribution of the source dataset.

## Run on AutoDL

Use the existing project environment; the scBaseCount optional Google packages
are **not** required:

```bash
cd /root/autodl-tmp/virtual_cell &&
git pull --ff-only &&
PIP_CACHE_DIR=/root/autodl-tmp/vcell-work/cache/pip .venv/bin/python -m pip install -e . &&
screen -dmS publicbg bash scripts/inspect_public_background_autodl.sh
```

Data and reports go under
`/root/autodl-tmp/vcell-work/background/public_inspect_01`.
The default total selected-file cap is 20 GiB, plus a 2 GiB disk reserve.
Downloads are sequential and use no GPU. A different output directory can be
specified with `--output`; do not start two processes for the same directory.

For a smaller first run, download just blood and bone marrow (3.60 GB):

```bash
screen -dmS publicbg_small bash scripts/inspect_public_background_autodl.sh \
  --datasets ts_blood ts_bone_marrow \
  --output /root/autodl-tmp/vcell-work/background/public_small_01
```

For an entirely offline catalogue report, use a separate directory:

```bash
bash scripts/inspect_public_background_autodl.sh --catalog-only \
  --output /root/autodl-tmp/vcell-work/background/public_catalog_01
```

Optional `--panel /path/to/output_genes.txt` accepts one exact gene ID/symbol per
line without a header. The inspector reports overlap independently for each
matrix, including the separate raw gene axis. Without it, panel overlap is
explicitly marked unassessed.

## Progress and interrupted downloads

```bash
tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/public_background_*.log | head -n 1)"
```

Rerun the same command to resume. Each range is tied to the source's strong ETag;
the client verifies exact byte ranges and lengths, rolls back incomplete ranges,
and retries. A server returning the whole file instead of the requested range
is rejected without appending it. Completed files are checked against their
locally recorded SHA256 before reuse. Source/configuration changes require a
different directory. Keep `.part` files and their sidecars for resume.

The source currently supplies multipart ETags, **not publisher SHA256 digests**.
ETags pin transfer identity; the local SHA256 detects changes on reuse. Do not
describe this as validation against a publisher-provided cryptographic checksum.
File-size limits do not cap cumulative network traffic after retries.

## Reports and scientific interpretation

Open `inspect_report.html`, or inspect `inspect_report.json`. Download
**`public_background_review.tar.gz`** to share for review; H5ADs are excluded.
`manifest.json` and `run_config.json` freeze the inspected sources and options.

Reports include:

- Status and errors for each file; successful files remain reusable if another fails.
- Separate sampled QC for `X`, `raw/X` and each layer, using the appropriate gene axis.
- Matrix dimensions, duplicate gene identifiers/symbols, missing panel genes,
  negative/nonfinite/noninteger entries and sampled cell totals where supported.
- Full, chunked counts of available donor, tissue, cell-type, assay, study,
  disease, condition and `is_primary_data` annotation labels. Label counts are
  not independently verified biological sample counts.

**Do not infer raw counts from a matrix name.** In the author's Tabula Sapiens
objects, raw counts and normalized/scaled matrices occupy different locations;
CELLxGENE conversions can change the layout. Integer/nonnegative samples alone
do not certify counts. CSR/dense QC samples at most 128 rows by default and
two million dense-equivalent values; CSC QC samples stored entries only. Full
cell-level QC and selection happen in a later, explicitly versioned step.

Whole Tabula Sapiens tissue objects may contain both 10x and Smart-seq cells.
This stage reports assay counts but does not yet perform the proposed 10x-only,
donor-stratified selection. It does not certify absence of every held-out cell.

Normal blood/immune/lung tissues are general background references, **not**
matched controls for K562/RPE1/H1/HepG2/Jurkat. Existing experiment controls and
the 1,834-gene output target remain unchanged. Before background pretraining,
review intervention status, donor/source overlaps and technology; reserve
donors/studies appropriately and keep HepG2/Jurkat training boundaries intact.

Only `inspect_complete` means all selected file inspections finished.
`catalog_only`, `partial_failure`, `blocked` and individual `pending` entries
are not successful dataset acquisition. Every inspected file remains quarantined
(`training_admitted: false`).
