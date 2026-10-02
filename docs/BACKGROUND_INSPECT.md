# scBaseCount acquisition and inspection

This is a **candidate inspection workflow**, not a background training dataset.
It downloads the human sample metadata for publication release `2026-01-12`,
finds K562/RPE1/H1 aliases, excludes metadata mentioning HepG2/Jurkat or obvious
mixed backgrounds, then downloads a bounded selection of H5AD files. No model
training, delta computation, or existing experiment modification occurs.

## 1. Cloud access (one-time setup)

Official references, checked 2026-10-02:

- [Arc access instructions](https://github.com/ArcInstitute/arc-virtual-cell-atlas#accessing-the-data)
- [scBaseCount metadata and count definitions](https://github.com/ArcInstitute/arc-virtual-cell-atlas/tree/main/scBaseCount)
- [Marketplace subscription](https://console.cloud.google.com/marketplace/product/bigquery-public-data/arc-institute)
- [Install Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
- [Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc)

Create/select a Google Cloud project with billing enabled, subscribe that exact
project to Arc Virtual Cell Atlas on Marketplace, and note its project ID.
The bucket is Requester Pays. Arc documents up to 2 TB/month of transfer credits
for the subscribed project; do not assume anonymous downloads or every cloud
charge is free. Check the subscription and Cloud Billing budget before transfer.
The script's GiB cap limits selected expression file sizes, not monetary charges
or total network bytes after retries. It never subscribes or enables billing.

Install the Google Cloud CLI on the server using Google's instructions. Then,
in your own terminal (replace the project ID):

```bash
gcloud init --no-launch-browser
gcloud auth application-default login --no-launch-browser
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

Follow the browser instructions printed by gcloud yourself. A plain `gcloud auth
login` is not sufficient for the Python SDK: this workflow uses ADC. An existing
authorized `GOOGLE_APPLICATION_CREDENTIALS` file also works. Do not paste tokens
or credential files into chat or commit them. Authentication, Marketplace
subscription and permission to charge the project must all be available.

## 2. Install and run on AutoDL

```bash
cd /root/autodl-tmp/virtual_cell
git pull --ff-only
PIP_CACHE_DIR=/root/autodl-tmp/vcell-work/cache/pip .venv/bin/python -m pip install -e '.[background]'
```

Run the bounded download and inspection in screen:

```bash
screen -dmS bginspect bash scripts/inspect_background_autodl.sh \
  --billing-project YOUR_PROJECT_ID \
  --output /root/autodl-tmp/vcell-work/background/scbasecount_inspect_01
```

Defaults: `Gene` (exonic) count feature; at most 2 files per background; 5 GiB per
file; 20 GiB cumulative expression-file size; 2 GiB disk reserve; 256 sampled
cells per CSR/dense file (further capped for memory). These are inspection
limits, not a representative sampling design. Data files remain in `h5ad/`.
Failed inspections still consume that invocation's size/file budget.

Use `--feature GeneFull_Ex50pAS` only in a **different output directory** when
investigating that counting convention. These are alternative counts of cells,
not additional independent biological samples. `Gene` is an initial inspection
choice, not a claim of equivalence to the existing datasets or Flex chemistry.

For metadata only, use a separate output directory:

```bash
bash scripts/inspect_background_autodl.sh \
  --billing-project YOUR_PROJECT_ID --metadata-only \
  --output /root/autodl-tmp/vcell-work/background/scbasecount_catalog_01
```

Sample metadata is capped at 256 MiB. No whole-bucket listing or recursive sync
is performed. Candidate selection uses cell-line aliases, prioritizes explicit
baseline claims and rotates source groups when available. It does **not** prove
control status. `none`, missing, unsure, study-level labels and ambiguous H1 or
subline identifiers all require source review. Metadata marked as a baseline
is still unverified. Samples with unknown/mixed interventions may be downloaded
for inspection; they are never admitted to training.

## 3. Progress, restart and review

```bash
tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/background_inspect_*.log | head -n 1)"
```

Rerun the **same command** after network interruption. Downloads use fixed object
generations and 16 MiB ranges, retain `.part` files, and verify MD5/CRC32C before
promotion. Finished files are rechecked by SHA256. Source identity/configuration
changes fail rather than silently combining runs. A corrupt partial file remains
for investigation; ordinary resume cannot repair bytes already written wrongly.

The output directory contains:

| File | Meaning |
| --- | --- |
| `inspect_report.html` | Open in a browser; status, coverage, exclusions and per-file QC |
| `inspect_report.json` | Machine-readable report; errors are preserved |
| `candidates.csv` | All matched/held-out metadata rows, decisions and source fields |
| `download_manifest.json` | Successful files, object generations, sizes and SHA256 |
| `run_config.json` | Pinned configuration |
| `background_inspect_review.tar.gz` | Small review package; excludes H5ADs and credentials |

Send only the review archive initially. `inspect_complete` means inspections
finished; it does not mean training admission. `no_files_inspected`,
`partial_failure`, and `blocked` must not be interpreted as success. The report
is refreshed after each file and on a caught error (not during a forced kill).

QC includes X shape/encoding, layers, duplicate gene identifiers, sampled finite/
nonnegative/integer checks, sampled cell totals/detected genes for CSR/dense,
and selected obs annotation values from the first 5,000 cells. CSC QC only samples
the first 200,000 stored entries. It does not scan all cells, certify raw counts,
or establish per-cell control membership, source independence or absence of
held-out cells. Cross-source overlap requires an accession-level follow-up audit.

Optional `--panel /path/to/output_genes.txt` takes one exact gene ID/symbol per
line, no header. It reports exact overlap and missing genes without silently
mapping aliases. If omitted, panel coverage is explicitly reported as unassessed.

## 4. Offline use

Local sample metadata CSV/Parquet must contain `srx_accession`, `file_path`,
`cell_line`, `perturbation`; keep the official pinned GCS paths in `file_path`.
The other source fields are preserved. To inspect already downloaded H5ADs by
their original basenames without Google credentials:

```bash
bash scripts/inspect_background_autodl.sh \
  --metadata /path/to/sample_metadata.parquet --local-dir /path/to/h5ad \
  --output /root/autodl-tmp/vcell-work/background/local_inspect_01
```

This first stage does not train a background encoder or modify Round8/Round9.
HepG2/Jurkat exclusion is a conservative metadata screen, not a provenance
guarantee. All collected files remain quarantined for the next review.
