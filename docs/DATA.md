# Data preparation and control-only inference

Run commands from the repository root in your activated environment. Raw data are
user-supplied and remain outside Git. Preserve raw files; write each preprocessing
configuration into a fresh prepared directory.

## Inspect inputs before configuring

The default experiment trains on K562/RPE1, validates on HepG2 and holds out Jurkat
for final evaluation. The template is not a guarantee that a downloaded file uses
the example column names. See [references](REFERENCES.md) for original releases.

```bash
python -m vcell catalogue
# Choose a file ID from the catalogue; this example placeholder must be replaced.
python -m vcell download --file-id FILE_ID --output /absolute/path/to/raw
python -m vcell inspect /absolute/path/to/raw/k562.h5ad
```

The catalogue/download commands target the Replogle Figshare article by default;
they do not download all contexts automatically. Supply the other contexts from
their respective releases. If files are already available, skip downloading.
Inspect every file's `.obs`, `.var`, count layers and example labels.

## Create a local manifest

```bash
mkdir -p configs/local
cp configs/data.example.yaml configs/local/data.yaml
```

Set `output_dir` and every `datasets[].path` to absolute server paths. The template
uses paths relative to the YAML's own directory: copying it into `configs/local/`
without editing those paths would point to the wrong location.

| Field | Meaning |
| --- | --- |
| `output_dir` | New prepared-data directory, relative to the YAML or absolute |
| `train_contexts`, `val_contexts`, `test_contexts` | Nonempty, disjoint context sets |
| `datasets[].id` | Dataset identifier used in group row IDs |
| `path`, `context` | Input `.h5ad` file and its assigned context |
| `gene_key` | `.var` column for gene IDs, or `null` to use `.var_names` |
| `perturbation_key` | `.obs` column with perturbation labels |
| `batch_key` | `.obs` column for matched batches; `null` explicitly declares one batch |
| `count_layer` | `X` or the exact `.layers` key containing raw counts |
| `control_values` | Exact control labels, e.g. `[non-targeting]` after verifying the file |
| `perturbation_map` | Optional mapping from source labels to single-target identifiers |
| `row_filter` | Optional `.obs` filters; overlapping source rows across entries are rejected |

Input counts must be finite, nonnegative and integer-valued. Do not feed already
log-normalized expression into the raw-count preprocessing path. Align gene ID
namespaces across files and resolve duplicate gene IDs explicitly. Compound guide
labels need explicit mapping; the current model supports training-known, single
perturbation IDs, not unseen targets or arbitrary combinations.

```bash
python -m vcell prepare --manifest configs/local/data.yaml
```

Prepared outputs:

- `dataset.npz`: control baselines, expression deltas, selected genes and perturbation vocabulary.
- `metadata.csv`: one row per dataset/batch/perturbation group, with context, split and cell counts.
- `data_audit.json`: source paths, effective manifest, feature selection, dropped groups and fingerprint.

Check the audit and metadata before training. All three splits must contain valid
groups, with enough perturbation cells and same-batch controls. The pipeline
normalizes each cell by its full-library count, applies `log1p`, then aggregates
group means. It selects high-variance features using training cells only and
retains held-out groups only for perturbations present in the training vocabulary.
The target is mean perturbed expression minus matched mean control expression.

For example, inspect the actual prepared directory from your manifest:

```bash
python -c "import pandas as pd; m=pd.read_csv('/absolute/path/to/prepared/metadata.csv'); print(m.groupby(['split','context']).agg(groups=('row_id','count'), targets=('perturbation','nunique')))"
```

Do not edit prepared NPZ/CSV files in place: their fingerprints are checked at load
time. Change the raw-data manifest and prepare a new directory instead.

## Control-only inference

Choose a model mode and seed using validation results, then freeze that choice.
Copy `configs/query.example.yaml` to `configs/local/query.yaml`, inspect the new
control dataset and update paths/fields. Use absolute paths; relative input paths
are resolved against the query manifest's directory. Only control cells are
needed, with enough controls per batch and the checkpoint's required genes.

Create a targets CSV with a `perturbation` header and unique IDs drawn from the
training vocabulary (replace these example IDs):

```csv
perturbation
TARGET_A
TARGET_B
```

```bash
python -m vcell predict-controls \
  --checkpoint runs/real_01/seed_0/supervised/best.pt \
  --manifest configs/local/query.yaml \
  --targets /absolute/path/to/targets.csv \
  --output runs/new_context_01 --device cpu
```

The output directory must not already exist. `mean_predictions.npz` contains
`delta`, `predicted_mean`, `baseline`, `genes` and individual-model deltas;
`prediction_metadata.csv` identifies each context/batch/perturbation query.
The checkpoint supplies normalization and validation-selected student ensemble
weights. These are group-mean predictions, not synthetic single-cell samples.
