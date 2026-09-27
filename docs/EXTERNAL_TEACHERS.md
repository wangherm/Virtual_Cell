# Import external teacher predictions

The default teachers are reference models trained from scratch. This interface
imports predictions generated in an external model's official environment; it
does not install State, GEARS or other pretrained models.

## Export queries

```bash
python -m vcell queries --data /absolute/path/to/prepared --output data/teacher_queries
```

The export contains control baselines, perturbation IDs, gene IDs and row IDs,
without true perturbed expression. Models needing control single-cell sets must
read the original control cells; do not repeat group means to fabricate cells.

## Generate and import deltas

Each row of `teacher_delta.npy` must be predicted mean log1p-normalized expression
minus the corresponding mean control expression, on the project's target scale.
Latent embeddings, raw counts and log-fold changes are not interchangeable with
these deltas. The row and gene CSVs must have `row_id` and `gene` columns;
the importer aligns by ID and requires full query/gene coverage.

Copy `configs/teacher_provenance.example.json`. Record the actual source,
revision, license, pretraining/fine-tuning contexts and explicitly excluded
held-out contexts. Declarations are checked, but the program cannot prove that
pretraining data were uncontaminated.

```bash
python -m vcell import-teacher \
  --data /absolute/path/to/prepared \
  --matrix data/external/teacher_delta.npy \
  --rows data/external/row_ids.csv --genes data/external/genes.csv \
  --provenance data/external/provenance.json --output data/teacher_external.npz
```

Keep the NPZ and its sibling JSON together. Replace a teacher entry in a local
training configuration, keeping the other teacher definitions:

```yaml
teachers:
  - {name: t_external, cache: /absolute/path/to/teacher_external.npz}
  - {name: t_module, architecture: module, hidden: 256, dropout: 0.1}
  - {name: t_mlp, architecture: mlp, hidden: 256, dropout: 0.1}
  - {name: t_bilinear, architecture: bilinear, hidden: 256, dropout: 0.1}
```

Only cached teachers skip training. Training-config cache paths are relative to
the shell's current directory, so absolute paths are preferable on a server.
A fixed external cache is shared across seeds; repeated teacher scores are not
independent retraining replicates.

Teachers with unknown provenance or exposure to held-out perturbation labels are
unsuitable for this strict transfer evaluation. Do not use validation/test truth
to fill missing outputs.
