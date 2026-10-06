# Round15 P4: full eligible Mixscale training comparison

This command trains a matched **pre-Mixscale reference vs full eligible Mixscale
expansion** comparison on the existing historical development tasks. It follows
the completed `round15_mixscale_ifng_04` handoff. The reference is the Round12
expanded dataset, not the smaller Round8 dataset; Round8 supplies the historical
split anchor only. The fixed output panel remains 1,834 genes.

## Start on AutoDL

```bash
cd /root/autodl-tmp/virtual_cell && \
git -c http.version=HTTP/1.1 pull --ff-only && \
screen -dmS vcell15p4 bash scripts/run_round15_p4_autodl.sh --name round15_p4_01 --parallel-jobs 3
```

All paths default to the data disk. The adapter must be complete and pass its
checksums. `--adapter-dir`, `--round10-run`, `--parent-knowledge`, and
`--background-dir` allow explicit relocated paths. Historical model configuration
must still resolve locally. No model or perturbation data download is attempted.
Incremental MyGene/STRING annotation queries may be needed for new target labels;
only gene identifiers are submitted, never expression or response labels.

The bundled `configs/round15_hgnc_targets.json` resolves all 59 audited Mixscale
target names offline, including the previous approved name `RARRES3` → `PLAAT4`
(Entrez5920; HGNC:9869). It records the full-table source SHA256, unique name
ownership and HGNC reports. Informal aliases are excluded. Only added-target cards
are updated; a conflicting cached Entrez identity still blocks the run. Targets
outside this bounded snapshot retain the strict MyGene fallback. STRING queries
are still needed for genuinely new target vectors.

If the earlier code stopped at annotation with `RARRES3`, update and use
`--name round15_p4_02`. `_01` remains an audit record; its frozen source identity
cannot be resumed after this naming fix. No training checkpoint was created by
that annotation-stage failure. Inspect `knowledge/target_annotation_audit.csv`
and `target_annotation_snapshot.json` in the new run for the applied evidence.

```bash
.venv/bin/python scripts/round15_p4_status.py
tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round15_p4_*.log | head -n 1)"
```

The master log prints each stage and process exit. Per-worker logs are under
`runs/round15_p4_01/logs/`, including background reconstruction loss and head
step/loss. `background_job_results.json` and `benchmark_job_results.json` preserve
both queue outcomes. A process failure is an incomplete run, not a success marker.

## Fixed comparisons and budget

- Data: Round12 reference and reference plus every QC-eligible Mixscale row.
- Tasks: historical context, target, and double holdout, preserving the identical
  historical calibration and outer rows. Jurkat and all HT29 sources stay closed.
- Seeds: 17, 29, 43. Three optimizer seeds are not biological replications.
- Methods: STRING interaction Ridge; no-ID local factor with response rank16;
  the same factor with rank32. Zero change and fit-only mean transfer are reported.
- The default matrix has **18 background fits, 18 benchmark workers, 54 fitted
  heads**. Background steps/batch copy the historical Round10 B2 budget. Each
  factor head uses 2,000 updates, batch128, AdamW lr5e-4/wd0.01; checkpoints and
  calibration selection occur every100 updates. Ridge alpha is selected from
  `[0.0001, 0.001, 0.01, 0.1, 1]` on source calibration only.
- Default concurrency is three independent processes, two CPU threads each.
  `--parallel-jobs 1/2/3/4` changes scheduling, not model settings.
- B2 is an expression projector trained on fit-only matched controls mixed 1:1
  with the existing public-background training cache. It uses Qwen-sized hidden
  dimensions from the local config; **this is not language-model/LoRA training**.
  No new Qwen text vectors are generated in this STRING-only experiment.

“Full” means all eligible **fit** rows can be sampled; excluded, calibration and
outer rows are never fitted. Training remains a fixed update-budget comparison,
not an assurance that every model has reached convergence. Rank bases, background
normalization/PCA and target PCA are refitted independently inside each fit fold.
Neural checkpoints and Ridge penalties use calibration, never outer scores.

## Annotation and target identity

Existing STRING vectors are copied exactly from the completed Round12 knowledge
artifact. Only genuinely added targets need STRING v12 top16 functional-neighbor
queries. Both data conditions use the same frozen per-gene vectors, while their
fit-only projections can differ. Coverage is explicit; a target with no measured
panel neighbors stays in the experiment with an unavailable flag.

`knowledge/target_id_audit.csv` compares stable Entrez identities (or an explicit
symbol fallback when the cached card lacks Entrez). Added aliases are joined to
the existing historical label, applying changes only to appended rows in a new
`harmonized/` dataset. Global target exclusion is then recomputed from the original
split anchor. Two historical labels mapping to one gene, or missing/ambiguous
added target annotation, block the experiment. No source prepared file is edited.

`training_weight_audit.csv` makes context weights explicit. Sampling retains the
historical equal-context/equal-target/equal-row rule. Five added IFNG contexts can
therefore receive substantial loss mass despite contributing relatively few
aggregate rows. This is a data/background/target-coverage experiment, **not a pure
cell-count effect**. Cell counts are not used as measurement-precision weights.

## Resume and outputs

```bash
cd /root/autodl-tmp/virtual_cell
screen -dmS vcell15p4resume bash scripts/run_round15_p4_autodl.sh --name round15_p4_01 --parallel-jobs 3 --resume
```

Use the same model/data/step options. Input, metadata, annotation, source code and
config identity are frozen. Resume verifies completed workers and continues
incomplete background/head checkpoints; it does not trust marker existence alone.
Changed identities require a new name. `--prepare-only` prepares annotation/cache
and configs without training; subsequently use the same command with `--resume`.

Success prints `P4 COMPLETE`. Review these files in `runs/round15_p4_01/`:

- `comparison_mean.csv`: task/method/data-condition means and seed dispersion.
- `paired_data_gain.csv`: reference-minus-expanded MSE and exploratory paired
  target-bootstrap intervals, averaging seeds before resampling whole targets.
- `context_scores.csv`, `retrieval.csv`, `response_diagnostics.csv`: MSE,
  perturbation specificity and response amplitude by context/target.
- `report.html` and **`round15_p4_review_light.tar.gz`**: review package with
  checksums and log tails, excluding model weights and expression/prediction arrays.

This completes the matched historical P4 comparison only. It does not run new
IFNG-background holdouts, matched one-vs-three-background strategies, cell
resampling diagnostics, native P5 teachers, P6 distillation or few-shot adaptation.
No automatic winner or external test release is produced. Broader Round15 stages
remain separate experiments rather than being silently claimed complete.

## Validation

`tests/test_round15_p4.py` performs real CPU background and factor/Ridge training,
full launcher/report/resume, old-vector preservation, canonical-target leakage
prevention, missing-identity rejection, historical-row/HT29 protection, outer-label
poisoning, exact interrupted-head resumption and orchestration recovery after one
worker fails while other workers have completed. Synthetic tests do not establish
real-data performance or annotation-service reachability on AutoDL.
