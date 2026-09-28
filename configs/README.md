# Configuration templates

Copy templates to `configs/local/` before adding server-specific paths or changing
an experiment. That directory is ignored by Git.

| File | Purpose |
| --- | --- |
| `smoke.yaml` | Synthetic check: one seed, up to 5 teacher/student epochs |
| `pilot.yaml` | Real-data pilot: one seed, up to 20 teacher/student epochs |
| `real.yaml` | Full experiment: three seeds, up to 100 teacher / 80 student epochs |
| `data.example.yaml` | Raw-count datasets, field names, context splits and preprocessing |
| `query.example.yaml` | Control-only data for inference in a new context |
| `teacher_provenance.example.json` | Provenance required to import external teacher predictions |
| `qwen_smoke.yaml` | Supervised pretrained Qwen plumbing check |
| `qwen_three_teachers.yaml` | Five comparison arms; requires all three audited teacher caches |
| `qwen_backbone.lock.json` | Pinned Qwen revision, file sizes and SHA256 hashes |
| `scgpt_backbone.lock.json` | Pinned author scGPT weights, arguments and vocabulary |
| `teacher_scgpt.example.yaml` | Manual frozen scGPT extraction; prefer the AutoDL launcher |

`scripts/run_scgpt_autodl.sh` writes effective export/head/Qwen configurations
under the selected teacher run directory. See [the scGPT guide](../docs/SCGPT_FIRST_RUN.md).

Training configuration paths (`data_dir`, `output_dir`, teacher `cache`) resolve
against the **shell's current working directory**. Run commands from the repository
root, or supply absolute paths. `--data`, `--output` and `--device` override the
corresponding training settings.

Data/query manifest paths (`datasets[].path`, and preprocessing `output_dir`)
resolve against the **manifest's directory**. Moving a manifest into `configs/local/`
changes what its relative paths mean. Use absolute paths in server manifests.

The `seed` in a data manifest controls preprocessing. Training uses the `seeds`
list, one independent experiment per entry. Do not change a run's settings in place
and then resume it: the runner checks the config, source, data and external caches.
