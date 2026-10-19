# Research model release: 19 October 2026

Download: https://github.com/wangherm/Virtual_Cell/releases/tag/research-models-2026-10-19

This release contains one trained Qwen adapter and seven source-frozen Round26
Python prediction methods. It is a research archive, not a new foundation model
or an independently validated deployment. Round27 is unfinished and excluded.

## Contents and configuration

| Artifact | Configuration | Additional requirements |
| --- | --- | --- |
| `qwen_student_a/adapter.pt` | Qwen3-0.6B-Base, 2,000 ordered output genes, 196 saved perturbation targets, best epoch 7 | Exact separately downloaded base and correctly preprocessed control queries |
| Seven `methods/*.py` files | Two Repeat and five Transfer algorithms; 1,809-gene panel; `fit_predict(data, budget)` | Original legal public source/query templates and refitting in the protected Linux worker |

The adapter is 54,748,485 bytes, SHA-256
`37ad93d5f3d736a816317b983fc02fa3387933eae2288867591e293afc6e2cf4`.
It contains 121 trainable tensors / 13,645,776 trainable parameters, saved prompts,
normalization and vocabularies. Frozen base weights and optimizer state are not
included. It is a custom VCell checkpoint, not a PEFT `adapter_config.json`
directory or a text-generation model.

The base is `Qwen/Qwen3-0.6B-Base`, revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. The checkpoint uses BF16, eight
continuous control tokens, hidden size 1,024, LoRA rank 8 / alpha 16 on `q_proj`
and `v_proj`, and a signed 2,000-gene regression head. CPU inference defaults to
float32; CUDA uses the saved BF16 setting unless overridden. Inference disables
gradient checkpointing and makes no DeepSeek calls.

`qwen_student_a/config.json` records exact base file hashes, response scale
`0.16227512061595917`, data/run fingerprints and checkpoint settings.
`genes.txt` and `perturbations.txt` preserve the saved order. Mean/std arrays are
loaded from the checkpoint. The complete raw preprocessing audit and original
training environment lock are not included; they must not be invented from the
weights. The historical server reported Python 3.12 / PyTorch 2.8.0+cu128, but
the export is not a complete environment capture.

The seven methods are identified in `manifest.json`:

| Manifest ID | Task | Module SHA-256 prefix |
| --- | --- | --- |
| `shinka_repeat_2710fa32d6` | Repeat | `2710fa32d6` |
| `repair_repeat_29e9a57329` | Repeat | `29e9a57329` |
| `shinka_transfer_e520dc6c8d` | Transfer | `e520dc6c8d` |
| `repair_transfer_3b6fa3a74a` | Transfer | `3b6fa3a74a` |
| `repair_transfer_bcf2681a74` | Transfer diagnostic | `bcf2681a74` |
| `meta_transfer_b3e0f6940b` | Transfer | `b3e0f6940b` |
| `meta_transfer_804ace7b90` | Transfer diagnostic | `804ace7b90` |

These are the existing source shortlist, including response-oriented Pareto
candidates. They are not all winners, and the strongest fixed comparators are
not included in this export. Each algorithm imports NumPy; the trusted worker
also uses PyTorch and the repository evaluation/runtime dependencies.

## Install and download

From the repository root, use an existing working environment if available.
For a new CPU checkout:

```bash
bash scripts/setup.sh cpu
.venv/bin/python -m pip install -e '.[foundation]'
```

The declared base dependencies are Python >=3.10, torch >=2.2,<3, numpy >=1.26,<3,
pandas >=2.1,<3, scipy >=1.11,<2, anndata >=0.10,<0.13, PyYAML >=6,<7 and
requests >=2.31,<3. Qwen inference additionally uses transformers==4.57.3,
peft==0.18.0 and the `foundation` extra (including Accelerate, safetensors and
huggingface-hub). These are compatibility constraints, not the historical
training lock. For CUDA, follow the repository server guide and select a wheel
compatible with the driver instead of replacing a functioning AutoDL install.

Download using authenticated or public GitHub CLI access:

```bash
mkdir -p /root/autodl-tmp/vcell-work/released_models
cd /root/autodl-tmp/vcell-work/released_models
gh release download research-models-2026-10-19 \
  --repo wangherm/Virtual_Cell \
  --pattern 'virtual-cell-research-models-20261019.tar.gz' \
  --pattern 'SHA256SUMS'
sha256sum -c SHA256SUMS
tar -xzf virtual-cell-research-models-20261019.tar.gz
```

The release page also supports a browser download. Keep the checksum file with
the archive. `manifest.json` identifies individual models; ARCHIVE_MANIFEST.json
records all package payload hashes. Neither archive is a resumable server backup.

## Qwen inference

Prepare the pinned base, or reuse the existing verified server snapshot:

```bash
cd /root/autodl-tmp/virtual_cell
.venv/bin/python scripts/prepare_qwen.py \
  --model-dir /root/autodl-tmp/vcell-work/models/Qwen3-0.6B-Base
```

An optional `--endpoint https://hf-mirror.com` changes the download endpoint;
the required revision and hashes stay fixed. The new inference entry point
supports a relocated base directory only when its content hashes match the
checkpoint. It never silently substitutes a different base or downloads one
during prediction.

Supply a query NPZ with exactly these four non-object arrays:

| Key | Shape | Meaning |
| --- | --- | --- |
| `baseline` | N x 2,000 float | Matched-control mean expression, using the original library normalization and per-cell log1p before averaging; not raw counts and not z-scores |
| `genes` | 2,000 strings | Exact saved Ensembl gene order |
| `targets` | N strings | Perturbation names from the saved 196-target vocabulary |
| `row_ids` | N unique strings | Caller-defined query identifiers |

Mean(log1p(normalized cell counts)) differs from log1p(mean counts). Preserve the
original preprocessing convention. The checkpoint's mean/std standardization
and response scaling are applied by the inference command. Missing output genes,
reordered genes and unseen targets are rejected; there is no silent zero fill.
No query response labels are required or accepted.

For a format example using a few original **training** control rows, not an
accuracy evaluation:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
data = Path('/root/autodl-tmp/vcell-work/prepared/real_min10_01')
meta = pd.read_csv(data / 'metadata.csv')
ix = np.flatnonzero(meta['split'].eq('train').to_numpy())[:3]
with np.load(data / 'dataset.npz', allow_pickle=False) as f:
    np.savez_compressed('/root/autodl-tmp/vcell-work/qwen_queries.npz',
        baseline=f['baseline'][ix], genes=f['genes'],
        targets=f['perturbations'][f['pert_idx'][ix]],
        row_ids=meta.iloc[ix]['row_id'].astype(str).to_numpy(dtype='U'))
PY

.venv/bin/python scripts/predict_release.py qwen \
  --release-dir /root/autodl-tmp/vcell-work/released_models/virtual-cell-research-models \
  --model-dir /root/autodl-tmp/vcell-work/models/Qwen3-0.6B-Base \
  --input /root/autodl-tmp/vcell-work/qwen_queries.npz \
  --output /root/autodl-tmp/vcell-work/qwen_predictions.npz \
  --device cuda --batch-size 16
```

The example prepared directory must have the exact checkpoint panel. An
1,809-gene later-round dataset is incompatible with this 2,000-gene adapter.
For a CPU software run use `--device cpu --dtype float32`; memory and runtime
depend on the separate full base model. Use a fresh output filename.

Outputs contain `delta`, `genes`, `row_ids` and `targets`, plus a provenance JSON.
Delta is the signed mean expression response in the original prepared response
space. It is not a raw count, a single-cell simulation or a text explanation.

## Frozen Round26 algorithm inference

The seven methods use legal source train/calibration responses and public query
features to fit a numerical method on each invocation. Their source templates,
condition labels, gene mapping, STRING/control projections and auxiliary arrays
are **not included** in the release. A `.py` file alone is insufficient.

Use an original public worker `input.npz` from the completed Round26 campaign,
for the matching task and 1,809-gene panel. Find available templates:

```bash
find /root/autodl-tmp/vcell-work/runs/round26_critic_02 \
  -path '*/workers/*/input.npz' -type f | head
```

Inspect the associated frozen fold and task before choosing. Do not supply
`private/grade.npz`, query truth, or an arbitrary new query whose transforms were
not generated by the original source-only protocol. Example command, replacing
the template path with an actual **Repeat** public input:

```bash
.venv/bin/python scripts/predict_release.py method \
  --release-dir /root/autodl-tmp/vcell-work/released_models/virtual-cell-research-models \
  --model repair_repeat_29e9a57329 \
  --input /ABSOLUTE/PATH/TO/MATCHING/REPEAT/WORKER/input.npz \
  --output /root/autodl-tmp/vcell-work/released_repeat_prediction_01 \
  --device cpu --steps 500 --seconds 120
```

Generated code runs only in the existing Linux Landlock/libseccomp worker;
missing kernel protection causes failure, not a fallback to unrestricted
execution. Use a fresh output directory. The helper rejects private query-truth
fields, reserved contexts and Transfer source/query cell-line overlap. These
checks do not independently audit caller-supplied biological provenance.
The release lacks the ordered 1,809-gene source template, so the helper checks
panel size and dimensions but cannot independently verify that panel's exact
historical order; reuse the original frozen input rather than rebuilding it ad hoc.
Repeat uses measured Rep1 query support. Transfer query support must be built
from legal other-cell-line donors, not the query's observed response. Neither
mode calls DeepSeek or requires a provider key.

## Checks performed and limits

The incoming export's 18 listed payload files were checked against their sizes
and SHA-256 values. All eight selected model files were verified. The Qwen
checkpoint matches the previously recorded Student A checksum, although its
legacy completion marker did not provide the newer per-file checksum mapping.
The restricted PyTorch loader succeeded and all trainable tensors were finite.
Seven generated modules passed syntax/entrypoint inspection and had recorded
assessment/completion checksum matches. Original entire runs were not rehashed.

Software checks exercise adapter reloading after moving a base snapshot and
prediction alignment using a tiny random test fixture. They do not reproduce
biological accuracy. The frozen production base and original expression/query
matrices are not contained in this archive, so original gene-level predictions
have not been independently regenerated during publication.

Historical Student A validation MSE was about 0.02590684 on 325 rows. It was
selected over Student B in that experiment; teacher pretraining overlap was
unverified. Round26's best new Repair Repeat development-confirmation MSE was
about 0.00442300 versus 0.00442612 for its fixed comparator, a small non-robust
difference. Round26 reported `scientific_pass=false`. These tasks/panels and
MSE values are not interchangeable. No new external or clinical validation is
claimed.

## Licensing and attribution

**The Qwen student is State-distilled and remains subject to the Arc Research
Institute State Model Non-Commercial License.** The full license, acceptable-use
policy, State code license, Apache-2.0 text and repository notices accompany the
archive. The repository's MIT code license does not remove model restrictions.
The separate Qwen base has Apache-2.0 terms at its pinned upstream revision.
Other datasets, priors and teacher models retain their respective terms.

State citation: Adduri, A. et al. (2025). *Predicting cellular responses to
perturbation across diverse contexts with State.*
https://doi.org/10.1101/2025.06.26.661135

Read `MODEL_CARD.md`, `licenses/MODEL_LICENSE.md` and THIRD_PARTY_NOTICES.md
before using or redistributing the artifacts.
