# First real scGPT teacher and Qwen distillation

There is **one student architecture**, Qwen3-0.6B-Base with LoRA and a numerical
response head. Each experiment arm trains its own copy from the same initial
weights. The supervised and scGPT arms run sequentially. Three teachers will feed
one student in the joint arm; they do not require three student models.

## Run on the existing AutoDL instance

The prepared real-data pilot and working Qwen environment must already exist.
In a JupyterLab terminal:

```bash
cd /root/autodl-tmp/virtual_cell
git pull --ff-only
screen -S scgpt-pilot
bash scripts/run_scgpt_autodl.sh
```

The script prints its timestamped log path immediately. It reuses the working
CUDA PyTorch and Qwen dependencies; it does not install scGPT's legacy dependency
stack. It downloads three locked files (about **207 MB**) from the public mirror,
checks SHA256, extracts control-cell embeddings, fits the response head for up
to 20 epochs, then runs Qwen supervised and scGPT-distilled smoke arms. The
existing local Qwen snapshot is verified and reused. No API token is required.
Set `HF_ENDPOINT` to use another endpoint. All large files stay on the data disk.

Detach with Ctrl+A then D. To watch from another terminal:

```bash
log=$(ls -t /root/autodl-tmp/vcell-work/logs/scgpt_pipeline_*.log | head -n 1)
tail -n 40 -f "$log"
```

Progress markers are `SCGPT SNAPSHOT VERIFIED`, `scgpt control_group=...`,
`scgpt head epoch=...`, `SCGPT TEACHER COMPLETE`, `supervised epoch=...`,
`scgpt epoch=...`, `QWEN RUN COMPLETE`, and `SCGPT PIPELINE COMPLETE`.
The teacher first needs to load the checkpoint and read raw controls; this
stage is not a Qwen epoch. `torch_dtype` deprecation messages are not failures.

Default outputs:

```text
vcell-work/models/scGPT-human/                   # Pinned weights, args, vocabulary
vcell-work/teachers/scgpt_01/
  plan.json                                    # Code, data, checkpoint identity
  provenance.json                              # Evidence and limitations
  features.npz + features.json                  # Real frozen control features
  head/head.pt                                 # Trained response head
  head/predictions.npz + head/predictions.json   # Aligned teacher delta cache
  teacher_summary.csv                          # All validation groups only
  qwen_config.json                             # Effective comparison config
vcell-work/runs/qwen_scgpt_01/
  supervised/                                  # Independent baseline student
  scgpt/                                       # Independently distilled student
  summary.csv                                  # Same selected validation groups
  COMPLETE.json
```

The default student smoke uses 128 training groups, 64 validation groups and two
epochs per arm. The teacher head and mean-transfer baseline use all training
groups. This is a workflow check, **not an equal-data-budget result**. After it
passes, use all groups for the student comparison:

```bash
bash scripts/run_scgpt_autodl.sh --name scgpt_full_01 --full-run
```

This uses all training and validation groups with up to ten student epochs.
It creates a separate experiment. Validation is used for teacher/student epoch
selection. Test expression is never used for fitting or reported in the scores.
Control observations from held-out contexts are allowed by this conditional
prediction protocol; their perturbation labels are not used for training.

Use `--teacher-only` to stop after the teacher and generated Qwen config. With
unchanged code/settings, `--resume` reuses verified completed feature/head stages
and resumes Qwen at epoch boundaries. An interrupted feature write or partially
trained teacher head is not resumable: choose a fresh `--name` (weights are
reused). Do not delete raw data or old results to retry.

## Exact adapter and validation boundary

The weight source is the author lab's
[wanglab/scGPT-human](https://huggingface.co/wanglab/scGPT-human/tree/a24c237737a40f3720f75abb555489e9fe753be6),
revision `a24c237737a40f3720f75abb555489e9fe753be6`.
`configs/scgpt_backbone.lock.json` pins all three files. The checkpoint SHA256 is
`6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922`.
The lock was checked against official Git/LFS object IDs. Mirror contents must
match it exactly before loading. Public downloads send no token.

The inference module follows the upstream gene/value encoders and post-norm
transformer. All **50,278,400 encoder parameters** load strictly; only the fused
QKV parameter names change for PyTorch attention. Task-decoder weights are
deliberately unused. This removes the need for old torchtext, FlashAttention,
scvi-tools and orbax installations. See `THIRD_PARTY_NOTICES.md` for attribution.

For each dataset/batch, sample eight actual control cells without replacement.
Normalize each cell using its full measured count library and log1p, quantile-bin
positive expression into 50 nonzero bins (zero is bin 0), then filter to the
scGPT vocabulary. Ties use the upstream random spreading rule with an explicit
seed. Uniformly sample at most 1,200 nonzero genes per cell. No CLS token is
inserted, matching the locked pretraining arguments. Mean-pool token embeddings
and then control cells. The conditional response head receives the pooled
embedding and perturbation identity, and learns the prepared gene-delta task.

This is **a custom frozen scGPT encoder plus response head**, not the official
scGPT perturbation decoder and not a claim to reproduce its published accuracy.
The measured gene panel is only about 8–10k genes, so input coverage differs from
the pretraining corpus. Whether these representations improve predictions is an
experimental question.

Local checks load the real checkpoint and produce finite 512-dimensional control
embeddings on CPU. Synthetic tests cover fused-QKV loading, missing-weight and
metadata rejection, deterministic binning, and raw-control extraction through
head training and cache validation. A separate integration check used the actual
scGPT and Qwen weights, synthetic control counts, a two-epoch response head, and
one epoch each of supervised and distilled Qwen (four training/validation groups).
Both arms completed on CPU; the 45-test local suite passed. These checks do not
measure biological accuracy or replace the server CUDA run.

## Provenance and the other two teachers

The [scGPT paper's data availability statement](https://www.nature.com/articles/s41592-024-02201-0)
identifies CELLxGENE Census 2023-05-15 for pretraining. The pinned arguments identify
a May 2023 whole-human training run. The
[GSE264667 record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667)
lists submission on 2024-04-23 and public release on 2024-05-05 for HepG2/Jurkat.
These published records support exclusion of this held-out perturbation dataset
from the reported pretraining corpus. This is an inference from source records,
**not an independent row-level pretraining audit**; unpublished exposure cannot
be excluded. The cache records this evidence level rather than claiming proof.
Automatic provenance is limited to the existing K562/RPE1 -> HepG2/Jurkat split
and the named GSE264667 inputs. Renaming other data is not a valid audit.

| Teacher | Current status |
| --- | --- |
| scGPT | Locked official weights load locally; one-command server teacher and Qwen comparison ready |
| State | Native adapter exists; requires a checkpoint excluding both held-out contexts and a compatible gene panel/expression scale |
| scFoundation | Native feature adapter exists; official MMF cell-checkpoint loading and provenance still need real-weight verification |

The current command runs **one real teacher**, not the joint three-teacher arm.
The latter remains gated on all three valid caches; no unverified or random
teacher is substituted. State checkpoints that held out only one of HepG2 and
Jurkat are insufficient for the current joint holdout protocol.
