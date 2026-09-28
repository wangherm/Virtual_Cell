# Pretrained teachers and a numerical Qwen student

## What is implemented

| Component | Implementation | Real-weight verification |
| --- | --- | --- |
| Qwen3 student | Pinned HF backbone, continuous control tokens, perturbation text, final prediction token, LoRA q/v projections, signed delta head | Run the AutoDL smoke command below |
| State | Native State Transition checkpoint inference on genuine matched control cells; aggregate expression and subtract the prepared baseline | Pending compatible checkpoint, gene-order/scale audit and holdout audit |
| scGPT | Locked official weights, binned inputs, frozen encoder and a conditional response head | Actual checkpoint loaded on CPU; [AutoDL launcher ready](SCGPT_FIRST_RUN.md) |
| scFoundation | Strictly load the official cell encoder checkpoint; freeze, pool actual control-cell embeddings, train a conditional response head | Pending official weights and isolated-environment check |
| Distillation | Same Qwen initialization and row selection; supervised, three individual-teacher arms, joint arm; no fake fallback | Offline test fixtures cover the training/cache/checkpoint path |

The two feature-head teachers are **custom task adaptations**, not reproductions
of the official scGPT perturbation decoder or scFoundation+GEARS result. Freezing
the backbone keeps this first integration small. This does not establish that the
adapted teachers will outperform the old reference models. The old v0.3 pipeline
and prepared data are unchanged.

Local validation on 2026-09-28 loaded the actual pinned Qwen3-0.6B-Base weights
and completed one CPU training epoch on four synthetic training groups and four
validation groups. This checks the real-weight loading/training path, not biology
or Blackwell CUDA performance. Tiny random Qwen fixtures separately test gradients,
interrupted resume, checkpoint reload and three-cache distillation without network.

## First run on AutoDL

If the supervised Qwen smoke has already passed, continue with
**[the scGPT teacher and distillation run](SCGPT_FIRST_RUN.md)**.

If `huggingface.co` is unreachable, use the public mirror recommended in
[AutoDL's networking guide](https://www.autodl.com/docs/network_turbo/):

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

Set these in the terminal that launches the script. The launcher now downloads
six required files into `vcell-work/models/Qwen3-0.6B-Base`, checks their sizes and
SHA256 hashes against `configs/qwen_backbone.lock.json`, then loads only local
files with offline mode enabled. The lock was checked against the official HF
revision's Git/LFS object IDs. Public downloads send no HF token. Verified local
files are reused without a network call; failed/incomplete downloads can be retried.
GPU checks, dependency installation and downloads are now logged from startup.

After a previous network failure, interrupt its retries with Ctrl+C, update the
code and choose a fresh output because the old attempt may have saved a manifest:

```bash
git pull --ff-only
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
bash scripts/run_qwen_autodl.sh --output /root/autodl-tmp/vcell-work/runs/qwen_smoke_02
```

The log remains `logs/qwen_smoke_01.log`. An alternative supported by AutoDL is
`source /etc/network_turbo` in the launching terminal. Mirror reachability still
depends on the server network; this is not a guarantee of connectivity.

In a JupyterLab terminal:

```bash
cd /root/autodl-tmp/virtual_cell
git pull --ff-only
screen -S qwen-smoke
bash scripts/run_qwen_autodl.sh
```

Detach with Ctrl+A then D; reconnect with `screen -r qwen-smoke`. In another terminal:

```bash
tail -n 30 -f /root/autodl-tmp/vcell-work/logs/qwen_smoke_01.log
nvidia-smi
```

Use `screen -r` only if a detached session exists. The smoke script uses the
working `.venv` or creates one with `--system-site-packages`. It does not request a
new CUDA wheel when an adequate torch is already installed. No FlashAttention,
bitsandbytes, Colab or API key is required for Qwen. Public HF downloads need a
working network connection. Cache and temporary files default to the data disk;
allow several GB of free disk for the backbone, dependencies and adapters.

The initial configuration uses actual pretrained Qwen weights, 128 training
groups, 64 validation groups, 2 epochs and batch size 4 with accumulation 4.
Small-sample scores are software diagnostics, not research conclusions. Scaling
statistics use all prepared **training** rows, never validation/test rows. The
summary includes no-change and mean-transfer baselines on the same selected
validation rows; mean-transfer uses the full training set. Smoke comparisons are
therefore not equal-data-budget benchmark comparisons.

Outputs under `/root/autodl-tmp/vcell-work/runs/qwen_smoke_01`:

- `run_manifest.json`: exact backbone/config/code/data/cache fingerprints and row indices.
- `supervised/history.csv`: per-epoch losses, validation delta MSE and elapsed seconds.
- `supervised/best.pt`: selected LoRA/input/output parameters and normalization metadata.
- `supervised/last.pt`: epoch-boundary resume state including optimizer.
- `supervised/validation_predictions.npz`: selected validation rows, with explicit IDs.
- `summary.csv`, `COMPLETE.json`: current comparison and successful completion marker.

The two checkpoints omit the frozen backbone; keep the original HF cache. They
are v0.4 Qwen checkpoints, not compatible with the old `predict-controls` command.
Training loss is scaled MSE; `val_mse_delta` and summary metrics use original
prepared delta units. All training targets are training rows only. There is no
word-vocabulary KL loss and no textual expression-value generation.

To resume after interruption, keep the exact code/config/cache and run:

```bash
bash scripts/run_qwen_autodl.sh --resume
```

An interrupted epoch is replayed from the previous saved epoch. If code or
configuration changed, choose a new output via `--output /absolute/new/path`.
The launcher always logs to `qwen_smoke_01.log`, including when overriding output.

To verify saved-model reload (prediction only; no test scoring):

```bash
export HF_HOME=/root/autodl-tmp/vcell-work/models/huggingface
.venv/bin/python -m vcell qwen-predict \
  --checkpoint /root/autodl-tmp/vcell-work/runs/qwen_smoke_01/supervised/best.pt \
  --data /root/autodl-tmp/vcell-work/prepared/real_min10_01 \
  --output /root/autodl-tmp/vcell-work/runs/qwen_smoke_01/reloaded_predictions.npz
```

This predicts all prepared control queries; matching gene order and perturbation
vocabulary are mandatory. It never reads perturbed outcomes to form predictions.

## Preparing the three real teachers

Use separate upstream environments. Do not install all teachers' historical
dependencies into the working Qwen environment. The native adapters are optional:
the Qwen smoke test does not import any of the teacher packages.

1. Obtain the **official** weights and pinned source checkout. Templates record
   source revisions whose APIs were inspected; those are not proof of checkpoint
   compatibility. Follow each upstream repository's installation instructions in
   its own environment, then install this repository there with `pip install -e
   /root/autodl-tmp/virtual_cell --no-deps` once its base dependencies are present.
2. Copy the appropriate `configs/teacher_*.example.yaml` to `configs/local/`.
   Set absolute paths, verify `sha256sum checkpoint`, and fill the hash. Source
   checkouts must match the configured revision and have no tracked modifications.
3. Copy `configs/foundation_provenance.example.json` for each teacher. Audit its
   perturbation-label training exposure, exclusions, weight license and target
   units. The template starts with `declared_no_holdout_perturbations: false`
   and is deliberately unusable without an audit. Do not change it to true just
   to pass validation. Broad unlabelled pretraining and perturbation supervision
   must be described separately in audit notes.
4. Export using the corresponding teacher environment:

```bash
python -u -m vcell teacher-export --config configs/local/teacher_scgpt.yaml
python -u -m vcell teacher-export --config configs/local/teacher_scfoundation.yaml
python -u -m vcell teacher-export --config configs/local/teacher_state.yaml
```

State writes a prediction cache directly. The other two commands write frozen
control features; they **cannot** be passed to Qwen as prediction caches. Set
the feature/output paths in two copies of `configs/teacher_head.example.yaml`:

```bash
python -u -m vcell teacher-fit-head --config configs/local/scgpt_head.yaml
python -u -m vcell teacher-fit-head --config configs/local/scfoundation_head.yaml
```

The response heads fit training labels only, use validation for early stopping,
then freeze and export predictions. Features are normalized on training rows only.
Each head saves `head.pt`, its history, validation result, and `predictions.npz`
plus a provenance sidecar. Sampled actual control-cell IDs and hashes are retained.
Default feature extraction samples 8 distinct controls per matched batch; it does
not repeat group means as fake cells. Larger control samples are a later ablation.

### State compatibility gate

The inspected public Replogle examples hold out HepG2 or Jurkat individually.
They do not establish exclusion of both in one checkpoint. The old prepared split
trains K562/RPE1, validates HepG2, and tests Jurkat. An incompatible public teacher
must be rejected. Obtain compatible jointly excluded weights or explicitly revise
the experiment's data/validation protocol and regenerate all data/caches. Fine-tuning
does not erase pretraining exposure. No automatic bypass is implemented here.

The State adapter supports gene-space outputs without a batch encoder. Supply
the checkpoint's exact ordered `genes.csv` (column `gene`), its perturbation map,
and documented normalization. Set `expression_space: log1p_full_library_10000`
only when the checkpoint was actually trained on that scale. Both input and output
gene panels must be covered; missing genes stop execution. This may require
preparing a common panel and rerunning baselines. `var_dims.pkl` must not simply
be assumed to contain the correct ordered output gene IDs.

### Teacher sources and differences

- [State code](https://github.com/ArcInstitute/state) and [Replogle weights](https://huggingface.co/arcinstitute/ST-HVG-Replogle).
- [scGPT weights and documentation](https://github.com/bowang-lab/scGPT): obtain `best_model.pt`, `args.json`, `vocab.json` from the official whole-human release. This adapter strictly loads all encoder components and converts fused QKV parameter names to PyTorch MHA names; it rejects partial loads. It uses log1p-normalized control expression and top-expressed genes, an adaptation choice rather than a claim to reproduce every pretraining preprocessing step.
- [scFoundation model instructions](https://github.com/biomap-research/scFoundation/tree/main/model): use the official full `models.ckpt` with its `cell` entry and matching source. A third-party encoder-only repack is not automatically interchangeable. The adapter uses the 19,264-symbol vocabulary and official singlecell/F/f1 cell-embedding preprocessing. Only absent encoder inputs are zero-padded; output predictions are never filled with zeros.
- [GEARS scope](https://github.com/snap-stanford/GEARS#A-note-on-usage): its default route is not designed for cross-cell-type transfer, so it is not used as a drop-in response teacher here.
- [scFoundation weight license](https://github.com/biomap-research/scFoundation/blob/main/MODEL_LICENSE): non-commercial research; code and weight licenses differ.

Official inference performed externally can also use the existing `vcell queries`
and `vcell import-teacher` path. Additionally declare `teacher_family` as `state`,
`scgpt` or `scfoundation`, and `prediction_space: prepared_log1p_delta`. These
fields identify audited predictions, not an automated proof of their source.

## Run the complete comparison after all caches pass

Inspect standalone teacher validation results before student distillation. Copy
and edit `configs/qwen_three_teachers.yaml`, then:

```bash
export HF_HOME=/root/autodl-tmp/vcell-work/models/huggingface
.venv/bin/python -u -m vcell qwen-run --config configs/local/qwen_three_teachers.yaml
```

Five arms run sequentially: supervised, State-only KD, scGPT-only KD,
scFoundation-only KD, and three-teacher KD. All start from the same Qwen and
adapter initialization, training rows and ordering. The default joint target is
the equal mean; `teacher_mix: validation` instead fits nonnegative simplex weights
on the selected validation rows only. Either choice must be disclosed and frozen
before final testing. A teacher may receive zero fitted weight. The loss is
supervised scaled MSE plus `kd_weight * MSE(student, frozen teacher target)`.

Use full training and validation rows for a pilot, then multiple seeds for
research conclusions. Qwen 0.6B can be larger than the biological teachers; do not
claim compression from its family name. Parameter counts and peak GPU allocation
are recorded. The first release intentionally keeps final test evaluation closed.
