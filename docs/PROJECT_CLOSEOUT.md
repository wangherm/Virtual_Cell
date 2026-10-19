# Server closeout and research model delivery

The project is paused and Round27 is unfinished. This workflow inventories the
server without training, API calls, cleanup or modification of existing runs.
All added scripts, comments, reports and model cards are in English.

## 1. Update the source checkout

```bash
cd /root/autodl-tmp/virtual_cell &&
git -c http.version=HTTP/1.1 pull --ff-only
```

If GitHub is unreachable, upload the supplied closeout `.bundle` and use its
actual absolute path:

```bash
cd /root/autodl-tmp/virtual_cell &&
git bundle verify /root/autodl-tmp/virtual-cell-closeout.bundle &&
git fetch /root/autodl-tmp/virtual-cell-closeout.bundle main &&
git switch main &&
git merge --ff-only FETCH_HEAD
```

Neither route resets local modifications. Resolve an update failure before
starting the inventory. No DeepSeek key is needed.

## 2. Check activity and collect progress

```bash
ps -eo pid,etime,%cpu,args | grep -E '[r]un_round27.py|[r]un_shinka_round27.py'
screen -ls
ls -d /root/autodl-tmp/vcell-work/runs/round27* 2>/dev/null
```

These commands inspect activity; they do not stop it. A screen session or its
absence alone is not proof that the experiment completed. If an active Round27
needs a graceful pause, use its existing controller:

```bash
.venv/bin/python scripts/round27_agent.py pause --name round27_agent_01
.venv/bin/python scripts/round27_status.py --name round27_agent_01
```

Pause takes effect at the controller's committed boundary, not immediately.
Wait for the process to exit before a consistent backup. The closeout command
itself is read-only and can inventory unfinished runs; such a snapshot may
reflect an evolving directory.

```bash
python3 -u scripts/project_closeout.py \
  --work-dir /root/autodl-tmp/vcell-work \
  --name project_closeout_01 \
  --unfinished-run round27_agent_01
```

Use the actual run name. Repeat `--unfinished-run` for every unfinished campaign.
Even a conflicting COMPLETE marker cannot override this explicit declaration.
Names are never reused or overwritten; use `project_closeout_02` on a rerun.

View progress from another terminal:

```bash
cat /root/autodl-tmp/vcell-work/closeout/project_closeout_01/progress.json
cat /root/autodl-tmp/vcell-work/closeout/project_closeout_01/REPORT.md
```

Download this package through JupyterLab:

```text
/root/autodl-tmp/vcell-work/closeout/project_closeout_01/project_closeout.tar.gz
```

Default collection lists filenames, sizes, declared stages and selected summary
flags. It does not hash all weights or read matrices. It does not automatically
copy logs, provider responses, local configs, credentials or raw metadata dumps.
Completion integrity is labeled `not_fully_verified`, rather than presented as
independently reproduced science. Old runs without COMPLETE markers are unknown.
Round27 remains explicitly unfinished.

## 3. Export an explicitly selected completed research model

Review `models.csv` before choosing. It lists checkpoints and valid assessed
Python methods, including incomplete runs for private preservation. It does
not rank or automatically promote them.

For the already completed Round26 campaign, you can export its existing
**source-only frozen algorithm shortlist** without manually choosing a path:

```bash
python3 -u scripts/project_closeout.py \
  --name project_models_01 \
  --unfinished-run round27_agent_01 \
  --frozen-run round26_critic_02
```

Use the actual completed Round26 name if different. This preserves all recorded
source finalists, including diagnostic Pareto methods; it does not select a
confirmation winner or claim that these are qualified deployment models.
Recorded module hashes are checked. Original absolute paths must still resolve.
Empty or incompatible shortlists fail explicitly rather than guessing a model.

Run a new snapshot with one or more paths relative to `work/runs`:

```bash
# Substitute a real completed-run path from models.csv; this is a template.
python3 -u scripts/project_closeout.py \
  --name project_models_explicit_01 \
  --unfinished-run round27_agent_01 \
  --model 'COMPLETED_RUN/path/from/models.csv/best.pt'
```

For an evolved algorithm, select its assessed `module.py` instead of `best.pt`.
Repeat `--model` to include multiple files. The tool rejects private/worker paths,
symbolic paths, unfinished/running runs and mismatched recorded file checksums.
Each selected file is hashed and checked for changes during copying. The
default per-model cap is 1,024 MiB; explicitly change `--max-model-mib` if necessary.
Only selected-file integrity is verified, not the whole original campaign.

A Python method may require original public source arrays and refitting; a
checkpoint may require separately licensed bases or adapters. The package is
an archival research delivery, **not a new universal inference interface**.
Read MODEL_CARD.md and retain original configuration, ordered gene panel,
preprocessing, parent run and source templates privately. Do not deserialize
an unknown pickle merely to inspect it.

## 4. Publish source versus model assets

The repository source and English README can be pushed independently of the
server. Actual server models can only be uploaded after their bytes are exported.
Do not commit raw data, caches or large weights into Git history.
GitHub documents release assets as a way to distribute binaries separately from
the Git repository: [GitHub large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

After reviewing the selected package and third-party redistribution terms,
upload it as a **draft research release** using an authenticated GitHub CLI.
The following command creates a draft, not a claim of a validated final model:

```bash
cd /root/autodl-tmp/vcell-work/closeout/project_models_01 &&
sha256sum -c SHA256SUMS &&
gh auth status &&
gh release create research-closeout-2026-10-19 \
  project_closeout.tar.gz SHA256SUMS MODEL_CARD.md \
  --repo wangherm/Virtual_Cell --target main --draft \
  --title 'Virtual Cell research closeout (Round27 unfinished)' \
  --notes-file REPORT.md
```

If `gh` is unavailable or the server cannot reach GitHub, download the package
and these companion files and upload from an authenticated local machine.
Do not paste access tokens into chat or save them in a command file.
Use a new release tag if that tag already exists; this command does not overwrite
an existing release. Review generated code, checkpoint contents and model/data
terms before making the draft public. No uploader is invoked by the inventory.

## 5. Preserve a real private backup

The review package is not a backup. Retain:

- `vcell-work/runs/`, including incomplete Round27, parent campaigns, native
  Shinka databases, proposal archives, memory and action journals.
- `prepared/`, `raw/`, `models/`, source templates, ordered panels and original
  data/annotation provenance.
- `agent_control/`, `agent_memory/`, predictions, private configs and logs.
- The repository and `.git`, source bundle, recorded dependency versions and
  environments needed to replay exact historical protocols.

These files can contain private data or credentials and belong in a private
backup. They are not automatically included in a public release. Preserve the
original mount paths, since historical provenance contains absolute paths.
Use the cloud provider's disk snapshot or a verified private copy after writers
have stopped. Verify backup restoration before deleting any source data.
This tool deliberately performs no deletion, migration or process termination.
