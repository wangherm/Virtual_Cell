# Server credential cleanup before retirement

`scripts/server_privacy_cleanup.py` is a standalone Python 3 standard-library
tool for the project's Linux root server. It defaults to a preview. It does not
read an audit report as deletion instructions: reported candidates can be source
examples, public notices, tokenizer configuration, or temporary download URLs.

## Before starting

Verify that research data, model releases, source code and required results have
been backed up. Close other shell/editor sessions when practical: a running
shell can write its old history back after the file is removed. Finish any
downloads before redacting the download log. Keep the final working connection
until backup and cleanup verification are finished.

The tool preserves model weights, raw/prepared data, research runs, code, licenses,
SSH private/public keys and SSH server configuration. It does not stop processes,
remove packages, change accounts, or alter AutoDL platform variables. It does
not delete arbitrary files containing the word "token" or edit Python source.

## Upload and preview

Upload `server_privacy_cleanup.py` to `/root/autodl-tmp/`, then run:

```bash
umask 077
python3 /root/autodl-tmp/server_privacy_cleanup.py --redact-download-log
```

An updated checkout can instead use `scripts/server_privacy_cleanup.py`. No
dependencies, API key, GPU or network connection are required. Output contains
fixed-path locations and action counts, never file contents or credential values.

## Apply ordinary credential cleanup

```bash
python3 /root/autodl-tmp/server_privacy_cleanup.py \
  --apply --redact-download-log
```

Only existing, regular, single-link files in the built-in allowlist are removed.
Every symlink path component is rejected. There is no recursive directory deletion.
The allowlist includes shell/Python history and known GitHub CLI, Git HTTPS,
Hugging Face, AWS, Azure, Google Cloud, Codex and Claude authentication stores.
Deleting a shared store logs out all locally stored accounts in that file. Token
caches at this project's known Hugging Face paths are included; weights are not.
Missing entries are harmless. Unsafe paths and operational errors return exit
code 2 and need local review; exit code 0 means the requested operations finished,
not that the entire server is free of secrets.

With `--redact-download-log`, only
`/root/autodl-tmp/vcell-work/logs/download_raw.log` is rewritten. HTTP URL query
strings and password-bearing URLs are replaced with markers while other log
text is retained. A private temporary file is atomically installed after checking
that the log did not change during the read. No secret backup is made. Logs over
256 MiB are skipped. Old blocks, snapshots and copies are not securely erased.

## Optional: remove local session credentials last

The focused audit also found Jupyter/VS Code runtime credentials. The following
step may disrupt reconnecting through those tools; run it from your final SSH
terminal after finishing Jupyter/editor work:

```bash
python3 /root/autodl-tmp/server_privacy_cleanup.py \
  --apply --redact-download-log --include-session-tokens
```

This additionally removes allowlisted Jupyter server JSON/cookie files and VS
Code agent endpoint JSON/token files. It does not stop existing servers or revoke
their in-memory tokens. Live services can regenerate the files. AutoDL's
`AutodlAutoPanelToken` is left to the provider-managed instance lifecycle; it is
not treated as a personal DeepSeek API key.

## Clear the current Bash session separately

A Python child process cannot erase the parent shell's variables or in-memory
history. After applying cleanup, run in the current Bash terminal:

```bash
history -c
unset HISTFILE
unset DEEPSEEK_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY \
  HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN GITHUB_TOKEN GH_TOKEN \
  AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
```

Other running shells/processes may still retain copies. Do not print their
environment or paste credential file contents into chat. Re-run the privacy audit
if retaining the instance, but expect provider and active-session findings until
the services stop.

## Retire credentials and storage

Local file deletion does not revoke remote credentials. Revoke or rotate personal
credentials used on this server at their issuing service after checking whether
other workloads use the same key. A shutdown stops running processes but does
not erase persistent disks. Once backups are verified, dispose of unwanted
instance, disk, image and snapshot copies using provider controls. Keep necessary
backups private; do not publish whole home-directory backups or audit reports.

This tool does not scrub arbitrary `.env` files, startup files, custom credential
stores, credential helpers, Git history, archives, databases, Trash, disk slack or
remote backups. It makes no complete-removal or secure-erasure claim.
