# Read-only server privacy audit before shutdown

For a separate allowlisted cleanup tool after reviewing the report, see
[Server credential cleanup](SERVER_PRIVACY_CLEANUP.md). The audit itself never
deletes anything.

`scripts/server_privacy_audit.py` is a standalone Python 3.10+ standard-library
tool. It inventories potential credentials and private metadata on your own
server. It does not delete, revoke, upload, execute configuration files, inspect
Git objects, or validate a credential online. Run it on the server, not against
your workstation's home directory.

## Run on AutoDL

From an updated checkout:

```bash
cd /root/autodl-tmp/virtual_cell
umask 077
python3 -u scripts/server_privacy_audit.py --output /root/private-audit-final
cat /root/private-audit-final/SUMMARY.txt
```

If updating Git is inconvenient, upload only `server_privacy_audit.py` to
`/root/autodl-tmp/` using JupyterLab and run:

```bash
umask 077
python3 -u /root/autodl-tmp/server_privacy_audit.py \
  --output /root/private-audit-final
```

The output directory must be new. For a second scan use a new name, such as
`/root/private-audit-final-02`. No pip installation, GPU, API key or network
connection is needed. Do not place reports in GitHub or a public model release.

Progress and detailed findings:

```bash
cat /root/private-audit-final/progress.json
cat /root/private-audit-final/SUMMARY.txt
# findings.json is a private machine-readable report; no values are included.
```

On Linux the report directory has mode 0700 and report files mode 0600. Locations,
categories, severity and line positions are reported; credential values and
source-line previews are not. Recognized credential patterns in filenames are
redacted as well. Reports can still reveal private directory structure and
process identifiers, so retain them privately.

## What is checked

Default roots are `/root`, `/etc`, `/tmp`, `/var/tmp` and `/var/log`. The normal
`/root/autodl-tmp` data disk is included through `/root`; directory symlinks are
not followed. If data were moved behind a symlink, supply the actual mounted
directory separately using another `--root` argument.

The scanner checks current environment variables and best-effort snapshots of
same-UID Linux process environments/arguments. It does not print command lines
or environment values. A stopped process has no inspectable environment, and
processes owned by other UIDs are excluded.

File checks include GitHub/Hugging Face credentials, cloud access files, SSH
private keys, private-key PEM markers, `.env` files, credential-bearing URLs,
literal password/token assignments, shell history, notebooks and common
configuration/log files. Jupyter and remote-editor runtime files are considered.
Public SSH `authorized_keys` and `known_hosts` are reported as metadata, not as
private keys. Email-like patterns are INFO findings and can be public authorship
notices. Default SSH host keys are server keys, not your personal login keys.

Severity is a review aid, not proof that a value is valid or compromised:
Counts refer to findings at locations, not to distinct valid credentials; one
token may be present in multiple processes or files.

- HIGH: credential material patterns, a nonempty credential environment variable,
  or a recognized credential/private-key store. A store can contain expired data.
- MEDIUM: a possible secret-bearing configuration, history, database, agent
  pointer or permissions issue. File presence alone does not prove a secret.
- INFO: personal/connection metadata and email-like strings, including potentially
  public or synthetic examples.

Review locations locally. Do not paste actual `.env`, history, private-key or
credential-store contents into chat. If discussing results, share counts and
categories first; review any paths for personal identifiers before sharing.

## Coverage limits

This is a bounded inventory, not a guarantee that a disk contains no private data.
It skips virtual `/proc`, `/sys`, `/dev` file trees, symlinks, Git object stores,
package environments and selected caches. `/proc` is read only through the
dedicated current-UID process snapshot.

Model checkpoints, raw matrices, archives, encrypted keyrings and database
contents are not inspected. History/credential database filenames are flagged,
but encrypted or embedded secrets may remain undetected. Shell history still
held only in memory is not covered. A program can also change a file during a
scan. This tool does not claim to find all personal, health or biological data.

Default limits are 16 MiB per complete text file, 1 GiB total reads and 100,000 files.
Larger text files get bounded head/tail inspection; their middle is not scanned.
Tail line numbers are relative to that segment. Limits, read failures and
partial-file samples appear in `findings.json`. Common token-usage counters and
obvious placeholders are ignored, but false positives and false negatives remain
possible. No match does not establish that a token was never present.

To include software dependency trees or raise the limits, use a fresh report:

```bash
python3 -u scripts/server_privacy_audit.py \
  --include-dependencies --max-file-mib 64 --max-total-mib 4096 \
  --max-files 300000 --output /root/private-audit-expanded
```

Use repeated `--root /absolute/path` to choose specific directories. Supplying
any `--root` replaces the default root list. `--no-env` and `--no-processes` disable
the corresponding runtime snapshots; they do not remove credentials.

## Closing the server

After reviewing the findings, revoke or rotate credentials that were used on the
server when appropriate. Clearing an environment variable or deleting a local
file does not invalidate a still-active token. Revocation can affect other jobs
using the same credential, so identify the account/key before doing it.

Keep private model/data backups you actually need. If you preserve a complete
disk image, it may preserve credentials too. Do not delete SSH login access
while it is still needed to finish the audit or backup. Once backups are checked,
use the provider controls to destroy unwanted instance/data-disk copies; merely
stopping an instance does not itself establish that its disks were erased.
The script performs none of these cleanup or account actions automatically.
