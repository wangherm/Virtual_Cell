#!/usr/bin/env python3
"""Preview or remove allowlisted Linux server credentials; preserve research files."""
import argparse
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile


# These are authentication stores or histories, never model/data directories.
HOME_FILES = (
    '.bash_history', '.zsh_history', '.python_history', '.lesshst',
    '.local/share/fish/fish_history', '.git-credentials', '.netrc',
    '.config/gh/hosts.yml', '.cache/huggingface/token',
    '.cache/huggingface/stored_tokens', '.huggingface/token',
    '.huggingface/stored_tokens', '.codex/auth.json', '.claude/.credentials.json',
    '.aws/credentials', '.azure/accessTokens.json', '.azure/msal_token_cache.json',
    '.config/gcloud/application_default_credentials.json',
    '.config/gcloud/credentials.db', '.config/gcloud/credentials.db-wal',
    '.config/gcloud/credentials.db-shm', '.config/gcloud/access_tokens.db',
    '.config/gcloud/access_tokens.db-wal', '.config/gcloud/access_tokens.db-shm',
)
WORK_FILES = (
    'models/huggingface/token', 'models/huggingface/stored_tokens',
    'cache/huggingface/token', 'cache/huggingface/stored_tokens',
)
SESSION_GLOBS = (
    '.local/share/jupyter/runtime/jpserver-*.json',
    '.local/share/jupyter/runtime/nbserver-*.json',
    '.config/Code/agent-host/local-endpoint/entries/*.json',
)
SESSION_FILES = (
    '.local/share/jupyter/runtime/jupyter_cookie_secret',
    '.local/share/jupyter/runtime/notebook_cookie_secret',
    '.vscode-server/.connection-token',
)
# Redact the whole URL query: download logs can contain presigned AWS credentials.
# Do not treat biological prose containing "Basic ..." as an authentication header.
QUERY_URL = re.compile(rb'https?://[^\s<>\x00"\x27?]+\?[^\s<>\x00"\x27]+')
PASSWORD_URL = re.compile(rb'(?:https?|ssh|ftp)://[^\s/@:]+:[^\s/@]+@[^\s<>\x00"\x27]+')


def safe_file(path, anchor):
    """Reject symlinks in every path component and anything outside its anchor."""
    path, anchor = Path(path).absolute(), Path(anchor).absolute()
    try:
        path.relative_to(anchor)
        for item in (path, *path.parents):
            if item.is_symlink():
                return False
        info = path.lstat()
    except (ValueError, OSError):
        return False
    return stat.S_ISREG(info.st_mode) and info.st_nlink == 1


def targets(home, work, sessions=False):
    for name in HOME_FILES:
        yield home / name, home, 'credential_or_history'
    for name in WORK_FILES:
        yield work / name, work, 'credential_or_history'
    if sessions:
        for name in SESSION_FILES:
            yield home / name, home, 'session_credential'
        for pattern in SESSION_GLOBS:
            for path in sorted(home.glob(pattern)):
                yield path, home, 'session_credential'


def clean_download_log(path, work, apply=False):
    """Preserve log text while removing URL passwords and signed query strings."""
    if not safe_file(path, work):
        return {'category': 'download_log', 'status': 'absent_or_unsafe'}
    before = path.stat()
    if before.st_size > 256 * 1024 * 1024:
        return {'category': 'download_log', 'status': 'skipped_over_256_mib'}
    count, temp = 0, None
    try:
        # No secret backup is created. Temp files use 0600 and are unlinked on failure.
        output = None
        if apply:
            fd, name = tempfile.mkstemp(prefix='.privacy-redact-', dir=path.parent)
            temp = Path(name)
            output = os.fdopen(fd, 'wb')
        try:
            flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
            with os.fdopen(os.open(path, flags), 'rb') as source:
                opened = os.fstat(source.fileno())
                if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    raise RuntimeError('Log changed before reading')
                for line in source:
                    line, n = PASSWORD_URL.subn(b'[REDACTED_CREDENTIAL_URL]', line)
                    line, m = QUERY_URL.subn(b'[REDACTED_QUERY_URL]', line)
                    count += n + m
                    if output:
                        output.write(line)
            if output:
                output.flush()
                os.fsync(output.fileno())
        finally:
            if output:
                output.close()
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size) != (
                after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size) or not safe_file(path, work):
            raise RuntimeError('Log changed during reading; no replacement performed')
        if apply and count:
            os.replace(temp, path)
            temp = None
        return {'category': 'download_log', 'status': 'redacted' if apply and count else 'preview' if count else 'unchanged', 'url_occurrences': count}
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def run(home, work, apply=False, sessions=False, redact_log=False):
    records = []
    for path, anchor, category in targets(home, work, sessions):
        if not path.exists() and not path.is_symlink():
            continue
        row = {'category': category, 'location': str(path)}
        try:
            if not safe_file(path, anchor):
                row['status'] = 'skipped_unsafe_path'
            elif apply:
                path.unlink()
                row['status'] = 'removed'
            else:
                row['status'] = 'would_remove'
        except OSError as error:
            row.update(status='failed', error_type=type(error).__name__)
        records.append(row)
    if redact_log:
        path = work / 'logs/download_raw.log'
        try:
            row = clean_download_log(path, work, apply)
        except (OSError, RuntimeError) as error:
            row = {'category': 'download_log', 'status': 'failed', 'error_type': type(error).__name__}
        row['location'] = str(path)
        records.append(row)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Delete allowlisted credential/history files; default is preview only')
    parser.add_argument('--include-session-tokens', action='store_true', help='Also remove local Jupyter/VS Code runtime credentials; may disrupt login')
    parser.add_argument('--redact-download-log', action='store_true', help='Remove URL queries/passwords from vcell-work/logs/download_raw.log')
    args = parser.parse_args()
    # Fixed server paths prevent accidentally applying the tool to a local workstation.
    if sys.platform != 'linux' or os.geteuid() != 0 or Path.home() != Path('/root'):
        parser.error('Run only on the intended Linux root server; no workstation cleanup is supported')
    os.umask(0o077)
    records = run(Path('/root'), Path('/root/autodl-tmp/vcell-work'), args.apply,
                  args.include_session_tokens, args.redact_download_log)
    print(json.dumps({'mode': 'APPLY' if args.apply else 'PREVIEW', 'actions': records,
                      'limitations': [
                          'No recursive deletion, process termination, account changes, or remote token revocation',
                          'SSH keys/access, AutoDL platform variables, source code, model weights, data, and research results are retained',
                          'Existing processes and shells may retain/recreate secrets or history',
                          'No shell startup edits, arbitrary .env cleanup, Git history, archive, Trash, snapshot, or secure-erasure claim',
                          'Use provider-side revocation and instance/disk disposal after verifying backups',
                      ]}, indent=2))
    if any(row['status'] in ('failed', 'skipped_unsafe_path', 'skipped_over_256_mib') for row in records):
        return 2
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('CLEANUP FAILED: ' + type(error).__name__ + '; details withheld to avoid credential disclosure.', file=sys.stderr)
        raise SystemExit(2)
