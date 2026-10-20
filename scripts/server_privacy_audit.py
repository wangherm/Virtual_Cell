"""Read-only, offline server credential inventory; never print credential values.

Only the standard library is used. No files are sourced, no archives or pickles
are loaded, no credentials are tested online, and nothing is deleted or revoked.
Reports are private operational records and must not be committed to GitHub.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat
import sys


PATTERNS = {
    'private_key_material': re.compile(r'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----'),
    'github_token_pattern': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,255}|github_pat_[A-Za-z0-9_]{30,255})\b'),
    'huggingface_token_pattern': re.compile(r'\bhf_[A-Za-z0-9]{20,255}\b'),
    'api_key_pattern': re.compile(r'\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,255}\b'),
    'aws_access_key_pattern': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'slack_token_pattern': re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{20,255}\b'),
    'bearer_credential_pattern': re.compile(r'\bBearer[ \t]+([A-Za-z0-9._~+/=-]{12,4096})', re.I),
    'basic_auth_pattern': re.compile(r'\bBasic[ \t]+[A-Za-z0-9+/]{12,4096}={0,2}', re.I),
    'url_with_password': re.compile(r'\b(?:https?|ssh|mysql|postgres(?:ql)?|mongodb(?:\+srv)?|redis)://[^\s/:@]+:[^\s/@]+@', re.I),
    'email_like_personal_metadata': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
}
ASSIGNMENT = re.compile(r'''(?<![A-Za-z0-9_.-])["']?([A-Za-z_][A-Za-z0-9_.-]{0,100})["']?\s*(?:=|:)\s*["']?([^\s"'`,;\}\)\]]{8,4096})''')
CLI_SECRET = re.compile(r'''--(?:api[-_]key|token|password|passwd|client[-_]secret)(?:=|\s+)["']?([^\s"']{8,4096})''', re.I)
SECRET_NAME = re.compile(r'(?:^|_)(?:API_KEY|ACCESS_TOKEN|REFRESH_TOKEN|AUTH_TOKEN|OAUTH_TOKEN|TOKEN|PASSWORD|PASSWD|CLIENT_SECRET|PRIVATE_KEY|SECRET_ACCESS_KEY|ACCOUNT_KEY|SECRET)$')
PLACEHOLDER = re.compile(r'(?i)(?:example|placeholder|redacted|dummy|test.only|test.key|your[_-]|insert[_-]|replace[_-]|changeme|not.a.real)')
DEPENDENCIES = {'.venv', 'venv', 'envs', 'node_modules', 'site-packages', '__pycache__', 'miniconda3', 'anaconda3', 'pkgs'}
BINARY = {'.pt', '.pth', '.ckpt', '.safetensors', '.h5ad', '.h5', '.hdf5', '.loom', '.npz', '.npy',
          '.rds', '.rda', '.bam', '.fastq', '.fq', '.gz', '.zip', '.tar', '.tgz', '.bz2', '.xz',
          '.7z', '.bundle', '.pack', '.idx', '.so', '.dll', '.exe', '.pyc', '.png', '.jpg', '.pdf',
          '.parquet', '.feather', '.sqlite', '.sqlite3', '.db', '.vscdb', '.keyring', '.kdbx'}
STORES = {
    '/.git-credentials': ('credential_store', 'HIGH'),
    '/.netrc': ('credential_store', 'HIGH'),
    '/.config/gh/hosts.yml': ('github_cli_credential_store', 'HIGH'),
    '/.aws/credentials': ('cloud_credential_store', 'HIGH'),
    '/etc/shadow': ('system_password_hash_store', 'HIGH'),
    '/etc/gshadow': ('system_password_hash_store', 'HIGH'),
    '/.codex/auth.json': ('coding_assistant_credential_store', 'HIGH'),
    '/.claude/.credentials.json': ('coding_assistant_credential_store', 'HIGH'),
    '/.config/rclone/rclone.conf': ('cloud_credential_store', 'HIGH'),
    '/.azure/accessTokens.json': ('cloud_credential_store', 'HIGH'),
    '/.azure/msal_token_cache.json': ('cloud_credential_store', 'HIGH'),
    '/.config/gcloud/application_default_credentials.json': ('cloud_credential_store', 'HIGH'),
    '/.config/gcloud/credentials.db': ('cloud_credential_database', 'HIGH'),
    '/.config/gcloud/access_tokens.db': ('cloud_credential_database', 'HIGH'),
    '/.cache/huggingface/token': ('huggingface_credential_store', 'HIGH'),
    '/.huggingface/token': ('huggingface_credential_store', 'HIGH'),
    '/.cache/huggingface/stored_tokens': ('huggingface_credential_store', 'HIGH'),
    '/jupyter/runtime/notebook_cookie_secret': ('jupyter_cookie_secret', 'HIGH'),
    '/jupyter/runtime/jupyter_cookie_secret': ('jupyter_cookie_secret', 'HIGH'),
    '/.kube/config': ('cluster_access_config', 'MEDIUM'),
    '/.docker/config.json': ('container_registry_config', 'MEDIUM'),
    '/.npmrc': ('package_registry_config', 'MEDIUM'),
    '/.pypirc': ('package_registry_config', 'MEDIUM'),
    '/.gitconfig': ('personal_git_configuration', 'INFO'),
    '/.ssh/authorized_keys': ('ssh_public_access_list_not_private_key', 'INFO'),
    '/.ssh/known_hosts': ('ssh_host_metadata_not_private_key', 'INFO'),
    '/.ssh/config': ('ssh_connection_metadata', 'INFO'),
    '/.bash_history': ('shell_history_may_contain_secrets', 'MEDIUM'),
    '/.zsh_history': ('shell_history_may_contain_secrets', 'MEDIUM'),
    '/.python_history': ('interactive_history_may_contain_secrets', 'MEDIUM'),
    '/fish/fish_history': ('shell_history_may_contain_secrets', 'MEDIUM'),
}


def credential_name(name):
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    normalized = re.sub(r'[^A-Z0-9]+', '_', name.upper()).strip('_')
    return (bool(SECRET_NAME.search(normalized)) or normalized == 'AUTH'
            or bool(re.search(r'(?:^|_)(?:APIKEY|ACCESSTOKEN|REFRESHTOKEN|CLIENTSECRET|ACCOUNTKEY|PRIVATEKEY)$', normalized)))


def plausible(value):
    if len(value) < 8 or PLACEHOLDER.search(value):
        return False
    if value.startswith(('$', '<', '{', 'os.', 'args.', 'self.', 'config.', 'process.', 'environ.', 'getenv(')):
        return False
    return value.lower() not in {'none', 'null', 'undefined', 'password', 'api_key', 'secret', 'false'}


def redacted(value):
    value = str(value).replace('\n', '[newline]').replace('\r', '[newline]')
    for expression in PATTERNS.values():
        value = expression.sub('[REDACTED]', value)
    value = CLI_SECRET.sub('--credential [REDACTED]', value)
    return ASSIGNMENT.sub(lambda m: m.group(1) + '=[REDACTED]' if credential_name(m.group(1)) else m.group(0), value)


def private_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True)
        stream.write('\n')
    temporary.replace(path)


class Audit:
    def __init__(self, output, max_file_mib=16, max_total_mib=1024, max_files=100000):
        self.output = Path(output).resolve()
        if self.output.exists():
            raise FileExistsError('Use a new private report directory')
        self.output.mkdir(parents=True, mode=0o700)
        os.chmod(self.output, 0o700)
        self.max_file = max_file_mib * 1024**2
        self.max_total = max_total_mib * 1024**2
        self.max_files = max_files
        self.findings = []
        self.seen = set()
        self.inodes = set()
        self.counts = Counter()
        self.coverage = []
        self.process_scanned = 0

    def add(self, category, severity, location, line=None, segment=None):
        location = redacted(location)
        key = (category, location, line, segment)
        if key in self.seen:
            return
        self.seen.add(key)
        if len(self.findings) >= 10000:
            self.counts['finding_limit_reached'] += 1
            return
        self.findings.append({'category': category, 'severity': severity,
                              'location': location, 'line_in_segment': line, 'segment': segment})

    def text(self, text, location, segment='full'):
        hits = 0
        for number, line in enumerate(text.splitlines(), 1):
            for category, expression in PATTERNS.items():
                match = expression.search(line)
                if match and (category in {'private_key_material', 'email_like_personal_metadata'} or plausible(match.group(0))):
                    self.add(category, 'INFO' if category == 'email_like_personal_metadata' else 'HIGH', location, number, segment)
                    hits += 1
            for match in CLI_SECRET.finditer(line):
                if plausible(match.group(1)):
                    self.add('possible_command_line_credential', 'HIGH', location, number, segment)
                    hits += 1
            for match in ASSIGNMENT.finditer(line):
                if credential_name(match.group(1)) and plausible(match.group(2)):
                    self.add('possible_literal_credential', 'HIGH', location, number, segment)
                    hits += 1
            if hits >= 100:
                self.counts['per_file_finding_limit'] += 1
                break

    def environment(self, values, location):
        for name, value in values.items():
            if not value:
                continue
            safe_name = name if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,100}', name) else '[nonstandard variable name]'
            place = location + ':' + safe_name
            if credential_name(name):
                self.add('credential_environment_variable_present', 'HIGH', place)
            if name in ('SSH_AUTH_SOCK', 'GOOGLE_APPLICATION_CREDENTIALS', 'AWS_SHARED_CREDENTIALS_FILE'):
                self.add('credential_or_agent_pointer_present', 'MEDIUM', place)
            self.text(value, place, 'environment_value_not_reported')

    def processes(self, proc_root=Path('/proc')):
        if sys.platform != 'linux' or not proc_root.is_dir():
            self.counts['process_scan_unavailable'] += 1
            return
        for process in proc_root.iterdir():
            if not process.name.isdigit():
                continue
            try:
                if process.stat().st_uid != os.getuid():
                    self.counts['other_uid_processes_not_scanned'] += 1
                    continue
                raw = (process / 'environ').read_bytes()
                env = {}
                for item in raw.split(b'\0'):
                    if b'=' in item:
                        key, value = item.split(b'=', 1)
                        env[key.decode('utf-8', 'replace')] = value.decode('utf-8', 'replace')
                self.environment(env, 'process:' + process.name)
                self.text((process / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace'),
                          'process:' + process.name + ':command_arguments', 'arguments_not_reported')
                self.process_scanned += 1
            except (FileNotFoundError, ProcessLookupError):
                self.counts['process_exited_during_scan'] += 1
            except (PermissionError, OSError):
                self.counts['processes_unreadable'] += 1

    def file(self, path):
        try:
            info = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                self.counts['nonregular_files_skipped'] += 1
                return
            identity = (info.st_dev, info.st_ino)
            if info.st_ino and identity in self.inodes:
                self.counts['hardlink_or_overlapping_root_duplicates'] += 1
                return
            self.inodes.add(identity)
            self.counts['files_examined'] += 1
            if not info.st_size:
                return
            name = path.as_posix()
            sensitive = False
            for ending, (category, severity) in STORES.items():
                if name.endswith(ending):
                    self.add(category, severity, path)
                    sensitive |= severity == 'HIGH'
            if path.name in {'.env', '.env.local', '.env.production'} or path.name.startswith('.env.'):
                self.add('environment_config_file', 'MEDIUM', path)
            if path.name in {'.connection-token', 'connection-token'}:
                self.add('remote_editor_connection_token_file', 'HIGH', path)
                sensitive = True
            if '/.ssh/' in name and path.name in {'id_rsa', 'id_ed25519', 'id_ecdsa', 'id_dsa'}:
                self.add('ssh_private_key_file', 'HIGH', path)
                sensitive = True
            if '/etc/ssh/' in name and path.name.startswith('ssh_host_') and path.name.endswith('_key'):
                self.add('server_host_private_key_not_personal_login_key', 'MEDIUM', path)
            if '/.gnupg/private-keys-v1.d/' in name:
                self.add('gpg_private_key_store', 'HIGH', path)
                sensitive = True
            if path.suffix.lower() in {'.sqlite', '.sqlite3', '.vscdb', '.keyring', '.kdbx'} or '/.local/share/keyrings/' in name:
                self.add('uninspected_history_or_credential_database', 'MEDIUM', path)
            if sensitive and stat.S_IMODE(info.st_mode) & 0o077:
                self.add('credential_file_accessible_to_group_or_others', 'MEDIUM', path)
            if path.suffix.lower() in BINARY or path.name.endswith(('.sqlite-wal', '.sqlite-shm')):
                self.counts['binary_archive_or_database_files_skipped'] += 1
                return
            remaining = self.max_total - self.counts['content_bytes_read']
            if remaining <= 0:
                self.counts['text_budget_exhausted_files'] += 1
                return
            # O_NOFOLLOW avoids reading a substituted symlink on supported systems.
            fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
            with os.fdopen(fd, 'rb') as stream:
                opened = os.fstat(stream.fileno())
                if not stat.S_ISREG(opened.st_mode):
                    self.counts['nonregular_files_skipped'] += 1
                    return
                probe = stream.read(min(4096, remaining))
                self.counts['content_bytes_read'] += len(probe)
                if b'\0' in probe:
                    self.counts['binary_content_files_skipped'] += 1
                    return
                stream.seek(0)
                remaining = self.max_total - self.counts['content_bytes_read']
                limit = min(self.max_file, remaining)
                if info.st_size > limit:
                    self.counts['partially_scanned_files'] += 1
                    if len(self.coverage) < 40:
                        self.coverage.append({'path': redacted(path), 'reason': 'Only bounded head/tail segments read'})
                    chunk = min(1024**2, max(0, limit // 2))
                    head = stream.read(chunk)
                    stream.seek(max(0, opened.st_size - chunk))
                    tail = stream.read(chunk)
                    self.counts['content_bytes_read'] += len(head) + len(tail)
                    self.text(head.decode('utf-8', 'replace'), path, 'head')
                    self.text(tail.decode('utf-8', 'replace'), path, 'tail_lines_are_segment_relative')
                else:
                    payload = stream.read(limit)
                    self.counts['content_bytes_read'] += len(payload)
                    self.text(payload.decode('utf-8', 'replace'), path)
                if os.fstat(stream.fileno()).st_mtime_ns != opened.st_mtime_ns:
                    self.counts['files_changed_during_scan'] += 1
        except (OSError, ValueError) as error:
            self.counts['files_unreadable'] += 1
            if len(self.coverage) < 40:
                self.coverage.append({'path': redacted(path), 'reason': type(error).__name__})

    def walk(self, roots, include_dependencies=False):
        for root in roots:
            root = Path(root).absolute()
            if root.is_symlink():
                self.counts['symlink_roots_skipped'] += 1
                continue
            if not root.exists():
                self.counts['missing_roots'] += 1
                continue
            if root.is_file():
                self.file(root)
                continue
            def failed(_error):
                self.counts['directories_unreadable'] += 1
            for directory, dirs, files in os.walk(root, followlinks=False, onerror=failed):
                parent = Path(directory)
                kept = []
                for name in sorted(dirs):
                    child = parent / name
                    excluded = (child.is_symlink() or child.resolve() == self.output
                                or child.as_posix() in {'/proc', '/sys', '/dev'}
                                or name.startswith('private-audit-')
                                or (parent.name == '.git' and name in {'objects', 'lfs'})
                                or (parent.name == '.cache' and name in {'pip', 'uv'})
                                or (not include_dependencies and name in DEPENDENCIES))
                    if excluded:
                        self.counts['directories_skipped'] += 1
                    else:
                        kept.append(name)
                dirs[:] = kept
                for name in sorted(files):
                    if self.counts['files_examined'] >= self.max_files:
                        self.counts['file_limit_reached'] += 1
                        return
                    self.file(parent / name)
                    if self.counts['files_examined'] and self.counts['files_examined'] % 500 == 0:
                        print('PROGRESS: files=' + str(self.counts['files_examined']) + '; findings=' + str(len(self.findings)), flush=True)
                        self.progress()

    def progress(self):
        private_json(self.output / 'progress.json', {'counts': dict(self.counts), 'findings': len(self.findings)})

    def finish(self, roots, env_requested, processes_requested):
        severity = Counter(f['severity'] for f in self.findings)
        value = {'created_at': datetime.now(timezone.utc).isoformat(), 'roots': [redacted(p) for p in roots],
                 'counts': dict(self.counts), 'severity_counts': dict(severity), 'processes_scanned': self.process_scanned,
                 'current_environment_requested': env_requested, 'process_scan_requested': processes_requested,
                 'finding_values_or_previews_included': False, 'online_validation_performed': False,
                 'deletion_or_revocation_performed': False, 'findings': self.findings, 'coverage_samples': self.coverage,
                 'limits': ['No Git object/history inspection; current Git config files are inspected',
                            'No archive, model, raw matrix, encrypted store or database content inspection',
                            'Skipped symlinks and software dependency trees unless explicitly included',
                            'Large text files use bounded head/tail reads; middle content may be missed',
                            'Process environment is a best-effort snapshot for the current UID only',
                            'Shell in-memory history, external cloud-side data and unknown secret formats can be missed',
                            'Patterns are candidates, not proof of validity; examples may be false positives'],
                 'no_findings_is_not_a_clean_disk_certificate': True}
        private_json(self.output / 'findings.json', value)
        lines = ['SERVER PRIVACY AUDIT', '', 'Read-only offline inventory. Credential values are not printed.',
                 f"HIGH={severity['HIGH']} MEDIUM={severity['MEDIUM']} INFO={severity['INFO']}",
                 f"Files examined={self.counts['files_examined']}; same-UID processes scanned={self.process_scanned}.",
                 '', 'Candidate locations (values omitted; tail line numbers are segment-relative):']
        for finding in self.findings:
            lines.append(f"[{finding['severity']}] {finding['category']} | {finding['location']} | line={finding['line_in_segment']} segment={finding['segment']}")
        lines += ['', 'COVERAGE LIMITS', *value['limits'], '',
                  'Unreadable files=' + str(self.counts['files_unreadable']) + '; partial text files=' + str(self.counts['partially_scanned_files']) +
                  '; text-budget skips=' + str(self.counts['text_budget_exhausted_files']) + '; file-limit hits=' + str(self.counts['file_limit_reached']),
                  '', 'No findings does NOT prove that no private data remain. This report is private; do not publish it.',
                  'Review candidates locally. Revoke/rotate credentials used on this server when appropriate.',
                  'Confirm private backups, then use the provider controls to destroy unwanted instance/disk copies.',
                  'Do not mistake public SSH authorized_keys for a private key or delete login access while still using it.']
        path = self.output / 'SUMMARY.txt'
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write('\n'.join(lines) + '\n')
        private_json(self.output / 'progress.json', {'stage': 'complete', 'severity_counts': dict(severity), 'counts': dict(self.counts)})
        print('AUDIT COMPLETE: HIGH=' + str(severity['HIGH']) + ' MEDIUM=' + str(severity['MEDIUM']) + ' INFO=' + str(severity['INFO']), flush=True)
        print('PRIVATE REPORT: ' + redacted(path), flush=True)
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', default=[], help='Repeat to inspect additional roots; default Linux roots: /root /etc /tmp /var/tmp /var/log')
    parser.add_argument('--output', default='/root/private-audit-' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--no-env', action='store_true', help='Do not inspect the current environment')
    parser.add_argument('--no-processes', action='store_true', help='Do not inspect same-UID Linux process snapshots')
    parser.add_argument('--include-dependencies', action='store_true')
    parser.add_argument('--max-file-mib', type=int, default=16)
    parser.add_argument('--max-total-mib', type=int, default=1024)
    parser.add_argument('--max-files', type=int, default=100000)
    args = parser.parse_args()
    if min(args.max_file_mib, args.max_total_mib, args.max_files) < 1:
        parser.error('Scan limits must be positive')
    if sys.platform != 'linux' and not args.root:
        parser.error('Default discovery is Linux-only; use explicit --root for a controlled local test')
    roots = args.root or ['/root', '/etc', '/tmp', '/var/tmp', '/var/log']
    if sys.platform == 'linux':
        os.umask(0o077)
    audit = Audit(args.output, args.max_file_mib, args.max_total_mib, args.max_files)
    print('Starting read-only offline audit; no credential values will be printed.', flush=True)
    if not args.no_env:
        audit.environment(dict(os.environ), 'current_environment')
    if not args.no_processes:
        audit.processes()
    audit.walk(roots, args.include_dependencies)
    audit.finish(roots, not args.no_env, not args.no_processes)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Do not allow a traceback to expose a credential-bearing filename.
        print('AUDIT FAILED: ' + type(error).__name__ + '; details omitted to avoid disclosure. Check paths and permissions locally.', file=sys.stderr)
        raise SystemExit(2)
