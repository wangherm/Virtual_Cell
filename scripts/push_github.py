"""Explicit, user-run source-only push. Never called by training or notebook Run All."""
import argparse
import base64
import getpass
import os
from pathlib import Path
import re
import subprocess


SOURCE_PATHS = ["src", "configs", "tests", "optional_tests", "scripts", "notebooks", "docs", ".github",
                "README.md", "pyproject.toml", ".gitignore", "LICENSE", "CITATION.cff"]


def git(*args, env=None, check=True):
    result = subprocess.run(["git", *args], capture_output=True, text=True, env=env)
    if check and result.returncode:
        # Never print credential-bearing diagnostics.
        raise RuntimeError(f"Git {args[0]} failed (exit {result.returncode}); inspect repository/permissions without exposing tokens.")
    return result


def push(remote, name, email, token, confirm=False):
    if not confirm:
        raise ValueError("Explicit confirmation is required.")
    if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", remote):
        raise ValueError("Use a plain HTTPS github.com repository URL, with no embedded credentials.")
    if "YOUR_USERNAME" in remote or not name or not email:
        raise ValueError("Set your existing GitHub repository, commit name and email first.")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if not (root / ".git").exists():
        git("init", "-b", "main")
    git("config", "user.name", name)
    git("config", "user.email", email)
    old = git("remote", "get-url", "origin", check=False)
    if old.returncode == 0 and old.stdout.strip().removesuffix(".git") != remote.removesuffix(".git"):
        raise ValueError("Existing origin is different. Refusing to change it automatically.")
    if old.returncode != 0:
        git("remote", "add", "origin", remote)
    # Refuse unrelated staged files, even in an existing local repository.
    staged = git("diff", "--cached", "--name-only").stdout.splitlines()
    def allowed(path):
        return any(path == s or path.startswith(s + "/") for s in SOURCE_PATHS)
    if any(not allowed(p) for p in staged):
        raise ValueError("Unrelated files already staged. Review/unstage them yourself first.")
    git("add", "--", *[p for p in SOURCE_PATHS if (root / p).exists()])
    if git("diff", "--cached", "--quiet", check=False).returncode != 0:
        git("commit", "-m", "Add VCell teacher-student experiments and deployment guide")
    env = os.environ.copy()
    auth = base64.b64encode(("x-access-token:" + token).encode()).decode()
    env.update(GIT_CONFIG_COUNT="2", GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
               GIT_CONFIG_VALUE_0="Authorization: Basic " + auth,
               GIT_CONFIG_KEY_1="credential.helper", GIT_CONFIG_VALUE_1="",
               GIT_TERMINAL_PROMPT="0")
    try:
        git("push", "-u", "origin", "HEAD:main", env=env)  # no force, no automatic pull/reset
    finally:
        env.pop("GIT_CONFIG_VALUE_0", None)
        token, auth = "", ""
    print("Source pushed successfully. Data, runs, checkpoints and tokens were not staged.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--remote", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--email", required=True)
    p.add_argument("--confirm", action="store_true")
    args = p.parse_args()
    if not args.confirm:
        p.error("Pass --confirm only after reviewing the source files.")
    push(args.remote, args.name, args.email, getpass.getpass("GitHub fine-grained token (hidden): "), args.confirm)
