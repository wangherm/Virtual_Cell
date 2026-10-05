#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
output="${1:?output directory required}"
work="${2:?work directory required}"
exec "$repo/.venv/bin/python" -u "$repo/scripts/setup_round15_r.py" --output "$output" --work-dir "$work"
