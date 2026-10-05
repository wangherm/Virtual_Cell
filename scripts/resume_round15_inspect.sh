#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
work="${VCELL_WORK:-/root/autodl-tmp/vcell-work}"
mkdir -p "$work/logs" "$work/tmp"
export TMPDIR="$work/tmp" PYTHONUNBUFFERED=1
log="$work/logs/round15_D1_recovery_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log") 2>&1
echo "LOG: $log"
exec bash "$repo/scripts/run_round15_mixscale.sh" "$work/raw/round15_D1/IFNG" "$work"
