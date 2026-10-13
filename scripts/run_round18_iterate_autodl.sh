#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# New run only. 18 trials per arm; 36 research requests per fold/seed.
# User flags can override these defaults; existing run budgets are never reset.
bash "$repo/scripts/run_round18_autodl.sh" --name round18_iterate_01 \
  --outer-contexts BXPC3_IFNG --seeds 17 --parallel-jobs 3 \
  --cycles 6 --trials 3 --patience-cycles 3 --live-routing --live-targets 3 "$@"
