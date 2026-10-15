#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=${VCELL_WORK_DIR:-/root/autodl-tmp/vcell-work}
name=${1:-round23_evolve_01}
data_name=${2:-round23_data_01}
: "${DEEPSEEK_API_KEY:?Export DEEPSEEK_API_KEY in the launching terminal}"
mkdir -p "$work/logs"
python="$repo/.venv/bin/python"
"$python" -u "$repo/scripts/round23.py" inventory --work-dir "$work"
"$python" -u "$repo/scripts/round23.py" data-agent --work-dir "$work" --dest "$work/data_agents/$data_name"
catalog="$work/data_agents/$data_name/ready_catalog.json"
"$python" -c 'import json,sys; d=json.load(open(sys.argv[1])); n=sum(e.get("role")=="source" and bool(e.get("bundle")) for e in d["sources"]); print("EXTERNAL SOURCE BUNDLES READY:", n); sys.exit(0 if n else "New source data blocked: inspect data-agent summary/adapter reports before research search")' "$catalog"
"$python" -u "$repo/scripts/round23.py" expand-catalog --work-dir "$work" --catalog "$catalog" --dest "$work/data_agents/$data_name/all_catalog.json"
catalog="$work/data_agents/$data_name/all_catalog.json"
bash "$repo/scripts/run_round23_autodl.sh" --name "$name" --catalog "$catalog" --mode smoke --resume
bash "$repo/scripts/run_round23_autodl.sh" --name "$name" --catalog "$catalog" --resume
