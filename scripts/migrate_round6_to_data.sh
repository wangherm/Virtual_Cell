#!/usr/bin/env bash
# Move completed Round 6 artifacts to the expanded data disk; retain old paths.
set -euo pipefail
src=/root/vcell-round6-work
dst=/root/autodl-tmp/vcell-round6-work
backup=/root/vcell-round6-work.migration-backup
lock=/root/autodl-tmp/.vcell-round6-migration-lock
command -v rsync >/dev/null || { echo 'rsync is required: apt-get update && apt-get install -y rsync'; exit 1; }
if [[ -L "$src" && "$(readlink -f "$src")" == "$dst" ]]; then
    echo "Already migrated: $src -> $dst"
    [[ ! -e "$backup" ]] || echo "A backup remains at $backup; inspect before removing it."
    df -h / /root/autodl-tmp
    exit 0
fi
[[ -d "$src" && ! -L "$src" ]] || { echo "Missing real source directory: $src"; exit 1; }
[[ "$(readlink -f "$src")" == "$src" ]] || exit 1
[[ "$(readlink -f /root/autodl-tmp)" == /root/autodl-tmp ]] || exit 1
[[ "$(stat -c %d "$src")" != "$(stat -c %d /root/autodl-tmp)" ]] || { echo 'Source and destination are on the same filesystem; inspect mounts.'; exit 1; }
[[ ! -e "$dst" && ! -L "$dst" && ! -e "$backup" && ! -L "$backup" ]] || { echo 'Destination or backup already exists; inspect it before retrying. Nothing deleted.'; exit 1; }
if pgrep -af '[r]un_round[0-9].*py|[v]cell qwen-run|[r]un_scgpt_teacher.py' ; then
    echo 'Training is active. Let it finish before migrating.'
    exit 1
fi
[[ -f "$src/runs/round6_01/COMPLETE.json" ]] || { echo 'Round 6 completion marker is missing.'; exit 1; }
need=$(du -sb "$src" | cut -f1)
free=$(df -B1 --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')
(( free > need + 1073741824 )) || { echo 'Insufficient destination space.'; exit 1; }
mkdir "$lock"
trap 'rmdir "$lock" 2>/dev/null || true' EXIT
echo "Copying $src -> $dst. Keep training stopped until completion."
mkdir "$dst"
rsync -aH --info=progress2 "$src/" "$dst/"
# Check every file's contents, metadata and directory membership before switching.
changes=$(rsync -aHnci --delete "$src/" "$dst/")
[[ -z "$changes" ]] || { printf 'Verification failed; source retained:\n%s\n' "$changes"; exit 1; }
sync
mv -- "$src" "$backup"
if ! ln -s -- "$dst" "$src"; then
    mv -- "$backup" "$src"
    echo 'Could not create compatibility symlink; restored source.'
    exit 1
fi
[[ -L "$src" && "$(readlink -f "$src")" == "$dst" ]] || exit 1
[[ -d "$backup" && ! -L "$backup" && "$(readlink -f "$backup")" == /root/vcell-round6-work.migration-backup ]] || exit 1
changes=$(rsync -aHnci --delete "$backup/" "$src/")
[[ -z "$changes" ]] || { echo 'Final verification failed; backup retained.'; exit 1; }
# Only the verified old system-disk copy is removed; raw data and models are untouched.
rm -rf -- "$backup"
echo "MIGRATION COMPLETE: $src -> $dst"
df -h / /root/autodl-tmp
ls -ld "$src" "$dst"
