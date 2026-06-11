#!/usr/bin/env bash
# Laptop-side: rsync results dirs back from the EC2 instance to this laptop.
# Default subset is the dirs your analyzers actually read; pass --all for the whole data/ tree.
#
# Usage:
#   bash scripts/ec2/sync_results.sh              # default subset (results + analysis)
#   bash scripts/ec2/sync_results.sh --all        # full data/ tree
#   bash scripts/ec2/sync_results.sh --start      # start instance first if it's stopped
#   bash scripts/ec2/sync_results.sh --dry-run    # preview without copying
#
# Notes:
#   - Uses rsync over ssh, preserves timestamps, deletes nothing on either side.
#   - Does NOT touch your local git index — files copy directly.

set -euo pipefail

# shellcheck source=_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

REPO_LOCAL="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

DO_START=0
DRY=""
SCOPE="default"
for a in "$@"; do
    case "$a" in
        --start)   DO_START=1 ;;
        --dry-run) DRY="--dry-run -v" ;;
        --all)     SCOPE="all" ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown arg: $a" >&2; exit 1 ;;
    esac
done

state="$(ec2_state)"
if [ "$state" != "running" ]; then
    if [ "$DO_START" -eq 1 ]; then
        ec2_start_and_wait
    else
        echo "[sync] instance is $state — pass --start to bring it up first."
        exit 1
    fi
fi

host="$(ec2_host)"

# rsync paths
RSYNC_OPTS="-az --info=stats2 -e \"ssh -i '$EC2_KEY_PATH' -o StrictHostKeyChecking=accept-new\" $DRY"

declare -a SUBDIRS
if [ "$SCOPE" = "all" ]; then
    SUBDIRS=("data/")
else
    SUBDIRS=(
        "data/results/"
        "data/msj/results/"
        "data/figures/"
        "data/analysis/"
    )
fi

for sub in "${SUBDIRS[@]}"; do
    src="$EC2_USER@$host:$EC2_REMOTE_REPO/$sub"
    dst="$REPO_LOCAL/$sub"
    mkdir -p "$dst"
    echo "[sync] $sub"
    eval rsync $RSYNC_OPTS "\"$src\"" "\"$dst\""
done

echo "[sync] done."
