#!/usr/bin/env bash
# Laptop-side: ssh in and attach to the running tmux chain session.
# Detach with Ctrl-b d (the chain keeps running on the instance).
#
# Usage:
#   bash scripts/ec2/attach.sh           # default session name "chain"
#   bash scripts/ec2/attach.sh <session> # custom name (matches --name passed to run_chain.sh)

set -euo pipefail

# shellcheck source=_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

SESSION="${1:-chain}"

state="$(ec2_state)"
if [ "$state" != "running" ]; then
    echo "[attach] instance is $state — cannot attach. Start it first or wait."
    exit 1
fi

host="$(ec2_host)"
exec ssh -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=accept-new -t "$EC2_USER@$host" \
    "tmux attach -t '$SESSION'"
