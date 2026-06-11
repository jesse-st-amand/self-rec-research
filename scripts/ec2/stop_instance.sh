#!/usr/bin/env bash
# Laptop-side: stop the EC2 instance to pause compute billing (EBS storage still charged).
# Safe to run; refuses if the chain tmux session is still active unless --force.
#
# Usage:
#   bash scripts/ec2/stop_instance.sh
#   bash scripts/ec2/stop_instance.sh --force    # stop even if a chain is running

set -euo pipefail

# shellcheck source=_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

state="$(ec2_state)"
if [ "$state" = "stopped" ] || [ "$state" = "stopping" ]; then
    echo "[stop] instance already $state."
    exit 0
fi

if [ "$state" != "running" ]; then
    echo "[stop] instance is in unexpected state: $state. Aborting."
    exit 1
fi

if [ "$FORCE" -ne 1 ]; then
    # Refuse if a tmux chain session is still active.
    if ec2_ssh "tmux ls 2>/dev/null | grep -q chain"; then
        echo "[stop] active 'chain' tmux session detected. Refusing to stop."
        echo "        - Wait for the chain to finish (it will auto-stop)."
        echo "        - Or use 'bash scripts/ec2/attach.sh' to inspect."
        echo "        - Or pass --force to stop anyway (cancels the chain)."
        exit 1
    fi
fi

echo "[stop] stopping instance $EC2_INSTANCE_ID ..."
aws ec2 stop-instances --region "$AWS_REGION" --instance-ids "$EC2_INSTANCE_ID" >/dev/null
aws ec2 wait instance-stopped --region "$AWS_REGION" --instance-ids "$EC2_INSTANCE_ID"
echo "[stop] stopped."
