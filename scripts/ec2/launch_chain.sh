#!/usr/bin/env bash
# Laptop-side: start the EC2 instance, ssh in, kick off run_chain.sh in detached tmux,
# and return control immediately. Closing the laptop will not affect the chain.
#
# Usage:
#   bash scripts/ec2/launch_chain.sh <chain_script_relative_to_repo> [--no-shutdown] [--name <session>]
#
# Example:
#   bash scripts/ec2/launch_chain.sh \
#       experiments_eval/ICA/SGTR_02_trained-OP_eval-on_self-same-OP/bash/run_remaining_chain.sh

set -euo pipefail

# shellcheck source=_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

if [ $# -lt 1 ]; then
    sed -n '2,11p' "$0"
    exit 1
fi

CHAIN="$1"
shift
EXTRA_ARGS=("$@")

ec2_start_and_wait

# Sanity: the chain script exists on the remote.
if ! ec2_ssh "test -f '$EC2_REMOTE_REPO/$CHAIN'"; then
    echo "ERROR: chain script not found on remote: $EC2_REMOTE_REPO/$CHAIN" >&2
    echo "Did you 'git pull' on the instance? (cd $EC2_REMOTE_REPO && git pull)" >&2
    exit 1
fi

# Forward to run_chain.sh on the remote.
REMOTE_CMD="cd '$EC2_REMOTE_REPO' && bash scripts/ec2/run_chain.sh '$CHAIN'"
for arg in "${EXTRA_ARGS[@]}"; do
    REMOTE_CMD+=" '$arg'"
done

echo "[launch] firing chain on remote ..."
ec2_ssh "$REMOTE_CMD"

echo ""
echo "[launch] chain is detached. You can close the laptop now."
echo "[launch] tail logs:    bash scripts/ec2/status.sh --tail"
echo "[launch] live attach:  bash scripts/ec2/attach.sh"
echo "[launch] pull results: bash scripts/ec2/sync_results.sh"
