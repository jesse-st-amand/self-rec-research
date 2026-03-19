#!/usr/bin/env bash
# Tutorial: launch a minimal SFT training job on RunPod with W&B logging.
#
# Prerequisites:
#   1. Set environment variables in .env:
#        RUNPOD_API_KEY=...
#        RUNPOD_NETWORK_VOLUME_ID=...
#        HF_TOKEN=...
#        WANDB_API_KEY=...
#
#   2. Extract training data (only needed once):
#        bash experiments_training/_data_processing/prepare.sh \
#            experiments_training/_tutorial/prepare_data.yaml
#
#   3. Push training data and configs to the repo so RunPod can clone them.
#
# Usage:
#   bash experiments_training/_tutorial/run_runpod.sh          # dry run
#   bash experiments_training/_tutorial/run_runpod.sh --launch  # actually launch
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/config_runpod.yaml"
RUNTIME="$SCRIPT_DIR/runpod_community.yaml"

if [[ "${1:-}" == "--launch" ]]; then
    uv run sgtr-runpod-launch \
        --config "$CONFIG" \
        --runtime "$RUNTIME"
else
    echo "=== DRY RUN (pass --launch to actually create the pod) ==="
    echo ""
    uv run sgtr-runpod-launch \
        --config "$CONFIG" \
        --runtime "$RUNTIME" \
        --dry-run
fi
