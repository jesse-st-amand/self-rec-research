#!/usr/bin/env bash
# Tutorial: run a minimal SFT training job.
#
# Prerequisites:
#   1. Extract training data (only needed once):
#      bash experiments_training/_data_processing/prepare.sh \
#          experiments_training/_tutorial/prepare_data.yaml
#
#   2. Run training:
#      bash experiments_training/_tutorial/run.sh
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"
RUNTIME="_external/SGTR-RL/runtimes/local_gpu.yaml"

uv run sgtr-train --config "$CONFIG" --runtime "$RUNTIME"
