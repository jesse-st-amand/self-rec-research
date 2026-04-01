#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/../../config.yaml"

uv run python scripts/alpaca_eval/analyze_uplift.py \
    --config "$CONFIG_FILE" \
    --output_dir data/alpaca_eval/analysis
