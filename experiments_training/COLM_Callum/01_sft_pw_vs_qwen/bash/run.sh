#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="$SCRIPT_DIR/../../runpod_a100.yaml"

uv run sgtr-runpod-launch \
    --config "$SCRIPT_DIR/../config.yaml" \
    --runtime "$RUNTIME" \
    --dry-run
