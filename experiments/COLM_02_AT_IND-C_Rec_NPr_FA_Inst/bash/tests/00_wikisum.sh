#!/bin/bash
# Analyze combined wikisum subsets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(basename "$(cd "$SCRIPT_DIR/../.." && pwd)")"

# List all wikisum subsets to combine
# Each subset adds more datapoints to the same evaluations

DATASET_PATHS=(
    "data/results/wikisum/debug/$EXP_DIR"
)

uv run srf-recognition-accuracy \
        --results_dir "${DATASET_PATHS[@]}" \
        --model_names -set test
