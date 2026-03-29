#!/bin/bash
# Sweep data generation script for multiple models

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/../../config.yaml"

# Read model_names from config.yaml
MODEL_NAMES=$(python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    config = yaml.safe_load(f)
print(' '.join(config.get('model_names', [])))
")

if [ -z "$MODEL_NAMES" ]; then
    echo "Error: No model_names found in $CONFIG_FILE"
    exit 1
fi

uv run srf-generate-sweep \
    --model_names $MODEL_NAMES \
    --dataset_path=data/input/sharegpt/english2_74/input.json \
    --dataset_config="$CONFIG_FILE"