#!/bin/bash
# Shared configuration for analysis scripts in this directory
# This file is sourced by all analysis scripts to get dataset subsets and model names

# Dataset subsets to combine (relative to data/results/{dataset_name}/)
# Each subset adds more datapoints to the same evaluations
DATASET_SUBSETS=(
    "english_26"
    "english2_74"
)

# Read MODEL_NAMES from root experiment config.yaml
_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_CONFIG_DIR/config.yaml" ]] && [[ "$_CONFIG_DIR" != "/" ]]; do
    _CONFIG_DIR="$(dirname "$_CONFIG_DIR")"
done
_ROOT_CONFIG="$_CONFIG_DIR/config.yaml"

MODEL_NAMES=($(python3 -c "
import yaml
with open('$_ROOT_CONFIG') as f:
    config = yaml.safe_load(f)
for item in config.get('model_names', []):
    print(item)
"))
