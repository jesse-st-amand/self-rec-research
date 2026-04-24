#!/bin/bash
# Shared configuration for ICL_01_test sweep scripts

# Find root config.yaml
_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_CONFIG_DIR/config.yaml" ]] && [[ "$_CONFIG_DIR" != "/" ]]; do
    _CONFIG_DIR="$(dirname "$_CONFIG_DIR")"
done
_ROOT_CONFIG="$_CONFIG_DIR/config.yaml"

MODEL_NAMES=($(uv run python scripts/utils/expand_config_models.py "$_ROOT_CONFIG" model_names 2>/dev/null))

GENERATOR_MODELS=($(uv run python scripts/utils/expand_config_models.py "$_ROOT_CONFIG" generator_models --no-trained 2>/dev/null))

TREATMENT_TYPE="other_models"
MAX_TASKS=3
SKIP_CONFIRMATION="true"
BATCH_MODE="false"
GPU_DISPATCH=""
