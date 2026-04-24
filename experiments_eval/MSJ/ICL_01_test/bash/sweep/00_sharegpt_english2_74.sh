#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Build arguments
MODEL_NAMES_ARG=("${MODEL_NAMES[@]}")
GENERATOR_MODELS_ARG=()
if [ ${#GENERATOR_MODELS[@]} -gt 0 ]; then
    GENERATOR_MODELS_ARG+=("--generator_models" "${GENERATOR_MODELS[@]}")
fi

YES_ARG=()
if [[ "$SKIP_CONFIRMATION" == "true" ]]; then
    YES_ARG=("-y")
fi

uv run srf-eval-sweep \
    --model_names "${MODEL_NAMES_ARG[@]}" \
    "${GENERATOR_MODELS_ARG[@]}" \
    --treatment_type "$TREATMENT_TYPE" \
    --dataset_dir_path data/input/sharegpt/english2_74 \
    --experiment_config "$SCRIPT_DIR/../../config.yaml" \
    --max-tasks "$MAX_TASKS" \
    "${YES_ARG[@]}"
