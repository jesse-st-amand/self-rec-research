#!/bin/bash
# Sweep experiment: Compare models against each other

uv run srf-eval-sweep \
    --model_names gpt-4.1-mini qwen-3.0-80b ll-3.1-70b deepseek-3.1 sonnet-3.7 gemini-2.0-flash \
    --treatment_type other_models \
    --dataset_dir_path data/input/pku_saferlhf/mismatch_1-20 \
    --experiment_config experiments/01_AT_PW-C_Rec_Pr/config.yaml \
    --batch
