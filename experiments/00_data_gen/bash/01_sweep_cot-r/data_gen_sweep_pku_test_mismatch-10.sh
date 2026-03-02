#!/bin/bash
# Example sweep data generation script for multiple models

uv run srf-generate-sweep \
    --model_names -set eval_cot-r \
    --dataset_path=data/input/pku_saferlhf/test-mismatch_10_100-200/input.json \
    --dataset_config=experiments/00_data_gen/config.yaml \
