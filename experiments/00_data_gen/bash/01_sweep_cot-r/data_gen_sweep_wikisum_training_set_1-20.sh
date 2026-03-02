#!/bin/bash
# Example sweep data generation script for multiple models

uv run srf-generate-sweep \
    --model_names -set eval_cot-r \
    --dataset_path=data/input/wikisum/training_set_1-20 \
    --dataset_config=experiments/00_data_gen/config.yaml \
