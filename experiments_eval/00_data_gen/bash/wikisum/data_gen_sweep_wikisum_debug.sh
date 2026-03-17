#!/bin/bash
# Example sweep data generation script for multiple models

uv run srf-generate-sweep \
    --model_names -set test_eval_cot-r_and_cot-i \
    --dataset_path=data/input/wikisum/debug \
    --dataset_config=experiments_eval/00_data_gen/configs/config.yaml \
