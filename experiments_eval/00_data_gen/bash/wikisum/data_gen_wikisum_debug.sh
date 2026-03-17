#!/bin/bash
# Generate WikiSum data using current architecture

uv run srf-generate \
    --model_name=ll-3.1-8b \
    --dataset_path=data/wikisum/debug/input.json \
    --dataset_config=experiments_eval/00_data_gen/configs/config.yaml
