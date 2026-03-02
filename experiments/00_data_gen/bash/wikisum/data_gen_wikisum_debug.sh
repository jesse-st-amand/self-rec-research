#!/bin/bash
# Generate WikiSum data using current architecture

uv run experiments/_scripts/gen/generate_data.py \
    --model_name=ll-3.1-8b \
    --dataset_path=data/wikisum/debug/input.json \
    --dataset_config=experiments/00_data_gen/configs/config.yaml
