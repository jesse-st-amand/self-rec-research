#!/bin/bash
# Generate PKU SaferLHF debug data using current architecture

uv run srf-generate \
    --model_name=gpt-oss-120b-thinking \
    --dataset_path=data/input/pku_saferlhf/debug/input.json \
    --dataset_config=experiments_eval/00_data_gen/configs/config.yaml \
    --overwrite
