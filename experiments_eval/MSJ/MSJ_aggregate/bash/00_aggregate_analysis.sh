#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

uv run python scripts/msj/aggregate_analysis.py \
    --results_root data/msj/results \
    --batches \
        MSJ_01_batch1 \
        MSJ_01_base_vs_trained_batch2 \
        MSJ_01_base_vs_trained_batch3 \
        MSJ_01_base_vs_trained_batch4 \
        MSJ_02_adversarial_batch1 \
        MSJ_02_adversarial_batch2 \
        MSJ_02_adversarial_batch3 \
        MSJ_02_adversarial_batch4 \
    --output_dir data/msj/results/aggregate
