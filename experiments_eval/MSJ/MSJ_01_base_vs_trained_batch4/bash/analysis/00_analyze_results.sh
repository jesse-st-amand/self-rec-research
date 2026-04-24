#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

uv run python scripts/msj/analyze_results.py \
    --results_dir data/msj/results/MSJ_01_base_vs_trained_batch3
