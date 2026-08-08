#!/usr/bin/env bash
# Generate the three figures used in the COLM 2026 main body.
#
# Tune text sizes in FIGURE_STYLE at the top of make_paper_figures.py, then
# re-run. Extra arguments are forwarded to the Python script, e.g.:
#
#   bash scripts/figures/COLM2026/bash/make_paper_figures.sh --only fig1
#   bash scripts/figures/COLM2026/bash/make_paper_figures.sh --copy-to-paper
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Figure 3 reads training_dir and data_subsets from the same config the
# AlpacaEval analysis pipeline uses. Override with CONFIG_FILE=... if you are
# pointing at a different set of training runs.
CONFIG_FILE="${CONFIG_FILE:-experiments_eval/COLM/COLM_AE-training-and-figures/config.yaml}"

uv run python scripts/figures/COLM2026/make_paper_figures.py \
    --copy-to-paper \
    --config "$CONFIG_FILE" \
    --results_dir data/alpaca_eval/results \
    --analysis_dir data/alpaca_eval/analysis \
    --output_dir data/figures/paper \
    "$@"
