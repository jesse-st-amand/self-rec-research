#!/usr/bin/env bash
# Generate the figures used in the COLM 2026 appendix.
#
# Companion to make_paper_figures.sh. Tune text sizes in APPENDIX_STYLE at the
# top of make_appendix_figures.py, then re-run. Extra arguments are forwarded to
# the Python script, e.g.:
#
#   bash scripts/figures/COLM2026/bash/make_appendix_figures.sh --only transfer
#   bash scripts/figures/COLM2026/bash/make_appendix_figures.sh --copy-to-paper
#
# Note: --only controlled-bar and --only controlled-scatter rerun the
# rank-distance analysis, which overwrites the experiment's aggregated
# rank_distance_* output in place. See the docstring in make_appendix_figures.py.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# The training-transfer figure reads training_dir and data_subsets from the same
# config the analysis pipeline uses. Override with CONFIG_FILE=... if you are
# pointing at a different set of training runs.
CONFIG_FILE="${CONFIG_FILE:-experiments_eval/COLM/COLM_AE-training-and-figures/config.yaml}"

uv run python scripts/figures/COLM2026/make_appendix_figures.py \
    --copy-to-paper \
    --config "$CONFIG_FILE" \
    --analysis_dir data/alpaca_eval/analysis \
    --output_dir data/figures/appendix \
    "$@"
