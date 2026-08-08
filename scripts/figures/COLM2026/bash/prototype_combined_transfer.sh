#!/usr/bin/env bash
# Main-body Figure 3, written to data/figures/prototypes instead of the paper
# path — for layout work, so a half-tuned figure never reaches the paper. The
# paper's copy comes from make_paper_figures.sh --only fig3.
#
# Tune the layout in the FIG_W/TOP_H/BOTTOM_H block at the top of
# prototype_combined_transfer.py and the text sizes in FIGURE_STYLE["fig3"] in
# make_paper_figures.py, then re-run.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CONFIG_FILE="${CONFIG_FILE:-experiments_eval/COLM/COLM_AE-training-and-figures/config.yaml}"

uv run python scripts/figures/COLM2026/prototype_combined_transfer.py \
    --config "$CONFIG_FILE" \
    --results_dir data/alpaca_eval/results \
    --output_dir data/figures/prototypes \
    "$@"
