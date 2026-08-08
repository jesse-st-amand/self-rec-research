#!/usr/bin/env bash
# What size is the text in the paper's figures once the page has shrunk them?
#
# COLM asks for nothing below \small, which prints at 8.9pt. The sizes in
# FIGURE_STYLE are not that number -- they are authored points, and the page
# scales them -- so this is the check that answers the question the format
# instructions actually ask. Exits non-zero if anything prints below the floor.
#
#   bash scripts/figures/COLM2026/bash/check_figure_text.sh
#   bash scripts/figures/COLM2026/bash/check_figure_text.sh some/figure.pdf
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

uv run python scripts/figures/COLM2026/check_figure_text.py "$@"
