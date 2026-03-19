#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CONFIG="${1:?Usage: $0 <config.yaml>}"
uv run python scripts/training/prepare_data.py --config "$CONFIG" "${@:2}"
