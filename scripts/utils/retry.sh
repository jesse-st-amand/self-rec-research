#!/usr/bin/env bash
# Retry a command every N seconds until it succeeds or max attempts reached.
#
# Usage:
#   scripts/utils/retry.sh <interval_seconds> [--max <N>] <command...>
#
# Examples:
#   scripts/utils/retry.sh 300 bash some_script.sh
#   scripts/utils/retry.sh 300 --max 20 bash some_script.sh
#   scripts/utils/retry.sh 60 --max 5 uv run python some_script.py

set -uo pipefail

INTERVAL="${1:?Usage: $0 <interval_seconds> [--max <N>] <command...>}"
shift

MAX_ATTEMPTS=0  # 0 = unlimited
if [[ "${1:-}" == "--max" ]]; then
    MAX_ATTEMPTS="${2:?--max requires a number}"
    shift 2
fi

ATTEMPT=1
while true; do
    echo "========================================"
    if [[ $MAX_ATTEMPTS -gt 0 ]]; then
        echo "Attempt $ATTEMPT/$MAX_ATTEMPTS — $(date)"
    else
        echo "Attempt $ATTEMPT — $(date)"
    fi
    echo "Command: $*"
    echo "========================================"

    if "$@"; then
        echo ""
        echo "✓ Succeeded on attempt $ATTEMPT"
        exit 0
    fi

    if [[ $MAX_ATTEMPTS -gt 0 && $ATTEMPT -ge $MAX_ATTEMPTS ]]; then
        echo ""
        echo "✗ Failed after $MAX_ATTEMPTS attempts. Giving up."
        exit 1
    fi

    echo ""
    echo "✗ Failed on attempt $ATTEMPT. Retrying in ${INTERVAL}s..."
    sleep "$INTERVAL"
    ATTEMPT=$((ATTEMPT + 1))
done
