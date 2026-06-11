#!/usr/bin/env bash
# Laptop-side: report instance state, active tmux sessions, latest chain log lines.
#
# Usage:
#   bash scripts/ec2/status.sh                 # one-shot summary
#   bash scripts/ec2/status.sh --tail          # follow the latest chain log

set -euo pipefail

# shellcheck source=_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

MODE="summary"
[ "${1:-}" = "--tail" ] && MODE="tail"

state="$(ec2_state)"
echo "[status] instance: $EC2_INSTANCE_ID  state: $state  region: $AWS_REGION"

if [ "$state" != "running" ]; then
    if [ "$MODE" = "tail" ]; then
        echo "[status] instance not running — cannot tail logs."
    fi
    exit 0
fi

host="$(ec2_host)"
echo "[status] host: $host"

if [ "$MODE" = "summary" ]; then
    echo "---"
    echo "[status] tmux sessions:"
    ec2_ssh "tmux ls 2>/dev/null || echo '  (none)'" || true
    echo "---"
    echo "[status] latest log files:"
    ec2_ssh "ls -lt ~/chain_logs/ 2>/dev/null | head -6 || echo '  (no chain_logs dir)'" || true
    echo "---"
    echo "[status] latest log tail (last 30 lines):"
    ec2_ssh "latest=\$(ls -1t ~/chain_logs/*.log 2>/dev/null | head -1); [ -n \"\$latest\" ] && tail -30 \"\$latest\" || echo '  (no logs)'" || true
else
    echo "[status] tailing latest log — Ctrl-C to stop."
    ec2_ssh "latest=\$(ls -1t ~/chain_logs/*.log 2>/dev/null | head -1); [ -n \"\$latest\" ] && tail -f \"\$latest\" || echo 'no logs yet'"
fi
