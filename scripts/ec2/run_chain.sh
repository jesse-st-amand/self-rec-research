#!/usr/bin/env bash
# Run a chain script inside a detached tmux session, with auto-shutdown on exit.
#
# Usage (on the EC2 instance):
#   bash scripts/ec2/run_chain.sh <chain_script_relative_to_repo> [--no-shutdown] [--name <session>]
#
# Example:
#   bash scripts/ec2/run_chain.sh experiments_eval/ICA/SGTR_02_trained-OP_eval-on_self-same-OP/bash/run_remaining_chain.sh
#
# Behavior:
#   - Sources $REPO_DIR/.env so chains see TINKER_API_KEY, OPENAI_API_KEY, etc.
#   - Generates a one-shot wrapper script under $HOME/chain_logs/<session>_<ts>.run
#     that tmux executes. The wrapper, on chain exit, calls `sudo shutdown -h now`
#     (unless --no-shutdown) so the instance transitions to `stopped` and pauses billing.
#   - Logs go to ~/chain_logs/<session>_<ts>.log

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/self-rec-research}"
SESSION="chain"
DO_SHUTDOWN=1

usage() {
    sed -n '2,16p' "$0"
    exit 1
}

CHAIN=""
while [ $# -gt 0 ]; do
    case "$1" in
        --no-shutdown) DO_SHUTDOWN=0; shift ;;
        --name) SESSION="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) CHAIN="$1"; shift ;;
    esac
done
[ -z "$CHAIN" ] && { echo "ERROR: chain script path required"; usage; }

# Resolve to absolute path against repo
if [ -f "$REPO_DIR/$CHAIN" ]; then
    CHAIN_ABS="$REPO_DIR/$CHAIN"
elif [ -f "$CHAIN" ]; then
    CHAIN_ABS="$(realpath "$CHAIN")"
else
    echo "ERROR: chain script not found: $CHAIN (looked in $REPO_DIR/$CHAIN and \$PWD)"
    exit 1
fi

if [ ! -f "$REPO_DIR/.env" ]; then
    echo "ERROR: $REPO_DIR/.env missing — run bootstrap.sh first"
    exit 1
fi
if ! command -v tmux >/dev/null; then
    echo "ERROR: tmux not installed"
    exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists. Attach with 'tmux attach -t $SESSION' or pick another --name."
    exit 1
fi

mkdir -p "$HOME/chain_logs"
TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
LOG_FILE="$HOME/chain_logs/${SESSION}_${TIMESTAMP}.log"
WRAPPER="$HOME/chain_logs/${SESSION}_${TIMESTAMP}.run"

# Generate a wrapper script that tmux will exec. Single-quoted heredoc + explicit
# variable substitution at the top — easier to reason about than nested escapes.
cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
set -uo pipefail

REPO_DIR='$REPO_DIR'
CHAIN_ABS='$CHAIN_ABS'
LOG_FILE='$LOG_FILE'
DO_SHUTDOWN=$DO_SHUTDOWN
EOF

cat >> "$WRAPPER" <<'EOF'

cd "$REPO_DIR"
set -a; . "$REPO_DIR/.env"; set +a

echo "[run_chain] starting at $(date -u +%FT%TZ)"
echo "[run_chain] chain: $CHAIN_ABS"
echo "[run_chain] log:   $LOG_FILE"

bash "$CHAIN_ABS"
EXIT_CODE=$?

echo ""
echo "[run_chain] chain exit code: $EXIT_CODE at $(date -u +%FT%TZ)"
if [ "$DO_SHUTDOWN" -eq 1 ]; then
    echo "[run_chain] auto-shutdown in 10s — Ctrl-C inside tmux to cancel."
    sleep 10
    sudo shutdown -h now
fi
EOF
chmod +x "$WRAPPER"

# tmux runs the wrapper, with stdout/stderr teed to the log file.
tmux new-session -d -s "$SESSION" "bash '$WRAPPER' 2>&1 | tee '$LOG_FILE'"

echo "[run_chain] launched tmux session '$SESSION'"
echo "[run_chain] live log: tail -f $LOG_FILE"
echo "[run_chain] attach:   tmux attach -t $SESSION"
echo "[run_chain] detach:   Ctrl-b d (inside tmux)"
if [ "$DO_SHUTDOWN" -eq 1 ]; then
    echo "[run_chain] instance will auto-stop when chain exits."
fi
