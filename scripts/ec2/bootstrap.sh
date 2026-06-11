#!/usr/bin/env bash
# Bootstrap a fresh EC2 instance to run Tinker eval chains for self-rec-research.
# Idempotent: safe to re-run.
#
# Usage (on the EC2 instance, as the default user — `ubuntu` on Ubuntu AMIs):
#   curl -fsSL https://raw.githubusercontent.com/<user>/self-rec-research/ec2/scripts/ec2/bootstrap.sh | bash
# OR (if repo already cloned):
#   bash scripts/ec2/bootstrap.sh
#
# What it installs/sets up:
#   - System packages (git, tmux, jq, rsync, build-essential)
#   - uv (Python package/version manager)
#   - Clones `self-rec-research` (main repo) and the editable `_external/` deps
#   - Runs `uv sync` to install Python deps
#   - Writes a `.env` template the user must populate with API keys

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/jesse-st-amand/self-rec-research.git}"
REPO_BRANCH="${REPO_BRANCH:-ec2}"
REPO_DIR="${REPO_DIR:-$HOME/self-rec-research}"

SGTR_RL_URL="${SGTR_RL_URL:-https://github.com/jesse-st-amand/SGTR-RL.git}"
SGTR_RL_BRANCH="${SGTR_RL_BRANCH:-js/local_editable}"

SRF_URL="${SRF_URL:-https://github.com/jesse-st-amand/self-rec-framework.git}"
SRF_BRANCH="${SRF_BRANCH:-public}"

log() { echo "[bootstrap] $*"; }

# 1. System packages
log "installing system packages"
if command -v apt-get >/dev/null; then
    sudo apt-get update -y
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
        git tmux jq rsync curl ca-certificates build-essential
elif command -v dnf >/dev/null; then
    sudo dnf install -y git tmux jq rsync curl ca-certificates "@Development tools"
else
    log "unsupported package manager — install git tmux jq rsync curl manually"
    exit 1
fi

# 2. uv
if ! command -v uv >/dev/null; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installs to ~/.local/bin — make available in this shell + future shells
    export PATH="$HOME/.local/bin:$PATH"
    if ! grep -q '\.local/bin' "$HOME/.bashrc" 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    fi
else
    log "uv already installed: $(uv --version)"
fi

# 3. Clone main repo if not present
if [ ! -d "$REPO_DIR/.git" ]; then
    log "cloning $REPO_URL ($REPO_BRANCH) → $REPO_DIR"
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
else
    log "main repo present at $REPO_DIR — pulling $REPO_BRANCH"
    git -C "$REPO_DIR" fetch origin "$REPO_BRANCH"
    git -C "$REPO_DIR" checkout "$REPO_BRANCH"
    git -C "$REPO_DIR" pull --ff-only origin "$REPO_BRANCH"
fi

# 4. Clone _external/ editable deps
mkdir -p "$REPO_DIR/_external"

clone_or_update() {
    local url="$1" branch="$2" dest="$3"
    if [ ! -d "$dest/.git" ]; then
        log "cloning $url ($branch) → $dest"
        git clone --branch "$branch" "$url" "$dest"
    else
        log "$dest present — fetching $branch"
        git -C "$dest" fetch origin "$branch"
        git -C "$dest" checkout "$branch"
        git -C "$dest" pull --ff-only origin "$branch" || log "  (pull skipped — local divergence)"
    fi
}

clone_or_update "$SGTR_RL_URL" "$SGTR_RL_BRANCH" "$REPO_DIR/_external/SGTR-RL"
clone_or_update "$SRF_URL"     "$SRF_BRANCH"     "$REPO_DIR/_external/self-rec-framework"

# 5. uv sync (installs Python deps + editable _external/ packages)
log "running uv sync in $REPO_DIR"
cd "$REPO_DIR"
uv sync

# 6. .env template
ENV_FILE="$REPO_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    log "writing .env template to $ENV_FILE — POPULATE BEFORE RUNNING CHAINS"
    cat > "$ENV_FILE" <<'ENV_EOF'
# Required for Tinker-routed eval calls
TINKER_API_KEY=

# Required for the OpenAI judge model used by MSJ + some evals
OPENAI_API_KEY=

# Optional — needed only if scripts/utils/sync_HF_data.py is used
HF_TOKEN=

# Optional — Anthropic, used by some downstream analysis scripts
ANTHROPIC_API_KEY=
ENV_EOF
    chmod 600 "$ENV_FILE"
else
    log ".env already present — leaving as-is"
fi

# 7. Install systemd hint for auto-shutdown (informational)
log ""
log "================================================================"
log "bootstrap complete."
log ""
log "Next steps:"
log "  1. Edit $ENV_FILE and fill in API keys."
log "  2. Verify the instance has 'instance-initiated-shutdown-behavior=stop'"
log "     (set on launch, or via aws ec2 modify-instance-attribute)."
log "  3. Sanity test: cd $REPO_DIR && uv run python -c 'import sgtr_rl, self_rec_framework; print(\"ok\")'"
log "  4. Launch a chain: bash scripts/ec2/run_chain.sh <chain_script>"
log "================================================================"
