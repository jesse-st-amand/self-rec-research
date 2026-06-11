# Sourced by other scripts/ec2/*.sh — not directly executable.
# Loads instance.env, resolves the live public DNS/IP, exposes helper fns.

set -euo pipefail

EC2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_ENV="$EC2_DIR/instance.env"

if [ ! -f "$INSTANCE_ENV" ]; then
    echo "ERROR: $INSTANCE_ENV missing. Copy instance.env.example to instance.env and fill in values."
    exit 1
fi

# shellcheck disable=SC1090
set -a; . "$INSTANCE_ENV"; set +a

: "${AWS_REGION:?AWS_REGION must be set in $INSTANCE_ENV}"
: "${EC2_INSTANCE_ID:?EC2_INSTANCE_ID must be set in $INSTANCE_ENV}"
: "${EC2_KEY_PATH:?EC2_KEY_PATH must be set in $INSTANCE_ENV}"
: "${EC2_USER:?EC2_USER must be set in $INSTANCE_ENV}"
: "${EC2_REMOTE_REPO:?EC2_REMOTE_REPO must be set in $INSTANCE_ENV}"

# Expand ~ in EC2_KEY_PATH
EC2_KEY_PATH="${EC2_KEY_PATH/#\~/$HOME}"

require_aws_cli() {
    if ! command -v aws >/dev/null; then
        echo "ERROR: aws CLI not found. Install it: https://aws.amazon.com/cli/"
        exit 1
    fi
}

# Get current state ("running", "stopped", "stopping", "pending", ...).
ec2_state() {
    require_aws_cli
    aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$EC2_INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text
}

# Resolve and echo the public DNS (or EC2_HOST if pinned). Errors if instance not running.
ec2_host() {
    if [ -n "${EC2_HOST:-}" ]; then
        echo "$EC2_HOST"
        return 0
    fi
    require_aws_cli
    local dns
    dns=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$EC2_INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicDnsName' \
        --output text 2>/dev/null)
    if [ -z "$dns" ] || [ "$dns" = "None" ]; then
        echo "ERROR: instance $EC2_INSTANCE_ID has no public DNS (probably not running). State: $(ec2_state)" >&2
        return 1
    fi
    echo "$dns"
}

# Start instance if stopped, wait until ssh is reachable.
ec2_start_and_wait() {
    require_aws_cli
    local state
    state=$(ec2_state)
    case "$state" in
        running) echo "[ec2] instance $EC2_INSTANCE_ID already running"; ;;
        stopped)
            echo "[ec2] starting instance $EC2_INSTANCE_ID ..."
            aws ec2 start-instances --region "$AWS_REGION" --instance-ids "$EC2_INSTANCE_ID" >/dev/null
            ;;
        stopping)
            echo "[ec2] instance is stopping — waiting before re-starting"
            aws ec2 wait instance-stopped --region "$AWS_REGION" --instance-ids "$EC2_INSTANCE_ID"
            aws ec2 start-instances --region "$AWS_REGION" --instance-ids "$EC2_INSTANCE_ID" >/dev/null
            ;;
        pending) echo "[ec2] instance is starting" ;;
        *) echo "[ec2] unexpected state: $state — proceeding anyway" ;;
    esac
    echo "[ec2] waiting for instance running state ..."
    aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$EC2_INSTANCE_ID"
    aws ec2 wait instance-status-ok --region "$AWS_REGION" --instance-ids "$EC2_INSTANCE_ID" || true

    local host
    host=$(ec2_host)
    echo "[ec2] host: $host"

    # Wait until SSH accepts a connection (status-ok doesn't always mean sshd is up).
    local i=0
    while [ $i -lt 30 ]; do
        if ssh -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -o BatchMode=yes \
            "$EC2_USER@$host" 'echo ok' >/dev/null 2>&1; then
            echo "[ec2] ssh ready"
            return 0
        fi
        i=$((i+1))
        sleep 2
    done
    echo "ERROR: ssh did not become ready within 60s" >&2
    return 1
}

# Run a command on the instance over ssh. Args after the function name are sent as a single command.
ec2_ssh() {
    local host
    host=$(ec2_host)
    ssh -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=accept-new "$EC2_USER@$host" "$@"
}
