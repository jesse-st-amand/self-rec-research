#!/usr/bin/env bash
# Laptop-side, ONE-TIME: create a t4g.small EC2 instance suitable for running
# Tinker eval chains. Outputs the instance ID + public DNS so you can fill in
# scripts/ec2/instance.env.
#
# What this provisions:
#   - t4g.small Ubuntu 22.04 ARM (Graviton) — cheapest sensible option.
#   - 30GB gp3 EBS root volume.
#   - New keypair (saved to ~/.ssh/<NAME>.pem).
#   - Security group allowing SSH from your current public IP only.
#   - InstanceInitiatedShutdownBehavior=stop (so `sudo shutdown -h now` from
#     inside the instance stops it instead of terminating).
#
# Prereqs:
#   - aws CLI configured (`aws configure` or env vars / SSO).
#   - A default VPC in the chosen region (Lightsail-friendly default — most accounts have one).
#
# Usage:
#   bash scripts/ec2/create_instance.sh                      # defaults
#   AWS_REGION=us-west-2 NAME=self-rec-ec2 bash scripts/ec2/create_instance.sh

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
NAME="${NAME:-self-rec-research-ec2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t4g.small}"
VOLUME_SIZE="${VOLUME_SIZE:-30}"
KEY_DIR="${KEY_DIR:-$HOME/.ssh}"
KEY_PATH="$KEY_DIR/$NAME.pem"

EC2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_ENV="$EC2_DIR/instance.env"

require() { command -v "$1" >/dev/null || { echo "ERROR: $1 not installed"; exit 1; }; }
require aws
require jq
require curl

# --- 0. Resolve current public IP for the security group's SSH rule ----------
MY_IP="$(curl -fsS https://checkip.amazonaws.com 2>/dev/null | tr -d '\r\n')"
[ -z "$MY_IP" ] && { echo "ERROR: could not detect public IP"; exit 1; }
echo "[create] your public IP: $MY_IP"

# --- 1. AMI: latest Ubuntu 22.04 ARM in this region --------------------------
echo "[create] looking up Ubuntu 22.04 ARM AMI in $AWS_REGION ..."
AMI_ID=$(aws ec2 describe-images --region "$AWS_REGION" \
    --owners 099720109477 \
    --filters \
        "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-arm64-server-*" \
        "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text)
[ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ] && { echo "ERROR: AMI lookup failed"; exit 1; }
echo "[create] AMI: $AMI_ID"

# --- 2. Keypair --------------------------------------------------------------
mkdir -p "$KEY_DIR"
if aws ec2 describe-key-pairs --region "$AWS_REGION" --key-names "$NAME" >/dev/null 2>&1; then
    echo "[create] keypair '$NAME' already exists in $AWS_REGION."
    if [ ! -f "$KEY_PATH" ]; then
        echo "ERROR: $KEY_PATH missing locally but keypair exists remotely. Delete the AWS keypair or restore the file."
        exit 1
    fi
else
    echo "[create] creating keypair '$NAME' → $KEY_PATH"
    aws ec2 create-key-pair --region "$AWS_REGION" --key-name "$NAME" \
        --query 'KeyMaterial' --output text > "$KEY_PATH"
    chmod 600 "$KEY_PATH"
fi

# --- 3. Security group with SSH ingress from MY_IP ---------------------------
SG_NAME="${NAME}-sg"
SG_ID=$(aws ec2 describe-security-groups --region "$AWS_REGION" \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)

if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
    echo "[create] creating security group $SG_NAME"
    SG_ID=$(aws ec2 create-security-group --region "$AWS_REGION" \
        --group-name "$SG_NAME" \
        --description "SSH access for $NAME" \
        --query 'GroupId' --output text)
fi
echo "[create] security group: $SG_ID"

# Add SSH rule for MY_IP if not already present.
EXISTING=$(aws ec2 describe-security-groups --region "$AWS_REGION" --group-ids "$SG_ID" \
    --query "SecurityGroups[0].IpPermissions[?FromPort==\`22\`].IpRanges[].CidrIp" --output text)
if ! echo "$EXISTING" | grep -q "${MY_IP}/32"; then
    echo "[create] authorizing SSH from $MY_IP/32"
    aws ec2 authorize-security-group-ingress --region "$AWS_REGION" \
        --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$MY_IP/32" >/dev/null \
        || echo "  (rule may already exist for a different IP — re-run from your new network if needed)"
else
    echo "[create] SSH from $MY_IP/32 already authorized"
fi

# --- 4. Launch instance ------------------------------------------------------
BLOCK_DEV='[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":'"$VOLUME_SIZE"',"VolumeType":"gp3","DeleteOnTermination":true}}]'

echo "[create] launching $INSTANCE_TYPE ..."
INSTANCE_ID=$(aws ec2 run-instances --region "$AWS_REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$NAME" \
    --security-group-ids "$SG_ID" \
    --instance-initiated-shutdown-behavior stop \
    --block-device-mappings "$BLOCK_DEV" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=self-rec-research}]" \
    --query 'Instances[0].InstanceId' --output text)
echo "[create] instance: $INSTANCE_ID"

aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
DNS=$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicDnsName' --output text)
echo "[create] public DNS: $DNS"

# --- 5. Write instance.env ---------------------------------------------------
echo "[create] writing $INSTANCE_ENV"
cat > "$INSTANCE_ENV" <<ENV_EOF
# Generated by create_instance.sh on $(date -u +%FT%TZ)
AWS_REGION=$AWS_REGION
EC2_INSTANCE_ID=$INSTANCE_ID
EC2_KEY_PATH=$KEY_PATH
EC2_USER=ubuntu
EC2_REMOTE_REPO=/home/ubuntu/self-rec-research
EC2_HOST=
ENV_EOF

# --- 6. Final pointers -------------------------------------------------------
cat <<EOF

================================================================
EC2 instance ready.

  Instance:  $INSTANCE_ID
  Region:    $AWS_REGION
  Type:      $INSTANCE_TYPE
  Public DNS: $DNS
  Key:       $KEY_PATH
  Config:    $INSTANCE_ENV

Next steps:
  1. SSH and run bootstrap (one-time):
       ssh -i $KEY_PATH ubuntu@$DNS
       curl -fsSL https://raw.githubusercontent.com/jesse-st-amand/self-rec-research/ec2/scripts/ec2/bootstrap.sh | bash
       # then edit ~/self-rec-research/.env and fill in TINKER_API_KEY etc.

  2. From the laptop, kick off a chain in detached tmux:
       bash scripts/ec2/launch_chain.sh experiments_eval/ICA/SGTR_02_trained-OP_eval-on_self-same-OP/bash/run_remaining_chain.sh

  3. Close laptop. Chain runs to completion and the instance auto-stops.

  4. Later: pull results back:
       bash scripts/ec2/sync_results.sh --start
================================================================
EOF
