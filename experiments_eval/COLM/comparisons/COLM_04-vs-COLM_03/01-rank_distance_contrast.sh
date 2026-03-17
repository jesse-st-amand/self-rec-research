#!/bin/bash
# Compare rank-distance vs accuracy between two experiments
# Uses pre-computed rank_distance_data.csv from each experiment's analysis

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENTRY_POINT="srf-rank-distance-contrast"

# Extract experiment names from directory name (e.g., COLM_04-vs-COLM_03)
COMPARISON_DIR="$(basename "$SCRIPT_DIR")"
if [[ "$COMPARISON_DIR" == *"-vs-"* ]]; then
    EXP1="${COMPARISON_DIR%%-vs-*}"  # Everything before "-vs-"
    EXP2="${COMPARISON_DIR#*-vs-}"   # Everything after "-vs-"
else
    echo "Error: Directory name must be in format 'exp1-vs-exp2'"
    exit 1
fi

# ============================================================================
# Load configuration from shared config file
# ============================================================================

CONFIG_FILE="$SCRIPT_DIR/config.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Please create config.yaml in the same directory."
    exit 1
fi

# Read MODEL_NAMES from YAML config
MODEL_NAMES=($(python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    config = yaml.safe_load(f)
for item in config.get('model_names', []):
    print(item)
"))

# ============================================================================
# Find most recent rank_distance_data.csv for each experiment
# ============================================================================

EXP1_BASE_DIR="data/analysis/_aggregated_data"
EXP1_PATTERN="${EXP1}_*"

# Find experiment directory (e.g., COLM_04_UT_PW-Q_Rec_NPr_FA_Rsn)
EXP1_EXP_DIR=$(ls -d "$EXP1_BASE_DIR"/$EXP1_PATTERN 2>/dev/null | head -1)

if [[ -z "$EXP1_EXP_DIR" ]] || [[ ! -d "$EXP1_EXP_DIR" ]]; then
    echo "Error: No experiment directory found for: $EXP1"
    echo "  Searched in: $EXP1_BASE_DIR for pattern: $EXP1_PATTERN"
    exit 1
fi

# Find most recent timestamp subdirectory within experiment directory
EXP1_LATEST_DIR=$(ls -td "$EXP1_EXP_DIR"/*/ 2>/dev/null | head -1)

if [[ -z "$EXP1_LATEST_DIR" ]] || [[ ! -d "$EXP1_LATEST_DIR" ]]; then
    echo "Error: No timestamp directories found in: $EXP1_EXP_DIR"
    exit 1
fi

# Remove trailing slash if present
EXP1_LATEST_DIR="${EXP1_LATEST_DIR%/}"

EXP1_FILE="$EXP1_LATEST_DIR/rank_distance_data.csv"
if [[ ! -f "$EXP1_FILE" ]]; then
    echo "Error: Rank distance data not found: $EXP1_FILE"
    exit 1
fi

# Find experiment directory for exp2
EXP2_PATTERN="${EXP2}_*"
EXP2_EXP_DIR=$(ls -d "$EXP1_BASE_DIR"/$EXP2_PATTERN 2>/dev/null | head -1)

if [[ -z "$EXP2_EXP_DIR" ]] || [[ ! -d "$EXP2_EXP_DIR" ]]; then
    echo "Error: No experiment directory found for: $EXP2"
    echo "  Searched in: $EXP1_BASE_DIR for pattern: $EXP2_PATTERN"
    exit 1
fi

# Find most recent timestamp subdirectory within experiment directory
EXP2_LATEST_DIR=$(ls -td "$EXP2_EXP_DIR"/*/ 2>/dev/null | head -1)

if [[ -z "$EXP2_LATEST_DIR" ]] || [[ ! -d "$EXP2_LATEST_DIR" ]]; then
    echo "Error: No timestamp directories found in: $EXP2_EXP_DIR"
    exit 1
fi

# Remove trailing slash if present
EXP2_LATEST_DIR="${EXP2_LATEST_DIR%/}"

EXP2_FILE="$EXP2_LATEST_DIR/rank_distance_data.csv"
if [[ ! -f "$EXP2_FILE" ]]; then
    echo "Error: Rank distance data not found: $EXP2_FILE"
    exit 1
fi

echo "Comparing rank-distance relationships:"
echo "  Exp1: $EXP1 ($EXP1_FILE)"
echo "  Exp2: $EXP2 ($EXP2_FILE)"
echo ""

# ============================================================================
# Create shared output directory for both rank and score contrast
# ============================================================================

COMPARISON_NAME="${EXP1}-vs-${EXP2}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="data/analysis/_aggregated_data/${COMPARISON_NAME}/${TIMESTAMP}"

# ============================================================================
# Run rank-distance contrast
# ============================================================================

uv run "$ENTRY_POINT" \
        --exp1_file "$EXP1_FILE" \
        --exp2_file "$EXP2_FILE" \
        --exp1_name "$EXP1" \
        --exp2_name "$EXP2" \
        --model_names "${MODEL_NAMES[@]}" \
        --output_dir "$OUTPUT_DIR"

# ============================================================================
# Run score-distance contrast (Elo score difference instead of rank)
# ============================================================================

EXP1_SCORE_FILE="${EXP1_LATEST_DIR}/score_distance_data.csv"
EXP2_SCORE_FILE="${EXP2_LATEST_DIR}/score_distance_data.csv"

if [[ -f "$EXP1_SCORE_FILE" ]] && [[ -f "$EXP2_SCORE_FILE" ]]; then
    echo ""
    echo "Comparing score-distance relationships:"
    echo "  Exp1: $EXP1 ($EXP1_SCORE_FILE)"
    echo "  Exp2: $EXP2 ($EXP2_SCORE_FILE)"
    echo ""

    uv run "$ENTRY_POINT" \
            --exp1_file "$EXP1_SCORE_FILE" \
            --exp2_file "$EXP2_SCORE_FILE" \
            --exp1_name "$EXP1" \
            --exp2_name "$EXP2" \
            --model_names "${MODEL_NAMES[@]}" \
            --distance_type score \
            --output_dir "$OUTPUT_DIR"
else
    echo ""
    echo "⚠ Score-distance data not found, skipping score-distance contrast."
    echo "  Run rank-distance analysis first to generate score_distance_data.csv."
fi
