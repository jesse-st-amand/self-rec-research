#!/bin/bash
# Automatically run all bash scripts needed to generate ICML figures
#
# This script reads the figure mappings from scripts/copy_figures.sh
# and runs the corresponding analysis scripts to generate each figure.

set +e  # Don't exit on error - we want to process all scripts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Extract and source the FIGURES array from copy_figures.sh
# Extract lines from "FIGURES=(" to the matching ")"
FIGURES_DEF=$(awk '/^FIGURES=\($/,/^\)$/ {print}' "$SCRIPT_DIR/copy_figures.sh")
eval "$FIGURES_DEF"

# Function to map figure filename to script path
map_figure_to_script() {
    local experiment_path="$1"
    local filename="$2"
    
    # Extract experiment name from path
    # e.g., "data/analysis/_aggregated_data/ICML_01_UT_PW-Q_Rec_NPr_FA_Inst" -> "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst"
    local exp_name=$(basename "$experiment_path")
    
    # Find the experiment directory under experiments_eval/ (searches all group subdirs)
    local exp_dir=""
    exp_dir=$(find "$PROJECT_ROOT/experiments_eval" -maxdepth 3 -type d -name "$exp_name" | head -1)
    if [[ -z "$exp_dir" ]]; then
        echo ""
        return
    fi
    exp_dir="${exp_dir#$PROJECT_ROOT/}"  # Make relative

    # Handle comparison experiments (e.g., ICML_01-vs-ICML_02)
    if [[ "$exp_name" == *"-vs-"* ]]; then
        local script_path="${exp_dir}/00-performance_contrast.sh"
        if [[ -f "$PROJECT_ROOT/$script_path" ]]; then
            echo "$script_path"
            return
        fi
    fi

    # Map figure filenames to script names
    case "$filename" in
        "aggregated_performance_grouped.png")
            echo "${exp_dir}/bash/analysis/_inter-dataset/00b-plot_performance_aggregate.sh"
            ;;
        "performance_vs_arena_ranking.png")
            echo "${exp_dir}/bash/analysis/_inter-dataset/02-performance_vs_size.sh"
            ;;
        "rank_distance_aggregated.png"|"rank_distance_adjusted.png"|"rank_distance_filtered_evaluator_rank.png"|"rank_distance_filtered_evaluator_rank_positive.png")
            echo "${exp_dir}/bash/analysis/_inter-dataset/03-rank-distance.sh"
            ;;
        "performance_contrast_grouped.png"|"performance_scatter.png")
            echo "${exp_dir}/00-performance_contrast.sh"
            ;;
        *)
            echo ""  # Unknown figure
            ;;
    esac
}

# FIGURES array is now sourced from copy_figures.sh above

echo "=============================================================================="
echo "Generating ICML Figures"
echo "=============================================================================="
echo ""

# Parse FIGURES array and collect unique scripts in order
# Use associative array to track seen scripts, regular array to preserve order
declare -A SCRIPTS_SEEN
SCRIPTS_TO_RUN=()

for entry in "${FIGURES[@]}"; do
    # Split by pipe character
    IFS='|' read -r experiment_path filename output_name <<< "$entry"
    
    script_path=$(map_figure_to_script "$experiment_path" "$filename")
    
    if [[ -n "$script_path" ]]; then
        full_script_path="$PROJECT_ROOT/$script_path"
        if [[ -f "$full_script_path" ]]; then
            # Only add if we haven't seen this script before
            if [[ -z "${SCRIPTS_SEEN[$script_path]}" ]]; then
                SCRIPTS_TO_RUN+=("$script_path")
                SCRIPTS_SEEN["$script_path"]=1
            fi
            echo "  ✓ Mapped: $filename -> $script_path"
        else
            echo "  ⚠  Script not found: $script_path"
        fi
    else
        echo "  ⚠  No script mapping for: $filename (from $experiment_path)"
    fi
done

echo ""
echo "=============================================================================="
echo "Running Analysis Scripts"
echo "=============================================================================="
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Run each script in order
success_count=0
failed_count=0
failed_scripts=()

for script_path in "${SCRIPTS_TO_RUN[@]}"; do
    full_script_path="$PROJECT_ROOT/$script_path"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: $script_path"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if bash "$full_script_path"; then
        echo ""
        echo "✓  Success: $script_path"
        ((success_count++))
    else
        echo ""
        echo "✗  Failed: $script_path"
        failed_scripts+=("$script_path")
        ((failed_count++))
    fi
    echo ""
done

echo "=============================================================================="
echo "Summary"
echo "=============================================================================="
echo "Successfully ran: $success_count"
echo "Failed: $failed_count"
echo "Total: $((success_count + failed_count))"
echo ""

if [[ $failed_count -gt 0 ]]; then
    echo "Failed scripts:"
    for script in "${failed_scripts[@]}"; do
        echo "  - $script"
    done
    echo ""
    exit 1
fi

echo "All figures generated successfully!"
echo ""
echo "Next step: Run scripts/copy_figures.sh to copy figures to ICML_figures/"
