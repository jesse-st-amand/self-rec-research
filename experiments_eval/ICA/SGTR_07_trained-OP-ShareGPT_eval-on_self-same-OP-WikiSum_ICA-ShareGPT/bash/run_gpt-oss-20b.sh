#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

ICA_DIR="experiments_eval/ICA/SGTR_07_trained-OP-ShareGPT_eval-on_self-same-OP-WikiSum_ICA-ShareGPT/gpt-oss-20b"

run() {
    local leaf="$1"
    local evaluator="$2"
    echo ""; echo "=== ${leaf} ==="
    uv run srf-eval-sweep \
        --model_names "$evaluator" \
        --generator_models gpt-oss-20b qwen-3.0-30b \
        --treatment_type other_models \
        --dataset_dir_path data/input/wikisum/training_set_1-20 \
        --experiment_config "$ICA_DIR/$leaf/config.yaml" \
        --max-tasks 1 -y
}

run "gpt-oss-20b_AT_IND_5shot_base_ica-alt" "gpt-oss-20b"
run "gpt-oss-20b_AT_IND_5shot_base_ica-ctrl" "gpt-oss-20b"
run "gpt-oss-20b_AT_IND_5shot_base_ica-ctrl2" "gpt-oss-20b"
run "gpt-oss-20b_AT_IND_5shot_base_ica-ctrl3" "gpt-oss-20b"
run "gpt-oss-20b_AT_IND_5shot_base_ica-self" "gpt-oss-20b"
run "gpt-oss-20b_AT_IND_5shot_trained-std_ica-alt" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "gpt-oss-20b_AT_IND_5shot_trained-std_ica-ctrl" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "gpt-oss-20b_AT_IND_5shot_trained-std_ica-ctrl2" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "gpt-oss-20b_AT_IND_5shot_trained-std_ica-ctrl3" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "gpt-oss-20b_AT_IND_5shot_trained-std_ica-self" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "gpt-oss-20b_AT_IND_base_no-ica" "gpt-oss-20b"
run "gpt-oss-20b_AT_IND_trained-std_no-ica" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "gpt-oss-20b_UT_IND_5shot_base_ica-alt" "gpt-oss-20b"
run "gpt-oss-20b_UT_IND_5shot_base_ica-ctrl" "gpt-oss-20b"
run "gpt-oss-20b_UT_IND_5shot_base_ica-ctrl2" "gpt-oss-20b"
run "gpt-oss-20b_UT_IND_5shot_base_ica-ctrl3" "gpt-oss-20b"
run "gpt-oss-20b_UT_IND_5shot_base_ica-self" "gpt-oss-20b"
run "gpt-oss-20b_UT_IND_5shot_trained-std_ica-alt" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_IND_ShareGPT"
run "gpt-oss-20b_UT_IND_5shot_trained-std_ica-ctrl" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_IND_ShareGPT"
run "gpt-oss-20b_UT_IND_5shot_trained-std_ica-ctrl2" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_IND_ShareGPT"
run "gpt-oss-20b_UT_IND_5shot_trained-std_ica-ctrl3" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_IND_ShareGPT"
run "gpt-oss-20b_UT_IND_5shot_trained-std_ica-self" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_IND_ShareGPT"
run "gpt-oss-20b_UT_IND_base_no-ica" "gpt-oss-20b"
run "gpt-oss-20b_UT_IND_trained-std_no-ica" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_IND_ShareGPT"

echo "SGTR_07_trained-OP-ShareGPT_eval-on_self-same-OP-WikiSum_ICA-ShareGPT/gpt-oss-20b complete."
