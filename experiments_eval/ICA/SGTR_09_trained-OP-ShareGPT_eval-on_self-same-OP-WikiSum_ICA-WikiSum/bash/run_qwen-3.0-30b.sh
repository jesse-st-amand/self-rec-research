#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

ICA_DIR="experiments_eval/ICA/SGTR_09_trained-OP-ShareGPT_eval-on_self-same-OP-WikiSum_ICA-WikiSum/qwen-3.0-30b"

run() {
    local leaf="$1"
    local evaluator="$2"
    echo ""; echo "=== ${leaf} ==="
    uv run srf-eval-sweep \
        --model_names "$evaluator" \
        --generator_models qwen-3.0-30b gpt-oss-120b \
        --treatment_type other_models \
        --dataset_dir_path data/input/wikisum/training_set_1-20 \
        --experiment_config "$ICA_DIR/$leaf/config.yaml" \
        --max-tasks 1 -y
}

run "qwen-3.0-30b_AT_IND_5shot_base_ica-alt" "qwen-3.0-30b"
run "qwen-3.0-30b_AT_IND_5shot_base_ica-ctrl" "qwen-3.0-30b"
run "qwen-3.0-30b_AT_IND_5shot_base_ica-ctrl2" "qwen-3.0-30b"
run "qwen-3.0-30b_AT_IND_5shot_base_ica-ctrl3" "qwen-3.0-30b"
run "qwen-3.0-30b_AT_IND_5shot_base_ica-self" "qwen-3.0-30b"
run "qwen-3.0-30b_AT_IND_5shot_trained-std_ica-alt" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_IND_ShareGPT"
run "qwen-3.0-30b_AT_IND_5shot_trained-std_ica-ctrl" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_IND_ShareGPT"
run "qwen-3.0-30b_AT_IND_5shot_trained-std_ica-ctrl2" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_IND_ShareGPT"
run "qwen-3.0-30b_AT_IND_5shot_trained-std_ica-ctrl3" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_IND_ShareGPT"
run "qwen-3.0-30b_AT_IND_5shot_trained-std_ica-self" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_IND_ShareGPT"
run "qwen-3.0-30b_AT_IND_base_no-ica" "qwen-3.0-30b"
run "qwen-3.0-30b_AT_IND_trained-std_no-ica" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_IND_ShareGPT"
run "qwen-3.0-30b_UT_IND_5shot_base_ica-alt" "qwen-3.0-30b"
run "qwen-3.0-30b_UT_IND_5shot_base_ica-ctrl" "qwen-3.0-30b"
run "qwen-3.0-30b_UT_IND_5shot_base_ica-ctrl2" "qwen-3.0-30b"
run "qwen-3.0-30b_UT_IND_5shot_base_ica-ctrl3" "qwen-3.0-30b"
run "qwen-3.0-30b_UT_IND_5shot_base_ica-self" "qwen-3.0-30b"
run "qwen-3.0-30b_UT_IND_5shot_trained-std_ica-alt" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_IND_ShareGPT"
run "qwen-3.0-30b_UT_IND_5shot_trained-std_ica-ctrl" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_IND_ShareGPT"
run "qwen-3.0-30b_UT_IND_5shot_trained-std_ica-ctrl2" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_IND_ShareGPT"
run "qwen-3.0-30b_UT_IND_5shot_trained-std_ica-ctrl3" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_IND_ShareGPT"
run "qwen-3.0-30b_UT_IND_5shot_trained-std_ica-self" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_IND_ShareGPT"
run "qwen-3.0-30b_UT_IND_base_no-ica" "qwen-3.0-30b"
run "qwen-3.0-30b_UT_IND_trained-std_no-ica" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_IND_ShareGPT"

echo "SGTR_09_trained-OP-ShareGPT_eval-on_self-same-OP-WikiSum_ICA-WikiSum/qwen-3.0-30b complete."
