#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

ICA_DIR="experiments_eval/ICA/SGTR_02_trained-OP_eval-on_self-same-OP/gpt-oss-20b"

run() {
    local leaf="$1"
    local evaluator="$2"
    echo ""; echo "=== ${leaf} ==="
    uv run srf-eval-sweep \
        --model_names "$evaluator" \
        --generator_models gpt-oss-20b qwen-3.0-30b \
        --treatment_type other_models \
        --dataset_dir_path data/input/sharegpt/english2_74 \
        --experiment_config "$ICA_DIR/$leaf/config.yaml" \
        --max-tasks 1 -y
}

run "gpt-oss-20b_AT_PW_1shot_base_ica-alt" "gpt-oss-20b"
run "gpt-oss-20b_AT_PW_1shot_base_ica-ctrl" "gpt-oss-20b"
run "gpt-oss-20b_AT_PW_1shot_base_ica-ctrl2" "gpt-oss-20b"
run "gpt-oss-20b_AT_PW_1shot_base_ica-ctrl3" "gpt-oss-20b"
run "gpt-oss-20b_AT_PW_1shot_base_ica-self" "gpt-oss-20b"
run "gpt-oss-20b_AT_PW_1shot_trained-std_ica-alt" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_PW_ShareGPT"
run "gpt-oss-20b_AT_PW_1shot_trained-std_ica-ctrl" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_PW_ShareGPT"
run "gpt-oss-20b_AT_PW_1shot_trained-std_ica-ctrl2" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_PW_ShareGPT"
run "gpt-oss-20b_AT_PW_1shot_trained-std_ica-ctrl3" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_PW_ShareGPT"
run "gpt-oss-20b_AT_PW_1shot_trained-std_ica-self" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_PW_ShareGPT"
run "gpt-oss-20b_AT_PW_5shot_base_ica-alt" "gpt-oss-20b"
run "gpt-oss-20b_AT_PW_5shot_base_ica-ctrl" "gpt-oss-20b"
run "gpt-oss-20b_AT_PW_5shot_base_ica-ctrl2" "gpt-oss-20b"
run "gpt-oss-20b_AT_PW_5shot_base_ica-ctrl3" "gpt-oss-20b"
run "gpt-oss-20b_AT_PW_5shot_base_ica-self" "gpt-oss-20b"
run "gpt-oss-20b_AT_PW_5shot_trained-std_ica-alt" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_PW_ShareGPT"
run "gpt-oss-20b_AT_PW_5shot_trained-std_ica-ctrl" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_PW_ShareGPT"
run "gpt-oss-20b_AT_PW_5shot_trained-std_ica-ctrl2" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_PW_ShareGPT"
run "gpt-oss-20b_AT_PW_5shot_trained-std_ica-ctrl3" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_PW_ShareGPT"
run "gpt-oss-20b_AT_PW_5shot_trained-std_ica-self" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_1shot_base_ica-alt" "gpt-oss-20b"
run "gpt-oss-20b_UT_PW_1shot_base_ica-ctrl" "gpt-oss-20b"
run "gpt-oss-20b_UT_PW_1shot_base_ica-ctrl2" "gpt-oss-20b"
run "gpt-oss-20b_UT_PW_1shot_base_ica-ctrl3" "gpt-oss-20b"
run "gpt-oss-20b_UT_PW_1shot_base_ica-self" "gpt-oss-20b"
run "gpt-oss-20b_UT_PW_1shot_trained-adv_ica-alt" "gpt-oss-20b_sft-as_qwen-3-30b_vs_gpt-oss-20b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_1shot_trained-adv_ica-ctrl" "gpt-oss-20b_sft-as_qwen-3-30b_vs_gpt-oss-20b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_1shot_trained-adv_ica-ctrl2" "gpt-oss-20b_sft-as_qwen-3-30b_vs_gpt-oss-20b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_1shot_trained-adv_ica-ctrl3" "gpt-oss-20b_sft-as_qwen-3-30b_vs_gpt-oss-20b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_1shot_trained-adv_ica-self" "gpt-oss-20b_sft-as_qwen-3-30b_vs_gpt-oss-20b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_1shot_trained-std_ica-alt" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_1shot_trained-std_ica-ctrl" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_1shot_trained-std_ica-ctrl2" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_1shot_trained-std_ica-ctrl3" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_1shot_trained-std_ica-self" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_5shot_base_ica-alt" "gpt-oss-20b"
run "gpt-oss-20b_UT_PW_5shot_base_ica-ctrl" "gpt-oss-20b"
run "gpt-oss-20b_UT_PW_5shot_base_ica-ctrl2" "gpt-oss-20b"
run "gpt-oss-20b_UT_PW_5shot_base_ica-ctrl3" "gpt-oss-20b"
run "gpt-oss-20b_UT_PW_5shot_base_ica-self" "gpt-oss-20b"
run "gpt-oss-20b_UT_PW_5shot_trained-adv_ica-alt" "gpt-oss-20b_sft-as_qwen-3-30b_vs_gpt-oss-20b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_5shot_trained-adv_ica-ctrl" "gpt-oss-20b_sft-as_qwen-3-30b_vs_gpt-oss-20b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_5shot_trained-adv_ica-ctrl2" "gpt-oss-20b_sft-as_qwen-3-30b_vs_gpt-oss-20b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_5shot_trained-adv_ica-ctrl3" "gpt-oss-20b_sft-as_qwen-3-30b_vs_gpt-oss-20b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_5shot_trained-adv_ica-self" "gpt-oss-20b_sft-as_qwen-3-30b_vs_gpt-oss-20b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_5shot_trained-std_ica-alt" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_5shot_trained-std_ica-ctrl" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_5shot_trained-std_ica-ctrl2" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_5shot_trained-std_ica-ctrl3" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "gpt-oss-20b_UT_PW_5shot_trained-std_ica-self" "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_PW_ShareGPT"

echo "ICA_b2 1/5-shot PW/gpt-oss-20b complete."
