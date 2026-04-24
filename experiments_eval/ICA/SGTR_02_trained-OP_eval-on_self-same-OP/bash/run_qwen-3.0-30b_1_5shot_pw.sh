#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

ICA_DIR="experiments_eval/ICA/SGTR_02_trained-OP_eval-on_self-same-OP/qwen-3.0-30b"

run() {
    local leaf="$1"
    local evaluator="$2"
    echo ""; echo "=== ${leaf} ==="
    uv run srf-eval-sweep \
        --model_names "$evaluator" \
        --generator_models qwen-3.0-30b gpt-oss-120b \
        --treatment_type other_models \
        --dataset_dir_path data/input/sharegpt/english2_74 \
        --experiment_config "$ICA_DIR/$leaf/config.yaml" \
        --max-tasks 1 -y
}

run "qwen-3.0-30b_AT_PW_1shot_base_ica-alt" "qwen-3.0-30b"
run "qwen-3.0-30b_AT_PW_1shot_base_ica-ctrl" "qwen-3.0-30b"
run "qwen-3.0-30b_AT_PW_1shot_base_ica-ctrl2" "qwen-3.0-30b"
run "qwen-3.0-30b_AT_PW_1shot_base_ica-ctrl3" "qwen-3.0-30b"
run "qwen-3.0-30b_AT_PW_1shot_base_ica-self" "qwen-3.0-30b"
run "qwen-3.0-30b_AT_PW_1shot_trained-std_ica-alt" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_PW_ShareGPT"
run "qwen-3.0-30b_AT_PW_1shot_trained-std_ica-ctrl" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_PW_ShareGPT"
run "qwen-3.0-30b_AT_PW_1shot_trained-std_ica-ctrl2" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_PW_ShareGPT"
run "qwen-3.0-30b_AT_PW_1shot_trained-std_ica-ctrl3" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_PW_ShareGPT"
run "qwen-3.0-30b_AT_PW_1shot_trained-std_ica-self" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_PW_ShareGPT"
run "qwen-3.0-30b_AT_PW_5shot_base_ica-alt" "qwen-3.0-30b"
run "qwen-3.0-30b_AT_PW_5shot_base_ica-ctrl" "qwen-3.0-30b"
run "qwen-3.0-30b_AT_PW_5shot_base_ica-ctrl2" "qwen-3.0-30b"
run "qwen-3.0-30b_AT_PW_5shot_base_ica-ctrl3" "qwen-3.0-30b"
run "qwen-3.0-30b_AT_PW_5shot_base_ica-self" "qwen-3.0-30b"
run "qwen-3.0-30b_AT_PW_5shot_trained-std_ica-alt" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_PW_ShareGPT"
run "qwen-3.0-30b_AT_PW_5shot_trained-std_ica-ctrl" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_PW_ShareGPT"
run "qwen-3.0-30b_AT_PW_5shot_trained-std_ica-ctrl2" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_PW_ShareGPT"
run "qwen-3.0-30b_AT_PW_5shot_trained-std_ica-ctrl3" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_PW_ShareGPT"
run "qwen-3.0-30b_AT_PW_5shot_trained-std_ica-self" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_1shot_base_ica-alt" "qwen-3.0-30b"
run "qwen-3.0-30b_UT_PW_1shot_base_ica-ctrl" "qwen-3.0-30b"
run "qwen-3.0-30b_UT_PW_1shot_base_ica-ctrl2" "qwen-3.0-30b"
run "qwen-3.0-30b_UT_PW_1shot_base_ica-ctrl3" "qwen-3.0-30b"
run "qwen-3.0-30b_UT_PW_1shot_base_ica-self" "qwen-3.0-30b"
run "qwen-3.0-30b_UT_PW_1shot_trained-adv_ica-alt" "qwen-3-30b_sft-as_gpt-oss-120b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_1shot_trained-adv_ica-ctrl" "qwen-3-30b_sft-as_gpt-oss-120b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_1shot_trained-adv_ica-ctrl2" "qwen-3-30b_sft-as_gpt-oss-120b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_1shot_trained-adv_ica-ctrl3" "qwen-3-30b_sft-as_gpt-oss-120b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_1shot_trained-adv_ica-self" "qwen-3-30b_sft-as_gpt-oss-120b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_1shot_trained-std_ica-alt" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_1shot_trained-std_ica-ctrl" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_1shot_trained-std_ica-ctrl2" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_1shot_trained-std_ica-ctrl3" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_1shot_trained-std_ica-self" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_5shot_base_ica-alt" "qwen-3.0-30b"
run "qwen-3.0-30b_UT_PW_5shot_base_ica-ctrl" "qwen-3.0-30b"
run "qwen-3.0-30b_UT_PW_5shot_base_ica-ctrl2" "qwen-3.0-30b"
run "qwen-3.0-30b_UT_PW_5shot_base_ica-ctrl3" "qwen-3.0-30b"
run "qwen-3.0-30b_UT_PW_5shot_base_ica-self" "qwen-3.0-30b"
run "qwen-3.0-30b_UT_PW_5shot_trained-adv_ica-alt" "qwen-3-30b_sft-as_gpt-oss-120b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_5shot_trained-adv_ica-ctrl" "qwen-3-30b_sft-as_gpt-oss-120b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_5shot_trained-adv_ica-ctrl2" "qwen-3-30b_sft-as_gpt-oss-120b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_5shot_trained-adv_ica-ctrl3" "qwen-3-30b_sft-as_gpt-oss-120b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_5shot_trained-adv_ica-self" "qwen-3-30b_sft-as_gpt-oss-120b_vs_qwen-3-30b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_5shot_trained-std_ica-alt" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_5shot_trained-std_ica-ctrl" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_5shot_trained-std_ica-ctrl2" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_5shot_trained-std_ica-ctrl3" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_PW_ShareGPT"
run "qwen-3.0-30b_UT_PW_5shot_trained-std_ica-self" "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_PW_ShareGPT"

echo "ICA_b2 1/5-shot PW/qwen-3.0-30b complete."
