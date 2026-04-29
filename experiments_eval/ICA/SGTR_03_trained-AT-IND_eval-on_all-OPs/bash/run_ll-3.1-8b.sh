#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

ICA_DIR="experiments_eval/ICA/SGTR_03_trained-AT-IND_eval-on_all-OPs/ll-3.1-8b"

run() {
    local leaf="$1"
    local evaluator="$2"
    echo ""; echo "=== ${leaf} ==="
    uv run srf-eval-sweep \
        --model_names "$evaluator" \
        --generator_models ll-3.1-8b qwen-3.0-30b \
        --treatment_type other_models \
        --dataset_dir_path data/input/sharegpt/english2_74 \
        --experiment_config "$ICA_DIR/$leaf/config.yaml" \
        --max-tasks 1 -y
}

run "ll-3.1-8b_AT_PW_1shot_trained-std_ica-alt" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_AT_PW_1shot_trained-std_ica-ctrl" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_AT_PW_1shot_trained-std_ica-ctrl2" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_AT_PW_1shot_trained-std_ica-ctrl3" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_AT_PW_1shot_trained-std_ica-self" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_AT_PW_5shot_trained-std_ica-alt" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_AT_PW_5shot_trained-std_ica-ctrl" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_AT_PW_5shot_trained-std_ica-ctrl2" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_AT_PW_5shot_trained-std_ica-ctrl3" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_AT_PW_5shot_trained-std_ica-self" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_AT_PW_trained-std_no-ica" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_IND_1shot_trained-std_ica-alt" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_IND_1shot_trained-std_ica-ctrl" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_IND_1shot_trained-std_ica-ctrl2" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_IND_1shot_trained-std_ica-ctrl3" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_IND_1shot_trained-std_ica-self" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_IND_5shot_trained-std_ica-alt" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_IND_5shot_trained-std_ica-ctrl" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_IND_5shot_trained-std_ica-ctrl2" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_IND_5shot_trained-std_ica-ctrl3" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_IND_5shot_trained-std_ica-self" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_IND_trained-std_no-ica" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_PW_1shot_trained-std_ica-alt" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_PW_1shot_trained-std_ica-ctrl" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_PW_1shot_trained-std_ica-ctrl2" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_PW_1shot_trained-std_ica-ctrl3" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_PW_1shot_trained-std_ica-self" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_PW_5shot_trained-std_ica-alt" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_PW_5shot_trained-std_ica-ctrl" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_PW_5shot_trained-std_ica-ctrl2" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_PW_5shot_trained-std_ica-ctrl3" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_PW_5shot_trained-std_ica-self" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"
run "ll-3.1-8b_UT_PW_trained-std_no-ica" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT"

echo "SGTR_03_trained-AT-IND_eval-on_all-OPs/ll-3.1-8b complete."
