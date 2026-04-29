#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

EXP_DIR="experiments_eval/ICA/MMLU_01_trained-OP_eval-on_self-same-OP/ll-3.1-8b"

run() {
    local leaf="$1"
    local evaluator="$2"
    echo ""; echo "=== ${leaf} ==="
    uv run srf-eval-sweep \
        --model_names "$evaluator" \
        --treatment_type other_models \
        --dataset_dir_path data/input/mmlu/mmlu_50 \
        --experiment_config "$EXP_DIR/$leaf/config.yaml" \
        --max-tasks 1 -y
}

run "ll-3.1-8b_trained-multi-op_1shot_ica-alt" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_UT-AT_PW-IND_ShareGPT"
run "ll-3.1-8b_trained-multi-op_1shot_ica-ctrl" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_UT-AT_PW-IND_ShareGPT"
run "ll-3.1-8b_trained-multi-op_1shot_ica-ctrl2" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_UT-AT_PW-IND_ShareGPT"
run "ll-3.1-8b_trained-multi-op_1shot_ica-ctrl3" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_UT-AT_PW-IND_ShareGPT"
run "ll-3.1-8b_trained-multi-op_1shot_ica-self" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_UT-AT_PW-IND_ShareGPT"
run "ll-3.1-8b_trained-multi-op_5shot_ica-alt" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_UT-AT_PW-IND_ShareGPT"
run "ll-3.1-8b_trained-multi-op_5shot_ica-ctrl" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_UT-AT_PW-IND_ShareGPT"
run "ll-3.1-8b_trained-multi-op_5shot_ica-ctrl2" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_UT-AT_PW-IND_ShareGPT"
run "ll-3.1-8b_trained-multi-op_5shot_ica-ctrl3" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_UT-AT_PW-IND_ShareGPT"
run "ll-3.1-8b_trained-multi-op_5shot_ica-self" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_UT-AT_PW-IND_ShareGPT"
run "ll-3.1-8b_trained-multi-op_no-ica" "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_UT-AT_PW-IND_ShareGPT"

echo "MMLU_01 multi-op/ll-3.1-8b complete."
