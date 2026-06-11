#!/usr/bin/env bash
# SGTR_02 multi-op evals for qwen-3.0-30b. Mirrors run_gpt-oss-20b_multi-op.sh.
# Reads the evaluator from each config's model_names field (always the qwen
# multi-op LoRA) and runs the full 4 ops × 3 shots × 5 conds + 4 no-ica = 64
# mini-batches.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

ICA_DIR="experiments_eval/ICA/SGTR_02_trained-OP_eval-on_self-same-OP/qwen-3.0-30b"
GENERATORS="qwen-3.0-30b gpt-oss-120b"

get_evaluator() {
    awk '/^model_names:/{f=1; next} f && /^- /{sub(/^- */,""); print; exit}' "$1"
}

run_eval() {
    local leaf="$1"
    local cfg="$ICA_DIR/$leaf/config.yaml"
    if [ ! -f "$cfg" ]; then echo "  SKIP (no config): $leaf"; return 0; fi
    local evaluator
    evaluator="$(get_evaluator "$cfg")"
    echo ""; echo "=== $leaf  (eval: $evaluator) ==="
    uv run srf-eval-sweep \
        --model_names "$evaluator" \
        --generator_models $GENERATORS \
        --treatment_type other_models \
        --dataset_dir_path data/input/sharegpt/english2_74 \
        --experiment_config "$cfg" \
        --max-tasks 1 -y \
        || echo "  ⚠ srf-eval-sweep failed for $leaf (continuing)"
}

for op in UT_PW UT_IND AT_PW AT_IND; do
    for shot in 1shot 5shot 10shot; do
        for cond in self alt ctrl ctrl2 ctrl3; do
            run_eval "qwen-3.0-30b_${op}_${shot}_trained-multi-op_ica-${cond}"
        done
    done
    run_eval "qwen-3.0-30b_${op}_trained-multi-op_no-ica"
done

echo ""
echo "SGTR_02 qwen-3.0-30b multi-op complete."
