#!/usr/bin/env bash
# 10-shot MMLU ICA for ll-3.1-8b: 30 mini-batches across base, trained-multi-op,
# trained-std-{UT,AT}-{IND,PW}. Mirrors the 1shot/5shot blocks of
# run_ll-3.1-8b.sh; reads the evaluator from each config's model_names field.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

ICA_DIR="experiments_eval/ICA/MMLU_01_trained-OP_eval-on_self-same-OP/ll-3.1-8b"

get_evaluator() {
    awk '/^model_names:/{f=1; next} f && /^- /{sub(/^- */,""); print; exit}' "$1"
}

run_eval() {
    local leaf="$1"
    local cfg="$ICA_DIR/$leaf/config.yaml"
    if [ ! -f "$cfg" ]; then
        echo "  SKIP (no config): $leaf"
        return 0
    fi
    local evaluator
    evaluator="$(get_evaluator "$cfg")"
    echo ""; echo "=== $leaf  (eval: $evaluator) ==="
    uv run srf-eval-sweep \
        --model_names "$evaluator" \
        --treatment_type other_models \
        --dataset_dir_path data/input/mmlu/mmlu_50 \
        --experiment_config "$cfg" \
        --max-tasks 1 -y \
        || echo "  ⚠ srf-eval-sweep failed for $leaf (continuing)"
}

# Base (5)
for cond in self alt ctrl ctrl2 ctrl3; do
    run_eval "ll-3.1-8b_base_10shot_ica-${cond}"
done

# Multi-op (5)
for cond in self alt ctrl ctrl2 ctrl3; do
    run_eval "ll-3.1-8b_trained-multi-op_10shot_ica-${cond}"
done

# Trained-std per-op (4 ops × 5 conds = 20)
for op in UT_PW UT_IND AT_PW AT_IND; do
    for cond in self alt ctrl ctrl2 ctrl3; do
        run_eval "ll-3.1-8b_trained-std-${op}_10shot_ica-${cond}"
    done
done

echo ""
echo "MMLU_01 ll-3.1-8b 10-shot complete."
