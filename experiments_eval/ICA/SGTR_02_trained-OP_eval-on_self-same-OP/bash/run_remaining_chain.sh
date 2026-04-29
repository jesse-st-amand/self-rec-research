#!/usr/bin/env bash
# Master chain script that runs:
#   Stage 1: gpt-oss-20b SGTR_02 multi-op missing AT_* + UT_IND_10shot stragglers (~32 dirs)
#   Stage 2: ll-3.1-8b SGTR_02 10shot (newly-created 60 dirs across base/std/multi-op)
#   Stage 3: 4 MSJ multi-op batches (multi-op-only configs, 2 models each)
#
# Failures in individual evals do not abort the chain — each stage continues.
# Log: /tmp/chain_remaining.log
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

ICA_DIR="experiments_eval/ICA/SGTR_02_trained-OP_eval-on_self-same-OP"
LL_BASE_GENS="ll-3.1-8b qwen-3.0-30b"
GPT_BASE_GENS="gpt-oss-20b qwen-3.0-30b"

# Read evaluator name from a mini-batch config.yaml
get_evaluator() {
    awk '/^model_names:/{f=1; next} f && /^- /{sub(/^- */,""); print; exit}' "$1"
}

run_eval() {
    local mini_batch_dir="$1"
    local generators="$2"
    local cfg="$mini_batch_dir/config.yaml"
    if [ ! -f "$cfg" ]; then
        echo "  SKIP (no config): $mini_batch_dir"
        return 0
    fi
    local evaluator
    evaluator="$(get_evaluator "$cfg")"
    local leaf
    leaf="$(basename "$mini_batch_dir")"
    echo ""; echo "=== $leaf  (eval: $evaluator) ==="
    uv run srf-eval-sweep \
        --model_names "$evaluator" \
        --generator_models $generators \
        --treatment_type other_models \
        --dataset_dir_path data/input/sharegpt/english2_74 \
        --experiment_config "$cfg" \
        --max-tasks 1 -y \
        || echo "  ⚠ srf-eval-sweep failed for $leaf (continuing)"
}

echo "============================================================"
echo "STAGE 1: gpt-oss-20b SGTR_02 multi-op missing dirs"
echo "============================================================"
GPT_BASE="$ICA_DIR/gpt-oss-20b"
for op in AT_IND AT_PW; do
    for shot in 1shot 5shot 10shot; do
        for cond in alt ctrl ctrl2 ctrl3 self; do
            run_eval "$GPT_BASE/gpt-oss-20b_${op}_${shot}_trained-multi-op_ica-${cond}" "$GPT_BASE_GENS"
        done
    done
done
# UT_IND_10shot stragglers: ica-alt missing entirely; ica-ctrl had only 1 of 2 evals
run_eval "$GPT_BASE/gpt-oss-20b_UT_IND_10shot_trained-multi-op_ica-alt"  "$GPT_BASE_GENS"
run_eval "$GPT_BASE/gpt-oss-20b_UT_IND_10shot_trained-multi-op_ica-ctrl" "$GPT_BASE_GENS"

echo ""
echo "============================================================"
echo "STAGE 2: ll-3.1-8b SGTR_02 10shot (all kinds)"
echo "============================================================"
LL_BASE="$ICA_DIR/ll-3.1-8b"
for op in UT_PW UT_IND AT_PW AT_IND; do
    for kind in base trained-std trained-multi-op; do
        for cond in self alt ctrl ctrl2 ctrl3; do
            run_eval "$LL_BASE/ll-3.1-8b_${op}_10shot_${kind}_ica-${cond}" "$LL_BASE_GENS"
        done
    done
done

echo ""
echo "============================================================"
echo "STAGE 3: MSJ multi-op (4 batches)"
echo "============================================================"
for batch in 1 2 3 4; do
    runner="experiments_eval/MSJ/MSJ_01_multi-op_batch${batch}/bash/attack/00_run_msj_sweep.sh"
    echo ""; echo "=== MSJ_01_multi-op_batch${batch} ==="
    if [ ! -x "$runner" ]; then
        echo "  ⚠ runner missing: $runner"
        continue
    fi
    bash "$runner" || echo "  ⚠ MSJ batch ${batch} failed (continuing)"
done

echo ""
echo "============================================================"
echo "ALL STAGES COMPLETE"
echo "============================================================"
