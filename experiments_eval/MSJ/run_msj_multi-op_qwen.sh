#!/usr/bin/env bash
# Re-run all 4 MSJ multi-op batches end-to-end so the new qwen multi-op model
# entry gets exercised alongside the existing llama + gpt-oss-20b multi-op
# entries. Each batch's run_attack.py overwrites attack_results.json from
# scratch, so this produces a clean 3-models × all-shot-counts dataset.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

for batch in 1 2 3 4; do
    runner="experiments_eval/MSJ/MSJ_01_multi-op_batch${batch}/bash/attack/00_run_msj_sweep.sh"
    echo ""; echo "=== MSJ_01_multi-op_batch${batch} ==="
    if [ ! -x "$runner" ]; then
        echo "  ⚠ runner missing: $runner"; continue
    fi
    bash "$runner" || echo "  ⚠ MSJ batch ${batch} failed (continuing)"
done

echo ""
echo "MSJ multi-op (3-model) sweep complete."
