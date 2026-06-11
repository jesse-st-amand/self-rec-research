#!/usr/bin/env bash
# Master chain: run all qwen multi-op evals in series.
#   Stage 1 — SGTR_02 (64 mini-batches: 4 ops × 3 shots × 5 conds + 4 no-ica)
#   Stage 2 — MMLU_01 (16 mini-batches: 3 shots × 5 conds + 1 no-ica)
#   Stage 3 — MSJ_01 (4 batches × 3 models × shot counts × objectives, full re-run)
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "============================================================"
echo "STAGE 1: SGTR_02 qwen-3.0-30b multi-op"
echo "============================================================"
bash experiments_eval/ICA/SGTR_02_trained-OP_eval-on_self-same-OP/bash/run_qwen-3.0-30b_multi-op.sh \
    || echo "  ⚠ SGTR_02 stage failed (continuing)"

echo ""
echo "============================================================"
echo "STAGE 2: MMLU_01 qwen-3.0-30b multi-op"
echo "============================================================"
bash experiments_eval/ICA/MMLU_01_trained-OP_eval-on_self-same-OP/bash/run_qwen-3.0-30b_multi-op.sh \
    || echo "  ⚠ MMLU_01 stage failed (continuing)"

echo ""
echo "============================================================"
echo "STAGE 3: MSJ_01 multi-op (re-run with qwen entry added)"
echo "============================================================"
bash experiments_eval/MSJ/run_msj_multi-op_qwen.sh \
    || echo "  ⚠ MSJ stage failed (continuing)"

echo ""
echo "============================================================"
echo "ALL QWEN MULTI-OP STAGES COMPLETE"
echo "============================================================"
