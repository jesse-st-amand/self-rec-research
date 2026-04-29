#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

bash experiments_eval/ICA/SGTR_02_trained-OP_eval-on_self-same-OP/bash/run_gpt-oss-20b_multi-op.sh
bash experiments_eval/ICA/SGTR_02_trained-OP_eval-on_self-same-OP/bash/run_ll-3.1-8b_multi-op.sh
echo "SGTR_02 multi-op complete (both bases)."
