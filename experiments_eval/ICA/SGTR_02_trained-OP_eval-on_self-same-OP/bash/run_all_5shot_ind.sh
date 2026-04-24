#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

bash experiments_eval/ICA/SGTR_02_trained-OP_eval-on_self-same-OP/bash/run_gpt-oss-20b_5shot_ind.sh
bash experiments_eval/ICA/SGTR_02_trained-OP_eval-on_self-same-OP/bash/run_qwen-3.0-30b_5shot_ind.sh
echo "All ICA_b2 5-shot IND runs complete."
