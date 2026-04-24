#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

bash experiments_eval/ICA/SGTR_02_trained-OP_eval-on_self-same-OP/bash/run_gpt-oss-20b_1_5shot_pw.sh
bash experiments_eval/ICA/SGTR_02_trained-OP_eval-on_self-same-OP/bash/run_qwen-3.0-30b_1_5shot_pw.sh
echo "All ICA_b2 1/5-shot PW runs complete."
