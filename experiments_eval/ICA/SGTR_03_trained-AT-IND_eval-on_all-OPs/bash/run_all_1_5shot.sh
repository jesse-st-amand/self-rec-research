#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

bash experiments_eval/ICA/SGTR_03_trained-AT-IND_eval-on_all-OPs/bash/run_gpt-oss-20b_1_5shot.sh
bash experiments_eval/ICA/SGTR_03_trained-AT-IND_eval-on_all-OPs/bash/run_qwen-3.0-30b_1_5shot.sh
echo "All ICA_b3 1/5-shot runs complete."
