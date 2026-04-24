#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

bash experiments_eval/ICA/SGTR_04_trained-UT-IND_eval-on_all-OPs/bash/run_gpt-oss-20b.sh
bash experiments_eval/ICA/SGTR_04_trained-UT-IND_eval-on_all-OPs/bash/run_qwen-3.0-30b.sh
echo "All SGTR_04_trained-UT-IND_eval-on_all-OPs runs complete."
