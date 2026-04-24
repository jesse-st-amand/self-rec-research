#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

bash experiments_eval/ICA/SGTR_03_trained-AT-IND_eval-on_all-OPs/bash/run_gpt-oss-20b.sh
bash experiments_eval/ICA/SGTR_03_trained-AT-IND_eval-on_all-OPs/bash/run_qwen-3.0-30b.sh
echo "All ICA_b3 runs complete."
