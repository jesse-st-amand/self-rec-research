#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

bash experiments_eval/ICA/SGTR_02_trained-OP_eval-on_self-same-OP/bash/run_ll-3.1-8b.sh
bash experiments_eval/ICA/SGTR_03_trained-AT-IND_eval-on_all-OPs/bash/run_ll-3.1-8b.sh
bash experiments_eval/ICA/MMLU_01_trained-OP_eval-on_self-same-OP/bash/run_ll-3.1-8b.sh
echo "All llama ICA runs complete."
