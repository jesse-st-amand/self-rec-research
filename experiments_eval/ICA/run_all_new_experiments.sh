#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

bash experiments_eval/ICA/SGTR_04_trained-UT-IND_eval-on_all-OPs/bash/run_all.sh
bash experiments_eval/ICA/SGTR_05_trained-AT-PW_eval-on_all-OPs/bash/run_all.sh
bash experiments_eval/ICA/SGTR_06_trained-UT-PW_eval-on_all-OPs/bash/run_all.sh
echo "All new SGTR experiments complete."
