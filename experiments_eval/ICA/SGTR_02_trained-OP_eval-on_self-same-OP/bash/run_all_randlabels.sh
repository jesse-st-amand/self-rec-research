#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

bash experiments_eval/ICA/SGTR_02_trained-OP_eval-on_self-same-OP/bash/run_ll-3.1-8b_randlabels.sh
bash experiments_eval/ICA/SGTR_02_trained-OP_eval-on_self-same-OP/bash/run_gpt-oss-20b_randlabels.sh

echo "All SGTR_02 randlabels evals complete."
