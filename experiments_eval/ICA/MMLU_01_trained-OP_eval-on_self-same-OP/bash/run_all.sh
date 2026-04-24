#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

bash experiments_eval/ICA/MMLU_01_trained-OP_eval-on_self-same-OP/bash/run_gpt-oss-20b.sh
bash experiments_eval/ICA/MMLU_01_trained-OP_eval-on_self-same-OP/bash/run_qwen-3.0-30b.sh
echo "All MMLU-ICA runs complete."
