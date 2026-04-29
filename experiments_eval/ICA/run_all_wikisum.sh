#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

bash experiments_eval/ICA/SGTR_07_trained-OP-ShareGPT_eval-on_self-same-OP-WikiSum_ICA-ShareGPT/bash/run_gpt-oss-20b.sh
bash experiments_eval/ICA/SGTR_07_trained-OP-ShareGPT_eval-on_self-same-OP-WikiSum_ICA-ShareGPT/bash/run_qwen-3.0-30b.sh
bash experiments_eval/ICA/SGTR_08_trained-AT-IND-ShareGPT_eval-on_all-OPs-WikiSum_ICA-ShareGPT/bash/run_gpt-oss-20b.sh
bash experiments_eval/ICA/SGTR_08_trained-AT-IND-ShareGPT_eval-on_all-OPs-WikiSum_ICA-ShareGPT/bash/run_qwen-3.0-30b.sh
bash experiments_eval/ICA/SGTR_09_trained-OP-ShareGPT_eval-on_self-same-OP-WikiSum_ICA-WikiSum/bash/run_gpt-oss-20b.sh
bash experiments_eval/ICA/SGTR_09_trained-OP-ShareGPT_eval-on_self-same-OP-WikiSum_ICA-WikiSum/bash/run_qwen-3.0-30b.sh
bash experiments_eval/ICA/SGTR_10_trained-AT-IND-ShareGPT_eval-on_all-OPs-WikiSum_ICA-WikiSum/bash/run_gpt-oss-20b.sh
bash experiments_eval/ICA/SGTR_10_trained-AT-IND-ShareGPT_eval-on_all-OPs-WikiSum_ICA-WikiSum/bash/run_qwen-3.0-30b.sh
echo "All wikisum SGTR runs complete."
