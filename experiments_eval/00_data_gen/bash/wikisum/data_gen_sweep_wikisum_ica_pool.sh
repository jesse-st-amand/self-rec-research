#!/bin/bash
# Generate gpt-oss-20b, qwen-3.0-30b, gpt-oss-120b responses on WikiSum
# training_set_1-20 + test_set_1-30, for use as self/alt/ICA sources in
# SGTR_07-10. Uses config_wikisum_ica.yaml so output dirs drop the
# _temp_0.0 suffix (matching the existing sharegpt/<model>/ layout).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CFG=experiments_eval/00_data_gen/config_wikisum_ica.yaml

for subset in training_set_1-20 test_set_1-30; do
  uv run srf-generate-sweep \
      --model_names gpt-oss-20b qwen-3.0-30b gpt-oss-120b \
      --dataset_path=data/input/wikisum/${subset} \
      --dataset_config=${CFG}
done
