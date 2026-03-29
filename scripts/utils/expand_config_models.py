"""Expand model names from a config YAML, including trained model discovery.

Used by bash scripts to resolve -set references and auto-discover trained
models from training_dir.

Usage:
    # Get evaluator models (with trained variants):
    python scripts/utils/expand_config_models.py config.yaml model_names

    # Get generator models (no trained expansion):
    python scripts/utils/expand_config_models.py config.yaml generator_models --no-trained

Prints one model name per line, with no debug output.
"""

import io
import sys


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <config.yaml> <field> [--no-trained]", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    field = sys.argv[2]
    expand_trained = "--no-trained" not in sys.argv

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    raw = config.get(field, [])
    if not raw:
        sys.exit(0)

    training_dir = config.get("training_dir") if expand_trained else None
    data_subsets = config.get("data_subsets") if expand_trained else None

    # Suppress debug output from expand_model_names
    _real_stdout = sys.stdout
    sys.stdout = io.StringIO()

    from self_rec_framework.scripts.utils import expand_model_names
    result = expand_model_names(raw, training_dir=training_dir, data_subsets=data_subsets)

    sys.stdout = _real_stdout
    for m in result:
        print(m)


if __name__ == "__main__":
    main()
