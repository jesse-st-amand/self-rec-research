"""Generate the MMLU-ICA experiment config tree + bash runner.

Produces:
  experiments_eval/MSJ/MMLU_ICA/<model>/<model>_<kind>_<condition>/config.yaml
  experiments_eval/MSJ/MMLU_ICA/bash/run_<model>.sh
  experiments_eval/MSJ/MMLU_ICA/bash/run_all.sh

Matrix (48 leaves):
  models:   gpt-oss-20b, qwen-3.0-30b
  kinds:    base, trained-std-UT_IND, trained-std-AT_IND, trained-adv-UT_IND
  conds:    no-ica, ica-self, ica-alt, ica-ctrl, ica-ctrl2, ica-ctrl3
  (adv only exists for UT_IND, so total = 2 base + 4 std + 2 adv = 8 evaluators × 6 = 48)

Everything sits on top of the ICA plumbing that was already built for SGTR-ICA;
the only knobs that change per cell are `model_names` and `icl_model`.
"""

import os
from pathlib import Path

import yaml


ROOT = Path("/home/jesse-st-amand/projects/python/self-rec-research")
EXP_DIR = ROOT / "experiments_eval/MSJ/MMLU_ICA"
BASH_DIR = EXP_DIR / "bash"


# Per base-model ICA author mapping (matches SGTR-ICA convention, non-thinking variants).
_ALT_MODEL = {
    "gpt-oss-20b": "qwen-3.0-30b",
    "qwen-3.0-30b": "gpt-oss-120b",
}

_CTRL = "deepseek-3.1"
_CTRL2 = "gpt-4o-mini"
_CTRL3 = "sonnet-4.5"


def trained_evaluator(model: str, kind: str) -> str:
    """Return the trained-LoRA evaluator name for a given (base, kind)."""
    if model == "gpt-oss-20b":
        if kind == "trained-std-UT_IND":
            return "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_IND_ShareGPT"
        if kind == "trained-std-AT_IND":
            return "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_IND_ShareGPT"
        if kind == "trained-std-UT_PW":
            return "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_PW_ShareGPT"
        if kind == "trained-std-AT_PW":
            return "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_PW_ShareGPT"
        if kind == "trained-adv-UT_IND":
            return "gpt-oss-20b_sft-as_qwen-3-30b_vs_gpt-oss-20b_UT_IND_ShareGPT"
        if kind == "trained-adv-UT_PW":
            return "gpt-oss-20b_sft-as_qwen-3-30b_vs_gpt-oss-20b_UT_PW_ShareGPT"
    if model == "qwen-3.0-30b":
        if kind == "trained-std-UT_IND":
            return "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_IND_ShareGPT"
        if kind == "trained-std-AT_IND":
            return "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_IND_ShareGPT"
        if kind == "trained-std-UT_PW":
            return "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_PW_ShareGPT"
        if kind == "trained-std-AT_PW":
            return "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_PW_ShareGPT"
        if kind == "trained-adv-UT_IND":
            return "qwen-3-30b_sft-as_gpt-oss-120b_vs_qwen-3-30b_UT_IND_ShareGPT"
        if kind == "trained-adv-UT_PW":
            return "qwen-3-30b_sft-as_gpt-oss-120b_vs_qwen-3-30b_UT_PW_ShareGPT"
    raise ValueError(f"no trained LoRA for model={model} kind={kind}")


def ica_author(model: str, kind: str, condition: str) -> str:
    """Resolve the ICA author model for a (base, kind, condition) triple.

    For std-trained models and the base model: self=base, alt=alt.
    For adv-trained models: self=alt (the labeled-self target), alt=base
      (the actual alt — what the trainer told the model to distinguish FROM self).
    """
    alt = _ALT_MODEL[model]
    is_adv = kind.startswith("trained-adv")
    if condition == "ica-self":
        return alt if is_adv else model
    if condition == "ica-alt":
        return model if is_adv else alt
    if condition == "ica-ctrl":
        return _CTRL
    if condition == "ica-ctrl2":
        return _CTRL2
    if condition == "ica-ctrl3":
        return _CTRL3
    raise ValueError(f"unknown ICA condition: {condition}")


def make_config(model: str, kind: str, condition: str, shots: int | None) -> dict:
    if kind == "base":
        evaluator = model
    else:
        evaluator = trained_evaluator(model, kind)

    cfg: dict = {
        "tags": "NA",
        "format": "MMLU-MC",
        "task": "MMLU",
        "priming": False,
        "temperature": 0.0,
        # Reasoning-heavy models (gpt-oss-*, qwen-3.0-30b) produce "analysis" chains
        # before ANSWER: X. 2048 gives enough headroom without risking truncation.
        "max_final_answer_tokens": 2048,
        "gpu_dispatch": "tinker",
        "model_names": [evaluator],
        "eval_dataset": "mmlu",
        "eval_subset": "mmlu_50",
    }
    if condition != "no-ica":
        assert shots is not None
        cfg["icl_count"] = shots
        cfg["icl_seed"] = 42
        cfg["icl_shuffle_per_sample"] = True
        cfg["icl_dataset"] = "sharegpt"
        cfg["icl_data_subset"] = "english2_74"
        cfg["icl_model"] = ica_author(model, kind, condition)
    return cfg


def main():
    kinds = [
        "base",
        "trained-std-UT_IND", "trained-std-AT_IND",
        "trained-std-UT_PW",  "trained-std-AT_PW",
        "trained-adv-UT_IND", "trained-adv-UT_PW",
    ]
    ica_conds = ["ica-self", "ica-alt", "ica-ctrl", "ica-ctrl2", "ica-ctrl3"]
    shot_counts = [1, 5, 10]
    models = ["gpt-oss-20b", "qwen-3.0-30b"]

    leaves_per_model: dict[str, list[str]] = {m: [] for m in models}
    created = 0
    for model in models:
        for kind in kinds:
            # One no-ICA baseline per (model, kind) — no shot tag
            leaf = f"{model}_{kind}_no-ica"
            leaves_per_model[model].append(leaf)
            cfg = make_config(model, kind, "no-ica", None)
            out_dir = EXP_DIR / model / leaf
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "config.yaml", "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            created += 1

            # ICA conditions × shot counts
            for shots in shot_counts:
                for cond in ica_conds:
                    leaf = f"{model}_{kind}_{shots}shot_{cond}"
                    leaves_per_model[model].append(leaf)
                    cfg = make_config(model, kind, cond, shots)
                    out_dir = EXP_DIR / model / leaf
                    out_dir.mkdir(parents=True, exist_ok=True)
                    with open(out_dir / "config.yaml", "w") as f:
                        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                    created += 1

    BASH_DIR.mkdir(parents=True, exist_ok=True)
    for model in models:
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'cd "$(git rev-parse --show-toplevel)"',
            "",
            f'EXP_DIR="experiments_eval/MSJ/MMLU_ICA/{model}"',
            "",
            "run() {",
            '    local leaf="$1"',
            '    local evaluator="$2"',
            '    echo ""; echo "=== ${leaf} ==="',
            "    uv run srf-eval-sweep \\",
            '        --model_names "$evaluator" \\',
            '        --treatment_type other_models \\',
            '        --dataset_dir_path data/input/mmlu/mmlu_50 \\',
            '        --experiment_config "$EXP_DIR/$leaf/config.yaml" \\',
            '        --max-tasks 1 -y',
            "}",
            "",
        ]
        for leaf in leaves_per_model[model]:
            cfg = yaml.safe_load(open(EXP_DIR / model / leaf / "config.yaml"))
            evaluator = cfg["model_names"][0]
            lines.append(f'run "{leaf}" "{evaluator}"')
        lines.append("")
        lines.append(f'echo "MMLU_ICA/{model} complete."')
        out = BASH_DIR / f"run_{model}.sh"
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.chmod(out, 0o755)

    with open(BASH_DIR / "run_all.sh", "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n"
                'cd "$(git rev-parse --show-toplevel)"\n\n')
        for model in models:
            f.write(f'bash experiments_eval/MSJ/MMLU_ICA/bash/run_{model}.sh\n')
        f.write('echo "All MMLU-ICA runs complete."\n')
    os.chmod(BASH_DIR / "run_all.sh", 0o755)

    print(f"Wrote {created} config YAMLs to {EXP_DIR}")
    print(f"Wrote bash runners to {BASH_DIR}")


if __name__ == "__main__":
    main()
