"""Analyze adversarial SGTR training effects.

Studies models trained to recognize a *foreign* model's text as their own.
For benchmark data from training runs, accuracy measures the rate the model
self-attributes the foreign model's text. We compute 1-accuracy to measure
how much actual self-recognition was *lost* (i.e., how much the model's
perception of its true self changed).

AlpacaEval results are unchanged — they measure actual SGTR on the
AlpacaEval dataset (the model judging its own real outputs vs others).

This script is a thin wrapper: it inverts training benchmark data, then
calls the same figure functions from analyze_self_preference.py and
prototype_uplift_figures.py.

Usage:
    uv run python scripts/alpaca_eval/analyze_adversarial_SGTR.py \
        --config experiments_eval/COLM/COLM_AE/config.yaml \
        --results_dir data/alpaca_eval/results \
        --output_dir data/alpaca_eval/analysis
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from self_rec_framework.scripts.utils import expand_model_names


def _invert_run_benchmarks(runs: list[dict]) -> list[dict]:
    """Invert benchmark accuracies for adversarial runs only.

    Only inverts when base_model != identity_model (the model is trained
    to recognize a foreign model's text as its own). When base == identity,
    the run is normal SGTR and accuracies are left unchanged.

    For inverted runs, replaces bp_data values:
        pre = 1 - original_pre
        post = 1 - original_post

    This converts "foreign-text self-attribution rate" into
    "actual self-recognition remaining".

    AE data (ae_base, ae_trained) is left unchanged.
    """
    for r in runs:
        if not r.get("is_adversarial", False):
            continue

        bp = r.get("bp_data", {})
        inverted = {}
        for bench, vals in bp.items():
            if isinstance(vals, dict) and "pre" in vals:
                inverted[bench] = {
                    "pre": 1 - vals["pre"] if vals["pre"] is not None else None,
                    "post": 1 - vals["post"] if vals["post"] is not None else None,
                }
            elif isinstance(vals, (list, tuple)) and len(vals) == 2:
                pre, post = vals
                inverted[bench] = (
                    1 - pre if pre is not None else None,
                    1 - post if post is not None else None,
                )
            else:
                inverted[bench] = vals
        r["bp_data"] = inverted
    return runs


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Analyze adversarial SGTR training effects")
    parser.add_argument("--config", required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--results_dir", default="data/alpaca_eval/results",
                        help="Root directory with AE results")
    parser.add_argument("--output_dir", default="data/alpaca_eval/analysis",
                        help="Root output directory for analysis")
    args = parser.parse_args()

    import yaml as _yaml
    with open(args.config) as f:
        config = _yaml.safe_load(f)

    data_subsets = config.get("data_subsets", None)
    training_dir = config.get("training_dir", "data/training")
    judge_modes = config.get("judge_mode", ["simple"])
    if isinstance(judge_modes, str):
        judge_modes = [judge_modes]

    raw_eval = config.get("evaluator_models", [])
    raw_gen = config.get("generator_models", [])
    judges = expand_model_names(raw_eval, training_dir=training_dir, data_subsets=data_subsets)
    generators = expand_model_names(raw_gen)

    print(f"Adversarial SGTR Analysis")
    print(f"Data subsets: {data_subsets}")
    print(f"Judge modes: {judge_modes}")
    print(f"Evaluator models: {', '.join(judges)}")

    # Import figure functions from the standard analysis pipeline
    import scripts.figures.prototype_uplift_figures as _uplift
    from scripts.alpaca_eval.analyze_self_preference import (
        analyze_ranking_mode,
        load_ranking_self_ranks,
        plot_ranking_delta,
        load_self_selection_rates,
        compute_self_preference_summary,
        plot_simple_self_preference_delta,
    )
    from scripts.alpaca_eval.run_self_preference import resolve_base_model
    from scripts.alpaca_eval.training_runs import discover_training_runs

    subsets_to_run = data_subsets if data_subsets else [None]

    # Build judge-to-subset mapping
    all_runs = discover_training_runs(training_dir, subsets=data_subsets)
    trained_to_subset = {r.trained_name: r.subset for r in all_runs}

    for subset in subsets_to_run:
        subset_label = subset or "all"
        subset_out = Path(args.output_dir) / subset_label
        subset_out.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"ADVERSARIAL ANALYSIS — {subset_label}")
        print(f"{'='*70}")

        # --- Non-AE figures (inverted benchmark data) ---
        # Save/restore uplift module state
        orig_out = _uplift.OUT_DIR
        orig_td = _uplift.TRAINING_DIR
        orig_subsets = getattr(_uplift, 'TRAINING_SUBSETS', None)

        _uplift.OUT_DIR = subset_out
        _uplift.TRAINING_DIR = training_dir
        _uplift.TRAINING_SUBSETS = [subset] if subset else None

        # Discover runs, invert benchmarks, then generate figures
        raw_runs = _uplift._discover_training_runs(
            training_dir=training_dir,
            subsets=[subset] if subset else None,
        )

        if not raw_runs:
            print(f"  ⚠ No training data for {subset_label}")
            _uplift.OUT_DIR = orig_out
            _uplift.TRAINING_DIR = orig_td
            _uplift.TRAINING_SUBSETS = orig_subsets
            continue

        inverted_runs = _invert_run_benchmarks(raw_runs)

        # Rename base models to distinguish adversarial vs normal runs.
        # Append the identity model so they plot as separate arrows
        # instead of being averaged together.
        # e.g., "Qwen 3.0 30B" → "Qwen 3.0 30B (as GPT-OSS 120B)" for adversarial
        #        "Qwen 3.0 30B" → "Qwen 3.0 30B (as self)" for normal
        from scripts.figures.prototype_uplift_figures import BASE_MODEL_COLORS

        IDENTITY_DISPLAY = {
            "gpt-oss-120b": "GPT-OSS 120B",
            "gpt-oss-20b": "GPT-OSS 20B",
            "ll-3.1-8b": "Llama 3.1 8B",
            "ll-3.3-70b": "Llama 3.3 70B",
            "qwen-3.0-30b": "Qwen 3.0 30B",
            "qwen-2.5-7b": "Qwen 2.5 7B",
        }

        for r in inverted_runs:
            if r.get("is_adversarial"):
                identity_short = r.get("identity_model", "?")
                identity_name = IDENTITY_DISPLAY.get(identity_short, identity_short)
                r["base"] = f"{r['base']} (as {identity_name})"
                BASE_MODEL_COLORS.setdefault(r["base"], "#C62828")
            else:
                r["base"] = f"{r['base']} (as self)"
                BASE_MODEL_COLORS.setdefault(r["base"], "#2E7D32")

        print(f"  Found {len(inverted_runs)} training runs ({sum(1 for r in inverted_runs if r.get('is_adversarial'))} adversarial, {sum(1 for r in inverted_runs if not r.get('is_adversarial'))} normal)")

        # Monkey-patch _discover_training_runs to return inverted data,
        # and _load_val_accuracy / _load_benchmark_accuracies to invert values
        _original_discover = _uplift._discover_training_runs
        _original_load_val = _uplift._load_val_accuracy
        _original_load_bp = _uplift._load_benchmark_accuracies

        # Build set of adversarial run_dirs for selective inversion
        adversarial_dirs = {r["run_dir"] for r in inverted_runs if r.get("is_adversarial")}

        def _patched_discover(**kwargs):
            return inverted_runs

        def _patched_load_val(run_dir):
            pre, post = _original_load_val(run_dir)
            if str(run_dir) in adversarial_dirs:
                pre = 1 - pre if pre is not None else None
                post = 1 - post if post is not None else None
            return pre, post

        def _patched_load_bp(run_dir):
            bp = _original_load_bp(run_dir)
            if str(run_dir) in adversarial_dirs:
                return {k: (1 - pre if pre is not None else None,
                            1 - post if post is not None else None)
                        for k, (pre, post) in bp.items()}
            return bp

        _uplift._discover_training_runs = _patched_discover
        _uplift._load_val_accuracy = _patched_load_val
        _uplift._load_benchmark_accuracies = _patched_load_bp

        try:
            print("\n  Task transfer figures (inverted)...")
            _uplift.fig_arrow_task_panels_real()
            print("\n  Dataset transfer figures (inverted)...")
            _uplift.fig_arrow_dataset_panels_real()
            print("\n  Transfer heatmap (inverted)...")
            _uplift.fig_task_transfer_heatmap_real()
        finally:
            # Restore original functions and state
            _uplift._discover_training_runs = _original_discover
            _uplift._load_val_accuracy = _original_load_val
            _uplift._load_benchmark_accuracies = _original_load_bp
            _uplift.OUT_DIR = orig_out
            _uplift.TRAINING_DIR = orig_td
            _uplift.TRAINING_SUBSETS = orig_subsets

        # --- AE figures (unchanged, per judge mode) ---
        def _judge_in_subset(judge_name, _subset=subset):
            if judge_name in trained_to_subset:
                return trained_to_subset[judge_name] == _subset
            return resolve_base_model(judge_name) == judge_name

        for mode in judge_modes:
            mode_results = Path(args.results_dir) / subset_label / mode
            mode_out = subset_out / mode
            mode_out.mkdir(parents=True, exist_ok=True)

            print(f"\n  --- {subset_label} / {mode} ---")

            if mode == "ranking" and mode_results.exists():
                available = sorted([
                    d.name for d in mode_results.iterdir()
                    if d.is_dir() and (d / "ranking.json").exists()
                    and _judge_in_subset(d.name)
                ])
                if available:
                    print(f"    Ranking: {len(available)} judges")
                    analyze_ranking_mode(mode_results, mode_out, available)
                else:
                    print(f"    No ranking results")

            elif mode == "simple" and mode_results.exists():
                available = sorted([
                    d.name for d in mode_results.iterdir()
                    if d.is_dir() and any(d.glob("vs_*.json"))
                    and _judge_in_subset(d.name)
                ])
                if available:
                    print(f"    Simple: {len(available)} judges")
                    matrix = load_self_selection_rates(mode_results, available, generators)
                    summary = compute_self_preference_summary(matrix)
                    if not summary.empty:
                        plot_simple_self_preference_delta(
                            summary, mode_out / "self_preference_delta_arrows.png")

    print(f"\n✓ Adversarial analysis complete. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
