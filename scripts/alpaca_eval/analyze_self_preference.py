"""Analyze self-preference evaluation results.

Aggregates pairwise annotations into a win-rate matrix and computes
self-preference bias metrics.

Usage:
    uv run python scripts/alpaca_eval/analyze_self_preference.py \
        --results_dir data/alpaca_eval/results \
        --model_names -set dr \
        --output_dir data/alpaca_eval/analysis
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from self_rec_framework.scripts.utils import expand_model_names


def load_win_rates(results_dir: Path, judges: list[str], generators: list[str] | None = None) -> pd.DataFrame:
    """Load pairwise results and compute win-rate matrix.

    Returns a DataFrame where rows = judges, columns = generators,
    values = judge's win rate (fraction of times judge's output was preferred
    over the generator's output).
    """
    if generators is None:
        generators = judges

    matrix = pd.DataFrame(index=judges, columns=generators, dtype=float)

    for judge in judges:
        judge_dir = results_dir / judge
        if not judge_dir.exists():
            continue
        for generator in generators:
            if generator == judge:
                continue
            result_path = judge_dir / f"vs_{generator}.json"
            if not result_path.exists():
                continue
            with open(result_path) as f:
                annotations = json.load(f)
            prefs = [a["preference"] for a in annotations]
            # preference=1 means judge's output (output_1) preferred
            # preference=2 means generator's output preferred
            # preference=1.5 means tie (count as 0.5 win for each)
            wins = sum(1 for p in prefs if p == 1)
            ties = sum(0.5 for p in prefs if p == 1.5)
            total = len(prefs)
            win_rate = (wins + ties) / total if total > 0 else float("nan")
            matrix.loc[judge, generator] = win_rate

    return matrix


def compute_self_preference_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute per-judge self-preference metrics.

    For each judge:
    - avg_self_win_rate: average win rate across all generators
      (how often judge prefers its own output over the generator's)
    - deviation_from_chance: avg_self_win_rate - 0.5 (positive = self-preference bias)
    """
    rows = []
    for judge in matrix.index:
        rates = matrix.loc[judge].dropna()
        if len(rates) == 0:
            continue
        avg_wr = rates.mean()
        rows.append({
            "judge": judge,
            "avg_self_win_rate": avg_wr,
            "deviation_from_chance": avg_wr - 0.5,
            "n_generators": len(rates),
            "min_win_rate": rates.min(),
            "max_win_rate": rates.max(),
        })
    return pd.DataFrame(rows).sort_values("avg_self_win_rate", ascending=False)


def plot_heatmap(matrix: pd.DataFrame, output_path: Path):
    """Plot a judge × opponent win-rate heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Mask diagonal (self vs self)
    mask = np.eye(len(matrix), dtype=bool)

    sns.heatmap(
        matrix.astype(float),
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0.5,
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Judge Win Rate (self-preference)"},
        linewidths=0.5,
    )

    ax.set_xlabel("Opponent (generator of output_2)", fontsize=12)
    ax.set_ylabel("Judge (also generator of output_1)", fontsize=12)
    ax.set_title("Self-Preference Win Rate Matrix\n"
                 "(Value = fraction of times judge preferred its own output)",
                 fontsize=13)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved heatmap to {output_path}")


def plot_deviation_bars(summary: pd.DataFrame, output_path: Path):
    """Plot per-judge self-preference deviation from chance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2ca02c" if v > 0 else "#d62728" for v in summary["deviation_from_chance"]]
    ax.barh(summary["judge"], summary["deviation_from_chance"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Self-Preference Deviation from Chance (0 = no bias)", fontsize=12)
    ax.set_ylabel("Judge Model", fontsize=12)
    ax.set_title("Self-Preference Bias per Judge\n"
                 "(Positive = prefers own output, Negative = prefers others')",
                 fontsize=13)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved deviation plot to {output_path}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Analyze self-preference evaluation results")
    parser.add_argument("--results_dir", default="data/alpaca_eval/results",
                        help="Directory with pairwise evaluation results")
    parser.add_argument("--model_names", nargs="+", default=None,
                        help="Model names to include in analysis")
    parser.add_argument("--config", default=None,
                        help="Path to experiment config YAML (reads model_names from it)")
    parser.add_argument("--output_dir", default="data/alpaca_eval/analysis",
                        help="Directory to save analysis outputs")
    args = parser.parse_args()

    # Resolve model names from --config or --model_names
    if args.config:
        import yaml as _yaml
        with open(args.config) as f:
            config = _yaml.safe_load(f)
        raw_eval = config.get("evaluator_models", config.get("model_names", []))
        raw_gen = config.get("generator_models", config.get("model_names", []))
        judges = expand_model_names(raw_eval)
        generators = expand_model_names(raw_gen)
    elif args.model_names:
        judges = expand_model_names(args.model_names)
        generators = judges
    else:
        parser.error("Provide either --config or --model_names")

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluator models (judges): {', '.join(judges)}")
    print(f"Generator models (opponents): {', '.join(generators)}")
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")

    # Load and compute — matrix rows = judges, columns = generators
    matrix = load_win_rates(results_dir, judges, generators)
    summary = compute_self_preference_summary(matrix)

    # Save CSVs
    matrix.to_csv(output_dir / "self_preference_matrix.csv")
    print(f"✓ Saved win-rate matrix to {output_dir / 'self_preference_matrix.csv'}")

    summary.to_csv(output_dir / "self_preference_summary.csv", index=False)
    print(f"✓ Saved summary to {output_dir / 'self_preference_summary.csv'}")

    # Print summary
    print(f"\n{'='*70}")
    print("SELF-PREFERENCE SUMMARY")
    print(f"{'='*70}")
    print(summary.to_string(index=False))
    print(f"\nOverall mean self-preference: {summary['avg_self_win_rate'].mean():.3f}")
    print(f"Overall mean deviation: {summary['deviation_from_chance'].mean():.3f}")

    # Generate figures
    if len(models) >= 2:
        plot_heatmap(matrix, output_dir / "self_preference_heatmap.png")
        plot_deviation_bars(summary, output_dir / "self_preference_deviation.png")

    print(f"\n✓ Analysis complete. Results in {output_dir}/")


if __name__ == "__main__":
    main()
