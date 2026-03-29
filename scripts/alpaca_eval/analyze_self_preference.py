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


def load_self_selection_rates(results_dir: Path, judges: list[str], generators: list[str] | None = None) -> pd.DataFrame:
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
            # None/NaN = failed to parse
            valid = [p for p in prefs if p is not None and not (isinstance(p, float) and np.isnan(p))]
            wins = sum(1 for p in valid if p == 1)
            ties = sum(0.5 for p in valid if p == 1.5)
            total = len(valid)
            null_count = len(prefs) - total
            if null_count > 0:
                print(f"  ⚠ {judge} vs {generator}: {null_count}/{len(prefs)} unparseable preferences")
            self_selection_rate = (wins + ties) / total if total > 0 else float("nan")
            matrix.loc[judge, generator] = self_selection_rate

    return matrix


def compute_self_preference_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute per-judge self-preference metrics.

    For each judge:
    - avg_self_selection_rate: average self-selection rate across all generators
      (how often judge selects its base model's output over the generator's)
    - deviation_from_chance: avg_self_selection_rate - 0.5 (positive = self-preference bias)
    """
    rows = []
    for judge in matrix.index:
        rates = matrix.loc[judge].dropna()
        if len(rates) == 0:
            continue
        avg_wr = rates.mean()
        rows.append({
            "judge": judge,
            "avg_self_selection_rate": avg_wr,
            "deviation_from_chance": avg_wr - 0.5,
            "n_generators": len(rates),
            "min_self_selection_rate": rates.min(),
            "max_self_selection_rate": rates.max(),
        })
    return pd.DataFrame(rows).sort_values("avg_self_selection_rate", ascending=False)


def plot_heatmap(matrix: pd.DataFrame, output_path: Path):
    """Plot a judge × opponent win-rate heatmap."""
    n_judges, n_gens = matrix.shape
    fig_w = max(6, n_gens * 1.5 + 3)
    fig_h = max(4, n_judges * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Mask cells where judge's base model == generator (NaN in the matrix)
    mask = matrix.astype(float).isna()

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
        cbar_kws={"label": "Judge Self-Selection Rate"},
        linewidths=0.5,
    )

    ax.set_xlabel("Opponent (generator)", fontsize=12)
    ax.set_ylabel("Judge", fontsize=12)
    ax.set_title("Judge Self-Selection Rate Matrix\n"
                 "(Value = fraction of times judge selected its base model's output)",
                 fontsize=13)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved heatmap to {output_path}")


def plot_delta_heatmap(matrix: pd.DataFrame, output_path: Path):
    """Plot a heatmap showing delta from base model self-selection rate.

    For each trained model row, subtracts the corresponding base model's
    rate for the same opponent. Base model rows show 0.0 (reference).
    """
    from scripts.alpaca_eval.run_self_preference import resolve_base_model

    # Identify base models (rows that are their own base)
    base_rows = {}
    for judge in matrix.index:
        base = resolve_base_model(judge)
        if base == judge:
            base_rows[judge] = matrix.loc[judge]

    # Build delta matrix (trained models only — skip base model rows)
    trained_judges = [j for j in matrix.index if resolve_base_model(j) != j]
    delta = matrix.loc[trained_judges].copy().astype(float)
    for judge in delta.index:
        base = resolve_base_model(judge)
        if base in base_rows:
            for col in delta.columns:
                base_val = base_rows[base].get(col)
                judge_val = delta.loc[judge, col]
                if pd.notna(judge_val) and pd.notna(base_val):
                    delta.loc[judge, col] = judge_val - base_val
                else:
                    delta.loc[judge, col] = float("nan")

    n_judges, n_gens = delta.shape
    fig_w = max(6, n_gens * 1.5 + 3)
    fig_h = max(4, n_judges * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    mask = delta.isna()

    # Diverging colormap centered at 0
    vmax = max(0.3, delta.abs().max().max())
    sns.heatmap(
        delta,
        mask=mask,
        annot=True,
        fmt="+.2f",
        cmap="RdYlGn",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        cbar_kws={"label": "Δ Self-Selection Rate (trained − base)"},
        linewidths=0.5,
    )

    ax.set_xlabel("Opponent (generator)", fontsize=12)
    ax.set_ylabel("Judge", fontsize=12)
    ax.set_title("Training Effect on Self-Selection Rate\n"
                 "(Δ = trained model rate − untrained base model rate)",
                 fontsize=13)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved delta heatmap to {output_path}")


def plot_training_effect_per_opponent(results_dir: Path, judges: list, output_path: Path):
    """Plot per-opponent self-selection rate: base (gray) vs trained avg (colored).

    One panel per base model. X-axis = opponent models. Arrows from base rate
    to trained average rate, similar to the uplift arrow plots.

    Only includes base models that have at least one trained version.
    """
    from scripts.alpaca_eval.run_self_preference import resolve_base_model, discover_trained_models

    def load_per_opponent_rates(judge_name):
        """Load {opponent: self_selection_rate} for a judge."""
        for name in [judge_name, judge_name.replace("_tinker_small", "")]:
            judge_dir = results_dir / name
            if not judge_dir.exists():
                continue
            rates = {}
            for f in judge_dir.glob("vs_*.json"):
                opp = f.stem.replace("vs_", "")
                with open(f) as fh:
                    data = json.load(fh)
                valid = [d["preference"] for d in data
                         if d.get("preference") is not None and d["preference"] == d["preference"]]
                if valid:
                    rates[opp] = sum(1 for p in valid if p == 1.0) / len(valid)
            return rates
        return {}

    # Find base models that have trained versions
    base_models = set()
    for j in judges:
        base = resolve_base_model(j)
        base_models.add(base)

    bases_with_trained = {}
    for base in sorted(base_models):
        trained = discover_trained_models([base])
        if trained:
            bases_with_trained[base] = trained

    if not bases_with_trained:
        print("  ⚠ No base models with trained versions found, skipping training effect plot")
        return

    base_colors_map = {
        "ll-3.1-8b": "#1565C0",
        "ll-3.3-70b": "#E65100",
        "qwen-3.0-30b": "#2E7D32",
    }

    n_bases = len(bases_with_trained)
    fig, axes = plt.subplots(1, n_bases, figsize=(n_bases * 5 + 1, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for b_idx, (base, trained_list) in enumerate(bases_with_trained.items()):
        ax = axes[b_idx]
        post_color = base_colors_map.get(base, "#1565C0")

        # Load base model rates
        base_rates = load_per_opponent_rates(base)

        # Load and average trained model rates per opponent
        trained_rates_per_opp = {}
        for t in trained_list:
            t_rates = load_per_opponent_rates(t)
            for opp, rate in t_rates.items():
                trained_rates_per_opp.setdefault(opp, []).append(rate)
        avg_trained_rates = {opp: np.mean(rates) for opp, rates in trained_rates_per_opp.items()}

        # Combine opponents that exist in both
        all_opps = sorted(set(base_rates.keys()) | set(avg_trained_rates.keys()))
        if not all_opps:
            continue

        for o_idx, opp in enumerate(all_opps):
            base_val = base_rates.get(opp)
            trained_val = avg_trained_rates.get(opp)

            if base_val is not None:
                ax.scatter(o_idx, base_val, color="gray", s=40, zorder=3,
                           edgecolors="black", linewidth=0.4)
            if trained_val is not None:
                ax.scatter(o_idx, trained_val, color=post_color, s=50, zorder=5,
                           edgecolors="black", linewidth=0.4)
            if base_val is not None and trained_val is not None:
                delta = trained_val - base_val
                arrow_color = "#2E7D32" if delta > 0 else "#C62828"
                ax.annotate("", xy=(o_idx, trained_val), xytext=(o_idx, base_val),
                            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5),
                            zorder=4)
                ax.text(o_idx + 0.15, (base_val + trained_val) / 2, f"{delta:+.2f}",
                        fontsize=6.5, color=arrow_color, va="center")

        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xticks(range(len(all_opps)))
        ax.set_xticklabels(all_opps, fontsize=7, rotation=40, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{base}\n(avg of {len(trained_list)} trained versions)",
                     fontsize=10, fontweight="bold")
        if b_idx == 0:
            ax.set_ylabel("Self-Selection Rate", fontsize=10)

    # Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='gray', markersize=6,
                    linestyle='None', markeredgecolor='black', markeredgewidth=0.4,
                    label="Base model (untrained)"),
        plt.Line2D([0], [0], marker='o', color='#1565C0', markersize=6,
                    linestyle='None', markeredgecolor='black', markeredgewidth=0.4,
                    label="Trained (avg over versions)"),
    ]
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Training Effect on Self-Selection Rate per Opponent (AlpacaEval)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved training effect per opponent to {output_path}")


def plot_score_distance_training_effect(output_path: Path, training_dir: str = "data/training",
                                        data_subsets: list[str] | None = None):
    """2×2 paneled score-distance scatter using real training benchmark data.

    Each panel = one task OP (UT PW, UT IND, AT PW, AT IND).
    Gray = epoch 0 (pre-training), Blue = final epoch (post-training).
    Each dot = one (evaluator, opponent) pair from a training run.
    X-axis = Elo score distance (evaluator - opponent).
    """
    from self_rec_framework.src.helpers.model_names import LM_ARENA_SCORES

    def get_score(model):
        if model in LM_ARENA_SCORES:
            return LM_ARENA_SCORES[model]
        base = model.replace("-thinking", "")
        return LM_ARENA_SCORES.get(base)

    OPPONENT_SHORT = {
        "qwen": "qwen-2.5-7b", "gpt_4o": "gpt-4o", "haiku_3_5": "haiku-3.5",
        "opus_4_1": "opus-4.1", "ll_3_1_70b": "ll-3.1-70b", "ll_3_3_70b": "ll-3.3-70b",
        "qwen3_30b": "qwen-3.0-30b", "ll_3_1_8b": "ll-3.1-8b",
    }

    # Map task OP panels to benchmark prediction dir names (_full preferred)
    panels = [
        ("(a) UT PW (Rec)", ["xeval_tag_at_pw_full", "xeval_tag_at_pw"]),
        ("(b) UT IND (Rec)", ["xeval_format_ind_full", "xeval_format_ind"]),
        ("(c) AT PW (Rec)", ["xeval_tag_at_pw_full", "xeval_tag_at_pw"]),
        ("(d) AT IND (Rec)", ["xeval_tag_at_ind_full", "xeval_tag_at_ind"]),
    ]

    # Collect data points from all training runs using unified module
    from scripts.alpaca_eval.training_runs import (
        discover_training_runs, get_benchmark_accuracy, map_benchmark_to_op,
    )

    runs = discover_training_runs(training_dir, subsets=data_subsets)
    if not runs:
        print("  ⚠ No training data found")
        return

    # Map panel titles to OP labels from map_benchmark_to_op
    PANEL_OP_MAP = {
        "(a) UT PW (Rec)": "UT PW",
        "(b) UT IND (Rec)": "UT IND",
        "(c) AT PW (Rec)": "AT PW",
        "(d) AT IND (Rec)": "AT IND",
    }

    panel_points = {title: {"pre": [], "post": []} for title, _ in panels}

    for run in runs:
        eval_score = get_score(run.base_model)
        opp_score = get_score(run.opponent)
        if eval_score is None or opp_score is None:
            continue
        score_dist = eval_score - opp_score

        for bench in run.benchmarks:
            op = map_benchmark_to_op(bench, run)
            if op is None:
                continue

            # Find which panel this OP belongs to
            for title, _ in panels:
                expected_op = PANEL_OP_MAP.get(title)
                if expected_op and op == expected_op:
                    pre = get_benchmark_accuracy(run, bench, epoch=0)
                    post = get_benchmark_accuracy(run, bench, epoch=None)
                    if pre is not None:
                        panel_points[title]["pre"].append((score_dist, pre))
                    if post is not None:
                        panel_points[title]["post"].append((score_dist, post))
                    break

    # Plot 2×2
    pre_color = "#AAAAAA"
    post_color = "#1565C0"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                              gridspec_kw={"wspace": 0.15, "hspace": 0.15})

    for idx, (title, _) in enumerate(panels):
        row, col = divmod(idx, 2)
        ax = axes[row][col]
        pre_pts = panel_points[title]["pre"]
        post_pts = panel_points[title]["post"]

        if pre_pts:
            xs, ys = zip(*pre_pts)
            ax.scatter(xs, ys, c=pre_color, alpha=0.5, s=35, edgecolors="black",
                       linewidth=0.3, label="Pre-training (epoch 0)", zorder=2)
            if len(xs) > 2:
                coeffs = np.polyfit(xs, ys, 1)
                x_line = np.linspace(min(xs), max(xs), 100)
                ax.plot(x_line, coeffs[0] * x_line + coeffs[1],
                        color="#888888", linewidth=1.5, linestyle="--", alpha=0.7, zorder=3)

        if post_pts:
            xs, ys = zip(*post_pts)
            ax.scatter(xs, ys, c=post_color, alpha=0.8, s=40, edgecolors="black",
                       linewidth=0.3, label="Post-training (final epoch)", zorder=4)
            if len(xs) > 2:
                coeffs = np.polyfit(xs, ys, 1)
                x_line_post = np.linspace(min(xs), max(xs), 100)
                ax.plot(x_line_post, coeffs[0] * x_line_post + coeffs[1],
                        color="#0D47A1", linewidth=2, alpha=0.9, zorder=5)

        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0.0, 1.05)

        if col == 0:
            ax.set_ylabel("Recognition Accuracy", fontsize=10)
        if row == 1:
            ax.set_xlabel("Elo Score Distance\n(Evaluator − Generator)", fontsize=10)
        else:
            ax.set_xticklabels([])
        if idx == 0:
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Training Effect on Score-Distance Relationship\n"
                 "(Real data from benchmark_predictions)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved score-distance training effect to {output_path}")


def plot_score_distance_heldout(output_path: Path, training_dir: str = "data/training",
                                data_subsets: list[str] | None = None):
    """2×2 paneled score-distance scatter using held-out validation accuracy.

    Each panel = one training OP (UT PW, UT IND, AT PW, AT IND).
    Only includes runs trained on that OP, using their val/accuracy
    (held-out test set from the same task). Gray = epoch 0, Blue = final.
    X-axis = Elo score distance (evaluator - opponent).
    """
    from self_rec_framework.src.helpers.model_names import LM_ARENA_SCORES
    from scripts.alpaca_eval.training_runs import discover_training_runs, get_val_accuracy

    def get_score(model):
        if model in LM_ARENA_SCORES:
            return LM_ARENA_SCORES[model]
        return LM_ARENA_SCORES.get(model.replace("-thinking", ""))

    runs = discover_training_runs(training_dir, subsets=data_subsets)
    if not runs:
        print("  ⚠ No training data found")
        return

    panels = [
        ("(a) UT PW", "ut", "pw"),
        ("(b) UT IND", "ut", "ind"),
        ("(c) AT PW", "at", "pw"),
        ("(d) AT IND", "at", "ind"),
    ]

    panel_points = {title: {"pre": [], "post": []} for title, _, _ in panels}

    for run in runs:
        eval_score = get_score(run.base_model)
        opp_score = get_score(run.opponent)
        if eval_score is None or opp_score is None:
            continue
        score_dist = eval_score - opp_score

        pre = get_val_accuracy(run, epoch=0)
        post = get_val_accuracy(run)

        for title, tag, fmt in panels:
            if run.tag == tag and run.fmt == fmt:
                if pre is not None:
                    panel_points[title]["pre"].append((score_dist, pre))
                if post is not None:
                    panel_points[title]["post"].append((score_dist, post))
                break

    pre_color = "#AAAAAA"
    post_color = "#1565C0"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                              gridspec_kw={"wspace": 0.15, "hspace": 0.15})

    for idx, (title, _, _) in enumerate(panels):
        row, col = divmod(idx, 2)
        ax = axes[row][col]
        pre_pts = panel_points[title]["pre"]
        post_pts = panel_points[title]["post"]

        if pre_pts:
            xs, ys = zip(*pre_pts)
            ax.scatter(xs, ys, c=pre_color, alpha=0.5, s=50, edgecolors="black",
                       linewidth=0.4, label="Pre-training (epoch 0)", zorder=2)
            if len(xs) > 2:
                coeffs = np.polyfit(xs, ys, 1)
                x_line = np.linspace(min(xs), max(xs), 100)
                ax.plot(x_line, coeffs[0] * x_line + coeffs[1],
                        color="#888888", linewidth=1.5, linestyle="--", alpha=0.7, zorder=3)

        if post_pts:
            xs, ys = zip(*post_pts)
            ax.scatter(xs, ys, c=post_color, alpha=0.8, s=55, edgecolors="black",
                       linewidth=0.4, label="Post-training (final epoch)", zorder=4)
            if len(xs) > 2:
                coeffs = np.polyfit(xs, ys, 1)
                x_line = np.linspace(min(xs), max(xs), 100)
                ax.plot(x_line, coeffs[0] * x_line + coeffs[1],
                        color="#0D47A1", linewidth=2, alpha=0.9, zorder=5)

        if not pre_pts and not post_pts:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=11, color="gray", fontstyle="italic", transform=ax.transAxes)

        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0.0, 1.05)

        if col == 0:
            ax.set_ylabel("Held-Out Val Accuracy", fontsize=10)
        if row == 1:
            ax.set_xlabel("Elo Score Distance\n(Evaluator − Opponent)", fontsize=10)
        else:
            ax.set_xticklabels([])
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Score-Distance vs Held-Out Accuracy\n"
                 "(validation set from same task as training)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved score-distance held-out plot to {output_path}")


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


def _shorten_trained_label(judge_name: str, base_model: str) -> str:
    """Shorten a trained model name for display on plots.

    Handles reorganized, archived, and original naming conventions.
    """
    import re
    # Strip -thinking suffix for label parsing, it's implied by the base model
    clean_name = judge_name.removesuffix("-thinking")
    suffix = clean_name.replace(base_model.removesuffix("-thinking") + "-", "")

    # Reorganized: "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-2-5-7b_UT_PW_ShareGPT"
    m = re.match(r".+_sft-as_(.+?)_vs_(.+?)_(UT|AT)_(PW|IND)_(\w+)$", suffix)
    if m:
        identity, opponent, tag, fmt, dataset = m.groups()
        label = f"vs {opponent} {tag} {fmt} {dataset}"
        # Only show identity if different from base (train_as case)
        from scripts.alpaca_eval.training_runs import REORG_MODEL_MAP
        identity_short = REORG_MODEL_MAP.get(identity, identity)
        if identity_short != base_model.removesuffix("-thinking"):
            label = f"as {identity} {label}"
        return label

    # Archived: "11_archived_ll8b_ut_pw_sharegpt_vs_qwen25"
    m = re.match(r"\d+_archived_\w+_(ut|at)_(pw|ind)_(\w+)_vs_(\w+)", suffix)
    if m:
        tag, fmt, dataset, opponent = m.groups()
        return f"vs {opponent} {tag} {fmt} {dataset}"

    # Archived train_as
    m = re.match(r"\d+_archived_\w+_(ut|at)_(pw|ind)_(\w+)_train_as_(\w+)_vs_(\w+)", suffix)
    if m:
        tag, fmt, dataset, train_as, opponent = m.groups()
        return f"as {train_as} vs {opponent} {tag} {fmt} {dataset}"

    # Original naming
    suffix = suffix.replace("01_sft_pw_vs_", "pw vs ")
    suffix = suffix.replace("02_sft_ind_vs_", "ind vs ")
    suffix = suffix.replace("03_sft_pw_vs_", "pw vs ")
    suffix = suffix.replace("01_sft_pw_", "pw ")
    suffix = suffix.replace("_tinker_small", "")
    return suffix


def plot_simple_self_preference_delta(summary: pd.DataFrame, output_path: Path):
    """Arrow plot showing change in avg self-selection rate from base to trained models.

    Similar to ranking_delta but for the simple pairwise mode.
    X-axis = self-selection rate (0.5 = chance).
    Gray dot = base model rate, colored dot = trained model rate.
    """
    from scripts.alpaca_eval.run_self_preference import resolve_base_model

    DISPLAY = {
        "ll-3.1-8b": "Llama 3.1 8B",
        "ll-3.3-70b": "Llama 3.3 70B",
        "qwen-3.0-30b": "Qwen 3.0 30B",
        "qwen-3.5-27b": "Qwen 3.5 27B",
        "gpt-oss-20b": "GPT-OSS 20B",
    }
    BASE_COLORS = {
        "ll-3.1-8b": "#1565C0",
        "ll-3.3-70b": "#E65100",
        "qwen-3.0-30b": "#2E7D32",
        "qwen-3.5-27b": "#7B1FA2",
        "gpt-oss-20b": "#C62828",
    }

    summary = summary.copy()
    summary["base_model"] = summary["judge"].apply(resolve_base_model)
    summary["is_trained"] = summary["judge"] != summary["base_model"]

    base_rates = {}
    for _, row in summary[~summary["is_trained"]].iterrows():
        base_rates[row["base_model"]] = row["avg_self_selection_rate"]

    trained = summary[summary["is_trained"]].copy()
    if trained.empty:
        print("  ⚠ No trained models for delta plot — skipping")
        return

    trained["base_rate"] = trained["base_model"].map(base_rates)
    trained = trained.dropna(subset=["base_rate"])
    if trained.empty:
        print("  ⚠ No matching base models — skipping")
        return

    trained["delta"] = trained["avg_self_selection_rate"] - trained["base_rate"]
    base_models = sorted(trained["base_model"].unique())

    n_rows = len(trained)
    fig, ax = plt.subplots(figsize=(10, max(5, n_rows * 0.5 + 2)))

    y_pos = 0
    y_ticks = []
    y_labels = []
    group_boundaries = []
    group_tops = {}

    for base in base_models:
        group = trained[trained["base_model"] == base].sort_values("avg_self_selection_rate")
        if group.empty:
            continue
        color = BASE_COLORS.get(base, "#666666")
        group_tops[base] = y_pos

        for _, row in group.iterrows():
            base_r = row["base_rate"]
            trained_r = row["avg_self_selection_rate"]

            ax.annotate("", xy=(trained_r, y_pos), xytext=(base_r, y_pos),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
            ax.scatter([base_r], [y_pos], color="#AAAAAA", s=40, zorder=5,
                       edgecolors="black", linewidth=0.5)
            ax.scatter([trained_r], [y_pos], color=color, s=50, zorder=5,
                       edgecolors="black", linewidth=0.5)

            label = _shorten_trained_label(row["judge"], base)
            y_ticks.append(y_pos)
            y_labels.append(label)
            y_pos += 1

        group_boundaries.append(y_pos - 0.5)
        y_pos += 1.0

    for yb in group_boundaries[:-1]:
        ax.axhline(y=yb, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Chance (0.5)")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Average Self-Selection Rate\n(0.5 = chance, higher = more self-preference)",
                  fontsize=10)
    ax.set_title("Change in Self-Selection Rate After Training\n"
                 "(gray dot = base, colored dot = trained)",
                 fontsize=12, fontweight="bold")

    for base in base_models:
        if base in group_tops:
            ax.text(1.02, group_tops[base], DISPLAY.get(base, base),
                    transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=9, fontweight="bold", color=BASE_COLORS.get(base, "#666666"))

    ax.invert_yaxis()
    plt.subplots_adjust(left=0.25, right=0.82)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved simple self-preference delta to {output_path}")


# ---------------------------------------------------------------------------
# Ranking mode analysis
# ---------------------------------------------------------------------------

def load_ranking_self_ranks(results_dir: Path, judges: list[str] | None = None) -> pd.DataFrame:
    """Load ranking results and compute average self-rank per judge.

    If judges is provided, only include those judges. Otherwise, scan the
    results directory for all available judges.

    Returns DataFrame with columns: judge, base_model, is_trained,
    avg_self_rank, pct_ranked_first, n_valid, n_total.
    """
    from scripts.alpaca_eval.run_self_preference import resolve_base_model

    # If no judges specified, discover from results directory
    if judges is None or not judges:
        judges = sorted([d.name for d in results_dir.iterdir()
                        if d.is_dir() and (d / "ranking.json").exists()])

    rows = []
    for judge in judges:
        ranking_file = results_dir / judge / "ranking.json"
        if not ranking_file.exists():
            continue
        with open(ranking_file) as f:
            data = json.load(f)

        base = resolve_base_model(judge)
        rank_col = f"rank_{base}"
        ranks = [d[rank_col] for d in data if d.get(rank_col) is not None]
        if not ranks:
            continue

        avg = sum(ranks) / len(ranks)
        n_first = sum(1 for r in ranks if r == 1)
        rows.append({
            "judge": judge,
            "base_model": base,
            "is_trained": judge != base,
            "avg_self_rank": avg,
            "pct_ranked_first": n_first / len(ranks),
            "n_valid": len(ranks),
            "n_total": len(data),
        })

    return pd.DataFrame(rows)


def plot_ranking_self_rank_comparison(df: pd.DataFrame, output_path: Path):
    """Bar chart comparing avg self-rank: base models vs trained variants.

    Groups by base model. Base model bar in gray, trained variants in color.
    Lower rank = more self-preference (rank 1 = best).
    """
    if df.empty:
        return

    base_models = sorted(df["base_model"].unique())

    # Clean display names
    DISPLAY = {
        "ll-3.1-8b": "Llama 3.1 8B",
        "ll-3.3-70b": "Llama 3.3 70B",
        "qwen-3.0-30b": "Qwen 3.0 30B",
        "qwen-3.5-27b": "Qwen 3.5 27B",
        "gpt-oss-20b": "GPT-OSS 20B",
    }

    fig, axes = plt.subplots(1, len(base_models), figsize=(5 * len(base_models), 5),
                              sharey=True, squeeze=False)

    for idx, base in enumerate(base_models):
        ax = axes[0][idx]
        subset = df[df["base_model"] == base].sort_values("avg_self_rank")

        colors = []
        labels = []
        for _, row in subset.iterrows():
            if not row["is_trained"]:
                colors.append("#AAAAAA")
                labels.append("Base (untrained)")
            else:
                colors.append("#1565C0")
                # Extract training description
                suffix = row["judge"].replace(base + "-", "")
                labels.append(suffix)

        bars = ax.barh(range(len(subset)), subset["avg_self_rank"], color=colors,
                       edgecolor="black", linewidth=0.5)

        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Avg Self-Rank (lower = more self-preference)", fontsize=9)
        ax.set_title(DISPLAY.get(base, base), fontsize=11, fontweight="bold")
        ax.axvline(x=df["avg_self_rank"].median(), color="gray", linestyle="--",
                   alpha=0.5, linewidth=0.8)

        # Add value labels on bars
        for bar, val in zip(bars, subset["avg_self_rank"]):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=7)

    fig.suptitle("Self-Rank in Multi-Model Ranking\n(Base vs Trained Evaluators)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved ranking self-rank comparison to {output_path}")


def plot_ranking_delta(df: pd.DataFrame, output_path: Path):
    """Arrow/lollipop chart showing change in self-rank from base to trained.

    X-axis = base model's avg self-rank → trained model's avg self-rank.
    One row per trained model, grouped by base.
    Arrows pointing left = more self-preference after training.
    """
    if df.empty:
        return

    from scripts.alpaca_eval.run_self_preference import resolve_base_model

    base_models = sorted(df["base_model"].unique())
    # Get base model ranks
    base_ranks = {}
    for _, row in df[~df["is_trained"]].iterrows():
        base_ranks[row["base_model"]] = row["avg_self_rank"]

    trained = df[df["is_trained"]].copy()
    if trained.empty:
        print("  ⚠ No trained models to compare — skipping delta plot")
        return

    trained["base_rank"] = trained["base_model"].map(base_ranks)
    trained["delta"] = trained["avg_self_rank"] - trained["base_rank"]
    trained = trained.sort_values(["base_model", "delta"])

    DISPLAY = {
        "ll-3.1-8b": "Llama 3.1 8B",
        "ll-3.3-70b": "Llama 3.3 70B",
        "qwen-3.0-30b": "Qwen 3.0 30B",
        "qwen-3.0-30b-thinking": "Qwen 3.0 30B",
        "qwen-3.5-27b": "Qwen 3.5 27B",
        "qwen-3.5-27b-thinking": "Qwen 3.5 27B",
        "gpt-oss-20b": "GPT-OSS 20B",
        "gpt-oss-20b-thinking": "GPT-OSS 20B",
        "gpt-oss-120b-thinking": "GPT-OSS 120B",
    }
    BASE_COLORS = {
        "ll-3.1-8b": "#1565C0",
        "ll-3.3-70b": "#E65100",
        "qwen-3.0-30b": "#2E7D32",
        "qwen-3.0-30b-thinking": "#2E7D32",
        "qwen-3.5-27b": "#7B1FA2",
        "qwen-3.5-27b-thinking": "#7B1FA2",
        "gpt-oss-20b": "#C62828",
        "gpt-oss-20b-thinking": "#C62828",
        "gpt-oss-120b-thinking": "#C62828",
    }

    n_rows = len(trained)
    fig, ax = plt.subplots(figsize=(10, max(5, n_rows * 0.5 + 2)))

    y_pos = 0
    y_ticks = []
    y_labels = []
    group_boundaries = []
    group_tops = {}

    for base in base_models:
        group = trained[trained["base_model"] == base]
        if group.empty:
            continue
        color = BASE_COLORS.get(base, "#666666")
        group_tops[base] = y_pos

        for _, row in group.iterrows():
            base_r = row["base_rank"]
            trained_r = row["avg_self_rank"]

            ax.annotate("", xy=(trained_r, y_pos), xytext=(base_r, y_pos),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
            ax.scatter([base_r], [y_pos], color="#AAAAAA", s=40, zorder=5,
                       edgecolors="black", linewidth=0.5)
            ax.scatter([trained_r], [y_pos], color=color, s=50, zorder=5,
                       edgecolors="black", linewidth=0.5)

            label = _shorten_trained_label(row["judge"], base)
            y_ticks.append(y_pos)
            y_labels.append(label)
            y_pos += 1

        group_boundaries.append(y_pos - 0.5)
        y_pos += 1.0

    for yb in group_boundaries[:-1]:
        ax.axhline(y=yb, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Average Self-Rank (lower = more self-preference)", fontsize=10)
    ax.set_title("Change in Self-Rank After Training\n(gray dot = base, colored dot = trained)",
                 fontsize=12, fontweight="bold")

    for base in base_models:
        if base in group_tops:
            ax.text(1.02, group_tops[base], DISPLAY.get(base, base),
                    transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=9, fontweight="bold", color=BASE_COLORS.get(base, "#666666"))

    ax.invert_yaxis()
    plt.subplots_adjust(left=0.25, right=0.82)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved ranking delta plot to {output_path}")


def analyze_ranking_mode(results_dir: Path, output_dir: Path, judges: list[str]):
    """Full analysis pipeline for ranking mode results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the filtered judges list passed from main()
    df = load_ranking_self_ranks(results_dir, judges)
    if df.empty:
        print("  ⚠ No ranking results found")
        return

    # Save summary CSV
    df.to_csv(output_dir / "ranking_summary.csv", index=False)
    print(f"✓ Saved ranking summary to {output_dir / 'ranking_summary.csv'}")

    # Print summary
    print(f"\n{'='*70}")
    print("RANKING SELF-PREFERENCE SUMMARY")
    print(f"{'='*70}")

    # Group by base model
    for base in sorted(df["base_model"].unique()):
        subset = df[df["base_model"] == base].sort_values("avg_self_rank")
        print(f"\n  {base}:")
        for _, row in subset.iterrows():
            trained_label = " (base)" if not row["is_trained"] else ""
            print(f"    {row['judge']:55s} avg_rank={row['avg_self_rank']:.2f}  "
                  f"#1={row['pct_ranked_first']*100:.1f}%  "
                  f"valid={row['n_valid']}/{row['n_total']}{trained_label}")

    # Overall stats
    base_df = df[~df["is_trained"]]
    trained_df = df[df["is_trained"]]
    if not base_df.empty and not trained_df.empty:
        base_avg = base_df["avg_self_rank"].mean()
        trained_avg = trained_df["avg_self_rank"].mean()
        print(f"\n  Mean self-rank — base: {base_avg:.2f}, trained: {trained_avg:.2f}, "
              f"delta: {trained_avg - base_avg:+.2f}")

    # Generate figures
    plot_ranking_self_rank_comparison(df, output_dir / "ranking_self_rank_comparison.png")
    plot_ranking_delta(df, output_dir / "ranking_delta.png")

    print(f"\n✓ Ranking analysis complete. Results in {output_dir}/")


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

    # Resolve model names and judge_mode from --config or --model_names
    judge_mode = "simple"
    data_subsets = None
    training_dir = "data/training"
    generators = []
    if args.config:
        import yaml as _yaml
        with open(args.config) as f:
            config = _yaml.safe_load(f)
        raw_eval = config.get("evaluator_models", config.get("model_names", []))
        raw_gen = config.get("generator_models", config.get("model_names", []))
        data_subsets = config.get("data_subsets", None)
        training_dir = config.get("training_dir", "data/training")
        judges = expand_model_names(raw_eval, training_dir=training_dir, data_subsets=data_subsets)
        generators = expand_model_names(raw_gen)
        judge_mode = config.get("judge_mode", "simple")
    elif args.model_names:
        judges = expand_model_names(args.model_names)
        generators = judges
    else:
        parser.error("Provide either --config or --model_names")

    # Normalize judge_mode to a list for permutation
    judge_modes = judge_mode if isinstance(judge_mode, list) else [judge_mode]

    print(f"Judge modes: {', '.join(judge_modes)}")
    if data_subsets:
        print(f"Data subsets: {', '.join(data_subsets)}")
    print(f"Evaluator models (judges): {', '.join(judges)}")

    # Build mapping: trained_name -> subset (for filtering judges per subset)
    # Include ae_aliases so old-style AE result names also map to the right subset
    from scripts.alpaca_eval.training_runs import discover_training_runs
    all_runs = discover_training_runs(training_dir, subsets=data_subsets)
    trained_to_subset = {}
    for r in all_runs:
        trained_to_subset[r.trained_name] = r.subset
        for alias in r.ae_aliases:
            trained_to_subset[alias] = r.subset

    base_results = Path(args.results_dir)
    base_output = Path(args.output_dir)
    subsets_to_run = data_subsets if data_subsets else [None]

    # Loop: data_subsets (outer) → non-AE figures → judge_modes (inner, AE figures)
    for subset in subsets_to_run:
        subset_label = subset or "all"
        subset_out = base_output / subset_label
        subset_out.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"DATA SUBSET: {subset_label}")
        print(f"{'='*70}")

        # Helper: check if a judge belongs to this subset
        def _judge_in_subset(judge_name, _subset=subset):
            if judge_name in trained_to_subset:
                return trained_to_subset[judge_name] == _subset
            # Trained thinking models have -thinking appended to the dir name
            # but trained_to_subset uses the name without -thinking
            clean = judge_name.removesuffix("-thinking")
            if clean != judge_name and clean in trained_to_subset:
                return trained_to_subset[clean] == _subset
            from scripts.alpaca_eval.run_self_preference import resolve_base_model
            return resolve_base_model(judge_name) == judge_name

        # --- Non-AE figures (saved directly in {subset}/, computed once) ---
        _generate_uplift_figures(subset_out, training_dir, [subset] if subset else None)
        plot_score_distance_heldout(
            subset_out / "score_distance_heldout.png",
            training_dir=training_dir,
            data_subsets=[subset] if subset else None,
        )

        # --- AE figures (per judge_mode, saved in {subset}/{mode}/) ---
        for mode in judge_modes:
            # Prefer subset-specific results dir (e.g., results/01_final/ranking/)
            # over flat dir (e.g., results/ranking/)
            if subset and (base_results / subset / mode).exists():
                mode_results = base_results / subset / mode
            else:
                mode_results = base_results / mode
            mode_out = subset_out / mode
            mode_out.mkdir(parents=True, exist_ok=True)

            print(f"\n  --- {subset_label} / {mode} ---")

            if mode == "ranking":
                if mode_results.exists() and any(mode_results.iterdir()):
                    available = sorted([
                        d.name for d in mode_results.iterdir()
                        if d.is_dir() and (d / "ranking.json").exists()
                        and _judge_in_subset(d.name)
                    ])
                    if available:
                        print(f"  Ranking: {len(available)} judges")
                        analyze_ranking_mode(mode_results, mode_out, available)
                    else:
                        print(f"  No ranking results for {subset_label}")

            elif mode == "simple":
                if mode_results.exists() and any(mode_results.iterdir()):
                    available = sorted([
                        d.name for d in mode_results.iterdir()
                        if d.is_dir() and any(d.glob("vs_*.json"))
                        and _judge_in_subset(d.name)
                    ])
                    if available:
                        print(f"  Simple: {len(available)} judges")
                        print(f"  Generators: {', '.join(generators)}")

                        matrix = load_self_selection_rates(mode_results, available, generators)
                        summary = compute_self_preference_summary(matrix)

                        matrix.to_csv(mode_out / "self_preference_matrix.csv")
                        summary.to_csv(mode_out / "self_preference_summary.csv", index=False)

                        print(f"\n  SELF-PREFERENCE SUMMARY")
                        print(summary.to_string(index=False))

                        if not summary.empty:
                            plot_heatmap(matrix, mode_out / "self_preference_heatmap.png")
                            plot_delta_heatmap(matrix, mode_out / "self_preference_delta_heatmap.png")
                            plot_deviation_bars(summary, mode_out / "self_preference_deviation.png")
                            plot_simple_self_preference_delta(summary, mode_out / "self_preference_delta_arrows.png")
                            plot_training_effect_per_opponent(
                                mode_results, available,
                                mode_out / "training_effect_per_opponent_real.png"
                            )
                    else:
                        print(f"  No simple results for {subset_label}")

            # Generate task transfer heatmap with AE data (mode-specific)
            _generate_ae_heatmap(mode_out, training_dir,
                                 [subset] if subset else None, mode,
                                 mode_results)

    print(f"\n✓ Analysis complete. Results in {base_output}/")


def _generate_ae_heatmap(output_dir: Path, training_dir: str,
                         data_subsets: list[str] | None,
                         ae_mode: str, ae_results_dir: Path):
    """Generate task transfer heatmap with AlpacaEval column.

    For simple mode: AE column shows avg self-selection rate change.
    For ranking mode: AE column shows avg self-rank change (inverted: lower = better).
    """
    try:
        from scripts.figures.COLM2026.prototype_uplift_figures import fig_task_transfer_heatmap_real
        import scripts.figures.COLM2026.prototype_uplift_figures as _uplift

        orig_out = _uplift.OUT_DIR
        orig_td = getattr(_uplift, 'TRAINING_DIR', 'data/training')
        orig_sub = getattr(_uplift, 'TRAINING_SUBSETS', None)

        output_dir.mkdir(parents=True, exist_ok=True)
        _uplift.OUT_DIR = output_dir
        _uplift.TRAINING_DIR = training_dir
        _uplift.TRAINING_SUBSETS = data_subsets
        # Pass AE mode info so the heatmap function can load the right data
        _uplift.AE_MODE = ae_mode
        _uplift.AE_RESULTS_DIR = str(ae_results_dir)

        fig_task_transfer_heatmap_real()

        _uplift.OUT_DIR = orig_out
        _uplift.TRAINING_DIR = orig_td
        _uplift.TRAINING_SUBSETS = orig_sub
        _uplift.AE_MODE = None
        _uplift.AE_RESULTS_DIR = None

    except Exception as e:
        print(f"\n⚠ Could not generate AE heatmap: {e}")
        import traceback
        traceback.print_exc()


def _generate_uplift_figures(output_dir: Path, training_dir: str, data_subsets: list[str] | None):
    """Generate uplift/transfer figures from training benchmark data.

    Saves directly to output_dir. Caller is responsible for per-subset routing.
    """
    try:
        from scripts.figures.COLM2026.prototype_uplift_figures import (
            fig_arrow_task_panels_real,
            fig_arrow_dataset_panels_real,
            fig_arrow_model_panels_real,
            fig_arrow_combined_panels_real,
            fig_arrow_combined_panels_real_horizontal,
        )
        import scripts.figures.COLM2026.prototype_uplift_figures as _uplift
        orig_out = _uplift.OUT_DIR
        orig_training_dir = getattr(_uplift, 'TRAINING_DIR', 'data/training')
        orig_subsets = getattr(_uplift, 'TRAINING_SUBSETS', None)

        output_dir.mkdir(parents=True, exist_ok=True)
        _uplift.OUT_DIR = output_dir
        _uplift.TRAINING_DIR = training_dir
        _uplift.TRAINING_SUBSETS = data_subsets

        subset_label = ", ".join(data_subsets) if data_subsets else "all"
        print(f"\n  TRAINING TRANSFER FIGURES ({subset_label})")

        fig_arrow_task_panels_real()
        fig_arrow_dataset_panels_real()
        fig_arrow_model_panels_real()
        fig_arrow_combined_panels_real()
        fig_arrow_combined_panels_real_horizontal()

        plot_score_distance_training_effect(
            output_dir / "score_distance_training_effect_real.png",
            training_dir=training_dir,
            data_subsets=data_subsets,
        )

        _uplift.OUT_DIR = orig_out
        _uplift.TRAINING_DIR = orig_training_dir
        _uplift.TRAINING_SUBSETS = orig_subsets

    except Exception as e:
        print(f"\n⚠ Could not generate uplift figures: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
