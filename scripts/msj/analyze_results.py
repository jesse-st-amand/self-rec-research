"""Analyze MSJ attack results: compute ASR, compare models, generate figures.

Reads attack_results.json produced by run_attack.py and generates:
- ASR (Attack Success Rate) by model × shot count
- ASR heatmap comparing base vs trained models
- Power-law curve fitting (ASR vs shot count)
- Per-objective breakdown

Usage:
    uv run python scripts/msj/analyze_results.py --results_dir <dir>
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path


def _save_fig(fig, output_dir: Path, stem: str):
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  ✓ Saved: {pdf_path} (+ .png)")


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load attack_results.json into a DataFrame."""
    results_file = results_dir / "attack_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"No results file at {results_file}")

    # Prefer rescored file if present
    rescored_file = results_dir / "attack_results_rescored.json"
    if rescored_file.exists():
        print(f"  Using rescored results: {rescored_file}")
        results_file = rescored_file

    with open(results_file) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["success"] = df["outcome"].str.contains("SUCCESS", case=False, na=False)
    return df


def compute_asr(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Attack Success Rate per model × shot count."""
    asr = df.groupby(["model", "n_shots"]).agg(
        n_attacks=("success", "count"),
        n_success=("success", "sum"),
    ).reset_index()
    asr["asr"] = asr["n_success"] / asr["n_attacks"]
    return asr


def _parse_model_label(label):
    """Parse model label into (base_model, format, tag, is_base).

    Examples:
        "GPT-OSS 20B (base)" → ("GPT-OSS 20B", None, None, True)
        "Llama 8B (UT PW ShareGPT)" → ("Llama 8B", "PW", "UT", False)
        "Qwen 30B (AT IND ShareGPT)" → ("Qwen 30B", "IND", "AT", False)
    """
    import re
    if "(base)" in label:
        base = label.replace("(base)", "").strip()
        # Normalize base names
        base = base.replace("Llama 3.1 8B", "Llama 8B").replace("Qwen 3.0 30B", "Qwen 30B")
        return base, None, None, True

    m = re.match(r'(.+?)\s*\((\w+)\s+(PW|IND)\s+\w+\)', label)
    if m:
        base, tag, fmt = m.group(1).strip(), m.group(2), m.group(3)
        return base, fmt, tag, False

    return label, None, None, False


def fig_asr_by_shots(asr: pd.DataFrame, output_dir: Path):
    """Line plot: ASR vs shot count.

    Encoding:
    - Color = base model (lighter tone for base, saturated for trained)
    - Line style = format (PW = solid, IND = dashed)
    - Marker shape = conversation tag (UT = circle, AT = square)
    - Base models = lighter color + thicker line
    """
    # Model family colors
    MODEL_COLORS = {
        "Llama 8B": ("#93c5fd", "#2563eb"),       # light blue, blue
        "GPT-OSS 20B": ("#6ee7b7", "#047857"),     # light teal, dark teal
        "Qwen 30B": ("#d8b4fe", "#7c3aed"),        # light purple, purple
    }

    # Format → line style
    FMT_LINESTYLE = {"PW": "-", "IND": "--"}
    # Tag → marker
    TAG_MARKER = {"UT": "o", "AT": "s"}

    fig, ax = plt.subplots(figsize=(12, 7))

    models = sorted(asr["model"].unique())

    for model_label in models:
        base, fmt, tag, is_base = _parse_model_label(model_label)

        # Color
        base_color, trained_color = MODEL_COLORS.get(base, ("#cccccc", "#666666"))
        color = base_color if is_base else trained_color

        # Line style
        if is_base:
            linestyle = "-"
            linewidth = 3.0
            alpha = 0.6
        else:
            linestyle = FMT_LINESTYLE.get(fmt, "-")
            linewidth = 2.0
            alpha = 0.9

        # Marker
        if is_base:
            marker = "D"  # diamond for base
            markersize = 8
        else:
            marker = TAG_MARKER.get(tag, "o")
            markersize = 7

        model_data = asr[asr["model"] == model_label].sort_values("n_shots")
        ax.plot(model_data["n_shots"], model_data["asr"],
                marker=marker, label=model_label, color=color,
                linestyle=linestyle, linewidth=linewidth, alpha=alpha,
                markersize=markersize, markeredgecolor="black", markeredgewidth=0.4)

    ax.set_xlabel("Number of Shots", fontsize=13)
    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=13)
    ax.set_title("Many-Shot Jailbreaking: ASR vs Shot Count", fontsize=15, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xscale("log")
    ax.grid(alpha=0.3, linestyle="--")
    ax.tick_params(axis="both", labelsize=11)

    # Legend outside the plot
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0, framealpha=0.9)

    _save_fig(fig, output_dir, "asr_by_shots")


def fig_asr_heatmap(asr: pd.DataFrame, output_dir: Path):
    """Heatmap: models × shot counts, cells = ASR."""
    models = sorted(asr["model"].unique())
    shot_counts = sorted(asr["n_shots"].unique())

    matrix = np.full((len(models), len(shot_counts)), np.nan)
    for ri, model in enumerate(models):
        for ci, shots in enumerate(shot_counts):
            match = asr[(asr["model"] == model) & (asr["n_shots"] == shots)]
            if not match.empty:
                matrix[ri, ci] = match["asr"].values[0]

    fig, ax = plt.subplots(figsize=(max(8, len(shot_counts) * 1.2), max(4, len(models) * 0.8)))
    im = ax.imshow(matrix, aspect="auto", cmap=plt.cm.RdYlGn_r, vmin=0, vmax=1)

    ax.set_xticks(range(len(shot_counts)))
    ax.set_xticklabels([str(s) for s in shot_counts], fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel("Number of Shots", fontsize=12)
    ax.set_title("Attack Success Rate", fontsize=14, fontweight="bold")

    for ri in range(len(models)):
        for ci in range(len(shot_counts)):
            v = matrix[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:.0%}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if v > 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8).set_label("ASR", fontsize=11)
    _save_fig(fig, output_dir, "asr_heatmap")


def fig_asr_delta_heatmap(asr: pd.DataFrame, output_dir: Path):
    """Per-model heatmaps showing ASR difference: trained - base.

    Generates one heatmap per base model family, with rows = trained variants
    and columns = shot counts.
    """
    models = sorted(asr["model"].unique())

    # Parse all models and group by base family
    base_map = {}  # normalized_base → base_label
    trained_map = {}  # normalized_base → [trained_labels]
    for m in models:
        base, fmt, tag, is_base = _parse_model_label(m)
        if is_base:
            base_map[base] = m
            trained_map.setdefault(base, [])
        else:
            trained_map.setdefault(base, []).append(m)

    if not base_map:
        print("  ⚠ No base/trained pairs found for delta heatmap — skipping")
        return

    shot_counts = sorted(asr["n_shots"].unique())

    for base_norm, base_label in base_map.items():
        trained_labels = trained_map.get(base_norm, [])
        if not trained_labels:
            continue

        pairs = [(base_label, t) for t in sorted(trained_labels)]
        pair_labels = [t.split("(")[1].rstrip(")") if "(" in t else t for _, t in pairs]
        base_labels_list = [base_norm] * len(pairs)

        matrix = np.full((len(pairs), len(shot_counts)), np.nan)
        for ri, (base, trained) in enumerate(pairs):
            for ci, shots in enumerate(shot_counts):
                base_asr_row = asr[(asr["model"] == base) & (asr["n_shots"] == shots)]
                trained_asr_row = asr[(asr["model"] == trained) & (asr["n_shots"] == shots)]
                if not base_asr_row.empty and not trained_asr_row.empty:
                    matrix[ri, ci] = trained_asr_row["asr"].values[0] - base_asr_row["asr"].values[0]

        vals = matrix[~np.isnan(matrix)]
        if len(vals) == 0:
            continue
        vmax = max(abs(vals.min()), abs(vals.max()), 0.01)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        fig, ax = plt.subplots(figsize=(max(8, len(shot_counts) * 1.5), max(3, len(pairs) * 0.8 + 1)))
        im = ax.imshow(matrix, aspect="auto", cmap=plt.cm.RdYlGn, norm=norm)

        ax.set_xticks(range(len(shot_counts)))
        ax.set_xticklabels([str(s) for s in shot_counts], fontsize=11)
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(pair_labels, fontsize=10)
        ax.set_xlabel("Number of Shots", fontsize=12)
        ax.set_title(f"{base_norm}: ASR Change (Trained − Base)\n(positive/red = more vulnerable)",
                     fontsize=13, fontweight="bold")

        for ri in range(len(pairs)):
            for ci in range(len(shot_counts)):
                v = matrix[ri, ci]
                if not np.isnan(v):
                    ax.text(ci, ri, f"{v:+.0%}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="white" if abs(v) > vmax * 0.5 else "black")

        fig.colorbar(im, ax=ax, shrink=0.8).set_label("Δ ASR", fontsize=11)
        safe_name = base_norm.replace(" ", "_").lower()
        _save_fig(fig, output_dir, f"asr_delta_heatmap_{safe_name}")


def print_summary(df: pd.DataFrame, asr: pd.DataFrame):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print(f"MSJ ATTACK RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total attacks: {len(df)}")
    print(f"Total successes: {df['success'].sum()} ({df['success'].mean():.1%})")
    print(f"Models: {', '.join(sorted(df['model'].unique()))}")
    print(f"Shot counts: {sorted(df['n_shots'].unique())}")
    print(f"Objectives: {df['objective'].nunique()}")

    print(f"\nASR by model:")
    for model in sorted(asr["model"].unique()):
        model_asr = asr[asr["model"] == model]
        print(f"\n  {model}:")
        for _, row in model_asr.sort_values("n_shots").iterrows():
            print(f"    {row['n_shots']:>4d} shots: {row['asr']:.1%} "
                  f"({int(row['n_success'])}/{int(row['n_attacks'])})")


def main():
    parser = argparse.ArgumentParser(description="Analyze MSJ attack results")
    parser.add_argument("--results_dir", required=True, help="Directory containing attack_results.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir}...")
    df = load_results(results_dir)
    asr = compute_asr(df)

    print_summary(df, asr)

    # Save ASR table
    asr.to_csv(output_dir / "asr_summary.csv", index=False)
    print(f"\n✓ Saved ASR summary to {output_dir / 'asr_summary.csv'}")

    # Generate figures
    print("\nGenerating figures...")
    fig_asr_by_shots(asr, output_dir)
    fig_asr_heatmap(asr, output_dir)
    fig_asr_delta_heatmap(asr, output_dir)

    print(f"\n✓ Analysis complete. Results in {output_dir}/")


if __name__ == "__main__":
    main()
