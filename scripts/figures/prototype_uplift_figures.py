"""
Prototype uplift figures for COLM 2026 paper.

Shows before/after contrast when models are trained on one operationalization
and tested on others. Uses proxy data: ll-3.1-8b (pre) → opus-4.1 (post).

Usage:
    uv run python scripts/figures/prototype_uplift_figures.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

from scripts.figures.prototype_compact_figures import (
    load_self_scores, adjust_ind_performance,
)

AGG_DIR = Path("data/analysis/_aggregated_data")
OUT_DIR = Path("data/figures/prototypes/uplift")
TRAINING_DIR = "data/training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Operationalizations to compare — trained on PW-Rec (UT), tested on all
OPS = {
    "PW-Rec (UT)": "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst",
    "IND-Rec (UT)": "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst",
    "PW-Pref (UT)": "ICML_05_UT_PW-Q_Pref-Q_NPr_FA_Inst",
    "IND-Pref (UT)": "ICML_06_UT_IND-Q_Pref-Q_NPr_FA_Inst",
    "PW-Rec (AT)": "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
    "IND-Rec (AT)": "COLM_02_AT_IND-C_Rec_NPr_FA_Inst",
}

# Short dataset names
DATASET_SHORT = {
    "wikisum/training_set_1-20+test_set_1-30": "WikiSum",
    "sharegpt/english_26+english2_74": "ShareGPT",
    "pku_saferlhf/mismatch_1-20+test_mismatch_1-20+test-mismatch_10_100-200": "PKU",
    "bigcodebench/instruct_1-50": "BigCode",
}

PRE_MODEL = "ll-3.1-8b"
POST_MODEL = "opus-4.1"

# For multi-model versions (simulate different model sizes)
MODEL_PAIRS = {
    "8B": ("ll-3.1-8b", "opus-4.1"),
    "80B": ("qwen-3.0-80b", "opus-4.1"),
    "Mini": ("gpt-4.1-mini", "opus-4.1"),
}


def load_model_performance(model_name):
    """
    Load mean performance (across datasets) for a given evaluator model
    across all operationalizations.

    Returns dict: {op_label: mean_accuracy}
    """
    results = {}
    for op_label, exp_name in OPS.items():
        exp_dir = AGG_DIR / exp_name
        if not exp_dir.exists():
            continue
        ts_dir = sorted(exp_dir.iterdir(), reverse=True)[0]
        perf_file = ts_dir / "aggregated_performance.csv"
        if not perf_file.exists():
            continue
        df = pd.read_csv(perf_file, index_col=0)
        df.columns = [DATASET_SHORT.get(c, c) for c in df.columns]
        if model_name in df.index:
            results[op_label] = df.loc[model_name].mean()
    return results


def load_all_model_pairs():
    """Load pre/post performance for all model pairs."""
    pair_data = {}
    for size_label, (pre, post) in MODEL_PAIRS.items():
        pre_perf = load_model_performance(pre)
        post_perf = load_model_performance(post)
        if pre_perf and post_perf:
            pair_data[size_label] = {"pre": pre_perf, "post": post_perf}
            print(f"  Loaded {size_label}: {pre} ({len(pre_perf)} OPs) → {post} ({len(post_perf)} OPs)")
    return pair_data


def load_per_dataset_performance(model_name):
    """
    Load per-dataset performance for a given evaluator model
    across all operationalizations.

    Returns dict: {op_label: {dataset: accuracy}}
    """
    results = {}
    for op_label, exp_name in OPS.items():
        exp_dir = AGG_DIR / exp_name
        if not exp_dir.exists():
            continue
        ts_dir = sorted(exp_dir.iterdir(), reverse=True)[0]
        perf_file = ts_dir / "aggregated_performance.csv"
        if not perf_file.exists():
            continue
        df = pd.read_csv(perf_file, index_col=0)
        df.columns = [DATASET_SHORT.get(c, c) for c in df.columns]
        if model_name in df.index:
            results[op_label] = df.loc[model_name].to_dict()
    return results


# ============================================================================
# FIGURE 1: Transfer heatmap — model sizes × test OPs, cell = delta accuracy
# ============================================================================
def fig_transfer_heatmap(pair_data):
    """
    Rows = model sizes, columns = test operationalizations.
    Cell = accuracy delta (post - pre). Green = improved, red = degraded.
    """
    op_labels = list(OPS.keys())
    size_labels = list(pair_data.keys())

    # Build matrix
    matrix = np.full((len(size_labels), len(op_labels)), np.nan)
    for i, size in enumerate(size_labels):
        pre = pair_data[size]["pre"]
        post = pair_data[size]["post"]
        for j, op in enumerate(op_labels):
            if op in pre and op in post:
                matrix[i, j] = post[op] - pre[op]

    df = pd.DataFrame(matrix, index=size_labels, columns=op_labels)

    fig, ax = plt.subplots(figsize=(10, 3))
    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)
    im = ax.imshow(df.values, cmap="RdYlGn", norm=norm, aspect="auto")

    # Annotations
    for i in range(len(size_labels)):
        for j in range(len(op_labels)):
            val = df.iloc[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.2 else "black"
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(range(len(op_labels)))
    ax.set_xticklabels(op_labels, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(size_labels)))
    ax.set_yticklabels(size_labels, fontsize=11)
    ax.set_ylabel("Model Size", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Accuracy Δ (post − pre)", fontsize=10)

    # Mark the training source OP
    train_idx = op_labels.index("PW-Rec (UT)")
    ax.add_patch(plt.Rectangle((train_idx - 0.5, -0.5), 1, len(size_labels),
                                fill=False, edgecolor="black", linewidth=2.5))

    ax.set_title("Transfer Uplift: Trained on PW-Rec (UT), Tested Across Operationalizations",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "transfer_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================================
# FIGURE 2: Grouped bar chart — pre/post bars per OP, grouped by model size
# ============================================================================
def fig_grouped_bars(pair_data):
    """
    X-axis = operationalizations, grouped bars for each model size.
    Solid bars = post-training, ghost bars = pre-training baseline.
    """
    op_labels = list(OPS.keys())
    size_labels = list(pair_data.keys())
    n_ops = len(op_labels)
    n_sizes = len(size_labels)

    colors = ["#1565C0", "#E65100", "#2E7D32"]
    pre_alpha = 0.25
    post_alpha = 0.85

    bar_width = 0.8 / (n_sizes * 2)  # pre + post per size
    fig, ax = plt.subplots(figsize=(12, 5))

    for s_idx, size in enumerate(size_labels):
        pre = pair_data[size]["pre"]
        post = pair_data[size]["post"]
        color = colors[s_idx % len(colors)]

        for o_idx, op in enumerate(op_labels):
            x_base = o_idx
            offset = (s_idx - n_sizes / 2 + 0.5) * bar_width * 2

            pre_val = pre.get(op, 0)
            post_val = post.get(op, 0)

            # Pre bar (ghost)
            ax.bar(x_base + offset - bar_width / 2, pre_val, bar_width,
                   color=color, alpha=pre_alpha, edgecolor=color, linewidth=0.8)
            # Post bar (solid)
            bar = ax.bar(x_base + offset + bar_width / 2, post_val, bar_width,
                         color=color, alpha=post_alpha, edgecolor="none")

            # Only label once per size
            if o_idx == 0:
                bar.set_label(f"{size}")

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(range(n_ops))
    ax.set_xticklabels(op_labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Recognition / Preference Accuracy", fontsize=10)
    ax.set_ylim(0.3, 1.0)

    # Custom legend
    handles = []
    for s_idx, size in enumerate(size_labels):
        color = colors[s_idx % len(colors)]
        handles.append(mpatches.Patch(facecolor=color, alpha=post_alpha, label=f"{size} (post)"))
        handles.append(mpatches.Patch(facecolor=color, alpha=pre_alpha, edgecolor=color,
                                       linewidth=0.8, label=f"{size} (pre)"))
    ax.legend(handles=handles, fontsize=8, loc="upper right", ncol=n_sizes)

    # Mark training source
    train_idx = op_labels.index("PW-Rec (UT)")
    ax.axvspan(train_idx - 0.45, train_idx + 0.45, color="gold", alpha=0.1, zorder=0)
    ax.text(train_idx, 0.97, "trained", ha="center", fontsize=8, fontstyle="italic", color="goldenrod")

    ax.set_title("Pre- vs Post-Training Accuracy Across Operationalizations",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "grouped_bars.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================================
# FIGURE 3a: Grouped arrow plot — all models in one panel, all OPs on x-axis
# ============================================================================
def _draw_grouped_arrows(ax, pair_data, op_labels, colors, show_xlabel=True,
                         show_delta=True, title=None, highlight_idx=None):
    """Draw grouped arrows for multiple models on a single axes."""
    size_labels = list(pair_data.keys())
    n_sizes = len(size_labels)
    group_width = 0.6
    offsets = np.linspace(-group_width / 2, group_width / 2, n_sizes)

    # Alternating column background with white/gray diagonal hatch
    for o_idx in range(len(op_labels)):
        if o_idx % 2 == 1:
            ax.axvspan(o_idx - 0.5, o_idx + 0.5, facecolor="white",
                       edgecolor="none", zorder=0)
            ax.axvspan(o_idx - 0.5, o_idx + 0.5, facecolor="none",
                       edgecolor="#cccccc", hatch="//", linewidth=0, zorder=0)

    for s_idx, size in enumerate(size_labels):
        pre = pair_data[size]["pre"]
        post = pair_data[size]["post"]
        color = colors[s_idx % len(colors)]

        for o_idx, op in enumerate(op_labels):
            pre_val = pre.get(op)
            post_val = post.get(op)
            if pre_val is None or post_val is None:
                continue

            x = o_idx + offsets[s_idx]
            delta = post_val - pre_val

            # Pre dot (gray)
            ax.scatter(x, pre_val, color="gray", s=35, zorder=3,
                       edgecolors="black", linewidth=0.4)

            # Arrow
            arrow_color = "#2E7D32" if delta > 0 else "#C62828"
            ax.annotate("", xy=(x, post_val), xytext=(x, pre_val),
                        arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.8),
                        zorder=4)

            # Post dot (colored)
            ax.scatter(x, post_val, color=color, s=45, zorder=5,
                       edgecolors="black", linewidth=0.4)

            # Delta label
            if show_delta:
                ax.text(x + 0.12, (pre_val + post_val) / 2, f"{delta:+.2f}",
                        fontsize=5.5, color=arrow_color, va="center")

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0.2, 1.05)
    ax.set_ylabel("Accuracy", fontsize=9)
    if show_xlabel:
        ax.set_xticks(range(len(op_labels)))
        ax.set_xticklabels(op_labels, fontsize=8, rotation=30, ha="right")
    else:
        ax.set_xticks(range(len(op_labels)))
        ax.set_xticklabels([])
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")


def fig_arrow_plot(pair_data):
    """All models in one panel, all OPs on x-axis."""
    op_labels = list(OPS.keys())
    size_labels = list(pair_data.keys())
    colors = ["#1565C0", "#E65100", "#2E7D32"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    _draw_grouped_arrows(ax, pair_data, op_labels, colors)

    # Mark training source
    train_idx = op_labels.index("PW-Rec (UT)")
    ax.axvspan(train_idx - 0.45, train_idx + 0.45, color="gold", alpha=0.1, zorder=0)
    ax.text(train_idx, 1.02, "trained", ha="center", fontsize=7, fontstyle="italic",
            color="goldenrod")

    # Legend
    handles = []
    for s_idx, size in enumerate(size_labels):
        pre_m, post_m = MODEL_PAIRS[size]
        handles.append(plt.Line2D([0], [0], marker='o', color=colors[s_idx],
                                   markersize=6, linestyle='None',
                                   label=f"{size} ({pre_m} → {post_m})"))
    handles.append(plt.Line2D([0], [0], marker='o', color='gray', markersize=5,
                               linestyle='None', markeredgecolor='black',
                               markeredgewidth=0.4, label="Pre-training"))
    ax.legend(handles=handles, fontsize=7, loc="upper right")

    ax.set_title("Training Uplift Across Operationalizations", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "arrow_grouped.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================================
# FIGURE 3b: Task OP transfer panels
# 4 panels — each panel = trained on a different task OP (PW_UT, IND_UT, PW_AT, IND_AT)
# x-axis = 6 test OPs: PW_UT, IND_UT, PW_UT(Pref), IND_UT(Pref), PW_AT, IND_AT
# ============================================================================
def fig_arrow_task_panels(pair_data):
    """
    Each panel simulates training on one of the 4 recognition OPs.
    The trained-on OP is highlighted. X-axis shows all 6 test OPs.
    """
    size_labels = list(pair_data.keys())
    colors = ["#1565C0", "#E65100", "#2E7D32"]

    # Training source OPs (one per panel)
    train_ops = [
        ("(a) Trained on: PW-Rec (UT)", "PW-Rec (UT)"),
        ("(b) Trained on: IND-Rec (UT)", "IND-Rec (UT)"),
        ("(c) Trained on: PW-Rec (AT)", "PW-Rec (AT)"),
        ("(d) Trained on: IND-Rec (AT)", "IND-Rec (AT)"),
    ]

    # Test OPs shown on x-axis (all 6) — pref columns last
    test_ops = ["PW-Rec (UT)", "IND-Rec (UT)", "PW-Rec (AT)", "IND-Rec (AT)",
                "PW-Pref (UT)", "IND-Pref (UT)"]
    # Shorter labels for display
    test_labels = ["PW (UT)", "IND (UT)", "PW (AT)", "IND (AT)",
                   "PW Pref (UT)", "IND Pref (UT)"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes_flat = axes.flatten()

    for idx, (title, trained_op) in enumerate(train_ops):
        ax = axes_flat[idx]
        show_x = idx >= 2  # bottom row only

        _draw_grouped_arrows(ax, pair_data, test_ops, colors,
                             show_xlabel=False, show_delta=True, title=title)

        # Highlight the trained-on OP
        trained_x = test_ops.index(trained_op)
        ax.axvspan(trained_x - 0.45, trained_x + 0.45, color="gold", alpha=0.1, zorder=0)
        ax.text(trained_x, ax.get_ylim()[1] * 0.98, "trained", ha="center",
                fontsize=6, fontstyle="italic", color="goldenrod")

        # Set x-axis labels
        ax.set_xticks(range(len(test_ops)))
        if show_x:
            ax.set_xticklabels(test_labels, fontsize=7.5, rotation=30, ha="right")
        else:
            ax.set_xticklabels([])

    # Shared legend at bottom
    handles = []
    for s_idx, size in enumerate(size_labels):
        pre_m, post_m = MODEL_PAIRS[size]
        handles.append(plt.Line2D([0], [0], marker='o', color=colors[s_idx],
                                   markersize=6, linestyle='None',
                                   label=f"{size} ({pre_m} → {post_m})"))
    handles.append(plt.Line2D([0], [0], marker='o', color='gray', markersize=5,
                               linestyle='None', markeredgecolor='black',
                               markeredgewidth=0.4, label="Pre-training"))
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Task Operationalization Transfer", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = OUT_DIR / "arrow_task_panels.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================================
# FIGURE 3c: Dataset domain transfer panels
# 4 panels — each panel = trained on a different dataset domain
# x-axis = 4 dataset domains (trained-on domain first, highlighted)
# Values = mean accuracy across the 4 recognition task OPs
# ============================================================================
def fig_arrow_dataset_panels(pair_data):
    """
    Each panel simulates training on one dataset domain.
    X-axis = 4 domains. Values = mean accuracy across 4 recognition task OPs
    (PW_UT, IND_UT, PW_AT, IND_AT).
    """
    size_labels = list(pair_data.keys())
    colors = ["#1565C0", "#E65100", "#2E7D32"]
    datasets = ["WikiSum", "ShareGPT", "PKU", "BigCode"]
    panel_labels_list = ["(a)", "(b)", "(c)", "(d)"]

    # The 4 recognition OPs to average over
    rec_ops = ["PW-Rec (UT)", "IND-Rec (UT)", "PW-Rec (AT)", "IND-Rec (AT)"]

    # Load per-dataset data for all models
    all_per_ds = {}
    for size_label, (pre_model, post_model) in MODEL_PAIRS.items():
        pre_ds = load_per_dataset_performance(pre_model)
        post_ds = load_per_dataset_performance(post_model)
        all_per_ds[size_label] = {"pre": pre_ds, "post": post_ds}

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    axes_flat = axes.flatten()

    for d_idx, train_dataset in enumerate(datasets):
        ax = axes_flat[d_idx]
        show_x = d_idx >= 2

        # For each model size, compute mean accuracy across rec_ops per dataset
        ds_pair_data = {}
        for size_label in size_labels:
            pre_ds = all_per_ds[size_label]["pre"]
            post_ds = all_per_ds[size_label]["post"]

            pre_means = {}
            post_means = {}
            for ds in datasets:
                pre_vals = [pre_ds.get(op, {}).get(ds) for op in rec_ops]
                post_vals = [post_ds.get(op, {}).get(ds) for op in rec_ops]
                pre_vals = [v for v in pre_vals if v is not None]
                post_vals = [v for v in post_vals if v is not None]
                if pre_vals:
                    pre_means[ds] = np.mean(pre_vals)
                if post_vals:
                    post_means[ds] = np.mean(post_vals)

            ds_pair_data[size_label] = {"pre": pre_means, "post": post_means}

        _draw_grouped_arrows(ax, ds_pair_data, datasets, colors,
                             show_xlabel=show_x, show_delta=True,
                             title=f"{panel_labels_list[d_idx]} Trained on: {train_dataset}")

        # Highlight the trained-on dataset
        trained_x = datasets.index(train_dataset)
        ax.axvspan(trained_x - 0.45, trained_x + 0.45, color="gold", alpha=0.1, zorder=0)
        ax.text(trained_x, ax.get_ylim()[1] * 0.98, "trained", ha="center",
                fontsize=6, fontstyle="italic", color="goldenrod")

    # Shared legend at bottom
    handles = []
    for s_idx, size in enumerate(size_labels):
        pre_m, post_m = MODEL_PAIRS[size]
        handles.append(plt.Line2D([0], [0], marker='o', color=colors[s_idx],
                                   markersize=6, linestyle='None',
                                   label=f"{size} ({pre_m} → {post_m})"))
    handles.append(plt.Line2D([0], [0], marker='o', color='gray', markersize=5,
                               linestyle='None', markeredgecolor='black',
                               markeredgewidth=0.4, label="Pre-training"))
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Dataset Domain Transfer (avg over 4 recognition OPs)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = OUT_DIR / "arrow_dataset_panels.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================================
# FIGURE 4: Radar/spider chart — pre vs post polygons per model size
# ============================================================================
def fig_radar(pair_data):
    """
    One polygon per condition (pre/post), axes = operationalizations.
    Pre = dashed, post = solid. One subplot per model size.
    """
    op_labels = list(OPS.keys())
    n_ops = len(op_labels)
    size_labels = list(pair_data.keys())
    n_sizes = len(size_labels)
    colors = ["#1565C0", "#E65100", "#2E7D32"]

    angles = np.linspace(0, 2 * np.pi, n_ops, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, axes = plt.subplots(1, n_sizes, figsize=(5 * n_sizes, 5),
                             subplot_kw=dict(polar=True))
    if n_sizes == 1:
        axes = [axes]

    for s_idx, size in enumerate(size_labels):
        ax = axes[s_idx]
        pre = pair_data[size]["pre"]
        post = pair_data[size]["post"]
        color = colors[s_idx % len(colors)]

        pre_vals = [pre.get(op, 0.5) for op in op_labels] + [pre.get(op_labels[0], 0.5)]
        post_vals = [post.get(op, 0.5) for op in op_labels] + [post.get(op_labels[0], 0.5)]

        ax.plot(angles, pre_vals, color="gray", linewidth=1.5, linestyle="--", label="Pre-training")
        ax.fill(angles, pre_vals, color="gray", alpha=0.1)

        ax.plot(angles, post_vals, color=color, linewidth=2, label="Post-training")
        ax.fill(angles, post_vals, color=color, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(op_labels, fontsize=8)
        ax.set_ylim(0.3, 1.0)
        ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        ax.set_yticklabels(["0.4", "0.5", "0.6", "0.7", "0.8", "0.9"], fontsize=7)

        # Highlight chance level
        chance_vals = [0.5] * (n_ops + 1)
        ax.plot(angles, chance_vals, color="red", linewidth=0.8, linestyle=":", alpha=0.5)

        ax.set_title(f"{size}", fontsize=11, fontweight="bold", pad=15)
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Transfer Profile: Pre- vs Post-Training Across Operationalizations",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = OUT_DIR / "radar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================================
# FIGURE 5: Delta matrix — training source OP × test OP (single row for now)
# ============================================================================
def fig_delta_matrix(pair_data):
    """
    Rows = model sizes (proxy for "trained on PW-Rec"),
    columns = test OP × dataset. Cell = delta with per-dataset breakdown.
    Shows which datasets benefit most from training transfer.
    """
    op_labels = list(OPS.keys())
    size_labels = list(pair_data.keys())

    # Load per-dataset performance for finer granularity
    all_data = {}
    for size_label, (pre_model, post_model) in MODEL_PAIRS.items():
        pre_ds = load_per_dataset_performance(pre_model)
        post_ds = load_per_dataset_performance(post_model)
        all_data[size_label] = {"pre": pre_ds, "post": post_ds}

    # Build columns: (OP, dataset) pairs
    datasets = ["WikiSum", "ShareGPT", "PKU", "BigCode"]
    col_tuples = []
    for op in op_labels:
        for ds in datasets:
            col_tuples.append((op, ds))

    # Build matrix
    matrix = np.full((len(size_labels), len(col_tuples)), np.nan)
    for i, size in enumerate(size_labels):
        pre_ds = all_data[size]["pre"]
        post_ds = all_data[size]["post"]
        for j, (op, ds) in enumerate(col_tuples):
            pre_val = pre_ds.get(op, {}).get(ds)
            post_val = post_ds.get(op, {}).get(ds)
            if pre_val is not None and post_val is not None:
                matrix[i, j] = post_val - pre_val

    col_labels = [f"{op}\n{ds}" for op, ds in col_tuples]
    df = pd.DataFrame(matrix, index=size_labels, columns=col_labels)

    fig, ax = plt.subplots(figsize=(18, 3.5))
    norm = TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.4)

    im = ax.imshow(df.values, cmap="RdYlGn", norm=norm, aspect="auto")

    for i in range(len(size_labels)):
        for j in range(len(col_tuples)):
            val = df.iloc[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.25 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=6.5, color=color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=6.5, rotation=45, ha="right")
    ax.set_yticks(range(len(size_labels)))
    ax.set_yticklabels(size_labels, fontsize=11)

    # Vertical separators between OPs
    for k in range(1, len(op_labels)):
        ax.axvline(x=k * len(datasets) - 0.5, color="black", linewidth=1.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cbar.set_label("Accuracy Δ", fontsize=10)

    ax.set_title("Per-Dataset Transfer Uplift: Trained on PW-Rec (UT)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "delta_matrix.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================================
# FIGURE 6: Task OP transfer heatmap — one per model
# y = trained-on task OP, x = tested-on task OP, cell = delta accuracy
# ============================================================================

def fig_task_transfer_heatmap(pair_data):
    """
    One heatmap per model. Rows = trained-on task OP, cols = tested-on task OP.
    Cell = accuracy delta (post − pre), averaged across datasets.
    Columns split into groups separated by whitespace: Rec | Pref
    """
    train_ops = ["PW-Rec (UT)", "IND-Rec (UT)", "PW-Rec (AT)", "IND-Rec (AT)"]
    train_labels = ["PW (UT)", "IND (UT)", "PW (AT)", "IND (AT)"]

    # Column groups: recognition block, then preference block
    rec_ops = ["PW-Rec (UT)", "IND-Rec (UT)", "PW-Rec (AT)", "IND-Rec (AT)"]
    rec_labels = ["PW (UT)", "IND (UT)", "PW (AT)", "IND (AT)"]
    pref_ops = ["PW-Pref (UT)", "IND-Pref (UT)"]
    pref_labels = ["PW Pref", "IND Pref"]

    size_labels = list(pair_data.keys())
    n_models = len(size_labels)
    n_train = len(train_ops)
    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)

    # Each model gets 2 sub-axes (rec block + pref block) plus gaps
    # width_ratios: [rec, gap, pref] per model, with model gaps between
    ratios = []
    for m in range(n_models):
        if m > 0:
            ratios.append(0.3)  # gap between models
        ratios.extend([len(rec_ops), 0.15, len(pref_ops)])

    fig, all_axes = plt.subplots(1, len(ratios), figsize=(2.2 * n_models * 3 + 1, 4),
                                  gridspec_kw={"width_ratios": ratios, "wspace": 0.0})
    if len(ratios) == 1:
        all_axes = [all_axes]

    last_im = None
    for m_idx, size in enumerate(size_labels):
        pre_m, post_m = MODEL_PAIRS[size]
        pre = pair_data[size]["pre"]
        post = pair_data[size]["post"]

        # Build rec and pref matrices
        rec_matrix = np.full((n_train, len(rec_ops)), np.nan)
        for i, train_op in enumerate(train_ops):
            for j, test_op in enumerate(rec_ops):
                pre_val = pre.get(test_op)
                post_val = post.get(test_op)
                if pre_val is not None and post_val is not None:
                    rec_matrix[i, j] = post_val - pre_val

        pref_matrix = np.full((n_train, len(pref_ops)), np.nan)
        for i, train_op in enumerate(train_ops):
            for j, test_op in enumerate(pref_ops):
                pre_val = pre.get(test_op)
                post_val = post.get(test_op)
                if pre_val is not None and post_val is not None:
                    pref_matrix[i, j] = post_val - pre_val

        # Axes indices for this model
        base = m_idx * 3 + (m_idx)  # account for inter-model gaps
        ax_rec = all_axes[base]
        ax_gap = all_axes[base + 1]
        ax_pref = all_axes[base + 2]

        # Hide gap axis
        ax_gap.set_axis_off()

        # Draw rec block
        last_im = ax_rec.imshow(rec_matrix, cmap="RdYlGn", norm=norm, aspect="auto")
        for i in range(n_train):
            for j in range(len(rec_ops)):
                val = rec_matrix[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 0.2 else "black"
                    ax_rec.text(j, i, f"{val:+.2f}", ha="center", va="center",
                                fontsize=8, fontweight="bold", color=color)

        ax_rec.set_xticks(range(len(rec_ops)))
        ax_rec.set_xticklabels(rec_labels, fontsize=8, rotation=35, ha="right")
        ax_rec.set_yticks(range(n_train))
        ax_rec.set_yticklabels(train_labels if m_idx == 0 else [], fontsize=8)
        if m_idx == 0:
            ax_rec.set_ylabel("Trained on", fontsize=10, fontweight="bold")
        ax_rec.set_title(f"{size} ({pre_m} → {post_m})", fontsize=9, fontweight="bold",
                         loc="left")

        # Draw pref block
        ax_pref.imshow(pref_matrix, cmap="RdYlGn", norm=norm, aspect="auto")
        for i in range(n_train):
            for j in range(len(pref_ops)):
                val = pref_matrix[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 0.2 else "black"
                    ax_pref.text(j, i, f"{val:+.2f}", ha="center", va="center",
                                 fontsize=8, fontweight="bold", color=color)

        ax_pref.set_xticks(range(len(pref_ops)))
        ax_pref.set_xticklabels(pref_labels, fontsize=8, rotation=35, ha="right")
        ax_pref.set_yticks([])

    # Hide any inter-model gap axes
    for m_idx in range(1, n_models):
        gap_idx = m_idx * 4 - 1  # gap axis before each model after the first
        if gap_idx < len(all_axes):
            all_axes[gap_idx].set_axis_off()

    cbar = fig.colorbar(last_im, ax=all_axes, shrink=0.8, pad=0.03, aspect=20)
    cbar.set_label("Accuracy Δ", fontsize=9)

    fig.suptitle("Task Operationalization Transfer", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    path = OUT_DIR / "heatmap_task_transfer.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================================
# FIGURE 6b: Task transfer heatmap WITH AlpacaEval column (proxy data)
# Same as fig_task_transfer_heatmap but with a 7th column for AlpacaEval
# self-selection rate. Uses opus-4.1 as post-trained proxy.
# ============================================================================
def load_alpaca_eval_self_selection(model_name, opponent="qwen-2.5-7b"):
    """Load AlpacaEval self-selection rate for a model vs an opponent."""
    result_path = Path(f"data/alpaca_eval/results/{model_name}/vs_{opponent}.json")
    if not result_path.exists():
        return None
    import json
    with open(result_path) as f:
        data = json.load(f)
    valid = [d["preference"] for d in data
             if d["preference"] is not None and d["preference"] == d["preference"]]
    if not valid:
        return None
    wins = sum(1 for p in valid if p == 1.0)
    return wins / len(valid)


def load_real_training_metrics(run_name):
    """Load pre/post training metrics from data/training/{run}__*/."""
    import json as _json
    import glob
    training_dir = Path("data/training")
    matches = glob.glob(str(training_dir / f"{run_name}__*"))
    if not matches:
        return None, None
    run_dir = Path(sorted(matches)[-1])

    pre_file = run_dir / "posthoc_benchmarks" / "check_configured_pre" / "metrics" / "metrics.jsonl"
    post_file = run_dir / "posthoc_benchmarks" / "check_configured_post" / "metrics" / "metrics.jsonl"

    def _load(path):
        if not path.exists():
            return {}
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return {}
        return _json.loads(lines[-1])

    return _load(pre_file), _load(post_file)


# Map benchmark metric keys to our OP labels
BENCHMARK_TO_OP = {
    "benchmark/xeval_dataset_wikisum/accuracy": "xeval_wikisum",  # dataset xeval
    "benchmark/xeval_dataset_bigcodebench/accuracy": "xeval_bigcode",
    "benchmark/xeval_dataset_pku/accuracy": "xeval_pku",
    "benchmark/xeval_task_pref_pw/accuracy": "PW-Pref (UT)",
    "benchmark/xeval_tag_at_pw/accuracy": "PW-Rec (AT)",
    "benchmark/xeval_source_numbered/accuracy": "xeval_source",
}


def fig_task_transfer_heatmap_with_ae(pair_data):
    """
    Same as fig_task_transfer_heatmap but with AlpacaEval self-selection column.
    Uses proxy data (opus-4.1 as post-trained).
    Three column groups separated by whitespace: Rec | Pref | AE
    """
    train_ops = ["PW-Rec (UT)", "IND-Rec (UT)", "PW-Rec (AT)", "IND-Rec (AT)"]
    train_labels = ["PW (UT)", "IND (UT)", "PW (AT)", "IND (AT)"]

    rec_ops = ["PW-Rec (UT)", "IND-Rec (UT)", "PW-Rec (AT)", "IND-Rec (AT)"]
    rec_labels = ["PW (UT)", "IND (UT)", "PW (AT)", "IND (AT)"]
    pref_ops = ["PW-Pref (UT)", "IND-Pref (UT)"]
    pref_labels = ["PW Pref", "IND Pref"]
    ae_ops = ["AE Self-Pref"]
    ae_labels = ["AlpacaEval"]

    size_labels = list(pair_data.keys())
    n_models = len(size_labels)
    n_train = len(train_ops)
    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)

    # Per model: [rec, gap, pref, gap, ae]; between models: [gap]
    ratios = []
    for m in range(n_models):
        if m > 0:
            ratios.append(0.3)  # gap between models
        ratios.extend([len(rec_ops), 0.15, len(pref_ops), 0.15, len(ae_ops)])

    fig, all_axes = plt.subplots(1, len(ratios), figsize=(2.2 * n_models * 3.5 + 1, 4),
                                  gridspec_kw={"width_ratios": ratios, "wspace": 0.0})

    last_im = None
    for m_idx, size in enumerate(size_labels):
        pre_m, post_m = MODEL_PAIRS[size]
        pre = pair_data[size]["pre"]
        post = pair_data[size]["post"]

        def _build_matrix(ops):
            mat = np.full((n_train, len(ops)), np.nan)
            for i, train_op in enumerate(train_ops):
                for j, test_op in enumerate(ops):
                    if test_op == "AE Self-Pref":
                        continue  # no AE data for proxy
                    pre_val = pre.get(test_op)
                    post_val = post.get(test_op)
                    if pre_val is not None and post_val is not None:
                        mat[i, j] = post_val - pre_val
            return mat

        rec_mat = _build_matrix(rec_ops)
        pref_mat = _build_matrix(pref_ops)
        ae_mat = _build_matrix(ae_ops)

        # Axis indices: each model uses 5 slots (rec, gap, pref, gap, ae)
        base = m_idx * 5 + m_idx  # +m_idx for inter-model gaps
        ax_rec = all_axes[base]
        ax_gap1 = all_axes[base + 1]
        ax_pref = all_axes[base + 2]
        ax_gap2 = all_axes[base + 3]
        ax_ae = all_axes[base + 4]

        ax_gap1.set_axis_off()
        ax_gap2.set_axis_off()

        def _draw_block(ax, mat, labels, show_yticks=False):
            nonlocal last_im
            last_im = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat[i, j]
                    if not np.isnan(val):
                        color = "white" if abs(val) > 0.2 else "black"
                        ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                                fontsize=8, fontweight="bold", color=color)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=7, rotation=35, ha="right")
            ax.set_yticks(range(n_train) if show_yticks else [])
            if show_yticks:
                ax.set_yticklabels(train_labels, fontsize=8)

        _draw_block(ax_rec, rec_mat, rec_labels, show_yticks=(m_idx == 0))
        _draw_block(ax_pref, pref_mat, pref_labels)
        _draw_block(ax_ae, ae_mat, ae_labels)

        if m_idx == 0:
            ax_rec.set_ylabel("Trained on", fontsize=10, fontweight="bold")
        ax_rec.set_title(f"{size} ({pre_m} → {post_m})", fontsize=9, fontweight="bold", loc="left")

    # Hide inter-model gap axes
    for m_idx in range(1, n_models):
        gap_idx = m_idx * 6 - 1
        if gap_idx < len(all_axes):
            all_axes[gap_idx].set_axis_off()

    cbar = fig.colorbar(last_im, ax=list(all_axes), shrink=0.8, pad=0.03, aspect=20)
    cbar.set_label("Accuracy Δ", fontsize=9)

    fig.suptitle("Task OP Transfer (with AlpacaEval)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    path = OUT_DIR / "heatmap_task_transfer_with_ae_proxy.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================================
# FIGURE 6c: Task transfer heatmap WITH AlpacaEval — REAL DATA
# Uses actual training metrics from data/training/ and AlpacaEval results.
# Only ll-3.1-8b has real data currently.
# ============================================================================
def _load_benchmark_accuracies(run_dir):
    """Load epoch 0 (pre) and max epoch (post) accuracies from benchmark_predictions/.

    Returns dict mapping benchmark_name → (pre_acc, post_acc).
    """
    import json as _json
    bp_dir = Path(run_dir) / "benchmark_predictions"
    if not bp_dir.exists():
        return {}
    results = {}
    for bench_dir in sorted(bp_dir.iterdir()):
        if not bench_dir.is_dir() or "mmlu" in bench_dir.name:
            continue
        epochs = {}
        for ef in bench_dir.glob("epoch_*.json"):
            ep = int(ef.stem.replace("epoch_", ""))
            with open(ef) as f:
                data = _json.load(f)
            acc = data.get("accuracy")
            if acc is not None:
                epochs[ep] = acc
        if epochs:
            pre = epochs.get(0)
            post = epochs.get(max(epochs.keys()))
            results[bench_dir.name] = (pre, post)
    return results


def _load_alpaca_eval_avg_self_selection(model_name):
    """Load average AlpacaEval self-selection rate across all opponents.

    Tries exact name first, then strips _tinker_small suffix for matching.
    """
    import json as _json
    # Try exact name first, then without _tinker_small
    candidates = [model_name]
    stripped = model_name.replace("_tinker_small", "")
    if stripped != model_name:
        candidates.append(stripped)

    for name in candidates:
        results_dir = Path(f"data/alpaca_eval/results/{name}")
        if not results_dir.exists():
            continue
        rates = []
        for f in results_dir.glob("vs_*.json"):
            with open(f) as fh:
                data = _json.load(fh)
            valid = [d["preference"] for d in data
                     if d.get("preference") is not None and d["preference"] == d["preference"]]
            if valid:
                rates.append(sum(1 for p in valid if p == 1.0) / len(valid))
        if rates:
            return np.mean(rates)
    return None


def fig_task_transfer_heatmap_real():
    """
    Task transfer heatmap using actual training data from all available runs.
    Uses benchmark_predictions epoch_0 (pre) and final epoch (post).
    AlpacaEval column averaged across all available opponents.
    """
    import glob as _glob

    # Discover all training runs and their metadata
    RUNS = []
    training_dir = Path("data/training")
    for run_path in sorted(training_dir.iterdir()):
        if not run_path.is_dir():
            continue
        run_name = run_path.name.split("__")[0]

        # Clean opponent names for display
        OPPONENT_DISPLAY = {
            "qwen": "Qwen 2.5 7B",
            "gpt_4o": "GPT-4o",
            "haiku_3_5": "Haiku 3.5",
            "opus_4_1": "Opus 4.1",
            "ll_3_1_70b": "Llama 3.1 70B",
            "ll_3_3_70b": "Llama 3.3 70B",
            "ll_3_1_8b": "Llama 3.1 8B",
            "qwen3_30b": "Qwen 3.0 30B",
            "multi_model_holdout_ll_3_1_70b": "Multi (holdout 70B)",
        }

        def _clean_opponent(raw):
            raw = raw.replace("_tinker_small", "")
            return OPPONENT_DISPLAY.get(raw, raw)

        # Determine base model from directory name conventions
        if run_name.startswith("01_sft_pw_ll_3_3_70b_vs"):
            base = "ll-3.3-70b"
            opp = _clean_opponent(run_name.split("vs_")[1])
            label = f"Llama 3.3 70B (PW → {opp})"
            trained_name = f"ll-3.3-70b-{run_name}"
        elif run_name.startswith("01_sft_pw_qwen3_30b_vs"):
            base = "qwen-3.0-30b"
            opp = _clean_opponent(run_name.split("vs_")[1])
            label = f"Qwen 3.0 30B (PW → {opp})"
            trained_name = f"qwen-3.0-30b-{run_name}"
        elif run_name.startswith("02_sft_ind_"):
            base = "ll-3.1-8b"
            opp = _clean_opponent(run_name.split("vs_")[1]) if "vs_" in run_name else "?"
            label = f"Llama 3.1 8B (IND → {opp})"
            trained_name = f"ll-3.1-8b-{run_name}"
        elif run_name.startswith("03_sft_pw_"):
            base = "ll-3.1-8b"
            opp = _clean_opponent(run_name.split("vs_")[1]) if "vs_" in run_name else "multi"
            label = f"Llama 3.1 8B (PW → {opp})"
            trained_name = f"ll-3.1-8b-{run_name}"
        else:
            base = "ll-3.1-8b"
            opp = _clean_opponent(run_name.split("vs_")[1]) if "vs_" in run_name else "?"
            label = f"Llama 3.1 8B (PW → {opp})"
            trained_name = f"ll-3.1-8b-{run_name}"

        RUNS.append({
            "run_dir": str(run_path),
            "run_name": run_name,
            "label": label,
            "base_model": base,
            "trained_name": trained_name,
        })

    # Columns: task transfer OPs + dataset xevals + AlpacaEval
    # Using benchmark_predictions directory names as keys
    TASK_COLS = [
        ("xeval_tag_at_pw", "PW Rec\n(AT)"),
        ("xeval_tag_at_ind", "IND Rec\n(AT)"),
        ("xeval_format_ind", "IND Rec\n(UT)"),
        ("xeval_format_pw", "PW Rec\n(UT)"),
        ("xeval_task_pref_pw", "PW Pref\n(UT)"),
        ("xeval_task_pref_ind", "IND Pref\n(UT)"),
    ]
    DATASET_COLS = [
        ("xeval_dataset_wikisum", "WikiSum"),
        ("xeval_dataset_bigcodebench", "BigCode"),
        ("xeval_dataset_pku", "PKU"),
    ]
    AE_COL = ("alpaca_eval", "AlpacaEval\nSelf-Pref")

    all_cols = TASK_COLS + DATASET_COLS + [AE_COL]
    col_keys = [c[0] for c in all_cols]
    col_labels = [c[1] for c in all_cols]
    n_cols = len(col_labels)

    # Separator positions (for visual grouping)
    task_end = len(TASK_COLS)
    dataset_end = task_end + len(DATASET_COLS)

    run_labels = []
    matrices = []

    for info in RUNS:
        bp_data = _load_benchmark_accuracies(info["run_dir"])
        if not bp_data:
            print(f"  ⚠ Skipping {info['run_name']}: no benchmark data")
            continue

        row = np.full(n_cols, np.nan)
        for j, key in enumerate(col_keys):
            if key == "alpaca_eval":
                # AlpacaEval: delta of avg self-selection rate
                base_ae = _load_alpaca_eval_avg_self_selection(info["base_model"])
                trained_ae = _load_alpaca_eval_avg_self_selection(info["trained_name"])
                if base_ae is not None and trained_ae is not None:
                    row[j] = trained_ae - base_ae
            elif key in bp_data:
                pre, post = bp_data[key]
                if pre is not None and post is not None:
                    row[j] = post - pre

        run_labels.append(info["label"])
        matrices.append(row)

    if not matrices:
        print("  ⚠ No real training data found, skipping real heatmap")
        return

    matrix = np.array(matrices)
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)

    fig_h = max(len(run_labels) * 0.55 + 2, 5)
    fig, ax = plt.subplots(figsize=(n_cols * 0.85 + 2.5, fig_h))

    # Draw with gaps between groups using pcolormesh-like approach
    im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")

    # Annotate cells
    for i in range(len(run_labels)):
        for j in range(n_cols):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.35 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=7, fontweight="bold", color=color)

    # Vertical separators between groups
    ax.axvline(x=task_end - 0.5, color="white", linewidth=2.5)
    ax.axvline(x=dataset_end - 0.5, color="white", linewidth=2.5)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=40, ha="right")
    ax.set_yticks(range(len(run_labels)))
    ax.set_yticklabels(run_labels, fontsize=7)
    ax.set_ylabel("Training Run", fontsize=10, fontweight="bold")

    # Group labels above columns
    ax.text(task_end / 2 - 0.5, -1.3, "Task Transfer", ha="center", fontsize=8,
            fontweight="bold", style="italic")
    ax.text(task_end + len(DATASET_COLS) / 2 - 0.5, -1.3, "Dataset Transfer", ha="center",
            fontsize=8, fontweight="bold", style="italic")
    ax.text(n_cols - 1, -1.3, "AE", ha="center", fontsize=8,
            fontweight="bold", style="italic")

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.03, aspect=25)
    cbar.set_label("Accuracy Δ (post − pre)", fontsize=8)

    ax.set_title("Training Transfer — All Models", fontsize=11, fontweight="bold", pad=20)
    plt.tight_layout()
    path = OUT_DIR / "heatmap_task_transfer_with_ae_real.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================================
# FIGURE 7: Dataset transfer heatmap — one per model
# y = trained-on dataset, x = tested-on dataset, cell = delta accuracy
# (averaged across the 4 recognition task OPs)
# ============================================================================
def fig_dataset_transfer_heatmap(pair_data):
    """
    One heatmap per model. Rows = trained-on dataset, cols = tested-on dataset.
    Cell = accuracy delta averaged across 4 recognition task OPs.
    """
    datasets = ["WikiSum", "ShareGPT", "PKU", "BigCode"]
    rec_ops = ["PW-Rec (UT)", "IND-Rec (UT)", "PW-Rec (AT)", "IND-Rec (AT)"]

    size_labels = list(pair_data.keys())
    n_models = len(size_labels)

    # Load per-dataset data
    all_per_ds = {}
    for size_label, (pre_model, post_model) in MODEL_PAIRS.items():
        pre_ds = load_per_dataset_performance(pre_model)
        post_ds = load_per_dataset_performance(post_model)
        all_per_ds[size_label] = {"pre": pre_ds, "post": post_ds}

    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)

    for m_idx, size in enumerate(size_labels):
        ax = axes[m_idx]
        pre_m, post_m = MODEL_PAIRS[size]
        pre_ds = all_per_ds[size]["pre"]
        post_ds = all_per_ds[size]["post"]

        n = len(datasets)
        matrix = np.full((n, n), np.nan)

        for i, train_ds in enumerate(datasets):
            for j, test_ds in enumerate(datasets):
                # Average delta across rec_ops for this dataset pair
                deltas = []
                for op in rec_ops:
                    pre_val = pre_ds.get(op, {}).get(test_ds)
                    post_val = post_ds.get(op, {}).get(test_ds)
                    if pre_val is not None and post_val is not None:
                        deltas.append(post_val - pre_val)
                if deltas:
                    matrix[i, j] = np.mean(deltas)

        im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="equal")

        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 0.2 else "black"
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=color)

        # Highlight diagonal
        for k in range(n):
            ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                        fill=False, edgecolor="black", linewidth=2))

        ax.set_xticks(range(n))
        ax.set_xticklabels(datasets, fontsize=9, rotation=35, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(datasets if m_idx == 0 else [], fontsize=9)
        if m_idx == 0:
            ax.set_ylabel("Trained on", fontsize=10, fontweight="bold")
        ax.set_xlabel("Tested on", fontsize=10, fontweight="bold")
        ax.set_title(f"{size} ({pre_m} → {post_m})", fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label("Accuracy Δ (post − pre)", fontsize=10)

    fig.suptitle("Dataset Domain Transfer (avg over 4 recognition OPs)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = OUT_DIR / "heatmap_dataset_transfer.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def fig_training_effect_panels(pre_model="ll-3.1-8b", post_model="opus-4.1"):
    """
    2×2 panel showing training effect on a single evaluator's score-distance
    relationship. Pre-training (light gray background) vs post-training (dark
    blue foreground) with trendlines and slope annotations.

    Uses proxy models: pre_model as "untrained", post_model as "trained" stand-in.

    Layout:
      (a) UT PW   (b) UT IND (adjusted)
      (c) AT PW   (d) AT IND (adjusted)
    """
    from self_rec_framework.src.helpers.model_names import LM_ARENA_SCORES

    panels = {
        "(a) User-Tag — Pairwise": "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst",
        "(b) User-Tag — Individual": "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst",
        "(c) Assistant-Tag — Pairwise": "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
        "(d) Assistant-Tag — Individual": "COLM_02_AT_IND-C_Rec_NPr_FA_Inst",
    }

    def get_score(model):
        if model in LM_ARENA_SCORES:
            return LM_ARENA_SCORES[model]
        base = model.replace("-thinking", "")
        if base in LM_ARENA_SCORES:
            return LM_ARENA_SCORES[base]
        return None

    panel_data = {}
    for title, exp_name in panels.items():
        exp_dir = AGG_DIR / exp_name
        if not exp_dir.exists():
            print(f"  ⚠ Missing dir: {exp_name}")
            return
        ts_dir = sorted(exp_dir.iterdir(), reverse=True)[0]
        csv_path = ts_dir / "rank_distance_data.csv"
        if not csv_path.exists():
            print(f"  ⚠ Missing: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        df["eval_score"] = df["evaluator"].apply(get_score)
        df["gen_score"] = df["generator"].apply(get_score)
        df = df.dropna(subset=["eval_score", "gen_score"])
        df["score_distance"] = df["eval_score"] - df["gen_score"]

        self_scores = load_self_scores(exp_name)
        if self_scores is not None:
            df = adjust_ind_performance(df, self_scores)

        pre_df = df[df["evaluator"] == pre_model].copy()
        post_df = df[df["evaluator"] == post_model].copy()
        panel_data[title] = {"pre": pre_df, "post": post_df}

    pre_color = "#AAAAAA"
    post_color = "#1565C0"
    pre_line_color = "#888888"
    post_line_color = "#0D47A1"

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8),
        gridspec_kw={"wspace": 0.15, "hspace": 0.1},
    )

    titles = list(panels.keys())
    is_ind = {1, 3}
    for idx, title in enumerate(titles):
        row, col = divmod(idx, 2)
        ax = axes[row][col]
        pre_df = panel_data[title]["pre"]
        post_df = panel_data[title]["post"]

        x_line = None
        coeffs_pre = None
        coeffs_post = None
        pre_agg = pd.DataFrame()
        post_agg = pd.DataFrame()

        if not pre_df.empty:
            ax.scatter(
                pre_df["score_distance"], pre_df["performance"],
                c=pre_color, alpha=0.35, s=20, edgecolors="none",
                label=f"{pre_model} (pre)", zorder=2,
            )
            pre_agg = pre_df.groupby("generator").agg(
                score_distance=("score_distance", "first"),
                performance=("performance", "mean"),
                weight=("n_samples", "sum"),
            ).reset_index()
            if len(pre_agg) > 2:
                x_v = pre_agg["score_distance"].values
                y_v = pre_agg["performance"].values
                w = np.sqrt(pre_agg["weight"].values)
                coeffs_pre = np.polyfit(x_v, y_v, 1, w=w)
                x_line = np.linspace(
                    min(x_v.min(), post_df["score_distance"].min() if not post_df.empty else x_v.min()),
                    max(x_v.max(), post_df["score_distance"].max() if not post_df.empty else x_v.max()),
                    100,
                )
                ax.plot(x_line, coeffs_pre[0] * x_line + coeffs_pre[1],
                        color=pre_line_color, linewidth=1.5, linestyle="--", alpha=0.7, zorder=3)

        if not post_df.empty:
            ax.scatter(
                post_df["score_distance"], post_df["performance"],
                c=post_color, alpha=0.7, s=25, edgecolors="none",
                label=f"{post_model} (post)", zorder=4,
            )
            post_agg = post_df.groupby("generator").agg(
                score_distance=("score_distance", "first"),
                performance=("performance", "mean"),
                weight=("n_samples", "sum"),
            ).reset_index()
            if len(post_agg) > 2:
                x_v = post_agg["score_distance"].values
                y_v = post_agg["performance"].values
                w = np.sqrt(post_agg["weight"].values)
                coeffs_post = np.polyfit(x_v, y_v, 1, w=w)
                if x_line is None:
                    x_line = np.linspace(x_v.min(), x_v.max(), 100)
                ax.plot(x_line, coeffs_post[0] * x_line + coeffs_post[1],
                        color=post_line_color, linewidth=2, alpha=0.9, zorder=5)

        slopes = []
        if coeffs_pre is not None and len(pre_agg) > 2:
            slopes.append(f"pre slope: {coeffs_pre[0]:.4f}")
        if coeffs_post is not None and len(post_agg) > 2:
            slopes.append(f"post slope: {coeffs_post[0]:.4f}")
        if len(slopes) == 2:
            delta = coeffs_post[0] - coeffs_pre[0]
            slopes.append(f"Δ slope: {delta:+.4f}")
        if slopes:
            ax.text(
                0.03, 0.03, "\n".join(slopes),
                transform=ax.transAxes, fontsize=7.5,
                verticalalignment="bottom", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            )

        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0.0, 1.05)

        y_label = "Adjusted Accuracy" if idx in is_ind else "Recognition Accuracy"
        if col == 0:
            ax.set_ylabel(y_label, fontsize=10)
        else:
            ax.set_ylabel("")

        if row == 1:
            ax.set_xlabel("Elo Score Distance\n(Evaluator − Generator)", fontsize=10)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

        if idx == 0:
            ax.legend(fontsize=8, loc="upper left", markerscale=1.5)

    fig.suptitle(
        f"Training Effect on Score-Distance Relationship\n"
        f"(proxy: {pre_model} → {post_model})",
        fontsize=13, fontweight="bold", y=1.0,
    )

    safe_pre = pre_model.replace(".", "_").replace("-", "_")
    path = OUT_DIR / f"training_effect_score_distance_{safe_pre}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def main():
    print("Loading model pair data...")
    pair_data = load_all_model_pairs()
    print(f"\nLoaded {len(pair_data)} model pairs.\n")

    print("Generating uplift prototype figures...")

    print("\n1. Transfer heatmap (delta per OP)")
    fig_transfer_heatmap(pair_data)

    print("\n2. Grouped bar chart (pre/post per OP)")
    fig_grouped_bars(pair_data)

    print("\n3a. Grouped arrow plot (all models, all OPs)")
    fig_arrow_plot(pair_data)

    print("\n3b. Task OP transfer panels (4 panels)")
    fig_arrow_task_panels(pair_data)

    print("\n3c. Dataset transfer panels (4 panels)")
    fig_arrow_dataset_panels(pair_data)

    print("\n4. Radar/spider chart (pre vs post polygon)")
    fig_radar(pair_data)

    print("\n5. Delta matrix (per OP × per dataset)")
    fig_delta_matrix(pair_data)

    print("\n6. Task OP transfer heatmap (one per model)")
    fig_task_transfer_heatmap(pair_data)

    print("\n6b. Task OP transfer heatmap with AlpacaEval (proxy data)")
    fig_task_transfer_heatmap_with_ae(pair_data)

    print("\n6c. Task OP transfer heatmap with AlpacaEval (REAL data)")
    fig_task_transfer_heatmap_real()

    print("\n7. Dataset transfer heatmap (one per model)")
    fig_dataset_transfer_heatmap(pair_data)

    print("\n8a. Arrow task panels (REAL data)")
    fig_arrow_task_panels_real()

    print("\n8b. Arrow dataset panels (REAL data)")
    fig_arrow_dataset_panels_real()

    print("\n8c. Arrow model panels (REAL data)")
    fig_arrow_model_panels_real()

    print("\n9a. Training effect score-distance (ll-3.1-8b → opus-4.1)")
    fig_training_effect_panels(pre_model="ll-3.1-8b", post_model="opus-4.1")

    print("\n9b. Training effect score-distance (gpt-4.1-mini → opus-4.1)")
    fig_training_effect_panels(pre_model="gpt-4.1-mini", post_model="opus-4.1")

    print(f"\nAll uplift prototypes saved to: {OUT_DIR}/")


# ============================================================================
# REAL DATA ARROW FIGURES
# ============================================================================

# Map benchmark_predictions dir names → display labels
BENCH_TO_TASK = {
    "xeval_tag_at_pw": "PW (AT)",
    "xeval_tag_at_ind": "IND (AT)",
    "xeval_format_ind": "IND (UT)",
    "xeval_format_pw": "PW (UT)",
    "xeval_task_pref_pw": "PW Pref (UT)",
    "xeval_task_pref_ind": "IND Pref (UT)",
}

BENCH_TO_DATASET = {
    "xeval_dataset_wikisum": "WikiSum",
    "xeval_dataset_bigcodebench": "BigCode",
    "xeval_dataset_pku": "PKU",
}

# Opponent display names
OPPONENT_DISPLAY = {
    "qwen": "Qwen 2.5 7B",
    "gpt_4o": "GPT-4o",
    "haiku_3_5": "Haiku 3.5",
    "opus_4_1": "Opus 4.1",
    "ll_3_1_70b": "Llama 3.1 70B",
    "ll_3_3_70b": "Llama 3.3 70B",
    "ll_3_1_8b": "Llama 3.1 8B",
    "qwen3_30b": "Qwen 3.0 30B",
    "multi_model_holdout_ll_3_1_70b": "Multi (holdout 70B)",
}


def _discover_training_runs(subsets=None, training_dir="data/training"):
    """Discover all training runs and return structured metadata.

    Uses the unified training_runs module to handle original, archived,
    and reorganized naming conventions.

    Args:
        subsets: List of subsets to include. None = all subsets.
        training_dir: Path to training data root.
    """
    from scripts.alpaca_eval.training_runs import (
        discover_training_runs, get_benchmark_accuracy, get_val_accuracy,
    )

    BASE_DISPLAY = {
        "ll-3.1-8b": "Llama 3.1 8B",
        "ll-3.3-70b": "Llama 3.3 70B",
        "qwen-3.0-30b": "Qwen 3.0 30B",
        "gpt-oss-120b": "GPT-OSS 120B",
    }

    unified_runs = discover_training_runs(training_dir, subsets=subsets)
    runs = []

    for r in unified_runs:
        base_display = BASE_DISPLAY.get(r.base_model, r.base_model)
        opponent_display = OPPONENT_DISPLAY.get(r.opponent, r.opponent)
        training_type = r.fmt.upper()  # "PW" or "IND"
        tag = r.tag.upper()  # "UT" or "AT"

        # Load benchmark accuracies using the unified helpers
        bp_data = {}
        for bench in r.benchmarks:
            pre = get_benchmark_accuracy(r, bench, epoch=0, prefer_full=True)
            post = get_benchmark_accuracy(r, bench, epoch=None, prefer_full=True)
            if pre is not None and post is not None:
                bp_data[bench] = {"pre": pre, "post": post}

        if not bp_data:
            continue

        # Load val/accuracy for held-out data
        val_pre = get_val_accuracy(r, epoch=0)
        val_post = get_val_accuracy(r)
        if val_pre is not None and val_post is not None:
            bp_data["val_accuracy"] = {"pre": val_pre, "post": val_post}

        # Load AlpacaEval data
        ae_base = _load_alpaca_eval_avg_self_selection(r.base_model)
        ae_trained = _load_alpaca_eval_avg_self_selection(r.trained_name)

        runs.append({
            "run_dir": str(r.run_path),
            "run_name": r.run_name,
            "base": base_display,
            "base_short": r.base_model,
            "opponent": opponent_display,
            "training_type": training_type,
            "tag": tag,
            "dataset": r.dataset,
            "subset": r.subset,
            "trained_name": r.trained_name,
            "label": f"{base_display} ({tag} {training_type} → {opponent_display})",
            "bp_data": bp_data,
            "ae_base": ae_base,
            "ae_trained": ae_trained,
        })
    return runs


def _draw_real_arrows(ax, runs, x_labels, get_pre_post_fn, colors,
                      show_xlabel=True, show_delta=True, title=None):
    """Draw grouped arrows for real training data on a single axes."""
    n_runs = len(runs)
    if n_runs == 0:
        return
    group_width = 0.6
    offsets = np.linspace(-group_width / 2, group_width / 2, max(n_runs, 2))
    if n_runs == 1:
        offsets = [0.0]

    # Alternating column background
    for o_idx in range(len(x_labels)):
        if o_idx % 2 == 1:
            ax.axvspan(o_idx - 0.5, o_idx + 0.5, facecolor="white", edgecolor="none", zorder=0)
            ax.axvspan(o_idx - 0.5, o_idx + 0.5, facecolor="none",
                       edgecolor="#cccccc", hatch="//", linewidth=0, zorder=0)

    for r_idx, run in enumerate(runs):
        color = colors[r_idx % len(colors)]

        for x_idx, label in enumerate(x_labels):
            pre_val, post_val = get_pre_post_fn(run, label)
            if pre_val is None or post_val is None:
                continue

            x = x_idx + offsets[r_idx]
            delta = post_val - pre_val

            ax.scatter(x, pre_val, color="gray", s=30, zorder=3,
                       edgecolors="black", linewidth=0.4)
            arrow_color = "#2E7D32" if delta > 0 else "#C62828"
            ax.annotate("", xy=(x, post_val), xytext=(x, pre_val),
                        arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5),
                        zorder=4)
            ax.scatter(x, post_val, color=color, s=40, zorder=5,
                       edgecolors="black", linewidth=0.4)
            if show_delta:
                ax.text(x + 0.1, (pre_val + post_val) / 2, f"{delta:+.2f}",
                        fontsize=5, color=arrow_color, va="center")

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=9)
    if show_xlabel:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=7.5, rotation=30, ha="right")
    else:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels([])
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")


def _avg_pre_post(runs, bench_key):
    """Average pre and post accuracy across multiple runs for a benchmark."""
    pres, posts = [], []
    for r in runs:
        if bench_key in r["bp_data"]:
            pre, post = r["bp_data"][bench_key]
            if pre is not None:
                pres.append(pre)
            if post is not None:
                posts.append(post)
    pre_avg = np.mean(pres) if pres else None
    post_avg = np.mean(posts) if posts else None
    return pre_avg, post_avg


def _avg_ae_pre_post(runs):
    """Average AlpacaEval pre/post across multiple runs."""
    bases = [r["ae_base"] for r in runs if r["ae_base"] is not None]
    trained = [r["ae_trained"] for r in runs if r["ae_trained"] is not None]
    return (np.mean(bases) if bases else None,
            np.mean(trained) if trained else None)


def fig_arrow_task_panels_real():
    """Real data: 2×2 panels by training OP (UT PW, UT IND, AT PW, AT IND).
    X-axis = test task OPs + AE. One dot-line per base model (averaged
    over training opponents). Panels without training data show placeholder.
    Uses _full benchmarks where available."""
    runs = _discover_training_runs(training_dir=TRAINING_DIR)
    if not runs:
        print("  ⚠ No training data found")
        return

    # X-axis: consistent across all panels. The column matching the training OP gets highlighted.
    test_ops = ["PW (UT)", "IND (UT)", "PW (AT)", "IND (AT)", "PW Pref", "IND Pref", "AE"]

    # Map test OP labels to benchmark keys — depends on the run's tag.
    # For UT-trained runs: AT tests use xeval_tag_at_*, UT tests use xeval_format_*
    # For AT-trained runs: UT tests use xeval_tag_ut_*, AT tests use xeval_format_*
    def _get_bench_for_label(label, run):
        """Get benchmark keys for a test OP label, given the training run's tag."""
        tag = run.get("tag", "UT")
        if label == "PW (UT)":
            if tag == "UT":
                return ["xeval_format_pw_full", "xeval_format_pw"]  # only for IND-trained
            else:
                return ["xeval_tag_ut_pw_full", "xeval_tag_ut_pw"]
        elif label == "IND (UT)":
            if tag == "UT":
                return ["xeval_format_ind_full", "xeval_format_ind"]
            else:
                return ["xeval_tag_ut_ind_full", "xeval_tag_ut_ind"]
        elif label == "PW (AT)":
            if tag == "AT":
                return ["xeval_format_pw_full", "xeval_format_pw"]  # only for IND-trained
            else:
                return ["xeval_tag_at_pw_full", "xeval_tag_at_pw"]
        elif label == "IND (AT)":
            if tag == "AT":
                return ["xeval_format_ind_full", "xeval_format_ind"]
            else:
                return ["xeval_tag_at_ind_full", "xeval_tag_at_ind"]
        elif label == "PW Pref":
            return ["xeval_task_pref_pw_full", "xeval_task_pref_pw"]
        elif label == "IND Pref":
            return ["xeval_task_pref_ind_full", "xeval_task_pref_ind"]
        return []

    base_models = ["Llama 3.1 8B", "Qwen 3.0 30B", "Llama 3.3 70B", "GPT-OSS 120B"]
    base_colors = {
        "Llama 3.1 8B": "#1565C0", "Qwen 3.0 30B": "#E65100",
        "Llama 3.3 70B": "#2E7D32", "GPT-OSS 120B": "#C62828",
    }

    # Training OP panels: (title, training_type, tag, highlighted_column)
    panels = [
        ("(a) Trained on: PW (UT)", "PW", "UT", "PW (UT)"),
        ("(b) Trained on: IND (UT)", "IND", "UT", "IND (UT)"),
        ("(c) Trained on: PW (AT)", "PW", "AT", "PW (AT)"),
        ("(d) Trained on: IND (AT)", "IND", "AT", "IND (AT)"),
    ]

    def _load_val_accuracy(run_dir):
        """Load pre (step 0) and post (final step) val/accuracy from metrics.jsonl."""
        import json as _json
        metrics_path = Path(run_dir) / "metrics" / "metrics.jsonl"
        if not metrics_path.exists():
            return None, None
        with open(metrics_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        pre, post = None, None
        for l in lines:
            d = _json.loads(l)
            if "val/accuracy" in d:
                if pre is None:
                    pre = d["val/accuracy"]
                post = d["val/accuracy"]  # keep updating — last one is final
        return pre, post

    def build_avg_pair_data(filtered_runs, bases, highlight_col):
        """Build pair_data. For the highlighted column (training OP),
        use val/accuracy (held-out data). For others, use benchmarks."""
        pair_data = {}
        for base in bases:
            base_runs = [r for r in filtered_runs if r["base"] == base]
            if not base_runs:
                continue
            pre_dict, post_dict = {}, {}
            for label in test_ops:
                if label == "AE":
                    pre_val, post_val = _avg_ae_pre_post(base_runs)
                elif label == highlight_col:
                    # This is the training OP — use val/accuracy (held-out)
                    pres, posts = [], []
                    for r in base_runs:
                        p, q = _load_val_accuracy(r["run_dir"])
                        if p is not None:
                            pres.append(p)
                        if q is not None:
                            posts.append(q)
                    pre_val = np.mean(pres) if pres else None
                    post_val = np.mean(posts) if posts else None
                else:
                    # Use tag-aware benchmark lookup — average across runs
                    # (runs in a group may have different tags if mixed, so check per-run)
                    pre_vals, post_vals = [], []
                    for r in base_runs:
                        bench_keys = _get_bench_for_label(label, r)
                        for bk in bench_keys:
                            bp = r["bp_data"].get(bk)
                            if bp:
                                pre_vals.append(bp["pre"])
                                post_vals.append(bp["post"])
                                break
                    pre_val = np.mean(pre_vals) if pre_vals else None
                    post_val = np.mean(post_vals) if post_vals else None
                if pre_val is not None:
                    pre_dict[label] = pre_val
                if post_val is not None:
                    post_dict[label] = post_val
            if pre_dict or post_dict:
                pair_data[base] = {"pre": pre_dict, "post": post_dict}
        return pair_data

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes_flat = axes.flatten()

    for p_idx, (title, fmt, tag, highlight_col) in enumerate(panels):
        ax = axes_flat[p_idx]
        filtered = [r for r in runs if r["training_type"] == fmt and r.get("tag", "UT") == tag]

        if not filtered:
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.text(0.5, 0.5, "No training data\navailable yet", ha="center", va="center",
                    fontsize=11, color="gray", fontstyle="italic", transform=ax.transAxes)
            ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_ylim(0.0, 1.05)
            ax.set_xticks(range(len(test_ops)))
            ax.set_xticklabels(test_ops, fontsize=7.5, rotation=30, ha="right")
            # Still highlight the training OP column even on empty panels
            hi_x = test_ops.index(highlight_col)
            ax.axvspan(hi_x - 0.45, hi_x + 0.45, color="gold", alpha=0.15, zorder=0)
            ax.text(hi_x, 0.98, "trained", ha="center",
                    fontsize=6, fontstyle="italic", color="goldenrod")
            continue

        pair_data = build_avg_pair_data(filtered, base_models, highlight_col)
        colors = [base_colors[b] for b in pair_data.keys()]

        _draw_real_arrows(ax, list(pair_data.values()), test_ops,
                          lambda run, label: (run["pre"].get(label), run["post"].get(label)),
                          colors, show_xlabel=True, show_delta=True, title=title)

        # Highlight the column matching the training OP
        hi_x = test_ops.index(highlight_col)
        ax.axvspan(hi_x - 0.45, hi_x + 0.45, color="gold", alpha=0.15, zorder=0)
        ax.text(hi_x, ax.get_ylim()[1] * 0.98, "trained", ha="center",
                fontsize=6, fontstyle="italic", color="goldenrod")

    handles = []
    for base in base_models:
        n = len([r for r in runs if r["base"] == base])
        if n > 0:
            handles.append(plt.Line2D([0], [0], marker='o', color=base_colors[base],
                                       markersize=6, linestyle='None',
                                       label=f"{base} (n={n})"))
    handles.append(plt.Line2D([0], [0], marker='o', color='gray', markersize=5,
                               linestyle='None', markeredgecolor='black',
                               markeredgewidth=0.4, label="Pre-training"))
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Task Transfer (averaged over training opponents)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    path = OUT_DIR / "arrow_task_panels_real.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def fig_arrow_dataset_panels_real():
    """Real data: 2×2 panels, one per training dataset domain.
    X-axis = 4 datasets. Dot-lines per base model (averaged over opponents).
    Currently ALL training is on ShareGPT — only panel (d) has data.
    For the highlighted trained-on column, use _full benchmark predictions
    (non-dataset benchmarks are implicitly ShareGPT-based).
    Cross-dataset columns use xeval_dataset_* benchmarks."""
    runs = _discover_training_runs(training_dir=TRAINING_DIR)
    if not runs:
        print("  ⚠ No training data found")
        return

    base_models = ["Llama 3.1 8B", "Qwen 3.0 30B", "Llama 3.3 70B", "GPT-OSS 120B"]
    base_colors = {
        "Llama 3.1 8B": "#1565C0", "Qwen 3.0 30B": "#E65100",
        "Llama 3.3 70B": "#2E7D32", "GPT-OSS 120B": "#C62828",
    }
    datasets = ["WikiSum", "BigCode", "PKU", "ShareGPT"]

    # Map display names to training run dataset field
    DS_DISPLAY_TO_FIELD = {
        "WikiSum": "wikisum", "BigCode": "bigcodebench",
        "PKU": "pku", "ShareGPT": "sharegpt",
    }

    # Map dataset labels to cross-dataset benchmark keys
    ds_bench_map = {
        "WikiSum": ["xeval_dataset_wikisum_full", "xeval_dataset_wikisum"],
        "BigCode": ["xeval_dataset_bigcodebench_full", "xeval_dataset_bigcodebench"],
        "PKU": ["xeval_dataset_pku_full", "xeval_dataset_pku"],
        "ShareGPT": ["xeval_dataset_sharegpt_full", "xeval_dataset_sharegpt"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    axes_flat = axes.flatten()
    panel_labels = ["(a) Trained on: WikiSum", "(b) Trained on: BigCode",
                    "(c) Trained on: PKU", "(d) Trained on: ShareGPT"]

    for d_idx, train_ds in enumerate(datasets):
        ax = axes_flat[d_idx]
        train_ds_field = DS_DISPLAY_TO_FIELD[train_ds]

        # Filter runs trained on this dataset
        filtered = [r for r in runs if r.get("dataset") == train_ds_field]

        if not filtered:
            ax.set_title(panel_labels[d_idx], fontsize=10, fontweight="bold")
            ax.text(0.5, 0.5, "No training data\navailable yet", ha="center", va="center",
                    fontsize=11, color="gray", fontstyle="italic", transform=ax.transAxes)
            ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_ylim(0.0, 1.05)
            ax.set_xticks(range(len(datasets)))
            ax.set_xticklabels(datasets, fontsize=8, rotation=30, ha="right")
            trained_x = datasets.index(train_ds)
            ax.axvspan(trained_x - 0.45, trained_x + 0.45, color="gold", alpha=0.1, zorder=0)
            continue

        # Build averaged pair_data per base model
        pair_data = {}
        for base in base_models:
            base_runs = [r for r in filtered if r["base"] == base]
            if not base_runs:
                continue
            pre_dict, post_dict = {}, {}
            for ds in datasets:
                ds_field = DS_DISPLAY_TO_FIELD[ds]
                if ds == train_ds:
                    # Training dataset = use val/accuracy (held-out data)
                    pres, posts = [], []
                    for r in base_runs:
                        bp = r["bp_data"]
                        if "val_accuracy" in bp:
                            pres.append(bp["val_accuracy"]["pre"])
                            posts.append(bp["val_accuracy"]["post"])
                    pre_val = np.mean(pres) if pres else None
                    post_val = np.mean(posts) if posts else None
                else:
                    # Cross-dataset: use xeval_dataset_* benchmarks
                    bench_keys = ds_bench_map.get(ds, [])
                    pre_vals, post_vals = [], []
                    for r in base_runs:
                        bp = r["bp_data"]
                        for bk in bench_keys:
                            if bk in bp:
                                pre_vals.append(bp[bk]["pre"])
                                post_vals.append(bp[bk]["post"])
                                break
                    pre_val = np.mean(pre_vals) if pre_vals else None
                    post_val = np.mean(post_vals) if post_vals else None

                if pre_val is not None:
                    pre_dict[ds] = pre_val
                if post_val is not None:
                    post_dict[ds] = post_val
            if pre_dict or post_dict:
                pair_data[base] = {"pre": pre_dict, "post": post_dict}

        colors = [base_colors[b] for b in pair_data.keys()]

        _draw_real_arrows(ax, list(pair_data.values()), datasets,
                          lambda run, label: (run["pre"].get(label), run["post"].get(label)),
                          colors, show_xlabel=True, show_delta=True,
                          title=panel_labels[d_idx])

        # Highlight the trained-on dataset
        trained_x = datasets.index(train_ds)
        ax.axvspan(trained_x - 0.45, trained_x + 0.45, color="gold", alpha=0.1, zorder=0)
        ax.text(trained_x, ax.get_ylim()[1] * 0.98, "trained", ha="center",
                fontsize=6, fontstyle="italic", color="goldenrod")

    handles = []
    for base in base_models:
        n = len([r for r in runs if r["base"] == base])
        if n > 0:
            handles.append(plt.Line2D([0], [0], marker='o', color=base_colors[base],
                                       markersize=6, linestyle='None',
                                       label=f"{base} (n={n})"))
    handles.append(plt.Line2D([0], [0], marker='o', color='gray', markersize=5,
                               linestyle='None', markeredgecolor='black',
                               markeredgewidth=0.4, label="Pre-training"))
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Dataset Domain Transfer (avg over training opponents)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    path = OUT_DIR / "arrow_dataset_panels_real.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def fig_arrow_model_panels_real():
    """Real data: one panel per training opponent.
    X-axis = generator models tested against (from AlpacaEval self-preference results).
    Each panel has dot-lines for 8B, 30B, 70B.
    Self-selection rate (from alpaca_eval results) as pre/post values.
    Highlights when the tested opponent matches the trained opponent."""
    import json as _json

    base_models = ["Llama 3.1 8B", "Qwen 3.0 30B", "Llama 3.3 70B"]
    base_colors = {"Llama 3.1 8B": "#1565C0", "Qwen 3.0 30B": "#E65100", "Llama 3.3 70B": "#2E7D32"}
    base_short = {"Llama 3.1 8B": "ll-3.1-8b", "Qwen 3.0 30B": "qwen-3.0-30b", "Llama 3.3 70B": "ll-3.3-70b"}

    runs = _discover_training_runs(training_dir=TRAINING_DIR)
    if not runs:
        print("  ⚠ No training data found")
        return

    pw_runs = [r for r in runs if r["training_type"] == "PW"]

    # Get unique training opponents
    opponents_8b = sorted(set(r["opponent"] for r in pw_runs if r["base"] == "Llama 3.1 8B"))

    # Collect all generator models from alpaca_eval results
    ae_results_dir = Path("data/alpaca_eval/results")

    # Get test opponents: all generators available in base model results
    test_generators = set()
    for base in base_models:
        short = base_short[base]
        base_dir = ae_results_dir / short
        if base_dir.exists():
            for f in base_dir.glob("vs_*.json"):
                test_generators.add(f.stem.replace("vs_", ""))
    test_generators = sorted(test_generators)

    # Helper: load self-selection rate for a specific judge vs opponent
    def _load_ae_rate(judge_name, opponent):
        # Try exact, then strip _tinker_small
        for name in [judge_name, judge_name.replace("_tinker_small", "")]:
            path = ae_results_dir / name / f"vs_{opponent}.json"
            if path.exists():
                with open(path) as f:
                    data = _json.load(f)
                valid = [d["preference"] for d in data
                         if d.get("preference") is not None and d["preference"] == d["preference"]]
                if valid:
                    return sum(1 for p in valid if p == 1.0) / len(valid)
        return None

    # One panel per training opponent
    n_panels = len(opponents_8b)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 4), sharey=True)
    if nrows * ncols == 1:
        axes = np.array([[axes]])
    axes_flat = axes.flatten()

    for p_idx, train_opponent in enumerate(opponents_8b):
        ax = axes_flat[p_idx]

        pair_data_list = []
        colors = []
        for base in base_models:
            short = base_short[base]
            # Find the trained run for this base+opponent
            matching = [r for r in pw_runs if r["base"] == base and r["opponent"] == train_opponent]
            if not matching:
                continue
            run = matching[0]
            trained_short = run["run_name"].replace("_tinker_small", "")
            trained_name = f"{short}-{trained_short}"

            pre_dict, post_dict = {}, {}
            for gen in test_generators:
                pre_val = _load_ae_rate(short, gen)  # base model as judge
                post_val = _load_ae_rate(trained_name, gen)  # trained model as judge
                if pre_val is not None:
                    pre_dict[gen] = pre_val
                if post_val is not None:
                    post_dict[gen] = post_val

            if pre_dict or post_dict:
                pair_data_list.append({"pre": pre_dict, "post": post_dict, "base": base})
                colors.append(base_colors[base])

        _draw_real_arrows(ax, pair_data_list, test_generators,
                          lambda run, label: (run["pre"].get(label), run["post"].get(label)),
                          colors, show_xlabel=True, show_delta=False,
                          title=f"Trained vs {train_opponent}")

        # Highlight column matching the training opponent
        # Map opponent display name → short name for matching
        OPPONENT_TO_SHORT = {v: k for k, v in OPPONENT_DISPLAY.items()}
        train_short_candidates = [OPPONENT_TO_SHORT.get(train_opponent, "")]
        # Also try direct matching
        for gen in test_generators:
            if train_opponent.lower().replace(" ", "-") in gen.lower().replace(".", "-"):
                train_short_candidates.append(gen)
        for candidate in train_short_candidates:
            if candidate in test_generators:
                hi_x = test_generators.index(candidate)
                ax.axvspan(hi_x - 0.45, hi_x + 0.45, color="gold", alpha=0.15, zorder=0)
                ax.text(hi_x, ax.get_ylim()[1] * 0.98, "trained", ha="center",
                        fontsize=6, fontstyle="italic", color="goldenrod")
                break

    # Hide unused
    for idx in range(len(opponents_8b), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    handles = []
    for base in base_models:
        if any(r["base"] == base for r in pw_runs):
            handles.append(plt.Line2D([0], [0], marker='o', color=base_colors[base],
                                       markersize=6, linestyle='None', label=base))
    handles.append(plt.Line2D([0], [0], marker='o', color='gray', markersize=5,
                               linestyle='None', markeredgecolor='black',
                               markeredgewidth=0.4, label="Pre-training"))
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Per-Opponent Self-Selection Transfer (AlpacaEval)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = OUT_DIR / "arrow_model_panels_real.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


if __name__ == "__main__":
    main()
