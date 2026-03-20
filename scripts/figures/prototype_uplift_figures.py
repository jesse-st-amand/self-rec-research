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

AGG_DIR = Path("data/analysis/_aggregated_data")
OUT_DIR = Path("data/figures/prototypes/uplift")
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

    print(f"\nAll uplift prototypes saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
