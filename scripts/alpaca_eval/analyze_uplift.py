"""Generate uplift visualization alternatives for SGTR training transfer.

Produces 6 figure types from training benchmark data:
1. Summary heatmap (training condition × test condition, averaged across models)
2. Grouped bar chart (test conditions grouped by training condition)
3. Simplified single-panel arrow plot (averaged across models)
4. Delta matrix with model facets (one heatmap per model)
5. Two-panel dot plot (delta distribution per test condition)
6. Paired strip plot (pre/post accuracy per test condition)

Usage:
    uv run python scripts/alpaca_eval/analyze_uplift.py --config <config.yaml> --output_dir <dir>
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from dotenv import load_dotenv


def _get_bench_for_label(label, run):
    """Get benchmark keys for a test OP label."""
    tag = run.get("tag", "UT")
    if label == "PW (UT)":
        if tag != "UT":
            return ["xeval_tag_ut_pw_full", "xeval_tag_ut_pw", "xeval_task_ut_pw_full", "xeval_task_ut_pw"]
        else:
            return ["xeval_format_pw_full", "xeval_format_pw", "xeval_task_ut_pw_full", "xeval_task_ut_pw"]
    elif label == "IND (UT)":
        if tag != "UT":
            return ["xeval_tag_ut_ind_full", "xeval_tag_ut_ind", "xeval_task_ut_ind_full", "xeval_task_ut_ind"]
        else:
            return ["xeval_format_ind_full", "xeval_format_ind", "xeval_task_ut_ind_full", "xeval_task_ut_ind"]
    elif label == "PW (AT)":
        if tag != "AT":
            return ["xeval_tag_at_pw_full", "xeval_tag_at_pw", "xeval_task_at_pw_full", "xeval_task_at_pw"]
        else:
            return ["xeval_format_pw_full", "xeval_format_pw", "xeval_task_at_pw_full", "xeval_task_at_pw"]
    elif label == "IND (AT)":
        if tag != "AT":
            return ["xeval_tag_at_ind_full", "xeval_tag_at_ind", "xeval_task_at_ind_full", "xeval_task_at_ind"]
        else:
            return ["xeval_format_ind_full", "xeval_format_ind", "xeval_task_at_ind_full", "xeval_task_at_ind"]
    elif label in ("PW Pref", "PW (UT) Pref"):
        return ["xeval_task_pref_pw_full", "xeval_task_pref_pw"]
    elif label in ("IND Pref", "IND (UT) Pref"):
        return ["xeval_task_pref_ind_full", "xeval_task_pref_ind"]
    return []


DS_BENCH_MAP = {
    "WikiSum": ["xeval_dataset_wikisum_full", "xeval_dataset_wikisum"],
    "BigCodeBench": ["xeval_dataset_bigcodebench_full", "xeval_dataset_bigcodebench"],
    "PKU": ["xeval_dataset_pku_full", "xeval_dataset_pku"],
    "ShareGPT": ["xeval_dataset_sharegpt_full", "xeval_dataset_sharegpt"],
}
DS_FIELD_MAP = {"WikiSum": "wikisum", "BigCodeBench": "bigcodebench", "PKU": "pku", "ShareGPT": "sharegpt"}

# All test conditions (task + dataset)
TASK_CONDITIONS = ["PW (UT)", "IND (UT)", "PW (AT)", "IND (AT)", "PW Pref", "IND Pref"]
DATASET_CONDITIONS = ["WikiSum", "BigCodeBench", "PKU", "ShareGPT"]

# Training conditions: (label, filter_fn)
TRAINING_CONDITIONS = [
    ("UT PW ShareGPT", lambda r: r["training_type"] == "PW" and r.get("tag") == "UT" and r.get("dataset") == "sharegpt"),
    ("UT IND ShareGPT", lambda r: r["training_type"] == "IND" and r.get("tag") == "UT" and r.get("dataset") == "sharegpt"),
    ("AT PW ShareGPT", lambda r: r["training_type"] == "PW" and r.get("tag") == "AT" and r.get("dataset") == "sharegpt"),
    ("AT IND ShareGPT", lambda r: r["training_type"] == "IND" and r.get("tag") == "AT" and r.get("dataset") == "sharegpt"),
    ("UT PW WikiSum", lambda r: r["training_type"] == "PW" and r.get("tag") == "UT" and r.get("dataset") == "wikisum"),
    ("UT PW BigCode", lambda r: r["training_type"] == "PW" and r.get("tag") == "UT" and r.get("dataset") == "bigcodebench"),
    ("UT PW PKU", lambda r: r["training_type"] == "PW" and r.get("tag") == "UT" and r.get("dataset") == "pku"),
]

# Model colors: grayscale-friendly (well-separated luminance) + logo hue
MODEL_COLORS = {
    "Llama 3.1 8B": "#4a90d9",       # blue, medium          (lum ~0.52)
    "GPT-OSS 20B": "#085c47",        # dark teal             (lum ~0.25)
    "Qwen 3.0 30B": "#c9a0e8",       # light lavender        (lum ~0.71)
    # Adversarial variants (same base color)
    "GPT-OSS 20B (as Qwen 3.0 30B)": "#085c47",
    "Qwen 3.0 30B (as GPT-OSS 120B)": "#c9a0e8",
}

# Auto-assign color for unknown models based on base name
def _get_model_color(model_label):
    if model_label in MODEL_COLORS:
        return MODEL_COLORS[model_label]
    # Try to match base name
    for key, color in MODEL_COLORS.items():
        if model_label.startswith(key.split(" (")[0]):
            return color
    return "#999999"


def _get_pre_post(run, test_label, _load_val_accuracy):
    """Get (pre, post) accuracy for a run on a test condition."""
    # Task conditions
    if test_label in TASK_CONDITIONS:
        # Skip if this IS the trained condition (not transfer)
        tag_map = {"PW (UT)": ("UT", "PW"), "IND (UT)": ("UT", "IND"),
                   "PW (AT)": ("AT", "PW"), "IND (AT)": ("AT", "IND"),
                   "PW Pref": ("UT", "PW"), "IND Pref": ("UT", "IND")}
        train_tag, train_fmt = run.get("tag", "UT"), run["training_type"]
        test_tag, test_fmt = tag_map.get(test_label, ("", ""))
        is_trained_cond = (train_tag == test_tag and train_fmt == test_fmt
                           and test_label not in ("PW Pref", "IND Pref")
                           and run.get("dataset") == "sharegpt")
        if is_trained_cond:
            return None, None  # Exclude: same condition as training
        bench_keys = _get_bench_for_label(test_label, run)
        for bk in bench_keys:
            bp = run["bp_data"].get(bk)
            if bp:
                return bp["pre"], bp["post"]
        return None, None
    # Dataset conditions
    elif test_label in DATASET_CONDITIONS:
        ds_field = DS_FIELD_MAP[test_label]
        is_trained_ds = (run.get("dataset") == ds_field
                         and run.get("tag") == "UT" and run["training_type"] == "PW")
        if is_trained_ds:
            return None, None  # Exclude: same dataset as training
        bench_keys = DS_BENCH_MAP.get(test_label, [])
        for bk in bench_keys:
            bp = run["bp_data"].get(bk)
            if bp:
                return bp["pre"], bp["post"]
        return None, None
    return None, None


def _build_delta_table(runs, _load_val_accuracy):
    """Build a list of dicts: {model, train_cond, test_cond, pre, post, delta}.

    Auto-detects training conditions from the runs. Uses model_label
    (which includes adversarial identity) when available.
    """
    rows = []
    all_test = TASK_CONDITIONS + DATASET_CONDITIONS

    # First try hardcoded conditions (works for 01_final)
    matched_runs = set()
    for train_label, train_filter in TRAINING_CONDITIONS:
        filtered = [r for r in runs if train_filter(r)]
        for r in filtered:
            matched_runs.add(id(r))
            for test_label in all_test:
                pre, post = _get_pre_post(r, test_label, _load_val_accuracy)
                if pre is not None and post is not None:
                    rows.append({
                        "model": r.get("model_label", r["base"]),
                        "train_cond": train_label,
                        "test_cond": test_label,
                        "pre": pre,
                        "post": post,
                        "delta": post - pre,
                    })

    # For any runs not matched by hardcoded conditions, auto-generate label
    for r in runs:
        if id(r) in matched_runs:
            continue
        tag = r.get("tag", "UT").upper()
        fmt = r["training_type"]
        ds_map = {"sharegpt": "ShareGPT", "wikisum": "WikiSum",
                  "bigcodebench": "BigCode", "pku": "PKU"}
        ds = ds_map.get(r.get("dataset", ""), r.get("dataset", "?"))
        train_label = f"{tag} {fmt} {ds}"
        for test_label in all_test:
            pre, post = _get_pre_post(r, test_label, _load_val_accuracy)
            if pre is not None and post is not None:
                rows.append({
                    "model": r.get("model_label", r["base"]),
                    "train_cond": train_label,
                    "test_cond": test_label,
                    "pre": pre,
                    "post": post,
                    "delta": post - pre,
                })
    return rows


def fig1_summary_heatmap(rows, output_dir):
    """Training condition × test condition heatmap, averaged across models."""
    from matplotlib.colors import TwoSlopeNorm
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return

    all_test = TASK_CONDITIONS + DATASET_CONDITIONS
    train_labels = [t for t, _ in TRAINING_CONDITIONS]

    matrix = np.full((len(train_labels), len(all_test)), np.nan)
    for ri, tl in enumerate(train_labels):
        for ci, tc in enumerate(all_test):
            matches = df[(df["train_cond"] == tl) & (df["test_cond"] == tc)]
            if not matches.empty:
                matrix[ri, ci] = matches["delta"].mean()

    vals = matrix[~np.isnan(matrix)]
    if len(vals) == 0:
        return
    vmax = max(abs(vals.min()), abs(vals.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(matrix, aspect="auto", cmap=plt.cm.RdYlGn, norm=norm)

    ax.set_xticks(range(len(all_test)))
    ax.set_xticklabels(all_test, fontsize=10, rotation=35, ha="right")
    ax.set_yticks(range(len(train_labels)))
    ax.set_yticklabels(train_labels, fontsize=11)

    for ri in range(matrix.shape[0]):
        for ci in range(matrix.shape[1]):
            v = matrix[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                        fontweight="bold", color="white" if abs(v) > vmax * 0.5 else "black")

    # Separator between task and dataset columns
    ax.axvline(x=len(TASK_CONDITIONS) - 0.5, color="black", linewidth=1.5)
    fig.colorbar(im, ax=ax, shrink=0.7).set_label("Δ Accuracy (post − pre)", fontsize=11)
    ax.set_title("Training Transfer: Avg Accuracy Change", fontsize=14, fontweight="bold")

    path = output_dir / "uplift_1_summary_heatmap.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def fig2_grouped_bar(rows, output_dir):
    """Grouped bar chart: test conditions grouped by training condition."""
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return

    all_test = TASK_CONDITIONS + DATASET_CONDITIONS
    train_labels = [t for t, _ in TRAINING_CONDITIONS]
    n_train = len(train_labels)
    n_test = len(all_test)

    fig, ax = plt.subplots(figsize=(16, 6))
    bar_width = 0.8 / n_train
    colors = plt.cm.tab10(np.linspace(0, 1, n_train))

    for ti, tl in enumerate(train_labels):
        means = []
        stds = []
        for tc in all_test:
            matches = df[(df["train_cond"] == tl) & (df["test_cond"] == tc)]
            means.append(matches["delta"].mean() if not matches.empty else 0)
            stds.append(matches["delta"].std() if len(matches) > 1 else 0)
        x = np.arange(n_test) + ti * bar_width
        ax.bar(x, means, bar_width, yerr=stds, label=tl, color=colors[ti],
               edgecolor="black", linewidth=0.3, capsize=2, alpha=0.8)

    ax.set_xticks(np.arange(n_test) + bar_width * (n_train - 1) / 2)
    ax.set_xticklabels(all_test, fontsize=9, rotation=35, ha="right")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axvline(x=len(TASK_CONDITIONS) - 0.5 + bar_width * (n_train - 1) / 2,
               color="gray", linestyle=":", linewidth=1.5)
    ax.set_ylabel("Δ Accuracy", fontsize=12)
    ax.set_title("Training Transfer: Grouped by Training Condition", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=4, loc="upper right")

    path = output_dir / "uplift_2_grouped_bar.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def fig3_simplified_arrow(rows, output_dir):
    """Single-panel horizontal arrow plot, averaged across models per training condition."""
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return

    all_test = TASK_CONDITIONS + DATASET_CONDITIONS
    train_labels = [t for t, _ in TRAINING_CONDITIONS]
    colors = plt.cm.tab10(np.linspace(0, 1, len(train_labels)))
    n_train = len(train_labels)

    fig, ax = plt.subplots(figsize=(12, 8))
    group_width = 0.7
    offsets = np.linspace(-group_width / 2, group_width / 2, max(n_train, 2))

    for ti, tl in enumerate(train_labels):
        for yi, tc in enumerate(all_test):
            matches = df[(df["train_cond"] == tl) & (df["test_cond"] == tc)]
            if matches.empty:
                continue
            pre_avg = matches["pre"].mean()
            post_avg = matches["post"].mean()
            y = yi + offsets[ti]
            arrow_color = "#2E7D32" if post_avg > pre_avg else "#C62828"
            ax.scatter(pre_avg, y, color="gray", s=25, zorder=3, edgecolors="black", linewidth=0.3)
            ax.annotate("", xy=(post_avg, y), xytext=(pre_avg, y),
                        arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5), zorder=4)
            ax.scatter(post_avg, y, color=colors[ti], s=35, zorder=5, edgecolors="black", linewidth=0.3)

    ax.set_yticks(range(len(all_test)))
    ax.set_yticklabels(all_test, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Training Transfer (averaged across models)", fontsize=14, fontweight="bold")

    # Separator
    ax.axhspan(len(TASK_CONDITIONS) - 0.5, len(TASK_CONDITIONS) - 0.5,
               color="black", linewidth=1)
    ax.axhline(y=len(TASK_CONDITIONS) - 0.5, color="black", linewidth=1)

    handles = [plt.Line2D([0], [0], marker="o", color=colors[i], markersize=6,
                           linestyle="None", label=tl) for i, tl in enumerate(train_labels)]
    handles.append(plt.Line2D([0], [0], marker="o", color="gray", markersize=5,
                               linestyle="None", markeredgecolor="black", label="Pre-training"))
    ax.legend(handles=handles, fontsize=8, loc="lower right", ncol=2)

    path = output_dir / "uplift_3_simplified_arrow.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def fig4_model_facet_heatmaps(rows, output_dir):
    """Three heatmaps side-by-side, one per model."""
    from matplotlib.colors import TwoSlopeNorm
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return

    models = sorted(df["model"].unique())
    all_test = TASK_CONDITIONS + DATASET_CONDITIONS
    train_labels = [t for t, _ in TRAINING_CONDITIONS]

    # Shared color scale
    all_deltas = df["delta"].dropna()
    if all_deltas.empty:
        return
    vmax = max(abs(all_deltas.min()), abs(all_deltas.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 7),
                              gridspec_kw={"wspace": 0.3})
    if len(models) == 1:
        axes = [axes]

    for mi, model in enumerate(models):
        ax = axes[mi]
        matrix = np.full((len(train_labels), len(all_test)), np.nan)
        for ri, tl in enumerate(train_labels):
            for ci, tc in enumerate(all_test):
                matches = df[(df["model"] == model) & (df["train_cond"] == tl) & (df["test_cond"] == tc)]
                if not matches.empty:
                    matrix[ri, ci] = matches["delta"].mean()

        im = ax.imshow(matrix, aspect="auto", cmap=plt.cm.RdYlGn, norm=norm)
        ax.set_xticks(range(len(all_test)))
        ax.set_xticklabels(all_test, fontsize=8, rotation=35, ha="right")
        if mi == 0:
            ax.set_yticks(range(len(train_labels)))
            ax.set_yticklabels(train_labels, fontsize=9)
        else:
            ax.set_yticks([])
        ax.set_title(model, fontsize=13, fontweight="bold",
                     color=_get_model_color(model))
        ax.axvline(x=len(TASK_CONDITIONS) - 0.5, color="black", linewidth=1)

        for ri in range(matrix.shape[0]):
            for ci in range(matrix.shape[1]):
                v = matrix[ri, ci]
                if not np.isnan(v):
                    ax.text(ci, ri, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > vmax * 0.5 else "black")

    fig.colorbar(im, ax=axes, shrink=0.6).set_label("Δ Accuracy", fontsize=11)
    fig.suptitle("Training Transfer by Model", fontsize=15, fontweight="bold")

    path = output_dir / "uplift_4_model_facets.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def fig5_dot_plot(rows, output_dir):
    """Two-panel dot plot: delta distribution per test condition.

    Shape = model. Color = training operationalization.
    Task panel: color = task OP trained on (ignoring dataset).
    Dataset panel: color = dataset trained on (ignoring task).
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return

    _BASE_MARKERS = {
        "Llama 3.1 8B": "o",
        "GPT-OSS 20B": "s",
        "Qwen 3.0 30B": "D",
    }
    # Adversarial variants get star markers with the same base shape logic
    _ADV_MARKERS = {
        "Llama 3.1 8B": "*",
        "GPT-OSS 20B": "X",
        "Qwen 3.0 30B": "P",  # plus (filled)
    }
    _ALL_MARKERS = ["o", "s", "D", "^", "v", "*", "p", "X"]
    # Build MODEL_MARKERS dynamically for all unique models in data
    unique_models = sorted(df["model"].unique())
    MODEL_MARKERS = {}
    mi = 0
    for m in unique_models:
        is_adv = "(as " in m and "(as self)" not in m
        base = m.split(" (")[0] if " (" in m else m
        if is_adv and base in _ADV_MARKERS:
            MODEL_MARKERS[m] = _ADV_MARKERS[base]
        elif base in _BASE_MARKERS:
            MODEL_MARKERS[m] = _BASE_MARKERS[base]
        else:
            MODEL_MARKERS[m] = _ALL_MARKERS[mi % len(_ALL_MARKERS)]
            mi += 1

    # Extract the task OP from train_cond (e.g., "UT PW ShareGPT" -> "UT PW")
    def _task_op(train_cond):
        parts = train_cond.split()
        return " ".join(parts[:2])  # "UT PW", "UT IND", "AT PW", "AT IND"

    # Extract the dataset from train_cond (e.g., "UT PW ShareGPT" -> "ShareGPT")
    def _dataset(train_cond):
        return train_cond.split()[-1]  # "ShareGPT", "WikiSum", "BigCode", "PKU"

    # 8 distinct colors chosen for grayscale distinguishability
    # (luminance values evenly spread when desaturated)
    TRAIN_COLORS = {
        # Task OPs (used in task panel)
        "UT PW": "#1a1a1a",     # near-black        (lum ~0.10)
        "UT IND": "#c44e52",    # muted red          (lum ~0.45)
        "AT PW": "#3070b0",     # darker blue        (lum ~0.38)
        "AT IND": "#d4a843",    # golden yellow      (lum ~0.67)
        # Datasets (used in dataset panel)
        "ShareGPT": "#8050b0",  # medium purple      (lum ~0.38)
        "WikiSum": "#e0a060",   # light orange       (lum ~0.65)
        "BigCode": "#2a6e2a",   # dark green         (lum ~0.30)
        "PKU": "#c0c0c0",       # silver/light gray  (lum ~0.75)
    }

    rng = np.random.default_rng(42)

    fig, (ax_task, ax_ds) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"wspace": 0.35})

    # ── Task transfer panel ──
    # Only include rows where training was on ShareGPT (task variation)
    task_df = df[df["train_cond"].str.endswith("ShareGPT") & df["test_cond"].isin(TASK_CONDITIONS)]

    # Box plots (behind scatter)
    task_box_data = [task_df[task_df["test_cond"] == tc]["delta"].dropna().values
                     for tc in TASK_CONDITIONS]
    bp = ax_task.boxplot(task_box_data, positions=range(len(TASK_CONDITIONS)), vert=False,
                         widths=0.5, patch_artist=True, zorder=1,
                         boxprops=dict(facecolor="#e0e0e0", alpha=0.5, edgecolor="#999999"),
                         whiskerprops=dict(color="#999999"), capprops=dict(color="#999999"),
                         medianprops=dict(color="#333333", linewidth=1.5),
                         flierprops=dict(marker="", markersize=0))  # hide outlier markers

    for yi, tc in enumerate(TASK_CONDITIONS):
        subset = task_df[task_df["test_cond"] == tc]
        for _, row in subset.iterrows():
            marker = MODEL_MARKERS.get(row["model"], "o")
            task_op = _task_op(row["train_cond"])
            color = TRAIN_COLORS.get(task_op, "#999999")
            jitter = rng.uniform(-0.2, 0.2)
            ax_task.scatter(row["delta"], yi + jitter, color=color, marker=marker,
                            s=50, alpha=0.8, edgecolors="black", linewidth=0.4, zorder=3)

    ax_task.axvline(x=0, color="black", linewidth=1, linestyle="-")
    ax_task.set_yticks(range(len(TASK_CONDITIONS)))
    ax_task.set_yticklabels(TASK_CONDITIONS, fontsize=12)
    ax_task.invert_yaxis()
    ax_task.set_xlabel("Δ Accuracy (post − pre)", fontsize=12)
    ax_task.set_title("Task Transfer", fontsize=14, fontweight="bold")
    ax_task.grid(axis="x", alpha=0.3, linestyle="--")
    for yi in range(1, len(TASK_CONDITIONS)):
        ax_task.axhline(y=yi - 0.5, color="#cccccc", linestyle=":", linewidth=0.8)

    # ── Dataset transfer panel ──
    # Only include rows where training was UT PW (dataset variation)
    ds_df = df[df["train_cond"].str.startswith("UT PW") & df["test_cond"].isin(DATASET_CONDITIONS)]

    # Box plots (behind scatter)
    ds_box_data = [ds_df[ds_df["test_cond"] == tc]["delta"].dropna().values
                   for tc in DATASET_CONDITIONS]
    bp2 = ax_ds.boxplot(ds_box_data, positions=range(len(DATASET_CONDITIONS)), vert=False,
                        widths=0.5, patch_artist=True, zorder=1,
                        boxprops=dict(facecolor="#e0e0e0", alpha=0.5, edgecolor="#999999"),
                        whiskerprops=dict(color="#999999"), capprops=dict(color="#999999"),
                        medianprops=dict(color="#333333", linewidth=1.5),
                        flierprops=dict(marker="", markersize=0))

    for yi, tc in enumerate(DATASET_CONDITIONS):
        subset = ds_df[ds_df["test_cond"] == tc]
        for _, row in subset.iterrows():
            marker = MODEL_MARKERS.get(row["model"], "o")
            ds = _dataset(row["train_cond"])
            color = TRAIN_COLORS.get(ds, "#999999")
            jitter = rng.uniform(-0.2, 0.2)
            ax_ds.scatter(row["delta"], yi + jitter, color=color, marker=marker,
                          s=50, alpha=0.8, edgecolors="black", linewidth=0.4, zorder=3)

    ax_ds.axvline(x=0, color="black", linewidth=1, linestyle="-")
    ax_ds.set_yticks(range(len(DATASET_CONDITIONS)))
    ax_ds.set_yticklabels(DATASET_CONDITIONS, fontsize=12)
    ax_ds.invert_yaxis()
    ax_ds.set_xlabel("Δ Accuracy (post − pre)", fontsize=12)
    ax_ds.set_title("Dataset Transfer", fontsize=14, fontweight="bold")
    ax_ds.grid(axis="x", alpha=0.3, linestyle="--")
    for yi in range(1, len(DATASET_CONDITIONS)):
        ax_ds.axhline(y=yi - 0.5, color="#cccccc", linestyle=":", linewidth=0.8)

    # ── Single shared legend ──
    # Model shapes
    model_handles = [plt.Line2D([0], [0], marker=mk, color="gray", markersize=8,
                                 linestyle="None", markeredgecolor="black", markeredgewidth=0.4,
                                 label=m) for m, mk in MODEL_MARKERS.items()]
    # All 8 training condition colors
    train_handles = [plt.Line2D([0], [0], marker="o", color=c, markersize=8,
                                 linestyle="None", markeredgecolor="black", markeredgewidth=0.4,
                                 label=f"Trained: {t}") for t, c in TRAIN_COLORS.items()]

    all_handles = model_handles + train_handles
    fig.legend(handles=all_handles, fontsize=9, loc="lower center",
               ncol=6, bbox_to_anchor=(0.5, -0.06), framealpha=0.9)

    path = output_dir / "uplift_5_dot_plot.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def fig5b_dot_plot_alt(rows, output_dir):
    """Two-panel dot plot (alternate encoding).

    Color = model (logo colors). Shape = training condition.
    Task panel: shape = task OP trained on (ignoring dataset).
    Dataset panel: shape = dataset trained on (ignoring task).
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return

    # Shape = training condition (8 distinct markers)
    TASK_OP_MARKERS = {
        "UT PW": "o",     # circle
        "UT IND": "s",    # square
        "AT PW": "D",     # diamond
        "AT IND": "^",    # triangle up
    }
    DATASET_MARKERS_B = {
        "ShareGPT": "o",      # circle
        "WikiSum": "v",       # triangle down
        "BigCode": "*",       # star
        "PKU": "p",           # pentagon
    }

    # Build unique model list with adversarial distinction
    unique_models_b = sorted(df["model"].unique())
    # Adversarial models get open/unfilled markers (edgecolor only, no fill)
    _is_adv_model = {m: "(as " in m and "(as self)" not in m for m in unique_models_b}

    def _task_op(train_cond):
        return " ".join(train_cond.split()[:2])

    def _dataset(train_cond):
        return train_cond.split()[-1]

    rng = np.random.default_rng(42)

    fig, (ax_task, ax_ds) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"wspace": 0.35})

    # ── Task transfer panel ──
    task_df = df[df["train_cond"].str.endswith("ShareGPT") & df["test_cond"].isin(TASK_CONDITIONS)]

    task_box_data = [task_df[task_df["test_cond"] == tc]["delta"].dropna().values
                     for tc in TASK_CONDITIONS]
    ax_task.boxplot(task_box_data, positions=range(len(TASK_CONDITIONS)), vert=False,
                    widths=0.5, patch_artist=True, zorder=1,
                    boxprops=dict(facecolor="#e0e0e0", alpha=0.5, edgecolor="#999999"),
                    whiskerprops=dict(color="#999999"), capprops=dict(color="#999999"),
                    medianprops=dict(color="#333333", linewidth=1.5),
                    flierprops=dict(marker="", markersize=0))

    for yi, tc in enumerate(TASK_CONDITIONS):
        subset = task_df[task_df["test_cond"] == tc]
        for _, row in subset.iterrows():
            task_op = _task_op(row["train_cond"])
            marker = TASK_OP_MARKERS.get(task_op, "o")
            color = _get_model_color(row["model"])
            is_adv = _is_adv_model.get(row["model"], False)
            jitter = rng.uniform(-0.2, 0.2)
            if is_adv:
                ax_task.scatter(row["delta"], yi + jitter, facecolors="none",
                                edgecolors=color, marker=marker,
                                s=70, linewidth=2.0, alpha=0.9, zorder=3)
            else:
                ax_task.scatter(row["delta"], yi + jitter, color=color, marker=marker,
                                s=60, alpha=0.8, edgecolors="black", linewidth=0.4, zorder=3)

    ax_task.axvline(x=0, color="black", linewidth=1, linestyle="-")
    ax_task.set_yticks(range(len(TASK_CONDITIONS)))
    ax_task.set_yticklabels(TASK_CONDITIONS, fontsize=12)
    ax_task.invert_yaxis()
    ax_task.set_xlabel("Δ Accuracy (post − pre)", fontsize=12)
    ax_task.set_title("Task Transfer", fontsize=14, fontweight="bold")
    ax_task.grid(axis="x", alpha=0.3, linestyle="--")
    for yi in range(1, len(TASK_CONDITIONS)):
        ax_task.axhline(y=yi - 0.5, color="#cccccc", linestyle=":", linewidth=0.8)

    # ── Dataset transfer panel ──
    ds_df = df[df["train_cond"].str.startswith("UT PW") & df["test_cond"].isin(DATASET_CONDITIONS)]

    ds_box_data = [ds_df[ds_df["test_cond"] == tc]["delta"].dropna().values
                   for tc in DATASET_CONDITIONS]
    ax_ds.boxplot(ds_box_data, positions=range(len(DATASET_CONDITIONS)), vert=False,
                  widths=0.5, patch_artist=True, zorder=1,
                  boxprops=dict(facecolor="#e0e0e0", alpha=0.5, edgecolor="#999999"),
                  whiskerprops=dict(color="#999999"), capprops=dict(color="#999999"),
                  medianprops=dict(color="#333333", linewidth=1.5),
                  flierprops=dict(marker="", markersize=0))

    for yi, tc in enumerate(DATASET_CONDITIONS):
        subset = ds_df[ds_df["test_cond"] == tc]
        for _, row in subset.iterrows():
            ds = _dataset(row["train_cond"])
            marker = DATASET_MARKERS_B.get(ds, "o")
            color = _get_model_color(row["model"])
            is_adv = _is_adv_model.get(row["model"], False)
            jitter = rng.uniform(-0.2, 0.2)
            if is_adv:
                ax_ds.scatter(row["delta"], yi + jitter, facecolors="none",
                              edgecolors=color, marker=marker,
                              s=70, linewidth=2.0, alpha=0.9, zorder=3)
            else:
                ax_ds.scatter(row["delta"], yi + jitter, color=color, marker=marker,
                              s=60, alpha=0.8, edgecolors="black", linewidth=0.4, zorder=3)

    ax_ds.axvline(x=0, color="black", linewidth=1, linestyle="-")
    ax_ds.set_yticks(range(len(DATASET_CONDITIONS)))
    ax_ds.set_yticklabels(DATASET_CONDITIONS, fontsize=12)
    ax_ds.invert_yaxis()
    ax_ds.set_xlabel("Δ Accuracy (post − pre)", fontsize=12)
    ax_ds.set_title("Dataset Transfer", fontsize=14, fontweight="bold")
    ax_ds.grid(axis="x", alpha=0.3, linestyle="--")
    for yi in range(1, len(DATASET_CONDITIONS)):
        ax_ds.axhline(y=yi - 0.5, color="#cccccc", linestyle=":", linewidth=0.8)

    # ── Single shared legend ──
    # Model colors
    # Model colors — standard (filled) and adversarial (open)
    model_handles = []
    for m in unique_models_b:
        color = _get_model_color(m)
        is_adv = _is_adv_model.get(m, False)
        if is_adv:
            model_handles.append(plt.Line2D([0], [0], marker="o", markerfacecolor="none",
                                             markeredgecolor=color, markersize=8,
                                             linestyle="None", markeredgewidth=2.0, label=m))
        else:
            model_handles.append(plt.Line2D([0], [0], marker="o", color=color, markersize=8,
                                             linestyle="None", markeredgecolor="black",
                                             markeredgewidth=0.4, label=m))
    # Task OP shapes
    task_shape_handles = [plt.Line2D([0], [0], marker=mk, color="gray", markersize=8,
                                      linestyle="None", markeredgecolor="black", markeredgewidth=0.4,
                                      label=f"Trained: {t}") for t, mk in TASK_OP_MARKERS.items()]
    # Dataset shapes
    ds_shape_handles = [plt.Line2D([0], [0], marker=mk, color="gray", markersize=8,
                                    linestyle="None", markeredgecolor="black", markeredgewidth=0.4,
                                    label=f"Trained: {d}") for d, mk in DATASET_MARKERS_B.items()]

    all_handles = model_handles + task_shape_handles + ds_shape_handles
    fig.legend(handles=all_handles, fontsize=15, loc="lower center",
               ncol=min(len(all_handles), 6), bbox_to_anchor=(0.5, -0.06), framealpha=0.9)

    path = output_dir / "uplift_5b_dot_plot_alt.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def fig5c_dot_plot_dual_color(rows, output_dir):
    """Two-panel dot plot (dual-color encoding for adversarial).

    Color = model (logo colors). Shape = training condition.
    Adversarial models: edgecolor = base model, facecolor = identity model.
    e.g., Qwen trained as GPT-OSS → purple edge, teal fill.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return

    # Base model color lookup (by display name prefix)
    _BASE_COLORS = {
        "Llama 3.1 8B": "#4a90d9",
        "GPT-OSS 20B": "#085c47",
        "GPT-OSS 120B": "#085c47",
        "Qwen 3.0 30B": "#c9a0e8",
    }

    def _model_color(label):
        """Get color for a model label, matching base name."""
        if label in _BASE_COLORS:
            return _BASE_COLORS[label]
        base = label.split(" (")[0] if " (" in label else label
        return _BASE_COLORS.get(base, "#999999")

    # Parse identity model from adversarial label: "GPT-OSS 20B (as Qwen 3.0 30B)" → "Qwen 3.0 30B"
    def _identity_from_label(label):
        if "(as " in label:
            return label.split("(as ")[1].rstrip(")")
        return None

    # Shape = training condition
    TASK_OP_MARKERS = {
        "UT PW": "o", "UT IND": "s", "AT PW": "D", "AT IND": "^",
    }
    DATASET_MARKERS_C = {
        "ShareGPT": "o", "WikiSum": "v", "BigCode": "*", "PKU": "p",
    }

    def _task_op(train_cond):
        return " ".join(train_cond.split()[:2])

    def _dataset(train_cond):
        return train_cond.split()[-1]

    unique_models_c = sorted(df["model"].unique())
    _is_adv = {m: "(as " in m and "(as self)" not in m for m in unique_models_c}

    rng = np.random.default_rng(42)

    # Shorthand y-axis labels
    TASK_SHORT = {
        "PW (UT)": "UT PW", "IND (UT)": "UT IND",
        "PW (AT)": "AT PW", "IND (AT)": "AT IND",
        "PW Pref": "PW Pref", "IND Pref": "IND Pref",
    }
    DS_SHORT = {"WikiSum": "WS", "BigCodeBench": "BCB", "PKU": "PKU", "ShareGPT": "S-GPT"}

    fig, (ax_task, ax_ds) = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={"wspace": 0.20})

    # ── Task transfer panel ──
    task_df = df[df["train_cond"].str.endswith("ShareGPT") & df["test_cond"].isin(TASK_CONDITIONS)]

    task_box_data = [task_df[task_df["test_cond"] == tc]["delta"].dropna().values
                     for tc in TASK_CONDITIONS]
    ax_task.boxplot(task_box_data, positions=range(len(TASK_CONDITIONS)), vert=False,
                    widths=0.5, patch_artist=True, zorder=1,
                    boxprops=dict(facecolor="#e0e0e0", alpha=0.5, edgecolor="#999999"),
                    whiskerprops=dict(color="#999999"), capprops=dict(color="#999999"),
                    medianprops=dict(color="#333333", linewidth=1.5),
                    flierprops=dict(marker="", markersize=0))

    for yi, tc in enumerate(TASK_CONDITIONS):
        subset = task_df[task_df["test_cond"] == tc]
        for _, row in subset.iterrows():
            task_op = _task_op(row["train_cond"])
            marker = TASK_OP_MARKERS.get(task_op, "o")
            base_color = _model_color(row["model"])
            is_adv = _is_adv.get(row["model"], False)
            jitter = rng.uniform(-0.2, 0.2)
            if is_adv:
                identity = _identity_from_label(row["model"])
                fill_color = _model_color(identity) if identity else base_color
                ax_task.scatter(row["delta"], yi + jitter, facecolors=fill_color,
                                edgecolors=base_color, marker=marker,
                                s=70, linewidth=2.5, alpha=0.9, zorder=3)
            else:
                ax_task.scatter(row["delta"], yi + jitter, color=base_color, marker=marker,
                                s=60, alpha=0.8, edgecolors="black", linewidth=0.4, zorder=3)

    ax_task.axvline(x=0, color="black", linewidth=1, linestyle="-")
    ax_task.set_yticks(range(len(TASK_CONDITIONS)))
    ax_task.set_yticklabels([TASK_SHORT.get(tc, tc) for tc in TASK_CONDITIONS],
                            fontsize=14, rotation=25, ha="right")
    ax_task.invert_yaxis()
    ax_task.set_xlabel("Δ Accuracy (post − pre)", fontsize=14)
    ax_task.set_title("Evaluation Format Transfer", fontsize=16, fontweight="bold")
    ax_task.grid(axis="x", alpha=0.3, linestyle="--")
    ax_task.tick_params(axis="x", labelsize=12)
    for yi in range(1, len(TASK_CONDITIONS)):
        ax_task.axhline(y=yi - 0.5, color="#cccccc", linestyle=":", linewidth=0.8)

    # ── Dataset transfer panel ──
    ds_df = df[df["train_cond"].str.startswith("UT PW") & df["test_cond"].isin(DATASET_CONDITIONS)]

    ds_box_data = [ds_df[ds_df["test_cond"] == tc]["delta"].dropna().values
                   for tc in DATASET_CONDITIONS]
    ax_ds.boxplot(ds_box_data, positions=range(len(DATASET_CONDITIONS)), vert=False,
                  widths=0.5, patch_artist=True, zorder=1,
                  boxprops=dict(facecolor="#e0e0e0", alpha=0.5, edgecolor="#999999"),
                  whiskerprops=dict(color="#999999"), capprops=dict(color="#999999"),
                  medianprops=dict(color="#333333", linewidth=1.5),
                  flierprops=dict(marker="", markersize=0))

    for yi, tc in enumerate(DATASET_CONDITIONS):
        subset = ds_df[ds_df["test_cond"] == tc]
        for _, row in subset.iterrows():
            ds = _dataset(row["train_cond"])
            marker = DATASET_MARKERS_C.get(ds, "o")
            star_bump = 25 if marker == "*" else 0
            base_color = _model_color(row["model"])
            is_adv = _is_adv.get(row["model"], False)
            jitter = rng.uniform(-0.2, 0.2)
            if is_adv:
                identity = _identity_from_label(row["model"])
                fill_color = _model_color(identity) if identity else base_color
                ax_ds.scatter(row["delta"], yi + jitter, facecolors=fill_color,
                              edgecolors=base_color, marker=marker,
                              s=70 + star_bump, linewidth=2.5, alpha=0.9, zorder=3)
            else:
                ax_ds.scatter(row["delta"], yi + jitter, color=base_color, marker=marker,
                              s=60 + star_bump, alpha=0.8, edgecolors="black", linewidth=0.4, zorder=3)

    ax_ds.axvline(x=0, color="black", linewidth=1, linestyle="-")
    ax_ds.set_yticks(range(len(DATASET_CONDITIONS)))
    ax_ds.set_yticklabels([DS_SHORT.get(tc, tc) for tc in DATASET_CONDITIONS],
                          fontsize=14, rotation=25, ha="right")
    ax_ds.invert_yaxis()
    ax_ds.set_xlabel("Δ Accuracy (post − pre)", fontsize=14)
    ax_ds.set_title("Task Domain Transfer", fontsize=16, fontweight="bold")
    ax_ds.grid(axis="x", alpha=0.3, linestyle="--")
    ax_ds.tick_params(axis="x", labelsize=12)
    for yi in range(1, len(DATASET_CONDITIONS)):
        ax_ds.axhline(y=yi - 0.5, color="#cccccc", linestyle=":", linewidth=0.8)

    # ── Legend (right side, vertical, ordered) ──
    MODEL_ORDER = [
        "Llama 3.1 8B (vs Qwen 3.0 30B)",
        "GPT-OSS 20B (vs Qwen 3.0 30B)",
        "GPT-OSS 20B (as Qwen 3.0 30B)",
        "Qwen 3.0 30B (vs GPT-OSS 120B)",
        "Qwen 3.0 30B (as GPT-OSS 120B)",
    ]
    model_handles = []
    for m in MODEL_ORDER:
        if m not in set(unique_models_c):
            continue
        # Wrap parenthesized text to next line
        display_label = m.replace(" (", "\n(")
        color = _model_color(m)
        is_adv = _is_adv.get(m, False)
        if is_adv:
            identity = _identity_from_label(m)
            fill = _model_color(identity) if identity else color
            model_handles.append(plt.Line2D([0], [0], marker="o", markerfacecolor=fill,
                                             markeredgecolor=color, markersize=9,
                                             linestyle="None", markeredgewidth=2.5, label=display_label))
        else:
            model_handles.append(plt.Line2D([0], [0], marker="o", color=color, markersize=8,
                                             linestyle="None", markeredgecolor="black",
                                             markeredgewidth=0.4, label=display_label))

    task_shape_handles = [plt.Line2D([0], [0], marker=mk, color="gray", markersize=8,
                                      linestyle="None", markeredgecolor="black", markeredgewidth=0.4,
                                      label=f"Trained: {t}") for t, mk in TASK_OP_MARKERS.items()]
    ds_shape_handles = [plt.Line2D([0], [0], marker=mk, color="gray", markersize=10 if mk == "*" else 8,
                                    linestyle="None", markeredgecolor="black", markeredgewidth=0.4,
                                    label=f"Trained: {d}") for d, mk in DATASET_MARKERS_C.items()]

    all_handles = model_handles + task_shape_handles + ds_shape_handles
    fig.legend(handles=all_handles, fontsize=10, loc="center right",
               ncol=1, bbox_to_anchor=(1.08, 0.5), framealpha=0.9)

    path = output_dir / "uplift_5c_dot_plot_dual_color.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def fig6_paired_strip(rows, output_dir):
    """Paired strip plot: pre/post accuracy connected by lines."""
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return

    fig, (ax_task, ax_ds) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"wspace": 0.3})

    for ax, conditions, title in [
        (ax_task, TASK_CONDITIONS, "Task Transfer"),
        (ax_ds, DATASET_CONDITIONS, "Dataset Transfer"),
    ]:
        for yi, tc in enumerate(conditions):
            subset = df[df["test_cond"] == tc]
            for _, row in subset.iterrows():
                color = _get_model_color(row["model"])
                jitter = np.random.uniform(-0.15, 0.15)
                y = yi + jitter
                ax.scatter(row["pre"], y, color="gray", s=25, zorder=3,
                           edgecolors="black", linewidth=0.3, marker="o")
                ax.scatter(row["post"], y, color=color, s=35, zorder=5,
                           edgecolors="black", linewidth=0.3, marker="o")
                ax.plot([row["pre"], row["post"]], [y, y], color=color,
                        linewidth=0.8, alpha=0.5, zorder=2)

        ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel("Accuracy", fontsize=12)
        ax.set_xlim(-0.02, 1.05)
        ax.set_title(title, fontsize=14, fontweight="bold")

    handles = [plt.Line2D([0], [0], marker="o", color="gray", markersize=6, linestyle="None",
                           markeredgecolor="black", markeredgewidth=0.3, label="Pre-training")]
    for m, c in MODEL_COLORS.items():
        handles.append(plt.Line2D([0], [0], marker="o", color=c, markersize=7, linestyle="None",
                                   markeredgecolor="black", markeredgewidth=0.3, label=f"{m} (post)"))
    fig.legend(handles=handles, fontsize=10, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.04))

    path = output_dir / "uplift_6_paired_strip.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate uplift visualization alternatives")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--output_dir", default="data/alpaca_eval/analysis", help="Root output directory")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    training_dir = config.get("training_dir", "data/training")
    data_subsets = config.get("data_subsets", None)

    from scripts.figures.COLM2026.prototype_uplift_figures import (
        _discover_training_runs, _load_val_accuracy,
    )

    runs = _discover_training_runs(training_dir=training_dir, subsets=data_subsets)
    if not runs:
        print("No training runs found")
        return

    # Label runs with identity info for adversarial distinction
    IDENTITY_DISPLAY = {
        "gpt-oss-120b": "GPT-OSS 120B", "gpt-oss-20b": "GPT-OSS 20B",
        "qwen-3.0-30b": "Qwen 3.0 30B", "qwen-3-30b": "Qwen 3.0 30B",
        "llama-3-1-8b": "Llama 3.1 8B", "ll-3.1-8b": "Llama 3.1 8B",
    }
    OPPONENT_DISPLAY = {
        "gpt-oss-120b": "GPT-OSS 120B", "gpt-oss-20b": "GPT-OSS 20B",
        "gpt-oss-120b-thinking": "GPT-OSS 120B", "gpt-oss-20b-thinking": "GPT-OSS 20B",
        "qwen-3.0-30b": "Qwen 3.0 30B", "qwen-3-30b": "Qwen 3.0 30B",
        "qwen-2.5-7b": "Qwen 2.5 7B", "qwen-2-5-7b": "Qwen 2.5 7B",
        "llama-3-1-8b": "Llama 3.1 8B", "ll-3.1-8b": "Llama 3.1 8B",
    }
    for r in runs:
        opponent = r.get("opponent", "")
        opponent_name = OPPONENT_DISPLAY.get(opponent, opponent)
        if r.get("is_adversarial"):
            identity = r.get("identity_model", "?")
            identity_name = IDENTITY_DISPLAY.get(identity, identity)
            r["model_label"] = f"{r['base']} (as {identity_name})"
        else:
            r["model_label"] = f"{r['base']} (vs {opponent_name})" if opponent_name else r["base"]
    print(f"Found {len(runs)} training runs ({sum(1 for r in runs if r.get('is_adversarial'))} adversarial)")

    subset_label = data_subsets[0] if data_subsets else "all"
    output_dir = Path(args.output_dir) / subset_label / "uplift"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building delta table...")
    rows = _build_delta_table(runs, _load_val_accuracy)
    print(f"  {len(rows)} data points")

    print("\n1. Summary heatmap")
    fig1_summary_heatmap(rows, output_dir)

    print("2. Grouped bar chart")
    fig2_grouped_bar(rows, output_dir)

    print("3. Simplified arrow plot")
    fig3_simplified_arrow(rows, output_dir)

    print("4. Model facet heatmaps")
    fig4_model_facet_heatmaps(rows, output_dir)

    print("5. Two-panel dot plot (color=training, shape=model)")
    fig5_dot_plot(rows, output_dir)

    print("5b. Two-panel dot plot alt (color=model, shape=training)")
    fig5b_dot_plot_alt(rows, output_dir)

    print("5c. Two-panel dot plot dual-color (adversarial = edge/fill)")
    fig5c_dot_plot_dual_color(rows, output_dir)

    print("6. Paired strip plot")
    fig6_paired_strip(rows, output_dir)

    print(f"\n✓ All uplift figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
