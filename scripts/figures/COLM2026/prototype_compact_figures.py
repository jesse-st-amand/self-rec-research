"""
Prototype compact figures for COLM 2026 paper.

Loads ICML_01-08 aggregated performance data and generates
multiple visualization candidates to find the most space-efficient
way to present recognition performance across operationalizations.

Usage:
    uv run python scripts/figures/prototype_compact_figures.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import seaborn as sns

AGG_DIR = Path("data/analysis/_aggregated_data")
OUT_DIR = Path("data/figures/prototypes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment metadata: short label, paradigm, task, model_type
EXPERIMENTS = {
    "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst": {
        "label": "PW-Rec\nInstruct",
        "short": "PW-Rec (Inst)",
        "paradigm": "PW", "task": "Rec", "model_type": "Instruct",
    },
    "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst": {
        "label": "IND-Rec\nInstruct",
        "short": "IND-Rec (Inst)",
        "paradigm": "IND", "task": "Rec", "model_type": "Instruct",
    },
    "ICML_03_UT_PW-Q_Rec_NPr_FA_Rsn": {
        "label": "PW-Rec\nReasoning",
        "short": "PW-Rec (Rsn)",
        "paradigm": "PW", "task": "Rec", "model_type": "Reasoning",
    },
    "ICML_04_UT_IND-Q_Rec_NPr_FA_Rsn": {
        "label": "IND-Rec\nReasoning",
        "short": "IND-Rec (Rsn)",
        "paradigm": "IND", "task": "Rec", "model_type": "Reasoning",
    },
    "ICML_05_UT_PW-Q_Pref-Q_NPr_FA_Inst": {
        "label": "PW-Pref\nInstruct",
        "short": "PW-Pref (Inst)",
        "paradigm": "PW", "task": "Pref", "model_type": "Instruct",
    },
    "ICML_06_UT_IND-Q_Pref-Q_NPr_FA_Inst": {
        "label": "IND-Pref\nInstruct",
        "short": "IND-Pref (Inst)",
        "paradigm": "IND", "task": "Pref", "model_type": "Instruct",
    },
    "ICML_07_UT_PW-Q_Rec_NPr_FA_Rsn-Inst": {
        "label": "PW-Rec\nRsn+Inst",
        "short": "PW-Rec (R+I)",
        "paradigm": "PW", "task": "Rec", "model_type": "Rsn+Inst",
    },
    "ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst": {
        "label": "IND-Rec\nRsn+Inst",
        "short": "IND-Rec (R+I)",
        "paradigm": "IND", "task": "Rec", "model_type": "Rsn+Inst",
    },
    # COLM experiments — assistant tag (AT), conversational format (C)
    "COLM_01_AT_PW-C_Rec_NPr_FA_Inst": {
        "label": "PW-Rec\nAT-Inst",
        "short": "PW-Rec (AT)",
        "paradigm": "PW", "task": "Rec", "model_type": "Instruct", "tag": "AT",
    },
    "COLM_02_AT_IND-C_Rec_NPr_FA_Inst": {
        "label": "IND-Rec\nAT-Inst",
        "short": "IND-Rec (AT)",
        "paradigm": "IND", "task": "Rec", "model_type": "Instruct", "tag": "AT",
    },
}

# Short dataset names
DATASET_SHORT = {
    "wikisum/training_set_1-20+test_set_1-30": "WikiSum",
    "sharegpt/english_26+english2_74": "ShareGPT",
    "pku_saferlhf/mismatch_1-20+test_mismatch_1-20+test-mismatch_10_100-200": "PKU",
    "bigcodebench/instruct_1-50": "BigCode",
}


def load_experiment(exp_name):
    """Load the latest aggregated_performance.csv for an experiment."""
    exp_dir = AGG_DIR / exp_name
    if not exp_dir.exists():
        return None
    timestamps = sorted(exp_dir.iterdir(), reverse=True)
    for ts_dir in timestamps:
        perf_file = ts_dir / "aggregated_performance.csv"
        if perf_file.exists():
            df = pd.read_csv(perf_file, index_col=0)
            df.columns = [DATASET_SHORT.get(c, c) for c in df.columns]
            return df
    return None


def load_all():
    """Load all experiments into a dict."""
    data = {}
    for exp_name, meta in EXPERIMENTS.items():
        df = load_experiment(exp_name)
        if df is not None:
            data[exp_name] = {"df": df, **meta}
            print(f"  Loaded {meta['short']}: {df.shape[0]} models × {df.shape[1]} datasets")
        else:
            print(f"  ⚠ Missing: {exp_name}")
    return data


# ============================================================================
# FIGURE 1: Heatmap — models × operationalizations (mean across datasets)
# ============================================================================
def fig_heatmap_ops(data):
    """
    Single heatmap: rows = models, columns = operationalizations.
    Cell value = mean accuracy across datasets. Color diverging at 0.5.
    """
    # Build matrix: models × experiments
    # Use ICML_01 models as the canonical row order
    exp_01 = data.get("ICML_01_UT_PW-Q_Rec_NPr_FA_Inst")
    if exp_01 is None:
        print("  ⚠ Skipping heatmap — ICML_01 not loaded")
        return

    # Collect mean performance per model per experiment
    rows = []
    for exp_name, info in data.items():
        df = info["df"]
        mean_perf = df.mean(axis=1)
        mean_perf.name = info["short"]
        rows.append(mean_perf)

    matrix = pd.DataFrame(rows).T
    matrix = matrix.dropna(how="all")

    # Sort by ICML_01 performance (descending)
    if "PW-Rec (Inst)" in matrix.columns:
        matrix = matrix.sort_values("PW-Rec (Inst)", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    norm = TwoSlopeNorm(vmin=0.3, vcenter=0.5, vmax=0.9)
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="RdYlGn", norm=norm,
        linewidths=0.5, ax=ax, cbar_kws={"label": "Mean Accuracy"},
        annot_kws={"size": 8},
    )
    ax.set_title("Recognition/Preference Performance Across Operationalizations\n(Mean across datasets)", fontsize=12)
    ax.set_ylabel("Evaluator Model")
    ax.set_xlabel("Operationalization")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = OUT_DIR / "heatmap_models_x_ops.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================================
# FIGURE 2: Compact grouped bar — PW vs IND for Rec (Inst), side by side
# ============================================================================
def fig_pw_vs_ind_bars(data):
    """
    Two-panel grouped bar chart comparing PW vs IND for recognition (instruct).
    Compact alternative to current Fig 1 (a) and (b).
    """
    pw = data.get("ICML_01_UT_PW-Q_Rec_NPr_FA_Inst")
    ind = data.get("ICML_02_UT_IND-Q_Rec_NPr_FA_Inst")
    if pw is None or ind is None:
        return

    pw_mean = pw["df"].mean(axis=1).sort_values(ascending=False)
    ind_mean = ind["df"].mean(axis=1).sort_values(ascending=False)

    # Use PW order for both
    models = pw_mean.index.tolist()
    ind_mean = ind_mean.reindex(models)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    bars1 = ax.bar(x - width / 2, pw_mean.values, width, label="Pairwise (PW)", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, ind_mean.values, width, label="Individual (IND)", color="#FF9800", alpha=0.85)

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance (0.5)")
    ax.set_ylabel("Mean Recognition Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Recognition Accuracy: Pairwise vs Individual Paradigm (Instruct Models)")
    plt.tight_layout()
    path = OUT_DIR / "pw_vs_ind_bars.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================================
# FIGURE 3: Dot plot — models × OPs with dots colored by performance
# ============================================================================
def fig_dotplot_ops(data):
    """
    Dot plot: x = operationalization, y = model.
    Dot size/color = mean accuracy. Very compact.
    """
    # Subset to core 6 experiments (Rec: PW/IND × Inst/Rsn/AT)
    subset_keys = [
        "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst",
        "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst",
        "ICML_03_UT_PW-Q_Rec_NPr_FA_Rsn",
        "ICML_04_UT_IND-Q_Rec_NPr_FA_Rsn",
        "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
        "COLM_02_AT_IND-C_Rec_NPr_FA_Inst",
    ]

    available = {k: data[k] for k in subset_keys if k in data}
    if len(available) < 2:
        return

    # Build long-form data
    records = []
    for exp_name, info in available.items():
        df = info["df"]
        mean_perf = df.mean(axis=1)
        for model, val in mean_perf.items():
            records.append({
                "model": model,
                "op": info["short"],
                "accuracy": val,
            })

    long_df = pd.DataFrame(records)

    # Get model order from PW-Rec (Inst) — descending
    pw_order = long_df[long_df["op"] == "PW-Rec (Inst)"].sort_values("accuracy", ascending=True)
    model_order = pw_order["model"].tolist()
    # Add any models not in PW-Rec
    all_models = long_df["model"].unique()
    for m in all_models:
        if m not in model_order:
            model_order.insert(0, m)

    op_order = [available[k]["short"] for k in subset_keys if k in available]

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, op in enumerate(op_order):
        op_data = long_df[long_df["op"] == op]
        for _, row in op_data.iterrows():
            y = model_order.index(row["model"])
            val = row["accuracy"]
            color = plt.cm.RdYlGn((val - 0.3) / 0.6)  # normalize 0.3-0.9
            size = max(20, val * 200)
            ax.scatter(i, y, s=size, c=[color], edgecolors="black", linewidth=0.5, zorder=3)

    ax.axvline(x=0.5, color="lightgray", linewidth=0)  # just for spacing
    ax.set_xticks(range(len(op_order)))
    ax.set_xticklabels(op_order, fontsize=9)
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order, fontsize=8)
    ax.set_xlabel("Operationalization")
    ax.set_title("Recognition Performance by Model and Operationalization\n(Dot size & color = mean accuracy, green = high)")

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0.3, 0.9))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Mean Accuracy", shrink=0.7)

    plt.tight_layout()
    path = OUT_DIR / "dotplot_models_x_ops.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================================
# FIGURE 4+: 2×2 panel heatmaps — reusable helper
# ============================================================================
def _fig_heatmap_2x2(data, panels, suptitle, filename):
    """
    Reusable 2×2 panel heatmap.
    panels: ordered dict of {panel_title: experiment_name}
    """
    panel_data = {}
    for title, exp_name in panels.items():
        info = data.get(exp_name)
        if info is None:
            print(f"  ⚠ Missing: {exp_name}")
            return
        df = info["df"].copy()
        df["Mean"] = df.mean(axis=1)
        panel_data[title] = df

    # Get common models (present in all 4 experiments)
    model_sets = [set(df.index) for df in panel_data.values()]
    common_models = sorted(model_sets[0].intersection(*model_sets[1:]))

    # Sort by first panel's mean (descending)
    first_title = list(panels.keys())[0]
    sort_order = (
        panel_data[first_title]
        .loc[common_models, "Mean"]
        .sort_values(ascending=False)
        .index.tolist()
    )

    norm = TwoSlopeNorm(vmin=0.3, vcenter=0.5, vmax=0.9)
    titles = list(panels.keys())

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 9),
        gridspec_kw={"wspace": 0.02, "hspace": 0.1},
    )

    for idx, title in enumerate(titles):
        row, col = divmod(idx, 2)
        ax = axes[row][col]
        df = panel_data[title].loc[sort_order]

        sns.heatmap(
            df, annot=True, fmt=".2f", cmap="RdYlGn", norm=norm,
            linewidths=0.5, ax=ax,
            cbar=False,
            annot_kws={"size": 8},
        )
        ax.set_title(title, fontsize=11, fontweight="bold")

        if col == 0:
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_ylabel("")

        if row == 1:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=9, rotation=30, ha="right")
        else:
            ax.set_xticklabels([])

    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, aspect=30, pad=0.02)
    cbar.set_label("Accuracy", fontsize=11)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.98)

    path = OUT_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def fig_heatmap_full(data):
    """4a: Recognition × Tag (UT/AT) × Paradigm (PW/IND)"""
    _fig_heatmap_2x2(data, {
        "(a) User-Tag — Pairwise": "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst",
        "(b) User-Tag — Individual": "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst",
        "(c) Assistant-Tag — Pairwise": "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
        "(d) Assistant-Tag — Individual": "COLM_02_AT_IND-C_Rec_NPr_FA_Inst",
    }, "Recognition Accuracy by Tag Framing and Paradigm", "heatmap_full_pw_ind.png")


def fig_heatmap_rec_vs_pref(data):
    """4b: Recognition vs Preference × Paradigm (PW/IND)"""
    _fig_heatmap_2x2(data, {
        "(a) Recognition — Pairwise": "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst",
        "(b) Recognition — Individual": "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst",
        "(c) Preference — Pairwise": "ICML_05_UT_PW-Q_Pref-Q_NPr_FA_Inst",
        "(d) Preference — Individual": "ICML_06_UT_IND-Q_Pref-Q_NPr_FA_Inst",
    }, "Recognition vs Preference Accuracy by Paradigm", "heatmap_rec_vs_pref.png")


# ============================================================================
# FIGURE 5: Small multiples — per-dataset performance for PW and IND
# ============================================================================
def fig_small_multiples(data):
    """
    2×4 grid: rows = PW/IND, columns = datasets.
    Each panel is a horizontal bar chart showing per-model performance.
    """
    pw = data.get("ICML_01_UT_PW-Q_Rec_NPr_FA_Inst")
    ind = data.get("ICML_02_UT_IND-Q_Rec_NPr_FA_Inst")
    if pw is None or ind is None:
        return

    datasets = list(pw["df"].columns)
    # Sort models by PW mean
    model_order = pw["df"].mean(axis=1).sort_values(ascending=True).index.tolist()

    fig, axes = plt.subplots(2, len(datasets), figsize=(14, 6), sharey=True)

    for col_idx, ds in enumerate(datasets):
        for row_idx, (exp_data, paradigm) in enumerate([(pw, "PW"), (ind, "IND")]):
            ax = axes[row_idx, col_idx]
            df = exp_data["df"]
            vals = df[ds].reindex(model_order)

            colors = ["#4CAF50" if v > 0.5 else "#F44336" for v in vals]
            ax.barh(range(len(model_order)), vals, color=colors, alpha=0.8, height=0.7)
            ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlim(0.2, 1.0)

            if row_idx == 0:
                ax.set_title(ds, fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.set_yticks(range(len(model_order)))
                ax.set_yticklabels(model_order, fontsize=7)
                ax.set_ylabel(paradigm, fontsize=11, fontweight="bold")
            else:
                ax.set_yticks([])

            ax.tick_params(axis="x", labelsize=7)

    plt.suptitle("Recognition Accuracy: Pairwise (top) vs Individual (bottom)", fontsize=12, y=1.01)
    plt.tight_layout()
    path = OUT_DIR / "small_multiples_pw_ind.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================================
# FIGURE 6: Compact summary — mean ± std across datasets, multiple OPs
# ============================================================================
def fig_summary_errorbar(data):
    """
    Single panel: models on x-axis, mean accuracy (across datasets) with
    error bars for std, separate lines/markers for each operationalization.
    Very compact — fits in half a column.
    """
    subset_keys = [
        "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst",
        "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst",
        "ICML_05_UT_PW-Q_Pref-Q_NPr_FA_Inst",
        "ICML_06_UT_IND-Q_Pref-Q_NPr_FA_Inst",
        "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
        "COLM_02_AT_IND-C_Rec_NPr_FA_Inst",
    ]
    available = {k: data[k] for k in subset_keys if k in data}

    markers = ["o", "s", "^", "D", "P", "X"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#795548"]

    # Get model order from PW-Rec
    pw = available.get("ICML_01_UT_PW-Q_Rec_NPr_FA_Inst")
    if pw is None:
        return
    model_order = pw["df"].mean(axis=1).sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(model_order))

    for i, (exp_name, info) in enumerate(available.items()):
        df = info["df"]
        means = df.mean(axis=1).reindex(model_order)
        stds = df.std(axis=1).reindex(model_order)

        offset = (i - len(available) / 2 + 0.5) * 0.15
        ax.errorbar(
            x + offset, means, yerr=stds,
            fmt=markers[i], color=colors[i], markersize=6,
            capsize=3, capthick=1, linewidth=0, elinewidth=1,
            label=info["short"], alpha=0.85,
        )

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Accuracy (± std across datasets)")
    ax.set_ylim(0.25, 1.0)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_title("Performance Across Operationalizations (mean ± std over 4 datasets)")
    plt.tight_layout()
    path = OUT_DIR / "summary_errorbar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================================
# Helpers for IND adjustment in distance plots
# ============================================================================

# Map from experiment name to the per-dataset accuracy_pivot.csv paths
# The pivot diagonal contains self-scores (evaluator == generator)
ANALYSIS_DIR = Path("data/analysis")

IND_PIVOT_PATTERNS = {
    "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst": {
        "wikisum": "wikisum/training_set_1-20+test_set_1-30/ICML_02_UT_IND-Q_Rec_NPr_FA_Inst_dr/recognition_accuracy/accuracy_pivot.csv",
        "sharegpt": "sharegpt/english_26+english2_74/ICML_02_UT_IND-Q_Rec_NPr_FA_Inst_dr/recognition_accuracy/accuracy_pivot.csv",
        "pku_saferlhf": "pku_saferlhf/mismatch_1-20+test_mismatch_1-20+test-mismatch_10_100-200/ICML_02_UT_IND-Q_Rec_NPr_FA_Inst_dr/recognition_accuracy/accuracy_pivot.csv",
        "bigcodebench": "bigcodebench/instruct_1-50/ICML_02_UT_IND-Q_Rec_NPr_FA_Inst_dr/recognition_accuracy/accuracy_pivot.csv",
    },
    "COLM_02_AT_IND-C_Rec_NPr_FA_Inst": {
        "wikisum": "wikisum/training_set_1-20+test_set_1-30/COLM_02_AT_IND-C_Rec_NPr_FA_Inst/recognition_accuracy/accuracy_pivot.csv",
        "sharegpt": "sharegpt/english_26+english2_74/COLM_02_AT_IND-C_Rec_NPr_FA_Inst/recognition_accuracy/accuracy_pivot.csv",
        "pku_saferlhf": "pku_saferlhf/mismatch_1-20+test_mismatch_1-20+test-mismatch_10_100-200/COLM_02_AT_IND-C_Rec_NPr_FA_Inst/recognition_accuracy/accuracy_pivot.csv",
        "bigcodebench": "bigcodebench/instruct_1-50/COLM_02_AT_IND-C_Rec_NPr_FA_Inst/recognition_accuracy/accuracy_pivot.csv",
    },
    "ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst": {
        "wikisum": "wikisum/training_set_1-20+test_set_1-30/ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst/recognition_accuracy/accuracy_pivot.csv",
        "sharegpt": "sharegpt/english_26+english2_74/ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst/recognition_accuracy/accuracy_pivot.csv",
        "pku_saferlhf": "pku_saferlhf/mismatch_1-20+test_mismatch_1-20+test-mismatch_10_100-200/ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst/recognition_accuracy/accuracy_pivot.csv",
        "bigcodebench": "bigcodebench/instruct_1-50/ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst/recognition_accuracy/accuracy_pivot.csv",
    },
}


def load_self_scores(exp_name):
    """
    Load self-scores for an IND experiment from accuracy pivot tables.
    Returns dict: evaluator -> weighted average self-score across datasets.
    Returns None if not an IND experiment.
    """
    if exp_name not in IND_PIVOT_PATTERNS:
        return None

    self_scores_per_model = {}  # model -> list of (score, n_datasets)
    for ds_name, rel_path in IND_PIVOT_PATTERNS[exp_name].items():
        pivot_path = ANALYSIS_DIR / rel_path
        if not pivot_path.exists():
            continue
        pivot = pd.read_csv(pivot_path, index_col=0)
        # Diagonal = self-scores
        for model in pivot.index:
            if model in pivot.columns:
                val = pivot.loc[model, model]
                if pd.notna(val):
                    self_scores_per_model.setdefault(model, []).append(val)

    # Average across datasets
    return {m: np.mean(scores) for m, scores in self_scores_per_model.items()}


def adjust_ind_performance(df, self_scores):
    """
    Adjust IND cross-model performance by averaging with evaluator's self-score.
    adjusted = (cross_performance + self_score) / 2

    This ensures a model that always accepts or always rejects scores 0.5.
    """
    if self_scores is None:
        return df

    df = df.copy()
    adjusted_perfs = []
    for _, row in df.iterrows():
        evaluator = row["evaluator"]
        if evaluator in self_scores:
            adjusted_perfs.append((row["performance"] + self_scores[evaluator]) / 2)
        else:
            adjusted_perfs.append(np.nan)
    df["performance"] = adjusted_perfs
    return df.dropna(subset=["performance"])


# ============================================================================
# FIGURE 7: 2×2 paneled Elo score distance scatter plots
# ============================================================================
def fig_score_distance_panels(data):
    """
    2×2 panel of score-distance vs recognition accuracy scatter plots:
      (a) UT PW   (b) UT IND (adjusted)
      (c) AT PW   (d) AT IND (adjusted)

    IND panels are adjusted by averaging with evaluator self-score,
    so a model that always accepts/rejects scores 0.5.
    """
    from self_rec_framework.src.helpers.model_names import LM_ARENA_SCORES
    from scipy import stats

    panels = {
        "(a) User-Tag — Pairwise": "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst",
        "(b) User-Tag — Individual": "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst",
        "(c) Assistant-Tag — Pairwise": "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
        "(d) Assistant-Tag — Individual": "COLM_02_AT_IND-C_Rec_NPr_FA_Inst",
    }

    # Load rank_distance_data.csv for each experiment
    panel_dfs = {}
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

        # Compute Elo score distance
        def get_score(model):
            if model in LM_ARENA_SCORES:
                return LM_ARENA_SCORES[model]
            base = model.replace("-thinking", "")
            if base in LM_ARENA_SCORES:
                return LM_ARENA_SCORES[base]
            return None

        df["eval_score"] = df["evaluator"].apply(get_score)
        df["gen_score"] = df["generator"].apply(get_score)
        df = df.dropna(subset=["eval_score", "gen_score"])
        df["score_distance"] = df["eval_score"] - df["gen_score"]

        # Apply IND adjustment if applicable
        self_scores = load_self_scores(exp_name)
        if self_scores is not None:
            df = adjust_ind_performance(df, self_scores)
            print(f"  Applied IND adjustment for {exp_name} ({len(self_scores)} self-scores)")

        panel_dfs[title] = df

    ds_colors = {
        "wikisum": "#2196F3",
        "sharegpt": "#FF9800",
        "pku_saferlhf": "#4CAF50",
        "bigcodebench": "#E91E63",
    }
    ds_labels = {
        "wikisum": "WikiSum",
        "sharegpt": "ShareGPT",
        "pku_saferlhf": "PKU",
        "bigcodebench": "BigCode",
    }

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8),
        gridspec_kw={"wspace": 0.15, "hspace": 0.1},
    )

    titles = list(panels.keys())
    is_ind = {1, 3}  # indices for IND panels
    for idx, title in enumerate(titles):
        row, col = divmod(idx, 2)
        ax = axes[row][col]
        df = panel_dfs[title]
        dataset_names = sorted(df["dataset"].unique())

        for ds_name in dataset_names:
            ds_data = df[df["dataset"] == ds_name]
            if ds_data.empty:
                continue
            ax.scatter(
                ds_data["score_distance"], ds_data["performance"],
                c=ds_colors.get(ds_name, "gray"),
                label=ds_labels.get(ds_name, ds_name),
                alpha=0.4, s=15, edgecolors="none",
            )

        # Aggregate per (evaluator, generator) pair across datasets
        agg = df.groupby(["evaluator", "generator"]).agg(
            score_distance=("score_distance", "first"),
            performance=("performance", "mean"),
            weight=("n_samples", "sum"),
        ).reset_index()

        if len(agg) > 2:
            w = agg["weight"].values
            x_vals = agg["score_distance"].values
            y_vals = agg["performance"].values
            coeffs = np.polyfit(x_vals, y_vals, 1, w=np.sqrt(w))
            slope, intercept = coeffs
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color="black", linewidth=1.5, alpha=0.8)

            # Bootstrap confidence band
            rng = np.random.default_rng(42)
            n_boot = 500
            boot_lines = np.zeros((n_boot, len(x_line)))
            for b in range(n_boot):
                idx_b = rng.choice(len(x_vals), size=len(x_vals), replace=True)
                c = np.polyfit(x_vals[idx_b], y_vals[idx_b], 1,
                               w=np.sqrt(w[idx_b]))
                boot_lines[b] = c[0] * x_line + c[1]
            lo = np.percentile(boot_lines, 2.5, axis=0)
            hi = np.percentile(boot_lines, 97.5, axis=0)
            ax.fill_between(x_line, lo, hi, color="black", alpha=0.1, zorder=1)

            r, p = stats.pearsonr(x_vals, y_vals)
            ax.text(
                0.03, 0.03, f"r = {r:.2f}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
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
            ax.legend(fontsize=7, loc="lower right", markerscale=1.5)

    fig.suptitle(
        "Recognition Accuracy vs LM Arena Elo Score Distance",
        fontsize=13, fontweight="bold", y=0.98,
    )

    path = OUT_DIR / "score_distance_panels.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================================
# FIGURE 8: 2×2 paneled rank distance scatter plots
# ============================================================================
def fig_rank_distance_panels(data):
    """
    2×2 panel of rank-distance vs recognition accuracy scatter plots:
      (a) UT PW   (b) UT IND (adjusted)
      (c) AT PW   (d) AT IND (adjusted)

    IND panels are adjusted by averaging with evaluator self-score.
    """
    from scipy import stats

    panels = {
        "(a) User-Tag — Pairwise": "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst",
        "(b) User-Tag — Individual": "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst",
        "(c) Assistant-Tag — Pairwise": "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
        "(d) Assistant-Tag — Individual": "COLM_02_AT_IND-C_Rec_NPr_FA_Inst",
    }

    panel_dfs = {}
    for title, exp_name in panels.items():
        exp_dir = AGG_DIR / exp_name
        if not exp_dir.exists():
            return
        ts_dir = sorted(exp_dir.iterdir(), reverse=True)[0]
        csv_path = ts_dir / "rank_distance_data.csv"
        if not csv_path.exists():
            return
        df = pd.read_csv(csv_path)

        # Apply IND adjustment if applicable
        self_scores = load_self_scores(exp_name)
        if self_scores is not None:
            df = adjust_ind_performance(df, self_scores)
            print(f"  Applied IND adjustment for {exp_name} ({len(self_scores)} self-scores)")

        panel_dfs[title] = df

    ds_colors = {
        "wikisum": "#2196F3",
        "sharegpt": "#FF9800",
        "pku_saferlhf": "#4CAF50",
        "bigcodebench": "#E91E63",
    }
    ds_labels = {
        "wikisum": "WikiSum",
        "sharegpt": "ShareGPT",
        "pku_saferlhf": "PKU",
        "bigcodebench": "BigCode",
    }

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8),
        gridspec_kw={"wspace": 0.15, "hspace": 0.1},
    )

    titles = list(panels.keys())
    is_ind = {1, 3}
    for idx, title in enumerate(titles):
        row, col = divmod(idx, 2)
        ax = axes[row][col]
        df = panel_dfs[title]
        dataset_names = sorted(df["dataset"].unique())

        for ds_name in dataset_names:
            ds_data = df[df["dataset"] == ds_name]
            ax.scatter(
                ds_data["distance"], ds_data["performance"],
                c=ds_colors.get(ds_name, "gray"),
                label=ds_labels.get(ds_name, ds_name),
                alpha=0.4, s=15, edgecolors="none",
            )

        agg = df.groupby(["evaluator", "generator"]).agg(
            distance=("distance", "first"),
            performance=("performance", "mean"),
            weight=("n_samples", "sum"),
        ).reset_index()

        if len(agg) > 2:
            w = agg["weight"].values
            x_vals = agg["distance"].values
            y_vals = agg["performance"].values
            coeffs = np.polyfit(x_vals, y_vals, 1, w=np.sqrt(w))
            slope, intercept = coeffs
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color="black", linewidth=1.5, alpha=0.8)
            r, p = stats.pearsonr(x_vals, y_vals)
            ax.text(
                0.03, 0.03, f"r = {r:.2f}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
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
            ax.set_xlabel("Rank Distance\n(Evaluator Rank − Generator Rank)", fontsize=10)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

        if idx == 0:
            ax.legend(fontsize=7, loc="lower right", markerscale=1.5)

    fig.suptitle(
        "Recognition Accuracy vs LM Arena Rank Distance",
        fontsize=13, fontweight="bold", y=0.98,
    )

    path = OUT_DIR / "rank_distance_panels.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


    # NOTE: fig_training_effect_panels moved to prototype_uplift_figures.py


# ============================================================================
# Raw per-pair pivot data loader (not aggregated — individual evaluator×treatment pairs)
# ============================================================================

# Dataset subsets that correspond to the combined analysis directories
DATASET_PIVOT_PATHS = {
    "WikiSum": "wikisum/training_set_1-20+test_set_1-30",
    "ShareGPT": "sharegpt/english_26+english2_74",
    "PKU": "pku_saferlhf/mismatch_1-20+test_mismatch_1-20+test-mismatch_10_100-200",
    "BigCode": "bigcodebench/instruct_1-50",
}

ANALYSIS_DIR = Path("data/analysis")


def load_pivot_values(exp_name, dataset_short=None, model_filter=None):
    """Load per-pair accuracy values from accuracy_pivot.csv files.

    For IND experiments, applies adjustment: (treatment_ij + control_j) / 2
    so that a model always accepting or always rejecting scores 0.5.

    Returns a flat list of adjusted accuracy values.
    If dataset_short is specified, loads only that dataset. Otherwise loads all.
    If model_filter is specified, loads only rows matching that evaluator model.
    """
    is_ind = "IND" in exp_name
    values = []
    datasets = {dataset_short: DATASET_PIVOT_PATHS[dataset_short]} if dataset_short else DATASET_PIVOT_PATHS

    for ds_name, ds_path in datasets.items():
        pivot_path = ANALYSIS_DIR / ds_path / exp_name / "recognition_accuracy" / "accuracy_pivot.csv"
        if not pivot_path.exists():
            continue
        df = pd.read_csv(pivot_path, index_col=0)

        if is_ind:
            # Extract control scores (diagonal: evaluator == treatment)
            control = {}
            for evaluator in df.index:
                if evaluator in df.columns:
                    val = df.loc[evaluator, evaluator]
                    if pd.notna(val):
                        control[evaluator] = val

            # Adjusted = (treatment_ij + control_j) / 2
            if model_filter:
                if model_filter in df.index and model_filter in control:
                    c_j = control[model_filter]
                    for col in df.columns:
                        if col == model_filter:
                            continue  # skip self
                        t_ij = df.loc[model_filter, col]
                        if pd.notna(t_ij):
                            values.append((t_ij + c_j) / 2)
            else:
                for evaluator in df.index:
                    if evaluator not in control:
                        continue
                    c_j = control[evaluator]
                    for col in df.columns:
                        if col == evaluator:
                            continue
                        t_ij = df.loc[evaluator, col]
                        if pd.notna(t_ij):
                            values.append((t_ij + c_j) / 2)
        else:
            # PW: use raw values directly
            if model_filter:
                if model_filter in df.index:
                    row_values = df.loc[model_filter].dropna().tolist()
                    values.extend(row_values)
            else:
                values.extend(df.values[~np.isnan(df.values)].tolist())
    return values


# ============================================================================
# FIGURE 10: Boxplots — 8 operationalizations
# First 4: PW, IND, UT, AT (pool across datasets)
# Last 4: WikiSum, ShareGPT, PKU, BigCode (pool across task OPs)
# ============================================================================

# Which experiments contribute to each task dimension
BOXPLOT_TASK_GROUPS = {
    "PW":  ["ICML_07_UT_PW-Q_Rec_NPr_FA_Rsn-Inst", "COLM_01_AT_PW-C_Rec_NPr_FA_Inst"],
    "IND": ["ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst", "COLM_02_AT_IND-C_Rec_NPr_FA_Inst"],
    "UT":  ["ICML_07_UT_PW-Q_Rec_NPr_FA_Rsn-Inst", "ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst"],
    "AT":  ["COLM_01_AT_PW-C_Rec_NPr_FA_Inst", "COLM_02_AT_IND-C_Rec_NPr_FA_Inst"],
}

# All 4 experiments used for dataset boxes
BOXPLOT_ALL_EXPS = [
    "ICML_07_UT_PW-Q_Rec_NPr_FA_Rsn-Inst",
    "ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst",
    "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
    "COLM_02_AT_IND-C_Rec_NPr_FA_Inst",
]

DATASET_NAMES = ["WikiSum", "ShareGPT", "PKU", "BigCode"]


def fig_boxplot_operationalizations(data):
    """
    8 boxplots: PW, IND, UT, AT, WikiSum, ShareGPT, PKU, BigCode.

    Uses raw per-pair pivot data (evaluator×treatment accuracy values)
    for maximum spread visibility.

    Task boxes pool all per-pair values from matching experiments across all datasets.
    Dataset boxes pool per-pair values for that dataset across all experiments.
    """
    box_labels = list(BOXPLOT_TASK_GROUPS.keys()) + DATASET_NAMES
    box_data = []

    # Task dimension boxes (pool all per-pair values across all datasets)
    for group_name, exp_list in BOXPLOT_TASK_GROUPS.items():
        values = []
        for exp_name in exp_list:
            values.extend(load_pivot_values(exp_name))
        box_data.append(values)

    # Dataset boxes (pool per-pair values for that dataset across all experiments)
    for ds_name in DATASET_NAMES:
        values = []
        for exp_name in BOXPLOT_ALL_EXPS:
            values.extend(load_pivot_values(exp_name, dataset_short=ds_name))
        box_data.append(values)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))

    # Colors: blue-ish for task dimensions, green-ish for datasets
    task_color = "#4C72B0"
    dataset_color = "#55A868"
    colors = [task_color] * 4 + [dataset_color] * 4

    # Data points behind boxes
    for i, (vals, color) in enumerate(zip(box_data, colors)):
        if not vals:
            continue
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            color=color, alpha=0.3, s=8, edgecolors="none", zorder=1,
        )

    # Boxes in front
    bp = ax.boxplot(
        box_data,
        positions=range(len(box_labels)),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5, zorder=4),
        whiskerprops=dict(color="gray", zorder=3),
        capprops=dict(color="gray", zorder=3),
        zorder=3,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        patch.set_zorder(3)

    # Chance line
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
    ax.text(len(box_labels) - 0.5, 0.505, "chance", fontsize=8, color="gray",
            ha="right", va="bottom")

    # Vertical separator between task and dataset groups
    ax.axvline(3.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)

    # Labels
    ax.set_xticks(range(len(box_labels)))
    ax.set_xticklabels(box_labels, fontsize=10, fontweight="bold")
    ax.set_ylabel("Recognition Accuracy", fontsize=11)
    ax.set_title("Performance Distribution Across Operationalizations", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=task_color, alpha=0.35, edgecolor=task_color, label="Task Dimension"),
        Patch(facecolor=dataset_color, alpha=0.35, edgecolor=dataset_color, label="Dataset"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Count annotations below each box
    for i, vals in enumerate(box_data):
        ax.text(i, -0.03, f"n={len(vals)}", ha="center", va="top", fontsize=7, color="gray")

    plt.tight_layout()
    path = OUT_DIR / "boxplot_operationalizations.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================================
# FIGURE 11: Boxplots — per-model distributions across all operationalizations
# 8 models selected for provider diversity and Elo spread
# ============================================================================

# Brand-inspired colors for model providers
BOXPLOT_MODELS = [
    ("ll-3.1-8b", "Llama 8B", "#0668E1"),             # Meta blue
    ("gpt-4o-mini", "GPT-4o Mini", "#10A37F"),         # OpenAI green
    ("gemini-2.0-flash", "Gemini Flash", "#4285F4"),    # Google blue
    ("qwen-3.0-80b", "Qwen 80B", "#6F42C1"),           # Alibaba purple
    ("deepseek-3.1", "DeepSeek 3.1", "#4D6BFE"),       # DeepSeek blue
    ("kimi-k2", "Kimi K2", "#FF6B35"),                  # Moonshot orange
    ("gpt-4o", "GPT-4o", "#10A37F"),                    # OpenAI green
    ("opus-4.1", "Opus 4.1", "#D97757"),                # Anthropic terracotta
]


def fig_boxplot_per_model(data):
    """
    8 boxplots, one per model. Each box shows the distribution of that model's
    per-pair accuracy across ALL operationalizations (4 experiments × 4 datasets).
    Uses raw pivot data for full spread. Models ordered by Elo score (low → high).
    """
    box_data = []
    box_labels = []
    box_colors = []

    for model_name, display_name, color in BOXPLOT_MODELS:
        values = []
        for exp_name in BOXPLOT_ALL_EXPS:
            values.extend(load_pivot_values(exp_name, model_filter=model_name))
        if not values:
            continue
        box_data.append(values)
        box_labels.append(display_name)
        box_colors.append(color)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Data points behind boxes
    for i, (vals, color) in enumerate(zip(box_data, box_colors)):
        if not vals:
            continue
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            color=color, alpha=0.3, s=10, edgecolors="none", zorder=1,
        )

    # Boxes in front
    bp = ax.boxplot(
        box_data,
        positions=range(len(box_labels)),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5, zorder=4),
        whiskerprops=dict(color="gray", zorder=3),
        capprops=dict(color="gray", zorder=3),
        zorder=3,
    )

    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        patch.set_zorder(3)

    # Chance line
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
    ax.text(len(box_labels) - 0.5, 0.505, "chance", fontsize=8, color="gray",
            ha="right", va="bottom")

    ax.set_xticks(range(len(box_labels)))
    ax.set_xticklabels(box_labels, fontsize=10, fontweight="bold", rotation=20, ha="right")
    ax.set_ylabel("Recognition Accuracy", fontsize=11)
    ax.set_title("Per-Model Performance Distribution Across All Operationalizations\n"
                 "(ordered by LM Arena Elo, low → high)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)

    # Count annotations
    for i, vals in enumerate(box_data):
        ax.text(i, -0.03, f"n={len(vals)}", ha="center", va="top", fontsize=7, color="gray")

    plt.tight_layout()
    path = OUT_DIR / "boxplot_per_model.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def fig_boxplot_combined(data):
    """Combined figure: operationalizations on top, per-model on bottom, saved as PDF.

    Uses only ICML_01, ICML_02, COLM_01, COLM_02 (instruct-only _dr sets).
    Only includes models present in all 4 experiments.
    """
    COMBINED_EXPS = [
        "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst_dr",
        "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst_dr",
        "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
        "COLM_02_AT_IND-C_Rec_NPr_FA_Inst",
    ]
    COMBINED_TASK_GROUPS = {
        "PW":  ["ICML_01_UT_PW-Q_Rec_NPr_FA_Inst_dr", "COLM_01_AT_PW-C_Rec_NPr_FA_Inst"],
        "IND": ["ICML_02_UT_IND-Q_Rec_NPr_FA_Inst_dr", "COLM_02_AT_IND-C_Rec_NPr_FA_Inst"],
        "UT":  ["ICML_01_UT_PW-Q_Rec_NPr_FA_Inst_dr", "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst_dr"],
        "AT":  ["COLM_01_AT_PW-C_Rec_NPr_FA_Inst", "COLM_02_AT_IND-C_Rec_NPr_FA_Inst"],
    }
    COMBINED_DS_ORDER = ["ShareGPT", "PKU", "BigCode", "WikiSum"]
    COMBINED_DS_DISPLAY = {
        "PKU": "PKU", "ShareGPT": "S-GPT",
        "BigCode": "BCB", "WikiSum": "WS",
    }

    # Use dr_colm model set: 13 instruct models common to all 4 experiments
    from self_rec_framework.src.helpers.model_sets import get_model_set
    common_models = set(get_model_set("dr_colm"))

    # Only include (evaluator, generator) pairs where both are in common_models
    def _load_filtered(exp_name, dataset_short=None, model_filter=None):
        """Load pivot values filtered to common model pairs.

        For the top panel (model_filter=None): loads values for all common
        models as evaluators, but only against other common models as generators.
        For the bottom panel (model_filter set): loads that model's row,
        but only columns matching common models.
        """
        is_ind = "IND" in exp_name
        values = []
        datasets = {dataset_short: DATASET_PIVOT_PATHS[dataset_short]} if dataset_short else DATASET_PIVOT_PATHS

        for ds_name, ds_path in datasets.items():
            pivot_path = ANALYSIS_DIR / ds_path / exp_name / "recognition_accuracy" / "accuracy_pivot.csv"
            if not pivot_path.exists():
                continue
            df = pd.read_csv(pivot_path, index_col=0)

            evaluators = [model_filter] if model_filter else sorted(common_models & set(df.index))
            generators = sorted(common_models & set(df.columns))

            if is_ind:
                control = {}
                for ev in df.index:
                    if ev in df.columns and pd.notna(df.loc[ev, ev]):
                        control[ev] = df.loc[ev, ev]
                for ev in evaluators:
                    if ev not in df.index or ev not in control:
                        continue
                    c_j = control[ev]
                    for gen in generators:
                        if gen == ev:
                            continue
                        t_ij = df.loc[ev, gen]
                        if pd.notna(t_ij):
                            values.append((t_ij + c_j) / 2)
            else:
                for ev in evaluators:
                    if ev not in df.index:
                        continue
                    for gen in generators:
                        if gen == ev:
                            continue
                        val = df.loc[ev, gen]
                        if pd.notna(val):
                            values.append(val)
        return values

    whisker_style = dict(color="black", linewidth=1.2, zorder=3)
    cap_style = dict(color="black", linewidth=1.2, zorder=3)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8),
                                          gridspec_kw={"hspace": 0.25})

    # ── Top panel: operationalizations ──
    task_labels = list(COMBINED_TASK_GROUPS.keys())
    ds_labels = [COMBINED_DS_DISPLAY[d] for d in COMBINED_DS_ORDER]
    box_labels_top = task_labels + ds_labels
    box_data_top = []

    for group_name, exp_list in COMBINED_TASK_GROUPS.items():
        values = []
        for exp_name in exp_list:
            values.extend(_load_filtered(exp_name))
        box_data_top.append(values)

    for ds_name in COMBINED_DS_ORDER:
        values = []
        for exp_name in COMBINED_EXPS:
            values.extend(_load_filtered(exp_name, dataset_short=ds_name))
        box_data_top.append(values)

    task_color = "#4C72B0"
    dataset_color = "#55A868"
    colors_top = [task_color] * 4 + [dataset_color] * 4

    for i, (vals, color) in enumerate(zip(box_data_top, colors_top)):
        if not vals:
            continue
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax_top.scatter(np.full(len(vals), i) + jitter, vals,
                       color=color, alpha=0.3, s=8, edgecolors="none", zorder=1)

    bp_top = ax_top.boxplot(
        box_data_top, positions=range(len(box_labels_top)), widths=0.5,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=2.0, zorder=4),
        whiskerprops=whisker_style, capprops=cap_style, zorder=3,
    )
    for patch, color in zip(bp_top["boxes"], colors_top):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        patch.set_zorder(3)

    ax_top.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
    ax_top.axvline(3.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax_top.set_xticks(range(len(box_labels_top)))
    ax_top.set_xticklabels(box_labels_top, fontsize=10, fontweight="bold",
                           rotation=25, ha="right")
    ax_top.set_ylabel("Recognition Accuracy", fontsize=11)
    ax_top.set_title("(a) Performance Distribution Across Operationalizations",
                     fontsize=12, fontweight="bold")
    ax_top.set_ylim(-0.05, 1.05)

    for i, vals in enumerate(box_data_top):
        ax_top.text(i, -0.01, f"n={len(vals)}", ha="center", va="top", fontsize=7, color="gray")

    # ── Bottom panel: per-model (only common models) ──
    filtered_models = [(m, d, c) for m, d, c in BOXPLOT_MODELS if m in common_models]

    box_data_bot = []
    box_labels_bot = []
    box_colors_bot = []

    for model_name, display_name, color in filtered_models:
        values = []
        for exp_name in COMBINED_EXPS:
            values.extend(load_pivot_values(exp_name, model_filter=model_name))
        if not values:
            continue
        box_data_bot.append(values)
        box_labels_bot.append(display_name)
        box_colors_bot.append(color)

    for i, (vals, color) in enumerate(zip(box_data_bot, box_colors_bot)):
        if not vals:
            continue
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax_bot.scatter(np.full(len(vals), i) + jitter, vals,
                       color=color, alpha=0.3, s=10, edgecolors="none", zorder=1)

    bp_bot = ax_bot.boxplot(
        box_data_bot, positions=range(len(box_labels_bot)), widths=0.5,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=2.0, zorder=4),
        whiskerprops=whisker_style, capprops=cap_style, zorder=3,
    )
    for patch, color in zip(bp_bot["boxes"], box_colors_bot):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        patch.set_zorder(3)

    ax_bot.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
    ax_bot.set_xticks(range(len(box_labels_bot)))
    ax_bot.set_xticklabels(box_labels_bot, fontsize=10, fontweight="bold", rotation=25, ha="right")
    ax_bot.set_ylabel("Recognition Accuracy", fontsize=11)
    ax_bot.set_title("(b) Per-Model Performance Distribution",
                     fontsize=12, fontweight="bold")
    ax_bot.set_ylim(-0.05, 1.05)

    for i, vals in enumerate(box_data_bot):
        ax_bot.text(i, -0.01, f"n={len(vals)}", ha="center", va="top", fontsize=7, color="gray")

    path = OUT_DIR / "boxplot_combined.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def fig_boxplot_combined_v2(data):
    """Variant with individual OP boxes: UT PW, UT IND, AT PW, AT IND (+ datasets + per-model)."""
    from self_rec_framework.src.helpers.model_sets import get_model_set

    COMBINED_EXPS = [
        "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst_dr",
        "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst_dr",
        "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
        "COLM_02_AT_IND-C_Rec_NPr_FA_Inst",
    ]
    TASK_INDIVIDUAL = {
        "UT PW":  ["ICML_01_UT_PW-Q_Rec_NPr_FA_Inst_dr"],
        "UT IND": ["ICML_02_UT_IND-Q_Rec_NPr_FA_Inst_dr"],
        "AT PW":  ["COLM_01_AT_PW-C_Rec_NPr_FA_Inst"],
        "AT IND": ["COLM_02_AT_IND-C_Rec_NPr_FA_Inst"],
    }
    DS_ORDER = ["ShareGPT", "PKU", "BigCode", "WikiSum"]
    DS_DISPLAY = {"PKU": "PKU", "ShareGPT": "S-GPT", "BigCode": "BCB", "WikiSum": "WS"}

    common_models = set(get_model_set("dr_colm"))

    def _load_filtered(exp_name, dataset_short=None, model_filter=None):
        is_ind = "IND" in exp_name
        values = []
        datasets = {dataset_short: DATASET_PIVOT_PATHS[dataset_short]} if dataset_short else DATASET_PIVOT_PATHS
        for ds_name, ds_path in datasets.items():
            pivot_path = ANALYSIS_DIR / ds_path / exp_name / "recognition_accuracy" / "accuracy_pivot.csv"
            if not pivot_path.exists():
                continue
            df = pd.read_csv(pivot_path, index_col=0)
            evaluators = [model_filter] if model_filter else sorted(common_models & set(df.index))
            generators = sorted(common_models & set(df.columns))
            if is_ind:
                control = {}
                for ev in df.index:
                    if ev in df.columns and pd.notna(df.loc[ev, ev]):
                        control[ev] = df.loc[ev, ev]
                for ev in evaluators:
                    if ev not in df.index or ev not in control:
                        continue
                    c_j = control[ev]
                    for gen in generators:
                        if gen == ev:
                            continue
                        t_ij = df.loc[ev, gen]
                        if pd.notna(t_ij):
                            values.append((t_ij + c_j) / 2)
            else:
                for ev in evaluators:
                    if ev not in df.index:
                        continue
                    for gen in generators:
                        if gen == ev:
                            continue
                        val = df.loc[ev, gen]
                        if pd.notna(val):
                            values.append(val)
        return values

    whisker_style = dict(color="black", linewidth=1.2, zorder=3)
    cap_style = dict(color="black", linewidth=1.2, zorder=3)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8),
                                          gridspec_kw={"hspace": 0.25})

    # ── Top panel: individual OPs + datasets ──
    task_labels = list(TASK_INDIVIDUAL.keys())
    ds_labels = [DS_DISPLAY[d] for d in DS_ORDER]
    box_labels_top = task_labels + ds_labels
    box_data_top = []

    for group_name, exp_list in TASK_INDIVIDUAL.items():
        values = []
        for exp_name in exp_list:
            values.extend(_load_filtered(exp_name))
        box_data_top.append(values)

    for ds_name in DS_ORDER:
        values = []
        for exp_name in COMBINED_EXPS:
            values.extend(_load_filtered(exp_name, dataset_short=ds_name))
        box_data_top.append(values)

    task_color = "#4C72B0"
    dataset_color = "#55A868"
    colors_top = [task_color] * 4 + [dataset_color] * 4

    for i, (vals, color) in enumerate(zip(box_data_top, colors_top)):
        if not vals:
            continue
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax_top.scatter(np.full(len(vals), i) + jitter, vals,
                       color=color, alpha=0.3, s=8, edgecolors="none", zorder=1)

    bp_top = ax_top.boxplot(
        box_data_top, positions=range(len(box_labels_top)), widths=0.5,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=2.0, zorder=4),
        whiskerprops=whisker_style, capprops=cap_style, zorder=3,
    )
    for patch, color in zip(bp_top["boxes"], colors_top):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        patch.set_zorder(3)

    ax_top.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
    ax_top.axvline(3.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax_top.set_xticks(range(len(box_labels_top)))
    ax_top.set_xticklabels(box_labels_top, fontsize=10, fontweight="bold",
                           rotation=25, ha="right")
    ax_top.set_ylabel("Recognition Accuracy", fontsize=11)
    ax_top.set_title("(a) Performance Distribution Across Operationalizations",
                     fontsize=12, fontweight="bold")
    ax_top.set_ylim(-0.05, 1.05)

    for i, vals in enumerate(box_data_top):
        ax_top.text(i, -0.01, f"n={len(vals)}", ha="center", va="top", fontsize=7, color="gray")

    # ── Bottom panel: per-model ──
    filtered_models = [(m, d, c) for m, d, c in BOXPLOT_MODELS if m in common_models]

    box_data_bot = []
    box_labels_bot = []
    box_colors_bot = []

    for model_name, display_name, color in filtered_models:
        values = []
        for exp_name in COMBINED_EXPS:
            values.extend(load_pivot_values(exp_name, model_filter=model_name))
        if not values:
            continue
        box_data_bot.append(values)
        box_labels_bot.append(display_name)
        box_colors_bot.append(color)

    for i, (vals, color) in enumerate(zip(box_data_bot, box_colors_bot)):
        if not vals:
            continue
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax_bot.scatter(np.full(len(vals), i) + jitter, vals,
                       color=color, alpha=0.3, s=10, edgecolors="none", zorder=1)

    bp_bot = ax_bot.boxplot(
        box_data_bot, positions=range(len(box_labels_bot)), widths=0.5,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=2.0, zorder=4),
        whiskerprops=whisker_style, capprops=cap_style, zorder=3,
    )
    for patch, color in zip(bp_bot["boxes"], box_colors_bot):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        patch.set_zorder(3)

    ax_bot.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
    ax_bot.set_xticks(range(len(box_labels_bot)))
    ax_bot.set_xticklabels(box_labels_bot, fontsize=10, fontweight="bold", rotation=25, ha="right")
    ax_bot.set_ylabel("Recognition Accuracy", fontsize=11)
    ax_bot.set_title("(b) Per-Model Performance Distribution",
                     fontsize=12, fontweight="bold")
    ax_bot.set_ylim(-0.05, 1.05)

    for i, vals in enumerate(box_data_bot):
        ax_bot.text(i, -0.01, f"n={len(vals)}", ha="center", va="top", fontsize=7, color="gray")

    path = OUT_DIR / "boxplot_combined_v2.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def fig_boxplot_combined_v3(data):
    """Variant with individual OP boxes + dataset distributions from PW experiments only."""
    from self_rec_framework.src.helpers.model_sets import get_model_set

    PW_EXPS = [
        "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst_dr",
        "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
    ]
    ALL_EXPS = [
        "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst_dr",
        "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst_dr",
        "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
        "COLM_02_AT_IND-C_Rec_NPr_FA_Inst",
    ]
    TASK_INDIVIDUAL = {
        "UT PW":  ["ICML_01_UT_PW-Q_Rec_NPr_FA_Inst_dr"],
        "UT IND": ["ICML_02_UT_IND-Q_Rec_NPr_FA_Inst_dr"],
        "AT PW":  ["COLM_01_AT_PW-C_Rec_NPr_FA_Inst"],
        "AT IND": ["COLM_02_AT_IND-C_Rec_NPr_FA_Inst"],
    }
    DS_ORDER = ["ShareGPT", "PKU", "BigCode", "WikiSum"]
    DS_DISPLAY = {"PKU": "PKU", "ShareGPT": "S-GPT", "BigCode": "BCB", "WikiSum": "WS"}

    common_models = set(get_model_set("dr_colm"))

    def _load_filtered(exp_name, dataset_short=None, model_filter=None):
        is_ind = "IND" in exp_name
        values = []
        datasets = {dataset_short: DATASET_PIVOT_PATHS[dataset_short]} if dataset_short else DATASET_PIVOT_PATHS
        for ds_name, ds_path in datasets.items():
            pivot_path = ANALYSIS_DIR / ds_path / exp_name / "recognition_accuracy" / "accuracy_pivot.csv"
            if not pivot_path.exists():
                continue
            df = pd.read_csv(pivot_path, index_col=0)
            evaluators = [model_filter] if model_filter else sorted(common_models & set(df.index))
            generators = sorted(common_models & set(df.columns))
            if is_ind:
                control = {}
                for ev in df.index:
                    if ev in df.columns and pd.notna(df.loc[ev, ev]):
                        control[ev] = df.loc[ev, ev]
                for ev in evaluators:
                    if ev not in df.index or ev not in control:
                        continue
                    c_j = control[ev]
                    for gen in generators:
                        if gen == ev:
                            continue
                        t_ij = df.loc[ev, gen]
                        if pd.notna(t_ij):
                            values.append((t_ij + c_j) / 2)
            else:
                for ev in evaluators:
                    if ev not in df.index:
                        continue
                    for gen in generators:
                        if gen == ev:
                            continue
                        val = df.loc[ev, gen]
                        if pd.notna(val):
                            values.append(val)
        return values

    whisker_style = dict(color="black", linewidth=1.2, zorder=3)
    cap_style = dict(color="black", linewidth=1.2, zorder=3)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8),
                                          gridspec_kw={"hspace": 0.25})

    # ── Top panel: individual OPs + datasets (PW only) ──
    task_labels = list(TASK_INDIVIDUAL.keys())
    ds_labels = [DS_DISPLAY[d] for d in DS_ORDER]
    box_labels_top = task_labels + ds_labels
    box_data_top = []

    for group_name, exp_list in TASK_INDIVIDUAL.items():
        values = []
        for exp_name in exp_list:
            values.extend(_load_filtered(exp_name))
        box_data_top.append(values)

    for ds_name in DS_ORDER:
        values = []
        for exp_name in PW_EXPS:  # Only UT PW and AT PW
            values.extend(_load_filtered(exp_name, dataset_short=ds_name))
        box_data_top.append(values)

    task_color = "#4C72B0"
    dataset_color = "#55A868"
    colors_top = [task_color] * 4 + [dataset_color] * 4

    for i, (vals, color) in enumerate(zip(box_data_top, colors_top)):
        if not vals:
            continue
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax_top.scatter(np.full(len(vals), i) + jitter, vals,
                       color=color, alpha=0.3, s=8, edgecolors="none", zorder=1)

    bp_top = ax_top.boxplot(
        box_data_top, positions=range(len(box_labels_top)), widths=0.5,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=2.0, zorder=4),
        whiskerprops=whisker_style, capprops=cap_style, zorder=3,
    )
    for patch, color in zip(bp_top["boxes"], colors_top):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        patch.set_zorder(3)

    ax_top.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
    ax_top.axvline(3.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax_top.set_xticks(range(len(box_labels_top)))
    ax_top.set_xticklabels(box_labels_top, fontsize=10, fontweight="bold",
                           rotation=25, ha="right")
    ax_top.set_ylabel("Recognition Accuracy", fontsize=11)
    ax_top.set_title("(a) Performance Distribution Across Operationalizations",
                     fontsize=12, fontweight="bold")
    ax_top.set_ylim(-0.05, 1.05)

    for i, vals in enumerate(box_data_top):
        ax_top.text(i, -0.01, f"n={len(vals)}", ha="center", va="top", fontsize=7, color="gray")

    # ── Bottom panel: per-model ──
    filtered_models = [(m, d, c) for m, d, c in BOXPLOT_MODELS if m in common_models]

    box_data_bot = []
    box_labels_bot = []
    box_colors_bot = []

    for model_name, display_name, color in filtered_models:
        values = []
        for exp_name in ALL_EXPS:
            values.extend(load_pivot_values(exp_name, model_filter=model_name))
        if not values:
            continue
        box_data_bot.append(values)
        box_labels_bot.append(display_name)
        box_colors_bot.append(color)

    for i, (vals, color) in enumerate(zip(box_data_bot, box_colors_bot)):
        if not vals:
            continue
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax_bot.scatter(np.full(len(vals), i) + jitter, vals,
                       color=color, alpha=0.3, s=10, edgecolors="none", zorder=1)

    bp_bot = ax_bot.boxplot(
        box_data_bot, positions=range(len(box_labels_bot)), widths=0.5,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=2.0, zorder=4),
        whiskerprops=whisker_style, capprops=cap_style, zorder=3,
    )
    for patch, color in zip(bp_bot["boxes"], box_colors_bot):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        patch.set_zorder(3)

    ax_bot.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
    ax_bot.set_xticks(range(len(box_labels_bot)))
    ax_bot.set_xticklabels(box_labels_bot, fontsize=10, fontweight="bold", rotation=25, ha="right")
    ax_bot.set_ylabel("Recognition Accuracy", fontsize=11)
    ax_bot.set_title("(b) Per-Model Performance Distribution",
                     fontsize=12, fontweight="bold")
    ax_bot.set_ylim(-0.05, 1.05)

    for i, vals in enumerate(box_data_bot):
        ax_bot.text(i, -0.01, f"n={len(vals)}", ha="center", va="top", fontsize=7, color="gray")

    path = OUT_DIR / "boxplot_combined_v3.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def fig_boxplot_combined_v4(data):
    """Variant: task OPs restricted to ShareGPT only, datasets restricted to PW only."""
    from self_rec_framework.src.helpers.model_sets import get_model_set

    PW_EXPS = [
        "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst_dr",
        "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
    ]
    ALL_EXPS = [
        "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst_dr",
        "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst_dr",
        "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
        "COLM_02_AT_IND-C_Rec_NPr_FA_Inst",
    ]
    TASK_INDIVIDUAL = {
        "UT PW":  ["ICML_01_UT_PW-Q_Rec_NPr_FA_Inst_dr"],
        "UT IND": ["ICML_02_UT_IND-Q_Rec_NPr_FA_Inst_dr"],
        "AT PW":  ["COLM_01_AT_PW-C_Rec_NPr_FA_Inst"],
        "AT IND": ["COLM_02_AT_IND-C_Rec_NPr_FA_Inst"],
    }
    DS_ORDER = ["ShareGPT", "PKU", "BigCode", "WikiSum"]
    DS_DISPLAY = {"PKU": "PKU", "ShareGPT": "S-GPT", "BigCode": "BCB", "WikiSum": "WS"}

    common_models = set(get_model_set("dr_colm"))

    def _load_filtered(exp_name, dataset_short=None, model_filter=None):
        is_ind = "IND" in exp_name
        values = []
        datasets = {dataset_short: DATASET_PIVOT_PATHS[dataset_short]} if dataset_short else DATASET_PIVOT_PATHS
        for ds_name, ds_path in datasets.items():
            pivot_path = ANALYSIS_DIR / ds_path / exp_name / "recognition_accuracy" / "accuracy_pivot.csv"
            if not pivot_path.exists():
                continue
            df = pd.read_csv(pivot_path, index_col=0)
            evaluators = [model_filter] if model_filter else sorted(common_models & set(df.index))
            generators = sorted(common_models & set(df.columns))
            if is_ind:
                control = {}
                for ev in df.index:
                    if ev in df.columns and pd.notna(df.loc[ev, ev]):
                        control[ev] = df.loc[ev, ev]
                for ev in evaluators:
                    if ev not in df.index or ev not in control:
                        continue
                    c_j = control[ev]
                    for gen in generators:
                        if gen == ev:
                            continue
                        t_ij = df.loc[ev, gen]
                        if pd.notna(t_ij):
                            values.append((t_ij + c_j) / 2)
            else:
                for ev in evaluators:
                    if ev not in df.index:
                        continue
                    for gen in generators:
                        if gen == ev:
                            continue
                        val = df.loc[ev, gen]
                        if pd.notna(val):
                            values.append(val)
        return values

    whisker_style = dict(color="black", linewidth=1.2, zorder=3)
    cap_style = dict(color="black", linewidth=1.2, zorder=3)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8),
                                          gridspec_kw={"hspace": 0.25})

    # ── Top panel: task OPs (ShareGPT only) + datasets (PW only) ──
    task_labels = list(TASK_INDIVIDUAL.keys())
    ds_labels = [DS_DISPLAY[d] for d in DS_ORDER]
    box_labels_top = task_labels + ds_labels
    box_data_top = []

    for group_name, exp_list in TASK_INDIVIDUAL.items():
        values = []
        for exp_name in exp_list:
            values.extend(_load_filtered(exp_name, dataset_short="ShareGPT"))
        box_data_top.append(values)

    for ds_name in DS_ORDER:
        values = []
        for exp_name in PW_EXPS:
            values.extend(_load_filtered(exp_name, dataset_short=ds_name))
        box_data_top.append(values)

    task_color = "#4C72B0"
    dataset_color = "#55A868"
    colors_top = [task_color] * 4 + [dataset_color] * 4

    for i, (vals, color) in enumerate(zip(box_data_top, colors_top)):
        if not vals:
            continue
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax_top.scatter(np.full(len(vals), i) + jitter, vals,
                       color=color, alpha=0.3, s=8, edgecolors="none", zorder=1)

    bp_top = ax_top.boxplot(
        box_data_top, positions=range(len(box_labels_top)), widths=0.5,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=2.0, zorder=4),
        whiskerprops=whisker_style, capprops=cap_style, zorder=3,
    )
    for patch, color in zip(bp_top["boxes"], colors_top):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        patch.set_zorder(3)

    ax_top.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
    ax_top.axvline(3.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax_top.set_xticks(range(len(box_labels_top)))
    ax_top.set_xticklabels(box_labels_top, fontsize=10, fontweight="bold",
                           rotation=25, ha="right")
    ax_top.set_ylabel("Recognition Accuracy", fontsize=11)
    ax_top.set_title("(a) Performance Distribution Across Operationalizations",
                     fontsize=12, fontweight="bold")
    ax_top.set_ylim(-0.05, 1.05)

    for i, vals in enumerate(box_data_top):
        ax_top.text(i, -0.01, f"n={len(vals)}", ha="center", va="top", fontsize=7, color="gray")

    # ── Bottom panel: per-model ──
    filtered_models = [(m, d, c) for m, d, c in BOXPLOT_MODELS if m in common_models]

    box_data_bot = []
    box_labels_bot = []
    box_colors_bot = []

    for model_name, display_name, color in filtered_models:
        values = []
        for exp_name in ALL_EXPS:
            values.extend(load_pivot_values(exp_name, model_filter=model_name))
        if not values:
            continue
        box_data_bot.append(values)
        box_labels_bot.append(display_name)
        box_colors_bot.append(color)

    for i, (vals, color) in enumerate(zip(box_data_bot, box_colors_bot)):
        if not vals:
            continue
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax_bot.scatter(np.full(len(vals), i) + jitter, vals,
                       color=color, alpha=0.3, s=10, edgecolors="none", zorder=1)

    bp_bot = ax_bot.boxplot(
        box_data_bot, positions=range(len(box_labels_bot)), widths=0.5,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=2.0, zorder=4),
        whiskerprops=whisker_style, capprops=cap_style, zorder=3,
    )
    for patch, color in zip(bp_bot["boxes"], box_colors_bot):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        patch.set_zorder(3)

    ax_bot.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
    ax_bot.set_xticks(range(len(box_labels_bot)))
    ax_bot.set_xticklabels(box_labels_bot, fontsize=10, fontweight="bold", rotation=25, ha="right")
    ax_bot.set_ylabel("Recognition Accuracy", fontsize=11)
    ax_bot.set_title("(b) Per-Model Performance Distribution",
                     fontsize=12, fontweight="bold")
    ax_bot.set_ylim(-0.05, 1.05)

    for i, vals in enumerate(box_data_bot):
        ax_bot.text(i, -0.01, f"n={len(vals)}", ha="center", va="top", fontsize=7, color="gray")

    path = OUT_DIR / "boxplot_combined_v4.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def fig_boxplot_with_grouped_bar(data):
    """V3 boxplot (panel a) + ICML_07 grouped bar chart (panel b)."""
    from self_rec_framework.src.helpers.model_sets import get_model_set
    from self_rec_framework.src.helpers.model_names import LM_ARENA_SCORES
    from self_rec_framework.scripts.utils import get_model_provider, provider_to_model_name
    from self_rec_framework.scripts.analysis.experiment_contrast import get_family_base_color

    # ═══════════ Panel (a): v3 boxplot (same as fig_boxplot_combined_v3) ═══════════
    PW_EXPS = [
        "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst_dr",
        "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
    ]
    ALL_EXPS = [
        "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst_dr",
        "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst_dr",
        "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
        "COLM_02_AT_IND-C_Rec_NPr_FA_Inst",
    ]
    TASK_INDIVIDUAL = {
        "UT PW":  ["ICML_01_UT_PW-Q_Rec_NPr_FA_Inst_dr"],
        "UT IND": ["ICML_02_UT_IND-Q_Rec_NPr_FA_Inst_dr"],
        "AT PW":  ["COLM_01_AT_PW-C_Rec_NPr_FA_Inst"],
        "AT IND": ["COLM_02_AT_IND-C_Rec_NPr_FA_Inst"],
    }
    DS_ORDER = ["ShareGPT", "PKU", "BigCode", "WikiSum"]
    DS_DISPLAY = {"PKU": "PKU", "ShareGPT": "S-GPT", "BigCode": "BCB", "WikiSum": "WS"}

    common_models = set(get_model_set("dr_colm"))

    def _load_filtered(exp_name, dataset_short=None, model_filter=None):
        is_ind = "IND" in exp_name
        values = []
        datasets = {dataset_short: DATASET_PIVOT_PATHS[dataset_short]} if dataset_short else DATASET_PIVOT_PATHS
        for ds_name, ds_path in datasets.items():
            pivot_path = ANALYSIS_DIR / ds_path / exp_name / "recognition_accuracy" / "accuracy_pivot.csv"
            if not pivot_path.exists():
                continue
            df = pd.read_csv(pivot_path, index_col=0)
            evaluators = [model_filter] if model_filter else sorted(common_models & set(df.index))
            generators = sorted(common_models & set(df.columns))
            if is_ind:
                control = {}
                for ev in df.index:
                    if ev in df.columns and pd.notna(df.loc[ev, ev]):
                        control[ev] = df.loc[ev, ev]
                for ev in evaluators:
                    if ev not in df.index or ev not in control:
                        continue
                    c_j = control[ev]
                    for gen in generators:
                        if gen == ev:
                            continue
                        t_ij = df.loc[ev, gen]
                        if pd.notna(t_ij):
                            values.append((t_ij + c_j) / 2)
            else:
                for ev in evaluators:
                    if ev not in df.index:
                        continue
                    for gen in generators:
                        if gen == ev:
                            continue
                        val = df.loc[ev, gen]
                        if pd.notna(val):
                            values.append(val)
        return values

    whisker_style = dict(color="black", linewidth=1.2, zorder=3)
    cap_style = dict(color="black", linewidth=1.2, zorder=3)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 9),
                                          gridspec_kw={"hspace": 0.20, "height_ratios": [1, 1.3]})

    # ── Panel (a): boxplots ──
    task_labels = list(TASK_INDIVIDUAL.keys())
    ds_labels = [DS_DISPLAY[d] for d in DS_ORDER]
    box_labels_top = task_labels + ds_labels
    box_data_top = []

    for group_name, exp_list in TASK_INDIVIDUAL.items():
        values = []
        for exp_name in exp_list:
            values.extend(_load_filtered(exp_name))
        box_data_top.append(values)

    for ds_name in DS_ORDER:
        values = []
        for exp_name in PW_EXPS:
            values.extend(_load_filtered(exp_name, dataset_short=ds_name))
        box_data_top.append(values)

    task_color = "#7B1FA2"
    # Shared dataset colors across both panels
    ds_bar_colors = {
        "WikiSum": "#1565C0", "ShareGPT": "#FF8F00",
        "PKU-SafeRLHF": "#43A047", "BigCodeBench": "#E57373",
    }
    # Match DS_ORDER keys to bar color keys
    DS_TO_BAR_KEY = {"ShareGPT": "ShareGPT", "PKU": "PKU-SafeRLHF", "BigCode": "BigCodeBench", "WikiSum": "WikiSum"}
    ds_colors_ordered = [ds_bar_colors[DS_TO_BAR_KEY[ds]] for ds in DS_ORDER]
    colors_top = [task_color] * 4 + ds_colors_ordered

    for i, (vals, color) in enumerate(zip(box_data_top, colors_top)):
        if not vals:
            continue
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax_top.scatter(np.full(len(vals), i) + jitter, vals,
                       color=color, alpha=0.3, s=8, edgecolors="none", zorder=1)

    bp_top = ax_top.boxplot(
        box_data_top, positions=range(len(box_labels_top)), widths=0.5,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=2.0, zorder=4),
        whiskerprops=whisker_style, capprops=cap_style, zorder=3,
    )
    for patch, color in zip(bp_top["boxes"], colors_top):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        patch.set_zorder(3)

    ax_top.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
    ax_top.axvline(3.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax_top.set_xticks(range(len(box_labels_top)))
    ax_top.set_xticklabels(box_labels_top, fontsize=13, fontweight="bold",
                           rotation=0, ha="center")
    ax_top.set_ylabel("Recognition Accuracy", fontsize=16)
    ax_top.set_title("(a) Performance Distribution Across Operationalizations",
                     fontsize=17, fontweight="bold")
    ax_top.set_ylim(-0.05, 1.05)
    ax_top.tick_params(axis="y", labelsize=13)

    for i, vals in enumerate(box_data_top):
        ax_top.text(i, -0.01, f"n={len(vals)}", ha="center", va="top", fontsize=9, color="gray")

    # ═══════════ Panel (b): ICML_07 grouped bar chart ═══════════
    perf_df = pd.read_csv(
        AGG_DIR / "ICML_07_UT_PW-Q_Rec_NPr_FA_Rsn-Inst" / "20260202_200109" / "aggregated_performance.csv",
        index_col=0,
    )
    counts_df = pd.read_csv(
        AGG_DIR / "ICML_07_UT_PW-Q_Rec_NPr_FA_Rsn-Inst" / "20260202_200109" / "aggregated_counts.csv",
        index_col=0,
    )

    # Short dataset names
    ds_short_map = {}
    for col in perf_df.columns:
        short = col.split("/")[0]
        ds_short_map[col] = {
            "wikisum": "WikiSum", "sharegpt": "ShareGPT",
            "pku_saferlhf": "PKU-SafeRLHF", "bigcodebench": "BigCodeBench",
        }.get(short, short)

    ds_display_order = ["WikiSum", "ShareGPT", "PKU-SafeRLHF", "BigCodeBench"]
    ds_bar_hatches = {
        "WikiSum": "", "ShareGPT": "",
        "PKU-SafeRLHF": "", "BigCodeBench": "",
    }

    # Sort models by Elo
    def _elo(m):
        return LM_ARENA_SCORES.get(m) or LM_ARENA_SCORES.get(m.replace("-thinking", ""), 0) or 0
    sorted_models = sorted(perf_df.index, key=_elo)

    # Display names
    MODEL_DISPLAY = {
        "ll-3.1-8b": "Llama 8B", "ll-3.1-70b": "Llama 70B", "ll-3.1-405b": "Llama 405B",
        "qwen-2.5-7b": "Qwen 7B", "qwen-2.5-72b": "Qwen 72B", "qwen-3.0-80b": "Qwen 80B",
        "qwen-3.0-80b-thinking": "Qwen 80B (R)", "gpt-4o-mini": "GPT-4o Mini",
        "gpt-4.1-mini": "GPT-4.1 Mini", "gpt-4o": "GPT-4o", "gpt-4.1": "GPT-4.1",
        "haiku-3.5": "Haiku 3.5", "sonnet-3.7": "Sonnet 3.7", "sonnet-4.5": "Sonnet 4.5",
        "opus-4.1": "Opus 4.1", "opus-4.1-thinking": "Opus 4.1 (R)",
        "gemini-2.0-flash-lite": "Flash Lite", "gemini-2.0-flash": "Flash 2.0",
        "gemini-2.5-pro-thinking": "Gemini Pro (R)",
        "deepseek-3.1": "DS 3.1", "deepseek-r1-thinking": "DS-R1 (R)",
        "kimi-k2": "Kimi K2", "kimi-k2-thinking": "Kimi K2 (R)",
        "gpt-oss-120b-thinking": "GPT-OSS 120B (R)",
    }

    n_models = len(sorted_models)
    n_datasets = len(ds_display_order)
    bar_width = 0.8 / n_datasets
    x_positions = np.arange(n_models)

    for d_idx, ds_name in enumerate(ds_display_order):
        # Find the column matching this dataset
        col = None
        for c, s in ds_short_map.items():
            if s == ds_name:
                col = c
                break
        if col is None:
            continue

        vals = [perf_df.loc[m, col] if m in perf_df.index and pd.notna(perf_df.loc[m, col]) else 0
                for m in sorted_models]
        offset = (d_idx - n_datasets / 2 + 0.5) * bar_width
        bars = ax_bot.bar(x_positions + offset, vals, bar_width * 0.9,
                          color=ds_bar_colors[ds_name], alpha=0.85,
                          edgecolor="black", linewidth=0.4,
                          hatch=ds_bar_hatches[ds_name], label=ds_name, zorder=2)

        # Significance asterisks (binomial test, two-sided, α=0.05)
        from scipy.stats import binomtest
        for m_idx, model in enumerate(sorted_models):
            acc = perf_df.loc[model, col] if model in perf_df.index and pd.notna(perf_df.loc[model, col]) else None
            n_samples = counts_df.loc[model, col] if model in counts_df.index and col in counts_df.columns and pd.notna(counts_df.loc[model, col]) else None
            if acc is not None and n_samples is not None and n_samples > 0:
                n = int(n_samples)
                k = round(acc * n)
                p = binomtest(k, n, 0.5).pvalue
                if p < 0.05:
                    bar_x = x_positions[m_idx] + offset
                    ax_bot.text(bar_x, acc + 0.01, "*", ha="center", va="bottom",
                                fontsize=12, fontweight="bold", color="black", zorder=5)

    ax_bot.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)
    ax_bot.set_xticks(x_positions)
    ax_bot.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in sorted_models],
                           fontsize=13, fontweight="bold", rotation=35, ha="right")

    # Color x-tick labels by model family (darkened for readability)
    def _darken(hex_color, factor=0.65):
        h = hex_color.lstrip("#")
        r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"

    for tick_label, model in zip(ax_bot.get_xticklabels(), sorted_models):
        tick_label.set_color(_darken(get_family_base_color(model)))

    ax_bot.set_ylabel("Recognition Accuracy", fontsize=16)
    ax_bot.set_title("(b) Per-Model Pairwise Recognition (UT) by Dataset",
                     fontsize=17, fontweight="bold")
    ax_bot.set_ylim(0, 1.05)
    ax_bot.tick_params(axis="y", labelsize=13)
    ax_bot.legend(fontsize=13, loc="upper left", ncol=2)
    ax_bot.set_xlim(-0.6, n_models - 0.4)

    path = OUT_DIR / "boxplot_with_grouped_bar.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def main():
    print("Loading experiment data...")
    data = load_all()
    print(f"\nLoaded {len(data)} experiments.\n")

    print("Generating prototype figures...")
    print("\n1. Heatmap: models × operationalizations")
    fig_heatmap_ops(data)

    print("\n2. Grouped bars: PW vs IND (Instruct)")
    fig_pw_vs_ind_bars(data)

    print("\n3. Dot plot: models × OPs (Rec only)")
    fig_dotplot_ops(data)

    print("\n4a. Full heatmap: 2×2 panel (UT/AT × PW/IND)")
    fig_heatmap_full(data)

    print("\n4b. Full heatmap: 2×2 panel (Rec/Pref × PW/IND)")
    fig_heatmap_rec_vs_pref(data)

    print("\n5. Small multiples: 2×4 grid (PW/IND × datasets)")
    fig_small_multiples(data)

    print("\n6. Summary errorbar: mean ± std with multiple OPs")
    fig_summary_errorbar(data)

    print("\n7. Score distance: 2×2 panel (UT/AT × PW/IND)")
    fig_score_distance_panels(data)

    print("\n8. Rank distance: 2×2 panel (UT/AT × PW/IND)")
    fig_rank_distance_panels(data)

    # NOTE: fig_training_effect_panels moved to prototype_uplift_figures.py

    print("\n10. Boxplots: 8 operationalizations")
    fig_boxplot_operationalizations(data)

    print("\n11. Boxplots: per-model across all OPs")
    fig_boxplot_per_model(data)

    print("\n12. Combined boxplots (operationalizations + per-model)")
    fig_boxplot_combined(data)

    print("\n13. Recognition vs Preference scatter (PW + IND stacked)")
    fig_rec_vs_pref_scatter()

    print(f"\nAll prototypes saved to: {OUT_DIR}/")


def fig_rec_vs_pref_scatter():
    """Vertically stacked scatter: Recognition vs Preference performance.

    Top panel: PW (ICML_01 vs ICML_05)
    Bottom panel: IND (ICML_02 vs ICML_06)

    Replicates the logic from srf-experiment-contrast's plot_performance_scatter
    with both panels sharing a single legend on the right.
    """
    from scipy import stats
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.ticker import MultipleLocator
    from self_rec_framework.scripts.analysis.experiment_contrast import (
        get_family_base_color, extract_dataset_name,
        format_dataset_display_name,
    )
    from self_rec_framework.scripts.utils import (
        get_model_provider, provider_to_model_name,
    )
    from self_rec_framework.scripts.utils import (
        calculate_binomial_ci, weighted_regression_with_ci, weighted_correlation,
    )

    # Data paths (latest timestamps)
    pairs = [
        {
            "rec_dir": "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst/20260128_224112",
            "pref_dir": "ICML_05_UT_PW-Q_Pref-Q_NPr_FA_Inst/20260127_163232",
            "label": "(c)",
        },
        {
            "rec_dir": "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst/20260127_151419",
            "pref_dir": "ICML_06_UT_IND-Q_Pref-Q_NPr_FA_Inst/20260127_163242",
            "label": "(d)",
        },
    ]

    fig, axes = plt.subplots(2, 1, figsize=(8, 16))

    # Shared state for legend
    all_families = set()
    all_datasets = set()
    family_colors_map = {}
    dataset_markers_map = {}
    dataset_line_colors_map = {}
    datasets_with_fit_all = {}

    markers = ['D', '^', 's', 'o', 'v', '<', '>', 'p', '*', 'h']
    ds_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                 "#8c564b", "#e377c2", "#7f7f7f"]

    for p_idx, pair in enumerate(pairs):
        ax = axes[p_idx]

        rec_perf = pd.read_csv(AGG_DIR / pair["rec_dir"] / "aggregated_performance.csv", index_col=0)
        pref_perf = pd.read_csv(AGG_DIR / pair["pref_dir"] / "aggregated_performance.csv", index_col=0)
        rec_counts = pd.read_csv(AGG_DIR / pair["rec_dir"] / "aggregated_counts.csv", index_col=0)
        pref_counts = pd.read_csv(AGG_DIR / pair["pref_dir"] / "aggregated_counts.csv", index_col=0)

        # Align columns by short dataset name
        def _align_counts(df_perf, df_counts):
            col_map = {}
            for pc in df_perf.columns:
                ps = extract_dataset_name(pc)
                for cc in df_counts.columns:
                    if extract_dataset_name(cc) == ps:
                        col_map[cc] = pc
                        break
            if col_map:
                df_counts = df_counts.rename(columns=col_map)
            return df_counts.reindex(index=df_perf.index, columns=df_perf.columns)

        rec_counts = _align_counts(rec_perf, rec_counts)
        pref_counts = _align_counts(pref_perf, pref_counts)

        # Build plot data
        plot_data = []
        for model in rec_perf.index:
            for dataset in rec_perf.columns:
                rec_val = rec_perf.loc[model, dataset]
                # Match pref column by short name
                ds_short = extract_dataset_name(dataset)
                pref_col = None
                for c in pref_perf.columns:
                    if extract_dataset_name(c) == ds_short:
                        pref_col = c
                        break
                if pref_col is None or model not in pref_perf.index:
                    continue
                pref_val = pref_perf.loc[model, pref_col]
                if pd.isna(rec_val) or pd.isna(pref_val):
                    continue

                # Error bars
                rec_err = pref_err = None
                n1 = rec_counts.loc[model, dataset] if dataset in rec_counts.columns and model in rec_counts.index else None
                n2 = pref_counts.loc[model, pref_col] if pref_col in pref_counts.columns and model in pref_counts.index else None
                if pd.notna(n1) and n1 > 0:
                    _, _, se = calculate_binomial_ci(rec_val, int(n1))
                    rec_err = 1.96 * se
                if pd.notna(n2) and n2 > 0:
                    _, _, se = calculate_binomial_ci(pref_val, int(n2))
                    pref_err = 1.96 * se

                plot_data.append({
                    "model": model,
                    "dataset": ds_short,
                    "rec": rec_val,     # y-axis
                    "pref": pref_val,   # x-axis
                    "family": get_model_provider(model),
                    "rec_err": rec_err,
                    "pref_err": pref_err,
                    "n1": int(n1) if pd.notna(n1) else None,
                    "n2": int(n2) if pd.notna(n2) else None,
                })

        if not plot_data:
            continue
        plot_df = pd.DataFrame(plot_data)

        unique_datasets = sorted(plot_df["dataset"].unique())
        unique_families = sorted(plot_df["family"].unique())

        # Assign markers/colors (consistent across panels)
        for i, ds in enumerate(unique_datasets):
            if ds not in dataset_markers_map:
                idx = len(dataset_markers_map)
                dataset_markers_map[ds] = markers[idx % len(markers)]
                dataset_line_colors_map[ds] = ds_colors[idx % len(ds_colors)]
        for fam in unique_families:
            all_families.add(fam)
            if fam not in family_colors_map:
                fam_models = plot_df[plot_df["family"] == fam]["model"].unique()
                family_colors_map[fam] = get_family_base_color(fam_models[0]) if len(fam_models) > 0 else "#9ca3af"

        all_datasets.update(unique_datasets)

        data_min = min(plot_df["rec"].min(), plot_df["pref"].min())
        data_max = max(plot_df["rec"].max(), plot_df["pref"].max())
        line_min = min(-0.05, data_min - 0.05)
        line_max = max(1.05, data_max + 0.05)

        datasets_with_fit = {}

        for dataset in unique_datasets:
            ds_data = plot_df[plot_df["dataset"] == dataset]
            marker = dataset_markers_map[dataset]

            for fam in ds_data["family"].unique():
                fam_data = ds_data[ds_data["family"] == fam]
                color = family_colors_map[fam]

                xerr = np.array([e if pd.notna(e) and e is not None else 0 for e in fam_data["pref_err"]])
                yerr = np.array([e if pd.notna(e) and e is not None else 0 for e in fam_data["rec_err"]])
                if np.all(xerr == 0):
                    xerr = None
                if np.all(yerr == 0):
                    yerr = None

                ax.errorbar(fam_data["pref"], fam_data["rec"], xerr=xerr, yerr=yerr,
                            fmt="none", ecolor=color, alpha=0.4, capsize=2, capthick=0.5,
                            elinewidth=0.5, zorder=1)
                ax.scatter(fam_data["pref"], fam_data["rec"], c=color, marker=marker,
                           s=100, alpha=0.7, edgecolors="black", linewidths=0.5, zorder=2)

            # Regression
            x_vals = ds_data["pref"].values.astype(float)
            y_vals = ds_data["rec"].values.astype(float)
            if len(x_vals) > 1:
                weights = None
                if "n1" in ds_data.columns and "n2" in ds_data.columns:
                    wl = []
                    for n1, n2 in zip(ds_data["n1"], ds_data["n2"]):
                        n1v = None if (pd.isna(n1) or n1 is None) else float(n1)
                        n2v = None if (pd.isna(n2) or n2 is None) else float(n2)
                        if n1v and n2v and n1v > 0 and n2v > 0:
                            wl.append(np.sqrt(n1v * n2v))
                        elif n1v and n1v > 0:
                            wl.append(n1v)
                        elif n2v and n2v > 0:
                            wl.append(n2v)
                        else:
                            wl.append(np.nan)
                    if not np.all(np.isnan(wl)):
                        weights = np.array(wl)
                        valid_w = weights[~np.isnan(weights)]
                        if len(valid_w) > 0:
                            weights = np.clip(weights, np.percentile(valid_w, 5), np.percentile(valid_w, 95))

                line_color = dataset_line_colors_map[dataset]
                valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
                if weights is not None:
                    valid_mask &= ~np.isnan(weights)

                if np.sum(valid_mask) >= 2:
                    xf, yf = x_vals[valid_mask], y_vals[valid_mask]
                    wf = weights[valid_mask] if weights is not None else None

                    reg = weighted_regression_with_ci(xf, yf, weights=wf, x_min=line_min, x_max=line_max)
                    if reg:
                        corr = weighted_correlation(xf, yf, wf) if wf is not None else stats.pearsonr(xf, yf)[0]
                        datasets_with_fit[dataset] = corr
                        ax.fill_between(reg["x"], reg["ci_lower"], reg["ci_upper"],
                                        color=line_color, alpha=0.15, zorder=0)
                        ax.plot(reg["x"], reg["y_pred"], color=line_color, linestyle="--",
                                linewidth=1.5, alpha=0.7, zorder=1)
                    else:
                        slope, intercept, r_value, _, _ = stats.linregress(xf, yf)
                        datasets_with_fit[dataset] = r_value
                        xl = np.linspace(line_min, line_max, 100)
                        ax.plot(xl, slope * xl + intercept, color=line_color,
                                linestyle="--", linewidth=1.5, alpha=0.7)

        # Store for legend
        for ds, corr in datasets_with_fit.items():
            key = (p_idx, ds)
            datasets_with_fit_all[key] = corr

        # Reference lines
        ax.plot([line_min, line_max], [line_min, line_max], color="#888888",
                linestyle=":", linewidth=2, alpha=0.7)
        ax.axhline(y=0.5, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axvline(x=0.5, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)

        ax.set_xlabel("Preference Score", fontsize=14)
        ax.set_ylabel("Performance Score", fontsize=14)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=12)

        # Panel label
        ax.text(0.02, 0.98, pair["label"], transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="top", ha="left")

        # Per-panel dataset legend (inside plot, bottom-right)
        ds_handles = []
        for ds in sorted(unique_datasets):
            marker = dataset_markers_map[ds]
            lc = dataset_line_colors_map[ds]
            h_marker = plt.Line2D([0], [0], marker=marker, color="w",
                                  markerfacecolor="gray", markersize=8,
                                  markeredgecolor="black", markeredgewidth=0.5)
            h_line = plt.Line2D([0], [0], linestyle="--", color=lc, linewidth=2)
            key = (p_idx, ds)
            r_str = f"r={datasets_with_fit_all[key]:.2f}" if key in datasets_with_fit_all else ""
            ds_handles.append(((h_marker, h_line), f"{format_dataset_display_name(ds)} ({r_str})"))

        ds_leg = ax.legend(
            handles=[h for h, _ in ds_handles],
            labels=[l for _, l in ds_handles],
            loc="lower right", fontsize=9, framealpha=0.9,
            handler_map={tuple: HandlerTuple(ndivide=None)},
        )
        ax.add_artist(ds_leg)

    # Shared legend on the right for model families + misc
    def _title_h(t):
        return plt.Line2D([], [], linestyle="", marker="", label=t)

    fam_handles = [_title_h("Model Name")]
    for fam in sorted(all_families):
        display = provider_to_model_name(fam)
        fam_handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                       markerfacecolor=family_colors_map[fam],
                                       markersize=10, markeredgecolor="black",
                                       markeredgewidth=0.5, label=display))

    misc_handles = [
        _title_h(""),
        plt.Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.0,
                   alpha=0.8, label="Chance (0.5)"),
        plt.Line2D([0], [0], color="#888888", linestyle=":", linewidth=2,
                   alpha=0.7, label="1:1 line"),
    ]

    all_handles = fam_handles + misc_handles
    all_labels = [h.get_label() for h in all_handles]

    fig.legend(handles=all_handles, labels=all_labels,
              loc="center right", bbox_to_anchor=(1.18, 0.5),
              fontsize=10, framealpha=0.9, borderpad=1.0,
              labelspacing=1.0)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    path = OUT_DIR / "rec_vs_pref_scatter.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def fig_quality_heuristic_combined(data=None):
    """2×3 combined figure: score-distance scatter (cols 0-1) + rec-vs-pref scatter (col 2).

    Row 0: PW panels — (a) UT-PW score-distance, (b) UT-IND score-distance, (c) rec vs pref PW
    Row 1: IND panels — (c) AT-PW score-distance, (d) AT-IND score-distance, (d) rec vs pref IND
    """
    from self_rec_framework.src.helpers.model_names import LM_ARENA_SCORES
    from scipy import stats
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.ticker import MultipleLocator
    from self_rec_framework.scripts.analysis.experiment_contrast import (
        get_family_base_color, extract_dataset_name,
        format_dataset_display_name,
    )
    from self_rec_framework.scripts.utils import (
        get_model_provider, provider_to_model_name,
        calculate_binomial_ci, weighted_regression_with_ci, weighted_correlation,
    )
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(20, 12))
    # Outer: 1×2 — left (score-distance 2×2) and right (scatter 2×1)
    # wspace gives room for the dotted separator + Performance Score y-label
    outer = GridSpec(1, 2, figure=fig, width_ratios=[2, 1], wspace=0.10,
                     left=0.05, right=0.97, top=0.95, bottom=0.08)
    gs_left = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], hspace=0.08, wspace=0.14)
    gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.08)

    axes_left = np.array([
        [fig.add_subplot(gs_left[0, 0]), fig.add_subplot(gs_left[0, 1])],
        [fig.add_subplot(gs_left[1, 0]), fig.add_subplot(gs_left[1, 1])],
    ])
    axes_right = [fig.add_subplot(gs_right[0, 0]), fig.add_subplot(gs_right[1, 0])]

    # ──────────────────────────────────────────────────────────────────
    # LEFT 2×2: Score distance panels (same logic as fig_score_distance_panels)
    # ──────────────────────────────────────────────────────────────────
    sd_panels = {
        "(a) User-Tag — Pairwise": "ICML_07_UT_PW-Q_Rec_NPr_FA_Rsn-Inst",
        "(b) User-Tag — Individual": "ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst",
        "(c) Assistant-Tag — Pairwise": "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
        "(d) Assistant-Tag — Individual": "COLM_02_AT_IND-C_Rec_NPr_FA_Inst",
    }

    panel_dfs = {}
    for title, exp_name in sd_panels.items():
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

        def get_score(model):
            if model in LM_ARENA_SCORES:
                return LM_ARENA_SCORES[model]
            base = model.replace("-thinking", "")
            if base in LM_ARENA_SCORES:
                return LM_ARENA_SCORES[base]
            return None

        df["eval_score"] = df["evaluator"].apply(get_score)
        df["gen_score"] = df["generator"].apply(get_score)
        df = df.dropna(subset=["eval_score", "gen_score"])
        df["score_distance"] = df["eval_score"] - df["gen_score"]

        self_scores = load_self_scores(exp_name)
        if self_scores is not None:
            df = adjust_ind_performance(df, self_scores)

        panel_dfs[title] = df

    sd_ds_colors = {
        "wikisum": "#2196F3", "sharegpt": "#FF9800",
        "pku_saferlhf": "#4CAF50", "bigcodebench": "#E91E63",
    }
    sd_ds_labels = {
        "wikisum": "WikiSum", "sharegpt": "ShareGPT",
        "pku_saferlhf": "PKU", "bigcodebench": "BigCode",
    }

    sd_titles = list(sd_panels.keys())
    is_ind = {1, 3}
    for idx, title in enumerate(sd_titles):
        row, col = divmod(idx, 2)
        ax = axes_left[row][col]
        df = panel_dfs[title]
        dataset_names = sorted(df["dataset"].unique())

        for ds_name in dataset_names:
            ds_data = df[df["dataset"] == ds_name]
            if ds_data.empty:
                continue
            ax.scatter(
                ds_data["score_distance"], ds_data["performance"],
                c=sd_ds_colors.get(ds_name, "gray"),
                label=sd_ds_labels.get(ds_name, ds_name),
                alpha=0.4, s=25, edgecolors="none",
            )

        agg = df.groupby(["evaluator", "generator"]).agg(
            score_distance=("score_distance", "first"),
            performance=("performance", "mean"),
            weight=("n_samples", "sum"),
        ).reset_index()

        if len(agg) > 2:
            w = agg["weight"].values
            x_vals = agg["score_distance"].values
            y_vals = agg["performance"].values
            coeffs = np.polyfit(x_vals, y_vals, 1, w=np.sqrt(w))
            slope, intercept = coeffs
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color="black", linewidth=4.5, alpha=0.8)

            rng = np.random.default_rng(42)
            n_boot = 500
            boot_lines = np.zeros((n_boot, len(x_line)))
            for b in range(n_boot):
                idx_b = rng.choice(len(x_vals), size=len(x_vals), replace=True)
                c = np.polyfit(x_vals[idx_b], y_vals[idx_b], 1, w=np.sqrt(w[idx_b]))
                boot_lines[b] = c[0] * x_line + c[1]
            lo = np.percentile(boot_lines, 2.5, axis=0)
            hi = np.percentile(boot_lines, 97.5, axis=0)
            ax.fill_between(x_line, lo, hi, color="black", alpha=0.1, zorder=1)

            r, p = stats.pearsonr(x_vals, y_vals)
            ax.text(0.03, 0.03, f"r = {r:.2f}", transform=ax.transAxes, fontsize=18,
                    verticalalignment="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(axis="both", labelsize=21)

        y_label = "Adjusted Accuracy" if idx in is_ind else "Recognition Accuracy"
        if col == 0:
            ax.set_ylabel(y_label, fontsize=23)
        if row == 1:
            ax.set_xlabel("Elo Score Distance", fontsize=23)
        else:
            ax.set_xticklabels([])
        if idx == 0:
            ax.legend(fontsize=12, loc="lower right", markerscale=1.5)

    # ──────────────────────────────────────────────────────────────────
    # RIGHT 2×1: Rec vs Pref scatter (same logic as fig_rec_vs_pref_scatter)
    # ──────────────────────────────────────────────────────────────────
    rp_pairs = [
        {
            "rec_dir": "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst/20260128_224112",
            "pref_dir": "ICML_05_UT_PW-Q_Pref-Q_NPr_FA_Inst/20260127_163232",
            "label": "(e)",
        },
        {
            "rec_dir": "ICML_02_UT_IND-Q_Rec_NPr_FA_Inst/20260127_151419",
            "pref_dir": "ICML_06_UT_IND-Q_Pref-Q_NPr_FA_Inst/20260127_163242",
            "label": "(f)",
        },
    ]

    rp_markers = ['D', '^', 's', 'o']
    rp_ds_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    rp_dataset_markers = {}
    rp_dataset_line_colors = {}
    all_families = set()
    family_colors = {}

    for p_idx, pair in enumerate(rp_pairs):
        ax = axes_right[p_idx]

        rec_perf = pd.read_csv(AGG_DIR / pair["rec_dir"] / "aggregated_performance.csv", index_col=0)
        pref_perf = pd.read_csv(AGG_DIR / pair["pref_dir"] / "aggregated_performance.csv", index_col=0)
        rec_counts = pd.read_csv(AGG_DIR / pair["rec_dir"] / "aggregated_counts.csv", index_col=0)
        pref_counts = pd.read_csv(AGG_DIR / pair["pref_dir"] / "aggregated_counts.csv", index_col=0)

        def _align_counts(df_perf, df_counts):
            col_map = {}
            for pc in df_perf.columns:
                ps = extract_dataset_name(pc)
                for cc in df_counts.columns:
                    if extract_dataset_name(cc) == ps:
                        col_map[cc] = pc
                        break
            if col_map:
                df_counts = df_counts.rename(columns=col_map)
            return df_counts.reindex(index=df_perf.index, columns=df_perf.columns)

        rec_counts = _align_counts(rec_perf, rec_counts)
        pref_counts = _align_counts(pref_perf, pref_counts)

        plot_data = []
        for model in rec_perf.index:
            for dataset in rec_perf.columns:
                rec_val = rec_perf.loc[model, dataset]
                ds_short = extract_dataset_name(dataset)
                pref_col = next((c for c in pref_perf.columns if extract_dataset_name(c) == ds_short), None)
                if pref_col is None or model not in pref_perf.index:
                    continue
                pref_val = pref_perf.loc[model, pref_col]
                if pd.isna(rec_val) or pd.isna(pref_val):
                    continue

                rec_err = pref_err = None
                n1 = rec_counts.loc[model, dataset] if dataset in rec_counts.columns and model in rec_counts.index else None
                n2 = pref_counts.loc[model, pref_col] if pref_col in pref_counts.columns and model in pref_counts.index else None
                if pd.notna(n1) and n1 > 0:
                    _, _, se = calculate_binomial_ci(rec_val, int(n1))
                    rec_err = 1.96 * se
                if pd.notna(n2) and n2 > 0:
                    _, _, se = calculate_binomial_ci(pref_val, int(n2))
                    pref_err = 1.96 * se

                plot_data.append({
                    "model": model, "dataset": ds_short,
                    "rec": rec_val, "pref": pref_val,
                    "family": get_model_provider(model),
                    "rec_err": rec_err, "pref_err": pref_err,
                    "n1": int(n1) if pd.notna(n1) else None,
                    "n2": int(n2) if pd.notna(n2) else None,
                })

        if not plot_data:
            continue
        plot_df = pd.DataFrame(plot_data)
        unique_datasets = sorted(plot_df["dataset"].unique())
        unique_fams = sorted(plot_df["family"].unique())

        for i, ds in enumerate(unique_datasets):
            if ds not in rp_dataset_markers:
                idx = len(rp_dataset_markers)
                rp_dataset_markers[ds] = rp_markers[idx % len(rp_markers)]
                rp_dataset_line_colors[ds] = rp_ds_colors[idx % len(rp_ds_colors)]
        for fam in unique_fams:
            all_families.add(fam)
            if fam not in family_colors:
                fm = plot_df[plot_df["family"] == fam]["model"].unique()
                family_colors[fam] = get_family_base_color(fm[0]) if len(fm) > 0 else "#9ca3af"

        data_min = min(plot_df["rec"].min(), plot_df["pref"].min())
        data_max = max(plot_df["rec"].max(), plot_df["pref"].max())
        line_min = min(-0.05, data_min - 0.05)
        line_max = max(1.05, data_max + 0.05)

        datasets_with_fit = {}
        for dataset in unique_datasets:
            ds_data = plot_df[plot_df["dataset"] == dataset]
            marker = rp_dataset_markers[dataset]
            for fam in ds_data["family"].unique():
                fam_data = ds_data[ds_data["family"] == fam]
                color = family_colors[fam]
                xerr = np.array([e if pd.notna(e) and e is not None else 0 for e in fam_data["pref_err"]])
                yerr = np.array([e if pd.notna(e) and e is not None else 0 for e in fam_data["rec_err"]])
                ax.errorbar(fam_data["pref"], fam_data["rec"],
                            xerr=xerr if not np.all(xerr == 0) else None,
                            yerr=yerr if not np.all(yerr == 0) else None,
                            fmt="none", ecolor=color, alpha=0.4, capsize=2,
                            capthick=0.5, elinewidth=0.5, zorder=1)
                ax.scatter(fam_data["pref"], fam_data["rec"], c=color, marker=marker,
                           s=140, alpha=0.7, edgecolors="black", linewidths=0.5, zorder=2)

            x_vals = ds_data["pref"].values.astype(float)
            y_vals = ds_data["rec"].values.astype(float)
            if len(x_vals) > 1:
                weights = None
                if "n1" in ds_data.columns and "n2" in ds_data.columns:
                    wl = []
                    for n1, n2 in zip(ds_data["n1"], ds_data["n2"]):
                        n1v = None if (pd.isna(n1) or n1 is None) else float(n1)
                        n2v = None if (pd.isna(n2) or n2 is None) else float(n2)
                        if n1v and n2v and n1v > 0 and n2v > 0:
                            wl.append(np.sqrt(n1v * n2v))
                        elif n1v and n1v > 0:
                            wl.append(n1v)
                        elif n2v and n2v > 0:
                            wl.append(n2v)
                        else:
                            wl.append(np.nan)
                    if not np.all(np.isnan(wl)):
                        weights = np.array(wl)
                        vw = weights[~np.isnan(weights)]
                        if len(vw) > 0:
                            weights = np.clip(weights, np.percentile(vw, 5), np.percentile(vw, 95))

                lc = rp_dataset_line_colors[dataset]
                vm = ~(np.isnan(x_vals) | np.isnan(y_vals))
                if weights is not None:
                    vm &= ~np.isnan(weights)
                if np.sum(vm) >= 2:
                    xf, yf = x_vals[vm], y_vals[vm]
                    wf = weights[vm] if weights is not None else None
                    reg = weighted_regression_with_ci(xf, yf, weights=wf, x_min=line_min, x_max=line_max)
                    if reg:
                        corr = weighted_correlation(xf, yf, wf) if wf is not None else stats.pearsonr(xf, yf)[0]
                        datasets_with_fit[dataset] = corr
                        ax.fill_between(reg["x"], reg["ci_lower"], reg["ci_upper"],
                                        color=lc, alpha=0.15, zorder=0)
                        ax.plot(reg["x"], reg["y_pred"], color=lc, linestyle="--",
                                linewidth=3.5, alpha=0.7, zorder=1)
                    else:
                        slope, intercept, r_value, _, _ = stats.linregress(xf, yf)
                        datasets_with_fit[dataset] = r_value
                        xl = np.linspace(line_min, line_max, 100)
                        ax.plot(xl, slope * xl + intercept, color=lc, linestyle="--",
                                linewidth=3.5, alpha=0.7)

        ax.plot([line_min, line_max], [line_min, line_max], color="#888888",
                linestyle=":", linewidth=4, alpha=0.7)
        ax.axhline(y=0.5, color="#555555", linestyle="--", linewidth=2.0, alpha=0.8)
        ax.axvline(x=0.5, color="#555555", linestyle="--", linewidth=2.0, alpha=0.8)

        ax.set_ylabel("Recognition Accuracy", fontsize=23)
        if p_idx == 1:
            ax.set_xlabel("Self-Selection Rate", fontsize=23)
        else:
            ax.set_xticklabels([])
        rp_panel_titles = ["(e) Pairwise — Rec. vs Pref.", "(f) Individual — Rec. vs Pref."]
        ax.set_title(rp_panel_titles[p_idx], fontsize=20, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=18)

        # Per-panel dataset legend (top-left for panel e, lower-right for panel f)
        ds_handles = []
        for ds in sorted(unique_datasets):
            mk = rp_dataset_markers[ds]
            lc = rp_dataset_line_colors[ds]
            h_m = plt.Line2D([0], [0], marker=mk, color="w", markerfacecolor="gray",
                             markersize=10, markeredgecolor="black", markeredgewidth=0.5)
            h_l = plt.Line2D([0], [0], linestyle="--", color=lc, linewidth=2)
            r_str = f"r={datasets_with_fit[ds]:.2f}" if ds in datasets_with_fit else ""
            ds_handles.append(((h_m, h_l), f"{format_dataset_display_name(ds)} ({r_str})"))
        ds_leg = ax.legend(handles=[h for h, _ in ds_handles], labels=[l for _, l in ds_handles],
                           loc="upper left", fontsize=13, framealpha=0.9,
                           handler_map={tuple: HandlerTuple(ndivide=None)})
        ax.add_artist(ds_leg)

    # ──────────────────────────────────────────────────────────────────
    # Shared legend for model families (right side)
    # ──────────────────────────────────────────────────────────────────
    fam_handles = []
    for fam in sorted(all_families):
        display = provider_to_model_name(fam)
        fam_handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                       markerfacecolor=family_colors[fam],
                                       markersize=12, markeredgecolor="black",
                                       markeredgewidth=0.5, label=display))

    # Place model family legend at bottom-right of panel (f)
    axes_right[1].legend(handles=fam_handles, loc="lower right",
                         fontsize=13, framealpha=0.9,
                         borderpad=0.8, labelspacing=0.5)

    # Vertical dotted separator — placed just left of the Performance Score y-label
    left_right_edge = axes_left[0][1].get_position().x1
    right_left_edge = axes_right[0].get_position().x0
    sep_x = left_right_edge + (right_left_edge - left_right_edge) * 0.3
    fig.add_artist(plt.Line2D([sep_x, sep_x], [0.05, 0.95], transform=fig.transFigure,
                              color="gray", linestyle=":", linewidth=1.5, zorder=10))

    path = OUT_DIR / "quality_heuristic_combined.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


if __name__ == "__main__":
    main()
