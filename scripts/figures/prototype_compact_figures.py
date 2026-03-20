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


# ============================================================================
# FIGURE 9: Training effect — single evaluator score-distance overlay
# ============================================================================
def fig_training_effect_panels(data, pre_model="ll-3.1-8b", post_model="opus-4.1"):
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
    from scipy import stats

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

    # Load and filter data for both models across all experiments
    panel_data = {}  # title -> {"pre": df, "post": df}
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

        # Compute score distance for all rows
        df["eval_score"] = df["evaluator"].apply(get_score)
        df["gen_score"] = df["generator"].apply(get_score)
        df = df.dropna(subset=["eval_score", "gen_score"])
        df["score_distance"] = df["eval_score"] - df["gen_score"]

        # Apply IND adjustment
        self_scores = load_self_scores(exp_name)
        if self_scores is not None:
            df = adjust_ind_performance(df, self_scores)

        # Filter to single evaluator for pre and post
        pre_df = df[df["evaluator"] == pre_model].copy()
        post_df = df[df["evaluator"] == post_model].copy()

        panel_data[title] = {"pre": pre_df, "post": post_df}

    # Colors
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

        # Background: pre-training data
        if not pre_df.empty:
            ax.scatter(
                pre_df["score_distance"], pre_df["performance"],
                c=pre_color, alpha=0.35, s=20, edgecolors="none",
                label=f"{pre_model} (pre)", zorder=2,
            )
            # Pre trendline
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

        # Foreground: post-training data
        if not post_df.empty:
            ax.scatter(
                post_df["score_distance"], post_df["performance"],
                c=post_color, alpha=0.7, s=25, edgecolors="none",
                label=f"{post_model} (post)", zorder=4,
            )
            # Post trendline
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
                ax.plot(x_line, coeffs_post[0] * x_line + coeffs_post[1],
                        color=post_line_color, linewidth=2, alpha=0.9, zorder=5)

        # Slope annotation
        slopes = []
        if not pre_df.empty and len(pre_agg) > 2:
            slopes.append(f"pre slope: {coeffs_pre[0]:.4f}")
        if not post_df.empty and len(post_agg) > 2:
            slopes.append(f"post slope: {coeffs_post[0]:.4f}")
        if len(slopes) == 2 and len(pre_agg) > 2 and len(post_agg) > 2:
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

    print("\n9. Training effect: score-distance overlay (proxy ll-3.1-8b → opus-4.1)")
    fig_training_effect_panels(data)

    print(f"\nAll prototypes saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
