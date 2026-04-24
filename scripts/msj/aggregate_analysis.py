"""Aggregate MSJ analysis across all batches (SGTR + adversarial).

Loads attack_results from multiple batch directories, merges them, and
generates combined figures distinguishing base / SGTR-trained / adversarial-
trained models across the full shot-count range.

Usage:
    uv run python scripts/msj/aggregate_analysis.py \
        --results_root data/msj/results \
        --batches MSJ_01_batch1 MSJ_01_base_vs_trained_batch2 \
                  MSJ_02_adversarial_batch1 MSJ_02_adversarial_batch2 \
        --output_dir data/msj/results/aggregate
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


MODEL_COLORS = {
    "Llama 8B": ("#93c5fd", "#2563eb"),
    "GPT-OSS 20B": ("#6ee7b7", "#047857"),
    "Qwen 30B": ("#d8b4fe", "#7c3aed"),
}
FMT_LINESTYLE = {"PW": "-", "IND": "--"}
TAG_MARKER = {"UT": "o", "AT": "s"}


def _save_fig(fig, output_dir: Path, stem: str):
    """Save figure as both PDF and PNG."""
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  ✓ Saved: {pdf_path} (+ .png)")


def _normalize_base(s: str) -> str:
    return s.replace("Llama 3.1 8B", "Llama 8B").replace("Qwen 3.0 30B", "Qwen 30B")


def _parse_label(label: str):
    """Return (base, fmt, tag, dataset, kind) where kind ∈ {'base','sgtr','adv'}."""
    if "(base)" in label:
        return _normalize_base(label.replace("(base)", "").strip()), None, None, None, "base"

    m = re.match(r"(.+?)\s*\(ADV:\s*as\s+[^,]+,\s*(\w+)\s+(PW|IND)\s+(\w+)\)", label)
    if m:
        return _normalize_base(m.group(1).strip()), m.group(3), m.group(2), m.group(4), "adv"

    m = re.match(r"(.+?)\s*\((\w+)\s+(PW|IND)\s+(\w+)\)", label)
    if m:
        return _normalize_base(m.group(1).strip()), m.group(3), m.group(2), m.group(4), "sgtr"

    return _normalize_base(label), None, None, None, "sgtr"


def load_all(results_root: Path, batches: list[str]) -> pd.DataFrame:
    frames = []
    for b in batches:
        d = results_root / b
        rescored = d / "attack_results_rescored.json"
        regular = d / "attack_results.json"
        f = rescored if rescored.exists() else regular
        if not f.exists():
            print(f"  ⚠ Skipping {b}: no results file")
            continue
        with open(f) as fh:
            data = json.load(fh)
        df = pd.DataFrame(data)
        df["batch"] = b
        df["used_rescored"] = f.name == "attack_results_rescored.json"
        frames.append(df)
        print(f"  ✓ Loaded {len(df)} rows from {b} ({f.name})")
    if not frames:
        raise SystemExit("No results loaded")
    df = pd.concat(frames, ignore_index=True)
    # Drop ERROR outcomes (e.g. Tinker 400 at 125-shot context overflow) so they
    # don't pollute ASR as pseudo-failures.
    n_err = (df["outcome"] == "ERROR").sum()
    if n_err:
        print(f"  ⚠ Dropping {n_err} ERROR rows from ASR computation")
        df = df[df["outcome"] != "ERROR"].reset_index(drop=True)
    df["success"] = df["outcome"].str.contains("SUCCESS", case=False, na=False)
    parsed = df["model"].apply(_parse_label)
    df["base_model"] = [p[0] for p in parsed]
    df["fmt"] = [p[1] for p in parsed]
    df["tag"] = [p[2] for p in parsed]
    df["dataset"] = [p[3] for p in parsed]
    df["kind"] = [p[4] for p in parsed]
    return df


def compute_asr(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ASR per (model, n_shots) — duplicates across batches are pooled."""
    asr = (
        df.groupby(["model", "base_model", "fmt", "tag", "dataset", "kind", "n_shots"], dropna=False)
        .agg(n_attacks=("success", "count"), n_success=("success", "sum"))
        .reset_index()
    )
    asr["asr"] = asr["n_success"] / asr["n_attacks"]
    return asr


def fig_asr_by_shots(asr: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(13, 8))
    for model_label in sorted(asr["model"].unique()):
        sub = asr[asr["model"] == model_label].sort_values("n_shots")
        if sub.empty:
            continue
        base, fmt, tag, kind = sub.iloc[0][["base_model", "fmt", "tag", "kind"]]
        light, dark = MODEL_COLORS.get(base, ("#cccccc", "#666666"))

        if kind == "base":
            color, ls, lw, alpha, marker, ms = light, "-", 3.0, 0.7, "D", 9
        elif kind == "adv":
            color, ls, lw, alpha = dark, FMT_LINESTYLE.get(fmt, "-"), 1.8, 0.9
            marker, ms = TAG_MARKER.get(tag, "o"), 7
        else:  # sgtr
            color, ls, lw, alpha = dark, FMT_LINESTYLE.get(fmt, "-"), 2.0, 0.9
            marker, ms = TAG_MARKER.get(tag, "o"), 7

        # Adversarial: hollow markers to distinguish from SGTR-trained
        mfc = "white" if kind == "adv" else color

        ax.plot(
            sub["n_shots"], sub["asr"],
            color=color, linestyle=ls, linewidth=lw, alpha=alpha,
            marker=marker, markersize=ms, markerfacecolor=mfc,
            markeredgecolor=color, markeredgewidth=1.2,
            label=model_label,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Shots", fontsize=13)
    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=13)
    ax.set_title("MSJ — Combined Across Batches\n(base = diamond, SGTR = filled, adversarial = hollow)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.9)

    _save_fig(fig, output_dir, "asr_by_shots_combined")


def fig_asr_heatmap(asr: pd.DataFrame, output_dir: Path):
    # Order rows: group by base model, then base→sgtr→adv
    kind_order = {"base": 0, "sgtr": 1, "adv": 2}
    keyed = asr.drop_duplicates("model").assign(
        _k=lambda d: d["kind"].map(kind_order)
    )
    models = keyed.sort_values(["base_model", "_k", "model"])["model"].tolist()
    shot_counts = sorted(asr["n_shots"].unique())

    matrix = np.full((len(models), len(shot_counts)), np.nan)
    for ri, m in enumerate(models):
        for ci, s in enumerate(shot_counts):
            row = asr[(asr["model"] == m) & (asr["n_shots"] == s)]
            if not row.empty:
                matrix[ri, ci] = row["asr"].values[0]

    fig, ax = plt.subplots(figsize=(max(8, len(shot_counts) * 1.1),
                                     max(5, len(models) * 0.45)))
    im = ax.imshow(matrix, aspect="auto", cmap=plt.cm.RdYlGn_r, vmin=0, vmax=1)
    ax.set_xticks(range(len(shot_counts)))
    ax.set_xticklabels([str(s) for s in shot_counts], fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("Number of Shots", fontsize=12)
    ax.set_title("ASR — All Batches Combined", fontsize=14, fontweight="bold")
    for ri in range(len(models)):
        for ci in range(len(shot_counts)):
            v = matrix[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:.0%}", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("ASR", fontsize=11)
    _save_fig(fig, output_dir, "asr_heatmap_combined")


def fig_kind_summary(asr: pd.DataFrame, output_dir: Path):
    """Per family: mean ASR by kind (base / sgtr / adv) over shot count."""
    families = sorted(asr["base_model"].unique())
    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    KIND_STYLE = {
        "base": ("Base", "#888888", "D", "-"),
        "sgtr": ("SGTR-trained (mean)", None, "o", "-"),
        "adv": ("Adversarial (mean)", None, "s", "--"),
    }

    for ax, fam in zip(axes, families):
        light, dark = MODEL_COLORS.get(fam, ("#cccccc", "#666666"))
        sub = asr[asr["base_model"] == fam]
        for kind, (label, fixed_color, marker, ls) in KIND_STYLE.items():
            kind_sub = sub[sub["kind"] == kind]
            if kind_sub.empty:
                continue
            grouped = kind_sub.groupby("n_shots")["asr"].agg(["mean", "std"]).reset_index()
            color = fixed_color if fixed_color else (dark if kind == "sgtr" else light)
            mfc = "white" if kind == "adv" else color
            ax.plot(grouped["n_shots"], grouped["mean"], color=color, linestyle=ls,
                    marker=marker, markersize=8, linewidth=2.0,
                    markerfacecolor=mfc, markeredgecolor=color, markeredgewidth=1.2,
                    label=label)
            if "std" in grouped and grouped["std"].notna().any():
                ax.fill_between(grouped["n_shots"],
                                grouped["mean"] - grouped["std"].fillna(0),
                                grouped["mean"] + grouped["std"].fillna(0),
                                color=color, alpha=0.12)
        ax.set_xscale("log")
        ax.set_title(fam, fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Shots", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize=9, loc="upper left")
    axes[0].set_ylabel("Attack Success Rate", fontsize=12)
    fig.suptitle("MSJ Vulnerability by Training Type", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, output_dir, "asr_by_kind")


def fig_delta_vs_base(asr: pd.DataFrame, output_dir: Path):
    """Per family heatmap: trained variants × shots, cell = ASR − base ASR."""
    shot_counts = sorted(asr["n_shots"].unique())

    for fam in sorted(asr["base_model"].unique()):
        sub = asr[asr["base_model"] == fam]
        base_rows = sub[sub["kind"] == "base"]
        trained = sub[sub["kind"] != "base"]
        if base_rows.empty or trained.empty:
            continue

        kind_order = {"sgtr": 0, "adv": 1}
        trained_models = sorted(
            trained["model"].unique(),
            key=lambda m: (kind_order[trained[trained["model"] == m]["kind"].iloc[0]], m),
        )

        matrix = np.full((len(trained_models), len(shot_counts)), np.nan)
        for ri, m in enumerate(trained_models):
            for ci, s in enumerate(shot_counts):
                t = trained[(trained["model"] == m) & (trained["n_shots"] == s)]
                b = base_rows[base_rows["n_shots"] == s]
                if not t.empty and not b.empty:
                    matrix[ri, ci] = t["asr"].values[0] - b["asr"].values[0]

        vals = matrix[~np.isnan(matrix)]
        if len(vals) == 0:
            continue
        vmax = max(abs(vals.min()), abs(vals.max()), 0.01)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        labels = [m.split("(", 1)[1].rstrip(")") if "(" in m else m for m in trained_models]
        fig, ax = plt.subplots(figsize=(max(8, len(shot_counts) * 1.3),
                                        max(3, len(trained_models) * 0.55 + 1)))
        im = ax.imshow(matrix, aspect="auto", cmap=plt.cm.RdYlGn, norm=norm)
        ax.set_xticks(range(len(shot_counts)))
        ax.set_xticklabels([str(s) for s in shot_counts], fontsize=11)
        ax.set_yticks(range(len(trained_models)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Number of Shots", fontsize=12)
        ax.set_title(f"{fam}: ASR Δ (trained − base)\n(red = more vulnerable than base)",
                     fontsize=13, fontweight="bold")
        for ri in range(len(trained_models)):
            for ci in range(len(shot_counts)):
                v = matrix[ri, ci]
                if not np.isnan(v):
                    ax.text(ci, ri, f"{v:+.0%}", ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="white" if abs(v) > vmax * 0.5 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8).set_label("Δ ASR", fontsize=11)
        _save_fig(fig, output_dir, f"asr_delta_combined_{fam.replace(' ', '_').lower()}")


def fig_delta_avg_across_models(asr: pd.DataFrame, output_dir: Path):
    """Heatmap: mean (trained − base) ASR delta collapsed across trained models within each family.

    SGTR-trained family rows (excluding adversarial), plus extra rows per family
    that has adversarial variants (averaged over adversarial ops), plus an
    overall "All (mean)" row that spans both SGTR and adversarial.
    """
    shot_counts = sorted(asr["n_shots"].unique())
    families = sorted(asr["base_model"].unique())

    # Build rows: SGTR per family, then Adv per family (only where adv exists)
    row_specs = [(fam, "sgtr", fam) for fam in families]
    for fam in families:
        if not asr[(asr["base_model"] == fam) & (asr["kind"] == "adv")].empty:
            row_specs.append((fam, "adv", f"{fam} (adv)"))

    matrix = np.full((len(row_specs), len(shot_counts)), np.nan)
    for ri, (fam, kind, _label) in enumerate(row_specs):
        sub = asr[asr["base_model"] == fam]
        base_rows = sub[sub["kind"] == "base"]
        trained = sub[sub["kind"] == kind]
        if base_rows.empty or trained.empty:
            continue
        for ci, s in enumerate(shot_counts):
            b = base_rows[base_rows["n_shots"] == s]
            if b.empty:
                continue
            base_asr_val = b["asr"].values[0]
            t = trained[trained["n_shots"] == s]
            if t.empty:
                continue
            matrix[ri, ci] = (t["asr"] - base_asr_val).mean()

    # Overall "All (mean)" row: averages every non-base trained-vs-base delta
    # across all families and kinds equally (family-level means first to avoid
    # over-weighting families with more trained variants).
    all_row = np.full(len(shot_counts), np.nan)
    for ci, s in enumerate(shot_counts):
        fam_deltas = []
        for fam in families:
            sub = asr[asr["base_model"] == fam]
            base_rows = sub[sub["kind"] == "base"]
            trained = sub[sub["kind"] != "base"]
            if base_rows.empty or trained.empty:
                continue
            b = base_rows[base_rows["n_shots"] == s]
            t = trained[trained["n_shots"] == s]
            if b.empty or t.empty:
                continue
            fam_deltas.append((t["asr"] - b["asr"].values[0]).mean())
        if fam_deltas:
            all_row[ci] = np.mean(fam_deltas)

    matrix_full = np.vstack([matrix, all_row[None, :]])
    row_labels = [lbl for _, _, lbl in row_specs] + ["All (mean)"]

    vals = matrix_full[~np.isnan(matrix_full)]
    if len(vals) == 0:
        print("  ⚠ No delta data for avg heatmap — skipping")
        return
    vmax = max(abs(vals.min()), abs(vals.max()), 0.01)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(8, len(shot_counts) * 1.3),
                                    max(3, len(row_labels) * 0.7 + 1)))
    im = ax.imshow(matrix_full, aspect="auto", cmap=plt.cm.RdYlGn, norm=norm)
    ax.set_xticks(range(len(shot_counts)))
    ax.set_xticklabels([str(s) for s in shot_counts], fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.axhline(y=len(row_specs) - 0.5, color="black", linewidth=1.2)
    ax.set_xlabel("Number of Shots", fontsize=12)
    ax.set_title("Mean ΔASR (trained − base), averaged across trained models\n(red = trained more vulnerable than base)",
                 fontsize=13, fontweight="bold")
    for ri in range(len(row_labels)):
        for ci in range(len(shot_counts)):
            v = matrix_full[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:+.0%}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if abs(v) > vmax * 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Δ ASR", fontsize=11)
    _save_fig(fig, output_dir, "asr_delta_avg_across_models")


def fig_delta_avg_across_families(asr: pd.DataFrame, output_dir: Path):
    """Heatmap: ops × shots, cells = mean ΔASR (trained − base) averaged across model families.

    Excludes adversarial models. Y-axis lists the 5 distinct SGTR operationalizations:
    UT_PW_ShareGPT, UT_PW_PKU, UT_IND_ShareGPT, AT_PW_ShareGPT, AT_IND_ShareGPT.
    For each op, we compute per-family (trained − base) at each shot count, then
    average across families.
    """
    shot_counts = sorted(asr["n_shots"].unique())
    sgtr = asr[asr["kind"] == "sgtr"].copy()
    bases = asr[asr["kind"] == "base"].copy()

    ops_order = [
        ("UT", "PW", "ShareGPT"),
        ("UT", "PW", "PKU"),
        ("UT", "IND", "ShareGPT"),
        ("AT", "PW", "ShareGPT"),
        ("AT", "IND", "ShareGPT"),
    ]
    op_labels = [f"{t}_{f}_{d}" for t, f, d in ops_order]

    matrix = np.full((len(ops_order), len(shot_counts)), np.nan)
    for ri, (tag, fmt, ds) in enumerate(ops_order):
        op_sub = sgtr[(sgtr["tag"] == tag) & (sgtr["fmt"] == fmt) & (sgtr["dataset"] == ds)]
        if op_sub.empty:
            continue
        for ci, s in enumerate(shot_counts):
            family_deltas = []
            for fam in op_sub["base_model"].unique():
                fam_sub = op_sub[(op_sub["base_model"] == fam) & (op_sub["n_shots"] == s)]
                base_row = bases[(bases["base_model"] == fam) & (bases["n_shots"] == s)]
                if fam_sub.empty or base_row.empty:
                    continue
                fam_mean = fam_sub["asr"].mean()
                family_deltas.append(fam_mean - base_row["asr"].values[0])
            if family_deltas:
                matrix[ri, ci] = np.mean(family_deltas)

    vals = matrix[~np.isnan(matrix)]
    if len(vals) == 0:
        print("  ⚠ No data for op-avg heatmap — skipping")
        return
    vmax = max(abs(vals.min()), abs(vals.max()), 0.01)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(8, len(shot_counts) * 1.3),
                                    max(3, len(ops_order) * 0.7 + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap=plt.cm.RdYlGn, norm=norm)
    ax.set_xticks(range(len(shot_counts)))
    ax.set_xticklabels([str(s) for s in shot_counts], fontsize=11)
    ax.set_yticks(range(len(ops_order)))
    ax.set_yticklabels(op_labels, fontsize=11)
    ax.set_xlabel("Number of Shots", fontsize=12)
    ax.set_title("Mean ΔASR (trained − base), averaged across model families\n(SGTR-trained only; adversarial excluded)",
                 fontsize=13, fontweight="bold")
    for ri in range(len(ops_order)):
        for ci in range(len(shot_counts)):
            v = matrix[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:+.0%}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if abs(v) > vmax * 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Δ ASR", fontsize=11)
    _save_fig(fig, output_dir, "asr_delta_avg_across_families")


def fig_dot_arrow_by_model(asr: pd.DataFrame, output_dir: Path):
    """Dot-and-arrow plot grouped by model family, one column per shot count.

    Excludes adversarial. Within each family group, each shot count has a
    side-by-side (base dot → SGTR-trained dot) pair connected by an arrow.
    Trained ASR is averaged across that family's SGTR operationalizations.
    """
    shot_counts = sorted(asr["n_shots"].unique())
    families = sorted(asr["base_model"].unique())

    sgtr_mean = (asr[asr["kind"] == "sgtr"]
                 .groupby(["base_model", "n_shots"])["asr"].mean()
                 .reset_index())
    base_asr = asr[asr["kind"] == "base"][["base_model", "n_shots", "asr"]]

    fig, ax = plt.subplots(figsize=(max(10, 1.0 * len(shot_counts) * len(families) + 3), 6))

    group_spacing = 1.5  # gap between family groups
    slot_width = 1.0     # width of one shot-count slot within a group
    family_width = len(shot_counts) * slot_width

    family_centers = []
    x_labels_shot = []
    x_ticks_shot = []

    for gi, fam in enumerate(families):
        light, dark = MODEL_COLORS.get(fam, ("#cccccc", "#666666"))
        group_start = gi * (family_width + group_spacing)
        family_centers.append(group_start + family_width / 2 - slot_width / 2)

        for si, shots in enumerate(shot_counts):
            x = group_start + si * slot_width
            x_ticks_shot.append(x)
            x_labels_shot.append(str(shots))

            base_row = base_asr[(base_asr["base_model"] == fam) & (base_asr["n_shots"] == shots)]
            sgtr_row = sgtr_mean[(sgtr_mean["base_model"] == fam) & (sgtr_mean["n_shots"] == shots)]
            if base_row.empty:
                continue
            by = base_row["asr"].values[0]

            ax.plot(x, by, marker="D", color=light,
                    markeredgecolor="black", markeredgewidth=0.8,
                    markersize=9, zorder=3,
                    label="Base" if (gi == 0 and si == 0) else None)

            if not sgtr_row.empty:
                ty = sgtr_row["asr"].values[0]
                ax.annotate("", xy=(x, ty), xytext=(x, by),
                            arrowprops=dict(arrowstyle="->", color=dark,
                                            lw=1.8, alpha=0.9))
                ax.plot(x, ty, marker="o", color=dark,
                        markeredgecolor="black", markeredgewidth=0.6,
                        markersize=8, zorder=3,
                        label="SGTR-trained" if (gi == 0 and si == 0) else None)

    # Bottom axis: shot-count labels
    ax.set_xticks(x_ticks_shot)
    ax.set_xticklabels(x_labels_shot, fontsize=24)
    ax.set_xlabel("Number of Shots (grouped by family)", fontsize=27)

    # Family labels below the xlabel to avoid overlap
    for center, fam in zip(family_centers, families):
        ax.text(center, -0.32, fam, ha="center", va="top",
                fontsize=27, fontweight="bold",
                transform=ax.get_xaxis_transform())

    # Dividers between groups
    for gi in range(1, len(families)):
        divider_x = gi * (family_width + group_spacing) - group_spacing / 2
        ax.axvline(divider_x, color="gray", linewidth=0.6, linestyle=":", alpha=0.7)

    ax.set_ylabel("ASR", fontsize=27)
    ymax = max(base_asr["asr"].max() if not base_asr.empty else 0,
               sgtr_mean["asr"].max() if not sgtr_mean.empty else 0)
    ax.set_ylim(-0.02, ymax + 0.1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.tick_params(axis="y", labelsize=24)
    ax.set_title("Base → SGTR-trained ASR shift, grouped by model family\n(adversarial excluded; trained averaged over operationalizations)",
                 fontsize=28, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), fontsize=25, framealpha=0.9,
                  borderaxespad=0.0)

    fig.tight_layout()
    _save_fig(fig, output_dir, "asr_dot_arrow_by_model")


def print_summary(df: pd.DataFrame, asr: pd.DataFrame):
    print(f"\n{'=' * 60}\nAGGREGATE MSJ SUMMARY\n{'=' * 60}")
    print(f"Total attacks: {len(df)}")
    print(f"Total successes: {df['success'].sum()} ({df['success'].mean():.1%})")
    print(f"Batches: {sorted(df['batch'].unique())}")
    print(f"Shot counts: {sorted(df['n_shots'].unique())}")
    print(f"Models: {df['model'].nunique()}")

    print("\nMean ASR by family × kind × shots:")
    summary = (asr.groupby(["base_model", "kind", "n_shots"])["asr"]
               .mean().reset_index())
    for fam in sorted(summary["base_model"].unique()):
        print(f"\n  {fam}")
        fam_sub = summary[summary["base_model"] == fam]
        pivot = fam_sub.pivot(index="kind", columns="n_shots", values="asr")
        print(pivot.to_string(float_format=lambda v: f"{v:.1%}" if pd.notna(v) else "  - "))


def main():
    parser = argparse.ArgumentParser(description="Aggregate MSJ analysis across batches")
    parser.add_argument("--results_root", default="data/msj/results")
    parser.add_argument(
        "--batches",
        nargs="+",
        default=[
            "MSJ_01_batch1",
            "MSJ_01_base_vs_trained_batch2",
            "MSJ_01_base_vs_trained_batch3",
            "MSJ_02_adversarial_batch1",
            "MSJ_02_adversarial_batch2",
            "MSJ_02_adversarial_batch3",
        ],
    )
    parser.add_argument("--output_dir", default="data/msj/results/aggregate")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(args.batches)} batches from {results_root}...")
    df = load_all(results_root, args.batches)
    asr = compute_asr(df)

    print_summary(df, asr)

    asr.to_csv(output_dir / "asr_summary_combined.csv", index=False)
    print(f"\n✓ Saved: {output_dir / 'asr_summary_combined.csv'}")

    print("\nGenerating figures...")
    fig_asr_by_shots(asr, output_dir)
    fig_asr_heatmap(asr, output_dir)
    fig_kind_summary(asr, output_dir)
    fig_delta_vs_base(asr, output_dir)
    fig_delta_avg_across_models(asr, output_dir)
    fig_delta_avg_across_families(asr, output_dir)
    fig_dot_arrow_by_model(asr, output_dir)

    print(f"\n✓ Aggregate analysis complete. Results in {output_dir}/")


if __name__ == "__main__":
    main()
