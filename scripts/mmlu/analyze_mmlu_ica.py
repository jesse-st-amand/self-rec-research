"""Analysis pipeline for the MMLU-ICA capabilities-control experiment.

Loads inspect_ai .eval logs from experiments_eval/ICA/MMLU_01_trained-OP_eval-on_self-same-OP/
(legacy path: experiments_eval/MSJ/MMLU_ICA/) and computes:
  - accuracy per (evaluator, kind, condition)
  - Δacc = acc(condition) − acc(no-ica baseline) within (model, kind)
  - author-dependence statistic: AD_self = Δacc_self − mean(Δacc_ctrl*)
  - bar plot of Δacc per (model, kind) split by condition
  - optional scatter: MSJ ASR-uplift vs AD_self (skipped if uplift CSV absent)

Usage:
  uv run python scripts/mmlu/analyze_mmlu_ica.py \\
      --experiment_dir experiments_eval/MSJ/MMLU_ICA \\
      --output_dir data/mmlu_ica/results/joint
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from inspect_ai.log import read_eval_log


_CONDS = ("no-ica", "ica-self", "ica-alt", "ica-ctrl", "ica-ctrl2", "ica-ctrl3")
_KINDS = (
    "base",
    "trained-std-UT_IND", "trained-std-AT_IND",
    "trained-std-UT_PW",  "trained-std-AT_PW",
    "trained-adv-UT_IND", "trained-adv-UT_PW",
    "trained-multi-op",
)

_KIND_COLORS = {
    "base":                  "#6b7280",
    "trained-std-UT_IND":    "#1e40af",
    "trained-std-AT_IND":    "#2563eb",
    "trained-std-UT_PW":     "#0369a1",
    "trained-std-AT_PW":     "#0891b2",
    "trained-adv-UT_IND":    "#7c3aed",
    "trained-adv-UT_PW":     "#a855f7",
    "trained-multi-op":      "#10b981",  # teal — matches SGTR analyzer
}

# Kind-group colors (per column group in dot-arrows figure)
_KV_GROUP_COLORS = {
    "base":             "#374151",
    "trained-std":      "#1e40af",
    "trained-adv":      "#7c3aed",
    "trained-multi-op": "#10b981",
}

_OPS = ("UT_PW", "UT_IND", "AT_PW", "AT_IND")  # matches SGTR analyzer's OP_DISPLAY_ORDER


def parse_leaf(leaf: str) -> dict | None:
    """Parse leaf dir name into {model, kind, shots, condition}.

    Two forms:
      - {model}_{kind}_no-ica                       (no-ICA baseline, shots=None)
      - {model}_{kind}_{shots}shot_{ica-condition}  (ICA cell)
    """
    m = re.match(
        r"^(?P<model>gpt-oss-20b|qwen-3\.0-30b|ll-3\.1-8b)"
        r"_(?P<kind>base|trained-multi-op|trained-(?:std|adv)-(?:UT|AT)_(?:IND|PW))"
        r"(?:_(?P<shots>\d+)shot)?"
        r"_(?P<condition>no-ica|ica-self|ica-alt|ica-ctrl3|ica-ctrl2|ica-ctrl)$",
        leaf,
    )
    if not m:
        return None
    d = m.groupdict()
    d["shots"] = int(d["shots"]) if d["shots"] else None
    return d


def kind_to_group_and_op(kind: str) -> tuple[str, str | None]:
    """Split "trained-std-UT_IND" → ("trained-std", "UT_IND"). Base → ("base", None).

    The op-agnostic kinds ("base", "trained-multi-op") return None for op,
    indicating their accuracy is the same across all op rows in figures.
    """
    if kind == "base":
        return "base", None
    if kind == "trained-multi-op":
        return "trained-multi-op", None
    # kind like "trained-std-UT_IND"
    parts = kind.split("-")
    # parts = ["trained", "std", "UT_IND"]
    return f"{parts[0]}-{parts[1]}", parts[2]


def extract_accuracy(log) -> tuple[float | None, int]:
    if log.status != "success" or log.results is None:
        return None, 0
    for eval_score in log.results.scores or []:
        metrics = eval_score.metrics or {}
        # inspect_ai's choice() scorer writes "accuracy"; SGTR-ICA uses "mean"
        m = metrics.get("accuracy") or metrics.get("mean")
        if m is None:
            continue
        n = eval_score.scored_samples or 0
        if n == 0:
            continue
        return float(m.value), int(n)
    return None, 0


def load_runs(experiment_dir: Path, results_root: Path) -> pd.DataFrame:
    rows = []
    for model_dir in sorted(experiment_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "bash":
            continue
        for leaf_dir in sorted(model_dir.iterdir()):
            if not leaf_dir.is_dir():
                continue
            parts = parse_leaf(leaf_dir.name)
            if not parts:
                continue
            cfg_path = leaf_dir / "config.yaml"
            if not cfg_path.exists():
                continue
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)

            eval_dataset = cfg.get("eval_dataset", "mmlu")
            eval_subset = cfg.get("eval_subset", "mmlu_50")
            # Two-level layout (introduced when ICA experiments were renamed):
            #   data/results/<dataset>/<subset>/<experiment_name>/<mini_batch>/*.eval
            # Fall back to the legacy flat layout if the nested dir is absent.
            experiment = cfg.get("experiment_name") or experiment_dir.name
            log_dir = (results_root / eval_dataset / eval_subset
                       / experiment / leaf_dir.name)
            if not log_dir.exists():
                legacy = results_root / eval_dataset / eval_subset / leaf_dir.name
                if not legacy.exists():
                    continue
                log_dir = legacy
            logs = sorted(log_dir.glob("*.eval"))
            if not logs:
                continue
            # Use the newest successful log
            for log_path in reversed(logs):
                try:
                    log = read_eval_log(str(log_path), header_only=True)
                except Exception:
                    continue
                acc, n = extract_accuracy(log)
                if acc is None:
                    continue
                rows.append({
                    "leaf": leaf_dir.name,
                    "model": parts["model"],
                    "kind": parts["kind"],
                    "shots": parts["shots"],
                    "condition": parts["condition"],
                    "evaluator": cfg["model_names"][0],
                    "icl_model": cfg.get("icl_model"),
                    "icl_count": cfg.get("icl_count"),
                    "accuracy": acc,
                    "n": n,
                    "log_path": str(log_path),
                })
                break
    return pd.DataFrame(rows)


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model, kind, shots, condition), compute Δacc vs the no-ICA baseline."""
    out = []
    for (model, kind), grp in df.groupby(["model", "kind"]):
        baseline = grp[grp["condition"] == "no-ica"]
        if baseline.empty:
            continue
        base_acc = baseline["accuracy"].iloc[0]
        for _, r in grp.iterrows():
            out.append({
                "model": model,
                "kind": kind,
                "shots": r["shots"],
                "condition": r["condition"],
                "icl_model": r["icl_model"],
                "acc": r["accuracy"],
                "base_acc": base_acc,
                "delta": r["accuracy"] - base_acc,
            })
    return pd.DataFrame(out)


def compute_author_dependence(deltas: pd.DataFrame) -> pd.DataFrame:
    """Per (model, kind, shots): AD_self / AD_alt vs. ctrl-averaged baseline."""
    rows = []
    ctrl_conds = {"ica-ctrl", "ica-ctrl2", "ica-ctrl3"}
    ica_df = deltas[deltas["condition"] != "no-ica"]
    for (model, kind, shots), grp in ica_df.groupby(["model", "kind", "shots"]):
        d = {r["condition"]: r["delta"] for _, r in grp.iterrows()}
        ctrl_vals = [d[c] for c in ctrl_conds if c in d and pd.notna(d[c])]
        if not ctrl_vals:
            continue
        ctrl_mean = float(np.mean(ctrl_vals))
        rows.append({
            "model": model,
            "kind": kind,
            "shots": shots,
            "delta_self": d.get("ica-self"),
            "delta_alt": d.get("ica-alt"),
            "delta_ctrl_mean": ctrl_mean,
            "AD_self": (d["ica-self"] - ctrl_mean) if "ica-self" in d else None,
            "AD_alt":  (d["ica-alt"]  - ctrl_mean) if "ica-alt"  in d else None,
        })
    return pd.DataFrame(rows)


def fig_delta_bars(deltas: pd.DataFrame, output_dir: Path):
    """Per base model × kind: bars of Δacc across ICA conditions."""
    models = sorted(deltas["model"].unique())
    kinds_present = [k for k in _KINDS if k in deltas["kind"].unique()]
    cond_order = [c for c in _CONDS if c != "no-ica"]

    fig, axes = plt.subplots(
        len(models), len(kinds_present),
        figsize=(2.2 * len(kinds_present), 2.8 * len(models)),
        sharey=True, squeeze=False,
    )
    x = np.arange(len(cond_order))
    for ri, model in enumerate(models):
        for ci, kind in enumerate(kinds_present):
            ax = axes[ri, ci]
            sub = deltas[(deltas["model"] == model) & (deltas["kind"] == kind)]
            vals = [sub[sub["condition"] == c]["delta"].iloc[0]
                    if not sub[sub["condition"] == c].empty else np.nan
                    for c in cond_order]
            ax.bar(x, vals, color=_KIND_COLORS.get(kind, "#333"), alpha=0.85)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(cond_order, rotation=45, ha="right", fontsize=8)
            if ri == 0:
                ax.set_title(kind, fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"{model}\nΔacc vs no-ICA", fontsize=9)
            ax.grid(alpha=0.3, linewidth=0.3)
    fig.suptitle("MMLU-ICA: Δaccuracy from no-ICA baseline",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0.02, 0.0, 1.0, 0.96])
    out = output_dir / "mmlu_ica_delta_bars.png"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out} (+ .pdf)")


def _plot_arrow(ax, x, y_from, y_to, color, alpha=0.85, lw=1.2):
    if pd.notna(y_from) and pd.notna(y_to):
        ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                    arrowprops=dict(arrowstyle="->", color=color,
                                    lw=lw, alpha=alpha))


def _draw_panel_separators(fig, axes, row_models, kv_order, n_cond_per_kv):
    """Draw figure-level horizontal bars between model groups and vertical bars
    between kv groups (mirrors scripts/ica/analyze_ica.py:_draw_panel_separators).
    """
    import matplotlib.lines as mlines

    # Vertical bars between kv groups
    for gi in range(1, len(kv_order)):
        ax_left = axes[0, gi * n_cond_per_kv]
        ax_below = axes[-1, gi * n_cond_per_kv]
        x = ax_left.get_position().x0 - 0.005
        y0 = ax_below.get_position().y0
        y1 = ax_left.get_position().y1
        line = mlines.Line2D([x, x], [y0, y1], transform=fig.transFigure,
                             color="#222", linewidth=2.0, zorder=10)
        fig.add_artist(line)

    # Horizontal bars between consecutive model groups
    starts = []
    cur = None
    for i, m in enumerate(row_models):
        if m != cur:
            starts.append(i)
            cur = m
    for i in starts[1:]:
        ax_above = axes[i - 1, 0]
        ax_below = axes[i, 0]
        ax_right_above = axes[i - 1, -1]
        y = (ax_above.get_position().y0 + ax_below.get_position().y1) / 2
        x0 = ax_above.get_position().x0
        x1 = ax_right_above.get_position().x1
        line = mlines.Line2D([x0, x1], [y, y], transform=fig.transFigure,
                             color="#222", linewidth=2.0, zorder=10)
        fig.add_artist(line)


def _draw_model_group_labels(fig, axes, row_models, x_pos=0.015):
    """Rotated label spanning each model's row range (left margin)."""
    starts = []
    current = None
    for i, m in enumerate(row_models):
        if m != current:
            starts.append((i, m))
            current = m
    starts.append((len(row_models), None))
    for j in range(len(starts) - 1):
        ri_start, model = starts[j]
        ri_end = starts[j + 1][0] - 1
        y_top = axes[ri_start, 0].get_position().y1
        y_bot = axes[ri_end, 0].get_position().y0
        fig.text(x_pos, (y_top + y_bot) / 2, model,
                 ha="center", va="center", rotation=90,
                 fontsize=12, fontweight="bold")


def fig_shot_dot_arrows(
    df: pd.DataFrame,
    output_dir: Path,
    kv_order: tuple = ("base", "trained-std", "trained-adv"),
    output_name: str = "mmlu_ica_dot_arrows",
    require_kind_prefix: str | None = None,
):
    """Combined dot-arrow figure mirroring SGTR-ICA b1/b2 layout.

    Layout (like data/ica/results/joint/all/<exp>/all/dot_arrows.png):
      rows = (model, op) for op ∈ {AT_IND, AT_PW, UT_IND, UT_PW}  → up to 12 rows
      col groups = kv_order                                        → 2–3 groups
      within each group: [ica-self, ica-alt, ica-ctrl-avg]         → 3 cols
      → len(kv_order) × 3 panels per row.

    Each panel: open dot = no-ICA accuracy, filled dot = ICA accuracy at the
    one shot count that was run, arrow connecting. ica-ctrl-avg averages the
    three control conditions.

    For op-agnostic column groups ("base", "trained-multi-op"), the same
    accuracy is shown in every op row (single LoRA / single base evaluator
    across ops). For "trained-std" / "trained-adv", each row uses the LoRA
    trained on that row's op; cells with no matching LoRA are left blank.

    `require_kind_prefix`: skip the figure entirely when no kind starts with
    this prefix (e.g. "trained-multi-op" — avoids emitting an empty multi-op
    figure when the multi-op evals haven't run yet).
    """
    if df.empty:
        return
    if require_kind_prefix is not None:
        if not df["kind"].astype(str).str.startswith(require_kind_prefix).any():
            return
    ctrl_suffixes = ("ica-ctrl", "ica-ctrl2", "ica-ctrl3")
    col_suffixes = ["ica-self", "ica-alt", "ica-ctrl-avg"]

    models = sorted(df["model"].unique())
    ops = list(_OPS)
    rows_spec = [(m, op) for m in models for op in ops]

    shots_present = sorted({int(s) for s in df["shots"].dropna().unique()})
    if not shots_present:
        shots_present = [5]
    lo, hi = min(shots_present), max(shots_present)
    span = max(hi - lo, 1)
    pad = max(1.0, span * 0.25)
    xlim = (lo - pad, hi + pad)

    n_rows = len(rows_spec)
    n_cols = len(kv_order) * len(col_suffixes)  # 9
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.9 * n_cols, 1.45 * n_rows + 1.4),
        sharey=True, squeeze=False,
    )

    for ri, (model, op) in enumerate(rows_spec):
        axes[ri, 0].set_ylabel(op, fontsize=10, rotation=0,
                               ha="right", va="center", fontweight="bold")
        m_df = df[df["model"] == model]

        for gi, kv_group in enumerate(kv_order):
            # Resolve which kind in this row maps to this column group.
            # Op-agnostic kinds (single LoRA shared across ops): "base" and
            # "trained-multi-op". For per-op trained kinds, append the row op.
            if kv_group in ("base", "trained-multi-op"):
                kind = kv_group
            else:
                kind = f"{kv_group}-{op}"
            has_kind = kind in df["kind"].values and not m_df[m_df["kind"] == kind].empty
            kv_color = _KV_GROUP_COLORS[kv_group]

            # Baseline (no-ica): shared across shots
            bl_acc = np.nan
            if has_kind:
                bl = m_df[(m_df["kind"] == kind) & (m_df["condition"] == "no-ica")]
                if not bl.empty:
                    bl_acc = bl["accuracy"].iloc[0]

            for si, suffix in enumerate(col_suffixes):
                ci = gi * len(col_suffixes) + si
                ax = axes[ri, ci]
                ax.set_xlim(*xlim)
                ax.set_ylim(-0.05, 1.05)
                ax.axhline(0.25, color="red", linestyle=":",
                           linewidth=0.5, alpha=0.5)
                ax.grid(alpha=0.3, linewidth=0.3)
                ax.tick_params(axis="both", labelsize=8)

                # Collect ICA accuracy per shot
                if has_kind:
                    if suffix == "ica-ctrl-avg":
                        sub = m_df[(m_df["kind"] == kind)
                                   & (m_df["condition"].isin(ctrl_suffixes))]
                        if not sub.empty:
                            ica_series = (sub.groupby("shots", as_index=False)
                                             ["accuracy"].mean()
                                             .sort_values("shots"))
                        else:
                            ica_series = pd.DataFrame(columns=["shots", "accuracy"])
                    else:
                        ica_series = (m_df[(m_df["kind"] == kind)
                                           & (m_df["condition"] == suffix)]
                                      .sort_values("shots"))
                else:
                    ica_series = pd.DataFrame(columns=["shots", "accuracy"])

                shots_arr = ica_series["shots"].values
                ica_vals = ica_series["accuracy"].values

                # Baseline dots at each shot (shared bl_acc)
                if pd.notna(bl_acc) and len(shots_arr):
                    ax.scatter(shots_arr, [bl_acc] * len(shots_arr),
                               marker="o", facecolors="none",
                               edgecolors=kv_color, s=55, linewidths=1.5, zorder=2)
                # Filled ICA dots
                if len(shots_arr):
                    ax.scatter(shots_arr, ica_vals, marker="o", color=kv_color,
                               s=55, linewidths=0.5, edgecolors="black",
                               zorder=3)
                    for x, v in zip(shots_arr, ica_vals):
                        _plot_arrow(ax, x, bl_acc, v, kv_color)

                if ri == 0:
                    ax.set_title(suffix, fontsize=9, fontweight="bold")
                if ri != n_rows - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xticks(shots_present)
                    ax.set_xticklabels([str(s) for s in shots_present])
                if si == 0 and gi > 0:
                    ax.spines["left"].set_linewidth(2.2)
                    ax.spines["left"].set_color("#444")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor="none",
                   markeredgecolor=_KV_GROUP_COLORS["base"],
                   markeredgewidth=1.6, markersize=8,
                   label="accuracy · no-ICA (open, kv color)"),
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=_KV_GROUP_COLORS["base"],
                   markeredgecolor="black", markeredgewidth=0.5,
                   markersize=8,
                   label="accuracy · ICA (filled, kv color)"),
    ]
    fig.legend(handles=handles, loc="lower center", fontsize=9,
               frameon=True, ncol=2, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("MMLU-ICA: accuracy with/without ICA (ctrl averaged)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.5, 0.028, "ICA shot count", ha="center", fontsize=11)
    plt.tight_layout(rect=[0.09, 0.06, 1.0, 0.93])

    # Kind-group banners above each 3-col block
    for gi, kv_group in enumerate(kv_order):
        ax_left = axes[0, gi * len(col_suffixes)]
        ax_right = axes[0, gi * len(col_suffixes) + len(col_suffixes) - 1]
        p_l = ax_left.get_position()
        p_r = ax_right.get_position()
        fig.text((p_l.x0 + p_r.x1) / 2, p_l.y1 + 0.035, kv_group,
                 ha="center", va="bottom",
                 fontsize=12, fontweight="bold",
                 color=_KV_GROUP_COLORS[kv_group])

    _draw_model_group_labels(fig, axes, [m for m, _ in rows_spec])
    _draw_panel_separators(fig, axes, [m for m, _ in rows_spec],
                           kv_order, len(col_suffixes))

    out = output_dir / f"{output_name}.png"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out} (+ .pdf)")


def fig_author_dependence(ad: pd.DataFrame, output_dir: Path):
    """Bar chart of AD_self (and AD_alt for adv) per (model, kind)."""
    if ad.empty:
        return
    models = sorted(ad["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.6),
                             squeeze=False)
    for ci, model in enumerate(models):
        ax = axes[0, ci]
        sub = ad[ad["model"] == model]
        labels = [f"{r['kind']}" for _, r in sub.iterrows()]
        ad_self = [r["AD_self"] if pd.notna(r["AD_self"]) else 0 for _, r in sub.iterrows()]
        ad_alt = [r["AD_alt"] if pd.notna(r["AD_alt"]) else 0 for _, r in sub.iterrows()]
        x = np.arange(len(labels))
        w = 0.4
        ax.bar(x - w / 2, ad_self, w, label="AD_self", color="#1e40af")
        ax.bar(x + w / 2, ad_alt, w, label="AD_alt",  color="#dc2626")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.set_ylabel("Δacc_self − mean(Δacc_ctrl)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linewidth=0.3)
    fig.suptitle("Author-dependence: does self-authored ICA degrade MMLU more than ctrl?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    out = output_dir / "mmlu_ica_author_dependence.png"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out} (+ .pdf)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--experiment_dir",
        default="experiments_eval/ICA/MMLU_01_trained-OP_eval-on_self-same-OP",
    )
    ap.add_argument("--results_root", default="data/results")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    experiment_dir = Path(args.experiment_dir)
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(experiment_dir, results_root)
    if df.empty:
        print("No runs found. Have the evals completed?")
        return
    df.to_csv(output_dir / "mmlu_ica_runs.csv", index=False)
    print(f"Loaded {len(df)} runs → {output_dir / 'mmlu_ica_runs.csv'}")

    # Per-(model,kind,condition) accuracy table
    print("\n--- Accuracy ---")
    piv = df.pivot_table(index=["model", "kind"],
                         columns="condition", values="accuracy")
    piv = piv.reindex(columns=[c for c in _CONDS if c in piv.columns])
    print(piv.to_string(float_format=lambda v: f"{v:.3f}" if pd.notna(v) else "  -  "))

    deltas = compute_deltas(df)
    deltas.to_csv(output_dir / "mmlu_ica_deltas.csv", index=False)
    print(f"\nΔacc table → {output_dir / 'mmlu_ica_deltas.csv'}")

    ad = compute_author_dependence(deltas)
    ad.to_csv(output_dir / "mmlu_ica_author_dependence.csv", index=False)
    if not ad.empty:
        print("\n--- Author-dependence ---")
        print(ad.to_string(float_format=lambda v: f"{v:+.3f}" if pd.notna(v) else "  -  ",
                           index=False))

    print("\nGenerating figures...")
    fig_delta_bars(deltas, output_dir)
    fig_author_dependence(ad, output_dir)
    fig_shot_dot_arrows(df, output_dir)
    # Multi-OP-only variant: base + trained-multi-op (skipped if no multi-op runs).
    fig_shot_dot_arrows(
        df, output_dir,
        kv_order=("base", "trained-multi-op"),
        output_name="mmlu_ica_dot_arrows_multi-op",
        require_kind_prefix="trained-multi-op",
    )
    print(f"\n✓ Outputs in {output_dir}")


if __name__ == "__main__":
    main()
