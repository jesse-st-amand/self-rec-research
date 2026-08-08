"""Generate the three figures that appear in the COLM 2026 main body.

This is a thin driver: every figure is produced by the same function that the
full analysis pipelines call, so the output is identical to what those pipelines
write. What this script adds is a single place to tune text size while sizing
figures for the page — see FIGURE_STYLE below.

    Figure 1  boxplot_with_grouped_bar.pdf     prototype_compact_figures.py
    Figure 2  quality_heuristic_combined.pdf   prototype_compact_figures.py
    Figure 3  combined_transfer_ranking.pdf    prototype_combined_transfer.py

Usage:
    bash scripts/figures/COLM2026/bash/make_paper_figures.sh
    bash scripts/figures/COLM2026/bash/make_paper_figures.sh --only fig1 fig2
    bash scripts/figures/COLM2026/bash/make_paper_figures.sh --copy-to-paper
"""

import argparse
import shutil
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_FIGURES = REPO_ROOT / "_external" / "COLM_2026_SGTR" / "figures"

# ============================================================================
# STYLE — edit these while sizing figures for the page.
# ============================================================================
# Every value is a font size in points, applied to one role within the figure.
# Set a role to None to leave whatever the plotting function chose untouched.
#
#   title         panel titles set via ax.set_title  (e.g. "(a) Standard Training")
#   axis_label    x/y axis labels, and colorbar labels
#   tick_label    x/y tick labels on axes and colorbars
#   legend        legend entry text
#   legend_title  legend heading text
#   annotation    text drawn inside the axes — heatmap cell values, "r = 0.73"
#                 boxes, significance asterisks
#   figure_text   figure-level text — suptitles, panel letters placed on the
#                 figure rather than an axes
#
# Three extra knobs per figure:
#   scale         multiplies every text size after the above are applied,
#                 including roles left at None. Use to shrink a whole figure
#                 uniformly without retuning each role.
#   figsize       (width, height) in inches, overriding the function's own
#                 figure size. None keeps it. Applied just before saving;
#                 the layout reflows because all four save with
#                 bbox_inches="tight".
#   margins       whitespace INSIDE the axes frame — use this when something
#                 drawn near an edge (an "n=" label, an asterisk, an annotation
#                 box) collides with the frame. Values are a fraction of the
#                 panel's current data range on that axis, so 0.05 adds 5% of
#                 the visible span. Three forms:
#
#                     "margins": 0.04                      every side, every panel
#                     "margins": {"(a)": 0.04}             every side of panel (a)
#                     "margins": {"(a)": {"bottom": 0.06}} one side of panel (a)
#
#                 A panel key is either a string matched against the start of
#                 the panel's title, or the 0-based position of the panel in
#                 creation order (colorbars are not counted and never padded).
#                 Sides are "left", "right", "bottom", "top"; omitted sides get
#                 0. Set None to leave every panel alone.
#
# COLM requires figure text to be no smaller than the 9pt caption text, so
# treat 9 as the floor for anything a reader has to read.

BASE_STYLE = {
    "title": 12,
    "axis_label": 12,
    "tick_label": 12,
    "legend": 12,
    "legend_title": 12,
    "annotation": 12,
    "figure_text": 12,
    "scale": 1.0,
    "figsize": None,
    "margins": None,
}

FIGURE_STYLE = {
    # Fig 1: box plots (a) + per-model grouped bar (b). Panel (b) is dense —
    # 24 evaluator labels across the full text width.
    "fig1": {
        **BASE_STYLE,
        "title": 14,
        "axis_label": 15,
        "tick_label": 14,
        "legend": 14,
        "annotation": 13,
        # Panel (a) writes "n=..." below each box, hanging off the bottom of the
        # data range; without the pad it sits on the frame.
        "margins": {"(a)": {"bottom": 0.04}},
    },
    # Fig 2: 2x3 scatter grid. Lots of panels, so text runs smaller.
    "fig2": {
        **BASE_STYLE,
        "title": 22,
        "axis_label": 25,
        "tick_label": 27,
        "legend": 18,
        "annotation": 21,
    },
    # Fig 3: four panels — the dot plots (a, b) above the heatmaps (c, d). The
    # two halves arrive at different natural sizes and set their own text at
    # different point sizes, so these are what makes them agree. Layout is in
    # prototype_combined_transfer.py; annotation covers both the dot plots'
    # box-plot labels and the delta value printed in each heatmap cell.
    "fig3": {
        **BASE_STYLE,
        "title": 20,
        "axis_label": 20,
        "tick_label": 20,
        "annotation": 20,
        "figure_text": 20,
    },
}

# ============================================================================
# Style application
# ============================================================================
# The plotting functions pass explicit fontsize= arguments, which override
# rcParams, and they save and close their own figures. So rather than trying to
# set defaults up front, we intercept savefig and restyle the finished figure.


def _axes_legends(ax):
    """Every Legend on `ax`, not just the one ax.get_legend() returns.

    A panel that needs two legends builds the first with ax.legend(), keeps it
    alive with ax.add_artist(), then calls ax.legend() again — which replaces
    ax.legend_. The first legend is still drawn but is only reachable through
    the child list, so restyling via get_legend() alone silently skips it.
    """
    from matplotlib.legend import Legend

    found, seen = [], set()
    for legend in [ax.get_legend()] + list(ax.get_children()):
        if isinstance(legend, Legend) and id(legend) not in seen:
            seen.add(id(legend))
            found.append(legend)
    return found


def _role_texts(fig):
    """Yield (role, Text artist) for every piece of text in a figure."""
    for ax in fig.axes:
        yield "title", ax.title
        for side in ("_left_title", "_right_title"):
            artist = getattr(ax, side, None)
            if artist is not None:
                yield "title", artist

        yield "axis_label", ax.xaxis.label
        yield "axis_label", ax.yaxis.label

        for minor in (False, True):
            for label in ax.get_xticklabels(minor=minor) + ax.get_yticklabels(minor=minor):
                yield "tick_label", label

        for legend in _axes_legends(ax):
            for text in legend.get_texts():
                yield "legend", text
            if legend.get_title() is not None:
                yield "legend_title", legend.get_title()

        for text in ax.texts:
            yield "annotation", text

    for legend in getattr(fig, "legends", []):
        for text in legend.get_texts():
            yield "legend", text
        if legend.get_title() is not None:
            yield "legend_title", legend.get_title()

    for text in fig.texts:
        yield "figure_text", text


_SIDES = ("left", "right", "bottom", "top")


def _panel_axes(fig):
    """The figure's data panels, in creation order, excluding colorbars."""
    return [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]


def _resolve_margins(spec, ax, index):
    """Look up the margin spec for one panel; None if it isn't padded."""
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return {side: float(spec) for side in _SIDES}

    title = ax.get_title().strip()
    for key, value in spec.items():
        if isinstance(key, int):
            if key != index:
                continue
        elif not (title and title.startswith(key)):
            continue
        if isinstance(value, (int, float)):
            return {side: float(value) for side in _SIDES}
        return {side: float(value.get(side, 0.0)) for side in _SIDES}
    return None


def _apply_margins(fig, spec):
    """Widen each panel's limits by the requested fraction of its data range."""
    for index, ax in enumerate(_panel_axes(fig)):
        margins = _resolve_margins(spec, ax, index)
        if margins is None:
            continue
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        # Signed spans, so an inverted axis (imshow) pads the right way round.
        x_span, y_span = x1 - x0, y1 - y0
        ax.set_xlim(x0 - margins["left"] * x_span, x1 + margins["right"] * x_span)
        ax.set_ylim(y0 - margins["bottom"] * y_span, y1 + margins["top"] * y_span)


def apply_style(fig, style):
    """Restyle a finished figure in place according to `style`."""
    scale = style.get("scale", 1.0) or 1.0

    for role, artist in _role_texts(fig):
        size = style.get(role)
        if size is not None:
            artist.set_fontsize(size)
        if scale != 1.0:
            artist.set_fontsize(artist.get_fontsize() * scale)

    # Tick label sizes are also set on the axis itself: a tick regenerated
    # during the save-time draw would otherwise fall back to the rcParam.
    tick_size = style.get("tick_label")
    if tick_size is not None:
        for ax in fig.axes:
            ax.tick_params(axis="both", which="both", labelsize=tick_size * scale)

    _apply_margins(fig, style.get("margins"))

    figsize = style.get("figsize")
    if figsize is not None:
        fig.set_size_inches(*figsize)

    # Last, and after the resize: anything a plotting function had to position by
    # measuring text cannot be placed until the text is at its final size. A
    # figure that has such an artist leaves the callback here for us to run.
    for hook in getattr(fig, "srf_after_style", ()):
        hook(fig)


@contextmanager
def styled(style):
    """Apply `style` to every figure saved inside the block."""
    if style is None:
        yield
        return

    original_savefig = Figure.savefig

    def savefig(self, *args, **kwargs):
        apply_style(self, style)
        return original_savefig(self, *args, **kwargs)

    Figure.savefig = savefig
    try:
        yield
    finally:
        Figure.savefig = original_savefig


# ============================================================================
# Figure builders — each returns the path the underlying function wrote to.
# ============================================================================

def build_fig1(_config, _args):
    """Recognition accuracy: box plots by operationalization + per-model bars."""
    from scripts.figures.COLM2026 import prototype_compact_figures as compact

    data = compact.load_all()
    with styled(FIGURE_STYLE["fig1"]):
        compact.fig_boxplot_with_grouped_bar(data)
    return compact.OUT_DIR / "boxplot_with_grouped_bar.pdf"


def build_fig2(_config, args):
    """Quality heuristic: score-distance scatter + recognition vs. preference."""
    from scripts.figures.COLM2026 import prototype_compact_figures as compact

    data = compact.load_all()
    with styled(FIGURE_STYLE["fig2"]):
        path = compact.fig_quality_heuristic_combined(data)
        # Alternative treatments of the IND panels (b, d), built at the same
        # style and size so they can be compared directly. See the ind_metric
        # docstring in prototype_compact_figures for what each one plots.
        for metric in args.ind_variants:
            compact.fig_quality_heuristic_combined(data, ind_metric=metric)
        # Same data under the pooled fit, for comparison against the paper's
        # evaluator x domain fixed-effects version.
        for reg in args.regression_variants:
            compact.fig_quality_heuristic_combined(data, regression=reg)
        # Alternative marker encodings for panels (e, f); the paper's version
        # colors every point by task domain.
        for encoding in args.rp_encoding_variants:
            compact.fig_quality_heuristic_combined(data, rp_encoding=encoding)
    return path


def build_fig3(config, args):
    """Training transfer (a, b) above AlpacaEval self-rank change (c, d).

    These were two figures until their captions had grown to repeat each other's
    setup; both report the same training runs. The panels are still drawn by the
    two functions that drew the standalone figures — see
    prototype_combined_transfer, which does the placing.
    """
    from scripts.figures.COLM2026 import prototype_combined_transfer as combined

    output_dir = Path(args.analysis_dir) / _subset_label(config.get("data_subsets")) / "training"
    return combined.build(config, args, output_dir)


# Order here is the order the figures appear in the paper.
FIGURES = {
    "fig1": ("Figure 1: recognition accuracy across operationalizations", build_fig1),
    "fig2": ("Figure 2: quality heuristic", build_fig2),
    "fig3": ("Figure 3: training transfer and AlpacaEval self-rank change", build_fig3),
}


# ============================================================================
# Helpers
# ============================================================================

def _subset_label(data_subsets):
    return data_subsets[0] if data_subsets else "all"


def _label_runs(runs):
    """Attach the display label each uplift figure expects (mirrors analyze_uplift.main)."""
    IDENTITY_DISPLAY = {
        "gpt-oss-120b": "GPT-OSS 120B", "gpt-oss-20b": "GPT-OSS 20B",
        "qwen-3.0-30b": "Qwen 3.0 30B", "qwen-3-30b": "Qwen 3.0 30B",
        "llama-3-1-8b": "Llama 3.1 8B", "ll-3.1-8b": "Llama 3.1 8B",
    }
    OPPONENT_DISPLAY = {
        **IDENTITY_DISPLAY,
        "gpt-oss-120b-thinking": "GPT-OSS 120B", "gpt-oss-20b-thinking": "GPT-OSS 20B",
        "qwen-2.5-7b": "Qwen 2.5 7B", "qwen-2-5-7b": "Qwen 2.5 7B",
    }
    for run in runs:
        opponent = run.get("opponent", "")
        opponent_name = OPPONENT_DISPLAY.get(opponent, opponent)
        if run.get("is_adversarial"):
            identity = run.get("identity_model", "?")
            run["model_label"] = f"{run['base']} (as {IDENTITY_DISPLAY.get(identity, identity)})"
        else:
            run["model_label"] = (f"{run['base']} (vs {opponent_name})"
                                  if opponent_name else run["base"])


def main():
    parser = argparse.ArgumentParser(
        description="Generate the COLM 2026 main-body figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Experiment config YAML (supplies training_dir and data_subsets "
                             "for the training figure)")
    parser.add_argument("--results_dir", default="data/alpaca_eval/results",
                        help="Root of the AlpacaEval judge results")
    parser.add_argument("--analysis_dir", default="data/alpaca_eval/analysis",
                        help="Root the AlpacaEval figures are written under")
    parser.add_argument("--output_dir", default="data/figures/paper",
                        help="Directory the finished figures are collected into")
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES), default=sorted(FIGURES),
                        help="Build only these figures")
    parser.add_argument("--ind-variants", nargs="*", choices=["raw", "attribution"],
                        default=["raw", "attribution"],
                        help="Alternative IND treatments to build alongside Figure 2. "
                             "Pass with no values to build only the paper version.")
    parser.add_argument("--regression-variants", nargs="*", choices=["pooled"], default=["pooled"],
                        help="Alternative regressions to build alongside Figure 2. "
                             "Pass with no values to build only the paper's FE fit.")
    parser.add_argument("--rp-encoding-variants", nargs="*", choices=["model"],
                        default=["model"],
                        help="Alternative marker encodings for Figure 2's panels (e, f), "
                             "whose paper version colors by task domain. Pass with no "
                             "values to build only the paper version.")
    parser.add_argument("--copy-to-paper", action="store_true",
                        help=f"Also copy the figures into {PAPER_FIGURES}")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    built = []
    for key in sorted(args.only, key=list(FIGURES).index):
        description, builder = FIGURES[key]
        print(f"\n{description}")
        path = builder(config, args)
        if not path.exists():
            raise RuntimeError(f"{key}: expected output at {path}, which was not written")

        collected = output_dir / path.name
        shutil.copy2(path, collected)
        print(f"  ✓ {path}")
        print(f"  → {collected}")
        built.append(collected)

    if args.copy_to_paper:
        PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
        print(f"\nCopying into {PAPER_FIGURES}")
        for path in built:
            shutil.copy2(path, PAPER_FIGURES / path.name)
            print(f"  ✓ {path.name}")

    print(f"\n{len(built)} figure(s) in {output_dir}/")


if __name__ == "__main__":
    main()
