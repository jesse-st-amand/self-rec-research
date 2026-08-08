"""Main-body Figure 3: training transfer and AlpacaEval self-rank, in one figure.

This began as a prototype of merging what were then two separate figures — the
training-transfer dot plot (two panels: evaluation-format transfer, task-domain
transfer) and the AlpacaEval self-rank heatmap (two panels: standard training,
adversarial training). Both say what SGTR training does to a model, over the
same set of training runs, and their captions repeated each other's setup, so
they are now one figure: (a) (b) the transfer panels above, (c) (d) the heatmap
panels below. make_paper_figures.build_fig3 calls build() below.

Nothing about either half's content is reimplemented here. The same two
functions the standalone figures used — fig5c_dot_plot_dual_color and
plot_ranking_delta_heatmap_dual_v2 — draw the panels; this script only decides
where on the page they land. So the merge changed the layout and nothing else.

Redirecting them takes two different tricks, because the two functions build
their figures differently:

  - The dot plot calls plt.subplots(1, 2) and lays out with a gridspec, so it
    is enough to hand it two axes that already exist. See draw_into_axes.

  - The heatmap places every axes explicitly in inches, because that is the only
    way to make a cell in panel (b) the same size as a cell in panel (a) when
    the panels have different row counts. Handing it axes would throw that away.
    Instead it is allowed to lay itself out as usual, into a virtual figure of
    the size it asks for, and each rect it places is mapped into a region of the
    real figure — scaled uniformly, so the cells stay square and equal. See
    draw_into_region.

Usage:
    bash scripts/figures/COLM2026/bash/make_paper_figures.sh --only fig3
    bash scripts/figures/COLM2026/bash/prototype_combined_transfer.sh   # writes
        # to data/figures/prototypes instead, for layout work off the paper path
"""

import argparse
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from scripts.figures.COLM2026.make_paper_figures import (
    FIGURE_STYLE, REPO_ROOT, _label_runs, apply_style,
)

# Text sizes live with the other figures', in make_paper_figures.FIGURE_STYLE.
# The two halves come in at different natural sizes and set their text at
# different point sizes, so that entry is what makes them agree.

# ============================================================================
# Layout
# ============================================================================
# Widths in inches. The heatmap half's natural width is fixed by its cell size
# and column count (~15.7in); the dot-plot half has no natural width, so it is
# stretched to match rather than the other way round.

FIG_W = 15.75
TOP_H = 4.6                     # band the dot plots get
BOTTOM_H = 4.4                  # band the heatmaps get
BAND_GAP = 0.2                  # inches between the two bands

# Margins inside the top band, as a fraction of the whole figure width. The
# left margin is what the rotated row labels ("UT PW", "S-GPT") hang into.
TOP_LEFT, TOP_RIGHT, TOP_WSPACE = 0.055, 0.995, 0.14


class _PanelDrawn(BaseException):
    """Raised in place of a panel function's save, to stop it there.

    Derives from BaseException so a broad `except Exception` inside the drawing
    code cannot swallow it.
    """


@contextmanager
def draw_into_axes(axes):
    """Make the next `plt.subplots()` hand back `axes`, and stop at the save.

    For a function that builds its figure with plt.subplots and takes an output
    path rather than an axes. The axes it would have made are replaced by ones
    that already sit where we want them; everything it draws into them is
    unchanged.
    """
    figure = axes[0].figure
    saved = (plt.subplots, plt.savefig, plt.close, plt.tight_layout, Figure.savefig)

    def subplots(*_args, **_kwargs):
        plt.sca(axes[0])
        return figure, (axes if len(axes) > 1 else axes[0])

    def stop(*_args, **_kwargs):
        raise _PanelDrawn

    plt.subplots = subplots
    plt.savefig = stop
    Figure.savefig = stop
    plt.close = lambda *a, **k: None
    plt.tight_layout = lambda *a, **k: None      # would reflow the whole figure
    try:
        yield
    finally:
        (plt.subplots, plt.savefig, plt.close,
         plt.tight_layout, Figure.savefig) = saved


@contextmanager
def draw_into_region(target, rect):
    """Draw a whole figure into `rect` of `target`, at its own proportions.

    `rect` is [left, bottom, width, height] in `target`'s figure coordinates.

    A function that places its axes in inches has already decided the geometry
    that matters — here, that a heatmap cell is the same size in both panels.
    Re-deriving that geometry for a different figure size would be a second copy
    of it, free to disagree. So the function lays itself out exactly as it
    normally would, against the figure size it asked plt.figure for, and each
    rect it places is mapped into `rect`.

    The mapping is a uniform scale, the largest that fits, centred: any other
    mapping would stretch one axis relative to the other and the cells would
    stop being square. So `rect` is a bound on the region, not a promise to
    fill it — give it the aspect the drawn figure has and nothing is wasted.
    """
    saved = (plt.figure, plt.savefig, plt.close, plt.tight_layout,
             Figure.savefig, Figure.add_axes)
    virtual = {}

    def figure(*_args, **kwargs):
        virtual["size"] = kwargs.get("figsize") or _args and _args[0]
        return target

    def add_axes(self, arg, *args, **kwargs):
        if self is not target or "size" not in virtual:
            return saved[5](self, arg, *args, **kwargs)
        left, bottom, width, height = arg
        fw, fh = virtual["size"]
        tw, th = target.get_size_inches()
        x, y, w, h = rect
        scale = min(w * tw / fw, h * th / fh)
        ow, oh = fw * scale / tw, fh * scale / th          # region actually used
        ox, oy = x + (w - ow) / 2, y + (h - oh) / 2
        return saved[5](self, [ox + left * ow, oy + bottom * oh,
                               width * ow, height * oh], *args, **kwargs)

    def stop(*_args, **_kwargs):
        raise _PanelDrawn

    plt.figure = figure
    Figure.add_axes = add_axes
    plt.savefig = stop
    Figure.savefig = stop
    plt.close = lambda *a, **k: None
    plt.tight_layout = lambda *a, **k: None
    try:
        yield
    finally:
        (plt.figure, plt.savefig, plt.close, plt.tight_layout,
         Figure.savefig, Figure.add_axes) = saved


def letter_panels(axes, letters="abcd"):
    """Letter the panels, in the title each already carries.

    In the title rather than floating above its top-left corner, which is where
    a panel letter usually goes: the adversarial heatmap is two columns wide and
    its centred title overhangs the axes on both sides, so a letter placed at the
    axes' left edge lands underneath it. The heatmap function already letters its
    own panels this way, since standalone it is a two-panel figure — those
    letters are (a) and (b), and here the same panels are (c) and (d).
    """
    for letter, ax in zip(letters, axes):
        title = ax.get_title()
        if title.startswith("(") and ") " in title[:5]:
            title = title.split(") ", 1)[1]
        ax.set_title(f"({letter}) {title}")


# ============================================================================
# Data — the same loads build_fig3 and build_fig4 do
# ============================================================================

def load_transfer_rows(config):
    from scripts.alpaca_eval.analyze_uplift import _build_delta_table
    from scripts.figures.COLM2026.prototype_uplift_figures import (
        _discover_training_runs, _load_val_accuracy,
    )

    training_dir = config.get("training_dir", "data/training")
    runs = _discover_training_runs(training_dir=training_dir,
                                   subsets=config.get("data_subsets"))
    if not runs:
        raise RuntimeError(f"No training runs found in {training_dir}")
    _label_runs(runs)
    print(f"  {len(runs)} training runs "
          f"({sum(1 for r in runs if r.get('is_adversarial'))} adversarial)")
    rows = _build_delta_table(runs, _load_val_accuracy)
    print(f"  {len(rows)} transfer data points")
    return rows


def load_ranking(config, args):
    from scripts.alpaca_eval.analyze_self_preference import (
        build_trained_subset_map, list_ranking_judges, load_ranking_self_ranks,
        resolve_mode_results_dir,
    )

    data_subsets = config.get("data_subsets")
    subset = data_subsets[0] if data_subsets else None
    results_dir = resolve_mode_results_dir(Path(args.results_dir), subset, "ranking")
    trained_to_subset = build_trained_subset_map(
        config.get("training_dir", "data/training"), data_subsets)
    judges = list_ranking_judges(results_dir, trained_to_subset, subset)
    if not judges:
        raise RuntimeError(f"No ranking results found in {results_dir}")

    df = load_ranking_self_ranks(results_dir, judges)
    print(f"  {len(df)} judges with self-rank data (of {len(judges)})")
    return df, results_dir


# ============================================================================
# The combined figure
# ============================================================================

def build(config, args, output_dir):
    from scripts.alpaca_eval.analyze_uplift import fig5c_dot_plot_dual_color
    from scripts.alpaca_eval.analyze_self_preference import (
        plot_ranking_delta_heatmap_dual_v2,
    )

    rows = load_transfer_rows(config)
    ranking_df, results_dir = load_ranking(config, args)

    total_h = TOP_H + BAND_GAP + BOTTOM_H
    fig = plt.figure(figsize=(FIG_W, total_h))

    # Top band: the two dot-plot panels, side by side across the full width.
    top_bottom = (BAND_GAP + BOTTOM_H) / total_h
    gs = fig.add_gridspec(1, 2, left=TOP_LEFT, right=TOP_RIGHT,
                          bottom=top_bottom + 0.05, top=0.94, wspace=TOP_WSPACE)
    top_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    with draw_into_axes(top_axes):
        try:
            fig5c_dot_plot_dual_color(rows, Path("."), simple=True)
        except _PanelDrawn:
            pass

    # Bottom band: the heatmap figure, mapped in whole so its cells stay square.
    before = set(map(id, fig.axes))
    with draw_into_region(fig, [0.0, 0.0, 1.0, BOTTOM_H / total_h]):
        try:
            plot_ranking_delta_heatmap_dual_v2(ranking_df, results_dir, Path("."))
        except _PanelDrawn:
            pass
    bottom_axes = [ax for ax in fig.axes if id(ax) not in before]

    letter_panels(top_axes + bottom_axes)
    apply_style(fig, FIGURE_STYLE["fig3"])

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "combined_transfer_ranking.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Experiment config YAML (training_dir and data_subsets)")
    parser.add_argument("--results_dir", default="data/alpaca_eval/results",
                        help="Root of the AlpacaEval judge results")
    parser.add_argument("--output_dir", default="data/figures/prototypes",
                        help="Directory the figure is written to")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    build(config, args, REPO_ROOT / args.output_dir)


if __name__ == "__main__":
    main()
