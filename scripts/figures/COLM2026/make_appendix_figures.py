"""Generate the figures that appear in the COLM 2026 appendix.

Companion to make_paper_figures.py, with the same shape and the same style
machinery — this one just points at the appendix's figures instead of the main
body's. Like that script it is a thin driver: every panel is drawn by the same
function the full analysis pipeline calls, so the content is identical to what
the pipeline produces. What this adds is one place to tune text size while
sizing figures for the page (APPENDIX_STYLE below) and one command that
rebuilds and collects the whole set, as vector PDFs.

    key                 paper file                                 appendix label
    paradigm            figures/figure2.pdf                        fig:paradigm_performance
    paradigm-diff       figures/figure2_appx.pdf                   fig:paradigm_performance_difference
    capability          figures/figure3.pdf                        fig:capability_relationship
    transfer            figures/uplift_5c_dot_plot_dual_color.pdf  fig:training_transfer_full
    controlled-bar      figures/figure_appx_controlled_bar.pdf     fig:app_controlled_bar
    controlled-scatter  figures/figure_appx_controlled_scatter.pdf fig:app_controlled_scatter

How the ICML figures are built: those come from bash wrappers under
experiments_eval/ that shell out to the srf-* console scripts, and a subprocess
is out of reach of the savefig interception the styling depends on. So the
driver runs each wrapper with `uv` shimmed out to record the command line it
builds, then calls that same entry point in-process with those arguments. The
pipeline's config handling is never reimplemented, so it cannot drift — which
matters more than it sounds: ICML_08's rank-distance wrapper passes
--exclude_self and ICML_07's does not.

Multipanel figures: every figure here is one, and they used to be assembled by
hand in Google Slides from PNG panels. They are matplotlib figures now. The
pipeline's plot functions each build a whole figure and take an output path
rather than an axes, so rather than fork a few hundred lines of drawing code per
panel, `draw_into` points plt.subplots at the panel we want and drops the
function's own save. See its docstring. Each run is narrowed with --figures to
the one figure that panel needs, which is what makes that redirection sound.

Legends: the analysis figures decode model family, dataset (with that dataset's
r) and the chance line in one legend per figure, of which only the dataset
entries differ between panels. So the panels are asked for --legend_sections
datasets and `shared_legend` draws the rest once at figure level.

Everything is written as PDF so the text stays vector on the page, matching the
main-body figures.

Rank-distance caveat: `controlled-bar` and `controlled-scatter` run the same
rank-distance analysis the pipeline runs, which writes IN PLACE into the
experiment's newest aggregated-data directory — including rank_distance_data.csv,
with no backup (see the warning in 03-rank-distance.sh, and the copies kept in
data/reference/rank_distance_backups/). Narrowing --figures narrows what is
drawn, not what is rewritten.

Usage:
    bash scripts/figures/COLM2026/bash/make_appendix_figures.sh
    bash scripts/figures/COLM2026/bash/make_appendix_figures.sh --only transfer
    bash scripts/figures/COLM2026/bash/make_appendix_figures.sh --copy-to-paper
"""

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from self_rec_framework.scripts.utils import dataset_name_style

from scripts.figures.COLM2026.make_paper_figures import (
    BASE_STYLE, ON_PAGE_PT, PAGE_SCALE, PAGE_TEXT_WIDTH_IN, PAPER_FIGURES,
    REPO_ROOT, _label_runs, _subset_label, apply_style, report_overflow,
)

# ============================================================================
# Experiments the appendix draws on
# ============================================================================
# The reasoning+instruct recognition pair — every ICML-derived appendix figure
# is one of these two experiments, or the contrast between them.

PW_EXPERIMENT = "ICML_07_UT_PW-Q_Rec_NPr_FA_Rsn-Inst"
IND_EXPERIMENT = "ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst"
CONTRAST_DIR = "ICML_07-vs-ICML_08"

EXPERIMENTS_ROOT = Path("experiments_eval/ICML")
AGGREGATED_ROOT = Path("data/analysis/_aggregated_data")

# ============================================================================
# STYLE — edit these while sizing figures for the page.
# ============================================================================
# Identical contract to FIGURE_STYLE in make_paper_figures.py; see the long
# comment there for what each role covers and how `scale`, `figsize` and
# `margins` behave.
#
# Every figure here is on the same footing as main-body Figures 2 and 3: drawn
# at exactly PAGE_SCALE times the size it is printed at, and saved on its canvas
# rather than a tight crop, so an authored ON_PAGE_PT divides by PAGE_SCALE to
# land on COLM's 9pt floor by construction rather than by iteration. `page()`
# below is what states a figsize in printed inches; the width is never anything
# but the text block, since that is what these are included at.
#
# What that footing costs is width. A figure authored this way has exactly 5.5
# printed inches to spend and no way to buy more, so where the old sizes were a
# free parameter the layout is now the only lever — which is why the panels
# below are stacked where they used to sit side by side, and why several of them
# are close to a full page tall. Height is nearly free in an appendix; width is
# not. See each figure's builder for what its own version of that trade was.

def page(width_in, height_in):
    """A figsize in printed inches, authored at PAGE_SCALE times that."""
    return (width_in * PAGE_SCALE, height_in * PAGE_SCALE)


_TEXT = {role: ON_PAGE_PT for role in
         ("title", "axis_label", "tick_label", "legend", "legend_title", "annotation",
          "figure_text")}

APPENDIX_STYLE = {
    # figure2: per-dataset grouped bars over 24 evaluators, pairwise above
    # individual. Very dense along x — two stacked full-width panels, with the
    # legend moved underneath so neither panel gives up width to it.
    "paradigm": {**BASE_STYLE, **_TEXT, "figsize": page(PAGE_TEXT_WIDTH_IN, 7.3)},
    # figure2_appx: same bar geometry, one panel.
    "paradigm-diff": {**BASE_STYLE, **_TEXT, "figsize": page(PAGE_TEXT_WIDTH_IN, 4.4)},
    # figure3: accuracy vs Arena Elo score, pairwise over individual. These were
    # side by side, which at 5.5in total gave each panel 2.4in of axes to carry
    # an x label, a y label, seven x ticks and a two-row legend. Stacked, each
    # gets the full width and the figure pays in height.
    "capability": {**BASE_STYLE, **_TEXT, "figsize": page(PAGE_TEXT_WIDTH_IN, 6.3)},
    # The fully encoded uplift dot plot. Same data as main-body Figure 3 but with
    # the model/condition legend kept, so it needs more room than fig3.
    "transfer": {**BASE_STYLE, **_TEXT, "figsize": page(PAGE_TEXT_WIDTH_IN, 5.0)},
    # Two stacked bar charts, 22 rotated evaluator names along each x axis.
    # Rotation is what lets those survive: what has to clear the label height is
    # the perpendicular gap between neighbouring baselines, not the tick spacing.
    "controlled-bar": {**BASE_STYLE, **_TEXT, "figsize": page(PAGE_TEXT_WIDTH_IN, 7.4)},
    # Four scatter panels: the two score-distance filters over the two paradigms.
    # Kept 2x2 rather than stacked — four full-width rows do not fit on a page,
    # and unlike `capability` these panels' x axis is shared down each column, so
    # only the bottom row spends height on labels.
    "controlled-scatter": {**BASE_STYLE, **_TEXT, "figsize": page(PAGE_TEXT_WIDTH_IN, 7.6)},
}


class _PanelDrawn(BaseException):
    """Raised in place of a panel function's first save, to stop it there.

    Derives from BaseException so a broad `except Exception` inside the
    analysis code cannot swallow it.
    """


@contextmanager
def draw_into(ax):
    """Make the next `plt.subplots()` hand back `ax`, and stop at the first save.

    The pipeline's plot functions each build a whole figure — `fig, ax =
    plt.subplots(...)` at the top, `plt.tight_layout()` and `plt.savefig(...)`
    at the bottom — and take an output path rather than an axes. Forking a few
    hundred lines of bar-chart drawing per panel just to change where it lands
    would give the appendix its own copy of the analysis, free to drift from the
    pipeline. Pointing plt.subplots at the panel we want costs nothing and keeps
    one implementation.

    Stopping at the first save matters as much as redirecting it. Several of
    these functions save the finished figure and then keep mutating it to write
    `_no_r` and `_minimal` variants — strip the legend, strip the axis labels,
    save again. Merely suppressing the saves would leave the panel in whichever
    state the last variant wanted, which is the stripped one. Raising out of the
    function at its first save leaves the labelled figure standing.

    Only sound when the run draws one figure before that save, which is what the
    `--figures` narrowing on each entry point guarantees.
    """
    figure = ax.figure
    saved = (plt.subplots, plt.savefig, plt.close, plt.tight_layout, Figure.savefig)

    def subplots(*_args, **_kwargs):
        plt.sca(ax)
        return figure, ax

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
def capture_save():
    """Hand back the figure a function was about to save, instead of saving it.

    `draw_into` is for functions whose panels we place ourselves. This is for the
    one figure we take whole — the uplift dot plot, which is already laid out the
    way the appendix wants it. What it is not already doing is fitting a page: it
    sizes itself in absolute inches and hangs its legend below the canvas at a
    negative anchor, both of which are answers to `bbox_inches="tight"` and both
    of which this module has to undo. Yields a one-element list that holds the
    figure once the function reaches its save.
    """
    captured = []
    saved = (plt.savefig, plt.close, Figure.savefig)

    def stop(self, *_args, **_kwargs):
        captured.append(self)
        raise _PanelDrawn

    Figure.savefig = stop
    plt.savefig = lambda *a, **k: stop(plt.gcf())
    plt.close = lambda *a, **k: None
    try:
        yield captured
    finally:
        plt.savefig, plt.close, Figure.savefig = saved


# ============================================================================
# Running the framework's analysis entry points in-process
# ============================================================================

SRF_MODULES = {
    "srf-plot-aggregated-performance":
        "self_rec_framework.scripts.analysis.plot_aggregated_performance",
    "srf-performance-vs-size":
        "self_rec_framework.scripts.analysis.performance_vs_size",
    "srf-rank-distance":
        "self_rec_framework.scripts.analysis.rank_distance",
    "srf-experiment-contrast":
        "self_rec_framework.scripts.analysis.experiment_contrast",
}


@contextmanager
def _argv(argv):
    """Run the block with sys.argv replaced — these mains are argparse-driven."""
    original = sys.argv
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = original


def run_srf(argv, into):
    """Call an srf-* entry point in-process, drawing its figure into `into`.

    `argv` is a full command line, entry point first — normally one that came
    back from capture_pipeline_argv, narrowed with --figures to the single
    figure this panel wants.
    """
    argv = [str(a) for a in argv]
    module = importlib.import_module(SRF_MODULES[argv[0]])
    print("  $ " + " ".join(argv[:5]) + (" ..." if len(argv) > 5 else ""))
    try:
        with _argv(argv), draw_into(into):
            module.main()
    except _PanelDrawn:
        pass                     # the panel is drawn; the rest of the run is not ours


# ============================================================================
# Taking the pipeline's arguments from the pipeline
# ============================================================================
# The bash wrappers under experiments_eval/ build their arguments from config.sh,
# the experiment's config.yaml, and whichever timestamped directory is newest.
# Rebuilding any of that here would be a second copy free to drift from the
# first — and it does drift: ICML_08's rank-distance wrapper passes
# --exclude_self and ICML_07's does not, a difference that changes what lands in
# the figure. So run the wrapper for real with `uv` shimmed out, and use the
# argument list it built. Nothing is reimplemented and nothing can fall behind.

_CAPTURE_SHIM = r"""#!/usr/bin/env bash
# Stands in for `uv` while a pipeline wrapper runs: records the command line the
# wrapper built, then exits without running it. \1 separates invocations, \0
# separates arguments; neither occurs in a path or a model name.
[ "$1" = run ] && shift
{ printf '%s\0' "$@"; printf '\1'; } >> "$SRF_ARGV_CAPTURE"
"""


def capture_pipeline_argv(script):
    """Every srf-* command line `script` would run, without running any of them."""
    script = REPO_ROOT / script
    if not script.exists():
        raise FileNotFoundError(f"No pipeline script at {script}")

    with tempfile.TemporaryDirectory() as tmp:
        shim_dir = Path(tmp) / "bin"
        shim_dir.mkdir()
        shim = shim_dir / "uv"
        shim.write_text(_CAPTURE_SHIM)
        shim.chmod(0o755)

        capture = Path(tmp) / "argv"
        capture.touch()

        result = subprocess.run(
            ["bash", str(script)], cwd=REPO_ROOT, capture_output=True, text=True,
            env={**os.environ,
                 "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
                 "SRF_ARGV_CAPTURE": str(capture)})
        if result.returncode != 0:
            raise RuntimeError(
                f"{script} exited {result.returncode}:\n"
                f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}")

        calls = []
        for record in capture.read_text().split("\1"):
            args = [a for a in record.split("\0") if a]
            if args:
                calls.append(args)
        return calls


def pipeline_argv(script, contains=None):
    """The one command line `script` builds, or the one matching `contains`.

    A wrapper that makes several calls (00b runs the plotter over the
    performance table and again over the deviation table) needs `contains` to
    say which one feeds the appendix.
    """
    calls = capture_pipeline_argv(script)
    if contains is not None:
        calls = [c for c in calls if any(contains in a for a in c)]
    if len(calls) != 1:
        raise RuntimeError(
            f"Expected exactly one matching invocation in {script}, found {len(calls)}"
            + (f" (contains={contains!r})" if contains else ""))
    return calls[0]


def argv_value(argv, flag):
    """The value following `flag`, as a Path."""
    return Path(argv[argv.index(flag) + 1])


def read_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def latest_aggregated_dir(name):
    """Newest timestamped directory under _aggregated_data/<name>/."""
    base = REPO_ROOT / AGGREGATED_ROOT / name
    if not base.is_dir():
        raise FileNotFoundError(
            f"No aggregated data for {name} — run 00a-performance_aggregate.sh first")
    stamps = sorted((d for d in base.iterdir() if d.is_dir()),
                    key=lambda d: d.stat().st_mtime, reverse=True)
    if not stamps:
        raise FileNotFoundError(f"No timestamp directories in {base}")
    return stamps[0]


def analysis_script(experiment, name):
    return EXPERIMENTS_ROOT / experiment / "bash/analysis/_inter-dataset" / name


# ============================================================================
# Assembling multipanel figures
# ============================================================================

def label_panels(axes, keep_legend=None, titles=None):
    """Drop the per-panel analysis titles and letter the panels instead.

    Each pipeline function titles its figure with the full experiment name over
    three or four lines, which is right for a standalone PNG and wrong inside a
    captioned figure. `titles` optionally gives each panel a short replacement,
    for a composite whose panels differ in a way the caption alone would make
    the reader count out. `keep_legend` is the index of the one panel that keeps
    its legend, or None to keep them all.
    """
    for index, ax in enumerate(axes):
        ax.set_title("")
        letter = f"({'abcdefgh'[index]})"
        label = f"{letter} {titles[index]}" if titles else letter
        ax.text(0.0, 1.01, label, transform=ax.transAxes,
                ha="left", va="bottom", fontweight="bold")
        if keep_legend is not None and index != keep_legend:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()


def _below_everything(fig, key, pad):
    """Figure y just under the lowest ink in `fig`, once `key`'s style is on.

    Where a figure-level legend goes has to be measured, not guessed: what it has
    to clear is the bottom panels' rotated tick labels, axis titles and (on the
    scatter figures) their own legends, all of which hang below the axes by an
    amount that depends on how long the longest label is.

    The style has to go on first, because the measurement is only meaningful at
    the sizes the figure will be saved at — the panel functions set their own
    18pt ticks and the appendix asks for 25. Applying it here and again in
    save_panels is harmless: every size in APPENDIX_STYLE is absolute.
    """
    apply_style(fig, APPENDIX_STYLE[key])
    fig.canvas.draw()                    # tight bboxes are only real once drawn
    renderer = fig.canvas.get_renderer()
    lowest = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes)
    return fig.transFigure.inverted().transform((0, lowest))[1] - pad


def legend_under_xaxis(ax, key, pad=0.04, ncol=None):
    """Drop a panel's own legend clear of its x-axis, ticks and label included.

    The analysis functions anchor it a fixed fraction of the axes height below
    the frame, which is a guess at how tall the x-axis furniture is; at appendix
    sizes the guess is short and the legend lands on the axis label. Measuring
    the axis instead means the placement survives a font-size change. `pad` is
    the gap left under it, as a fraction of the panel's height.

    `ncol` redraws it at a given number of columns, keeping the entries and their
    order. Worth doing wherever a legend fits on one row: under a stacked panel
    every row it saves is a row the panels get back, and height is what a
    full-width layout is short of.
    """
    legend = ax.get_legend()
    if legend is None:
        raise RuntimeError("Panel has no legend to move — was it drawn with --no_legend?")
    if ncol is not None:
        handles = list(legend.legend_handles)
        labels = [text.get_text() for text in legend.get_texts()]
        title = legend.get_title().get_text() or None
        legend.remove()
        legend = ax.legend(handles=handles, labels=labels, title=title, ncol=ncol,
                           loc="upper center", framealpha=0.9)
    apply_style(ax.figure, APPENDIX_STYLE[key])     # measure at the saved sizes
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    # The x-axis alone, so the legend's current position is not what we measure.
    bottom = ax.xaxis.get_tightbbox(renderer).y0
    y = ax.transAxes.inverted().transform((0, bottom))[1] - pad
    legend.set_bbox_to_anchor((0.5, y), transform=ax.transAxes)
    return legend


def legend_below(fig, ax, key, ncol=None, pad=0.02, **legend_kwargs):
    """Move a panel's legend out to a row under the whole figure.

    The analysis functions hang their legend in a column to the right of the
    axes, which is right for a standalone figure and wrong here: that column is
    charged against the figure's width, and everything in the appendix is dense
    along x — two dozen rotated evaluator names to a panel. Underneath, the
    legend costs height instead, which nothing else is competing for.

    Takes the legend the panel already drew rather than rebuilding it, so the
    entries and their order stay whatever the analysis function decided, and
    anchors it below everything else in the figure — see _below_everything.
    `pad` is the gap left under that, as a fraction of figure height.
    """
    legend = ax.get_legend()
    if legend is None:
        raise RuntimeError("Panel has no legend to move — was it drawn with --no_legend?")
    handles = list(legend.legend_handles)
    labels = [text.get_text() for text in legend.get_texts()]
    title = legend.get_title().get_text() or None
    legend.remove()                      # so it is not measured where it was

    y = _below_everything(fig, key, pad)
    return fig.legend(handles=handles, labels=labels, title=title,
                      loc="upper center", bbox_to_anchor=(0.5, y),
                      ncol=ncol or len(handles), **legend_kwargs)


def shared_legend(fig, ax, key, sections=("models", "misc"), pad=0.02, **legend_kwargs):
    """Draw the legend sections every panel shares once, at figure level.

    The analysis scatter figures decode three things: point colour (model
    family), point shape and fit-line colour (dataset, carrying that dataset's
    r), and the chance line. Only the dataset entries differ between panels, so
    the panels are asked for that section alone (--legend_sections datasets) and
    the rest is drawn here from the section handles the framework stashes on the
    axes. Four copies of a seven-entry model-family column is most of what makes
    the standalone versions of these figures legend-heavy.

    Goes below everything else, the panels' own dataset legends included — those
    are two rows deep and hang well under the axes, so a fixed anchor lands on
    top of their second row. See _below_everything.
    """
    groups = getattr(ax, "srf_legend_sections", None)
    if not groups:
        raise RuntimeError(
            "No legend sections on the panel axes — the analysis function is "
            "expected to stash them via add_section_legend")

    handles, labels = [], []
    for name in sections:
        section = groups.get(name)
        if not section:
            continue
        handles.extend(section[0])
        labels.extend(section[1])

    y = _below_everything(fig, key, pad)
    return fig.legend(handles=handles, labels=labels,
                      loc="upper center", bbox_to_anchor=(0.5, y), **legend_kwargs)


def save_panels(fig, key, name, output_dir):
    """Style, lay out and write a multipanel figure as PDF."""
    apply_style(fig, APPENDIX_STYLE[key])
    path = Path(output_dir) / "_build" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    report_overflow(fig, name)
    # The canvas, not bbox_inches="tight" — see the note above APPENDIX_STYLE.
    # A tight crop is set by whichever artist reaches furthest, so the saved
    # width, and with it the factor LaTeX scales every glyph by, moves whenever
    # a label gets a character longer.
    fig.savefig(path)
    plt.close(fig)
    return path


def new_panels(key, nrows, ncols, **gridspec):
    """A blank multipanel figure at this key's figsize.

    The panel functions' own `plt.tight_layout()` is suppressed while they draw,
    so spacing is set here — `hspace` especially, since a panel whose evaluator
    names are rotated 45 degrees needs room below it or they run into the next
    panel.
    """
    figsize = APPENDIX_STYLE[key].get("figsize")
    fig, _ = plt.subplots(nrows, ncols, figsize=figsize,
                          gridspec_kw=gridspec or None)
    return fig, fig.axes


# ============================================================================
# Figure builders — each returns [(written path, name in the paper), ...]
# ============================================================================

def build_paradigm(_config, args):
    """figure2.pdf: recognition accuracy by dataset, pairwise (a) over individual (b)."""
    # Margins here are a budget, not a look. Saving the canvas means whatever
    # hangs below a panel has to be given room rather than discovered afterwards,
    # and what hangs below these is 24 evaluator names rotated 45 degrees: the
    # longest is "Gemini 2.0 Flash Lite (R)", 1.5in at 9pt, so 1.06in of drop.
    # The bottom margin holds that for the lower panel plus the shared legend;
    # hspace holds it for the upper one.
    fig, axes = new_panels("paradigm", 2, 1, hspace=0.58, bottom=0.245,
                           left=0.115, right=0.99, top=0.96)
    for ax, experiment in zip(axes, (PW_EXPERIMENT, IND_EXPERIMENT)):
        # The wrapper plots the performance table and then the deviation table;
        # only the first is a panel of figure2.
        argv = pipeline_argv(analysis_script(experiment, "00b-plot_performance_aggregate.sh"),
                             contains="aggregated_performance.csv")
        run_srf([*argv, "--figures", "aggregated_performance_grouped"], into=ax)

    # The two panels order their models differently, so both keep their tick
    # labels; only the shared axis title and legend are deduplicated.
    label_panels(axes, keep_legend=0)
    axes[0].set_xlabel("")
    legend_below(fig, axes[0], "paradigm", framealpha=0.9)
    return [(save_panels(fig, "paradigm", "figure2.pdf", args.output_dir), "figure2.pdf")]


def build_paradigm_diff(_config, args):
    """figure2_appx.pdf: pairwise minus individual, per dataset."""
    fig, axes = new_panels("paradigm-diff", 1, 1, bottom=0.38,
                           left=0.125, right=0.99, top=0.965)
    argv = pipeline_argv(EXPERIMENTS_ROOT / "comparisons" / CONTRAST_DIR
                         / "00-performance_contrast.sh")
    run_srf([*argv, "--figures", "performance_contrast_grouped"], into=axes[0])

    axes[0].set_title("")
    legend_below(fig, axes[0], "paradigm-diff", framealpha=0.9)
    return [(save_panels(fig, "paradigm-diff", "figure2_appx.pdf", args.output_dir),
             "figure2_appx.pdf")]


def build_capability(_config, args):
    """figure3.pdf: recognition accuracy against Arena Elo score, pairwise (a) over individual (b).

    Stacked rather than side by side. Half of 5.5in leaves each panel about 2.4in
    of axes, and at 9pt "Evaluator LM Arena Score" alone is 1.4in of that; the
    per-dataset legend under each panel is wider still. Both fit at full width,
    and the second row is cheaper here than anywhere else in the paper.
    """
    fig, axes = new_panels("capability", 2, 1, hspace=0.42, bottom=0.20,
                           left=0.115, right=0.985, top=0.955)
    for ax, experiment in zip(axes, (PW_EXPERIMENT, IND_EXPERIMENT)):
        argv = pipeline_argv(analysis_script(experiment, "02-performance_vs_size.sh"))
        run_srf([*argv, "--figures", "performance_vs_arena_score",
                 "--legend_sections", "datasets"], into=ax)

    label_panels(axes, titles=("Pairwise", "Individual"))
    axes[0].set_xlabel("")               # the panels share an x axis now
    for ax in axes:
        # One row of four rather than two of two: "BCB (r=0.65)" and its handle
        # come to 1.1in at 9pt, so four of them fit across 5.5in with room over.
        legend_under_xaxis(ax, "capability", ncol=4)
    shared_legend(fig, axes[0], "capability", ncol=4, framealpha=0.9)
    return [(save_panels(fig, "capability", "figure3.pdf", args.output_dir), "figure3.pdf")]


def build_transfer(config, args):
    """Training transfer with every point's provenance encoded (appendix version).

    Same data and same function as main-body Figure 3; simple=False keeps the
    model/condition encoding instead of collapsing to one marker.
    """
    from scripts.alpaca_eval.analyze_uplift import _build_delta_table, fig5c_dot_plot_dual_color
    from scripts.figures.COLM2026.prototype_uplift_figures import (
        _discover_training_runs, _load_val_accuracy,
    )

    training_dir = config.get("training_dir", "data/training")
    data_subsets = config.get("data_subsets")

    runs = _discover_training_runs(training_dir=training_dir, subsets=data_subsets)
    if not runs:
        raise RuntimeError(f"No training runs found in {training_dir} (subsets={data_subsets})")
    _label_runs(runs)
    print(f"  {len(runs)} training runs "
          f"({sum(1 for r in runs if r.get('is_adversarial'))} adversarial)")

    rows = _build_delta_table(runs, _load_val_accuracy)
    print(f"  {len(rows)} data points")

    output_dir = Path(args.analysis_dir) / _subset_label(data_subsets) / "uplift"
    output_dir.mkdir(parents=True, exist_ok=True)

    with capture_save() as captured:
        try:
            fig5c_dot_plot_dual_color(rows, output_dir, simple=False)
        except _PanelDrawn:
            pass
    if not captured:
        raise RuntimeError("fig5c_dot_plot_dual_color returned without saving a figure")
    fig = captured[0]

    # Its own legend is one block of thirteen entries, which columns badly: the
    # five model entries carry a wrapped two-line label and the eight shape
    # entries a single line, so any column count leaves one column twice the
    # height of the others and the rest of the box empty. Split by the two things
    # being keyed — who the model is, and what it was trained on — each row is
    # then uniform and can be packed to its own width.
    legend = fig.legends[0]
    entries = list(zip(legend.legend_handles, (t.get_text() for t in legend.get_texts())))
    legend.remove()
    models = [e for e in entries if not e[1].startswith("Trained:")]
    trained = [e for e in entries if e[1].startswith("Trained:")]

    # Three columns for the models, four for the shapes: a model entry is a name
    # over a parenthesised opponent, about 1.6in at 9pt, and a shape entry is
    # "Trained: UT IND" at 1.0in. Both come to roughly 4.8in of the 5.5in there is.
    #
    # Stacked by measuring rather than at two fixed anchors. What the first row
    # has to clear is the panels' x label, and what the second has to clear is
    # however tall the first turned out to be; neither is known until the text is
    # at its final size, and a guess at either lands a legend on top of the thing
    # above it, inside the canvas, where nothing warns about it.
    fig.subplots_adjust(left=0.125, right=0.99, top=0.945, bottom=0.36, wspace=0.22)
    y = _below_everything(fig, "transfer", pad=0.02)
    for row, ncol in ((models, 3), (trained, 4)):
        legend = fig.legend(handles=[h for h, _ in row], labels=[l for _, l in row],
                            loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
                            framealpha=0.9, columnspacing=1.5, handletextpad=0.5)
        # Restyle before measuring: a legend built here starts at the rcParam
        # size, and measured there it reports a box less than half the height it
        # will be once save_panels puts it at ON_PAGE_PT. The row below would
        # then be placed inside it.
        apply_style(fig, APPENDIX_STYLE["transfer"])
        fig.canvas.draw()
        bottom = legend.get_window_extent(fig.canvas.get_renderer()).y0
        y = fig.transFigure.inverted().transform((0, bottom))[1] - 0.02

    name = "uplift_5c_dot_plot_dual_color.pdf"
    return [(save_panels(fig, "transfer", name, args.output_dir), name)]


_RANK_DISTANCE_ARGV = {}


def rank_distance_argv(experiment, *extra):
    """The command line 03-rank-distance.sh builds for `experiment`, plus `extra`.

    Cached because each panel of the two score-controlled figures needs the same
    line with a different --figures, and capturing it runs the wrapper.

    Every one of those runs rewrites rank_distance_data.csv and
    score_distance_data.csv in the experiment's newest aggregated directory —
    the same thing 03-rank-distance.sh does, and the reason that script carries
    a warning about it. --figures narrows which figures are drawn, not what the
    run writes beside them, so the directory's other rank-distance figures are
    left as whatever the last full run produced.
    """
    if experiment not in _RANK_DISTANCE_ARGV:
        _RANK_DISTANCE_ARGV[experiment] = pipeline_argv(
            analysis_script(experiment, "03-rank-distance.sh"))
    return [*_RANK_DISTANCE_ARGV[experiment], *extra]


def build_controlled_bar(_config, args):
    """Accuracy by evaluator, restricted to pairings within +/-20 Arena Elo points.

    Stacked rather than side by side: 22 evaluator names along x is far too dense
    to survive being squeezed into half a text width.
    """
    fig, axes = new_panels("controlled-bar", 2, 1, hspace=0.54, bottom=0.228,
                           left=0.115, right=0.99, top=0.96)
    for index, (ax, experiment) in enumerate(zip(axes, (PW_EXPERIMENT, IND_EXPERIMENT))):
        # The panels' dataset/significance/chance entries are identical, so only
        # the first draws them.
        argv = rank_distance_argv(experiment, "--figures", "score_distance_grouped_bar_chart")
        if index:
            argv.append("--no_legend")
        print(f"  ⚠ rewriting rank-distance data in {argv_value(argv, '--output_dir')}")
        run_srf(argv, into=ax)

    # The individual panel is adjusted for self-recognition bias; that belongs in
    # the caption, not spelled out in a panel title long enough to reach halfway
    # across the panel above's evaluator names.
    label_panels(axes, titles=("Pairwise", "Individual"))
    axes[0].set_xlabel("")           # both panels order evaluators the same way
    legend_below(fig, axes[0], "controlled-bar", framealpha=0.9)
    name = "figure_appx_controlled_bar.pdf"
    return [(save_panels(fig, "controlled-bar", name, args.output_dir), name)]


def build_controlled_scatter(_config, args):
    """Accuracy vs evaluator Arena Elo score under the two score-distance filters.

    Rows are the filters — symmetric window above, the asymmetric one that drops
    the easy pairings below — and columns are the two paradigms.
    """
    within = "score_distance_filtered_evaluator_score"
    positive = "score_distance_filtered_evaluator_score_positive"
    panels = [
        (PW_EXPERIMENT, within), (IND_EXPERIMENT, within),
        (PW_EXPERIMENT, positive), (IND_EXPERIMENT, positive),
    ]

    # hspace has three things to clear, not one: the top row's tick labels, then
    # its two-row dataset legend, then the bottom row's two-line title. Sized for
    # the ticks alone the legend lands on the title — which is inside the canvas,
    # so nothing warns about it.
    fig, axes = new_panels("controlled-scatter", 2, 2, hspace=0.62, wspace=0.30,
                           bottom=0.185, left=0.105, right=0.985, top=0.93)
    for ax, (experiment, figure) in zip(axes, panels):
        run_srf(rank_distance_argv(experiment, "--figures", figure,
                                   "--legend_sections", "datasets"), into=ax)

    # Wrapped after the paradigm: half of 5.5in is 2.6in, and
    # "(b) Individual, score distance ±20" set on one line at 9pt is 3.0in, which
    # ran off the right edge of the figure.
    label_panels(axes, titles=("Pairwise,\nscore distance ±20",
                               "Individual,\nscore distance ±20",
                               "Pairwise,\nscore distance < 20",
                               "Individual,\nscore distance < 20"))
    for ax in (axes[0], axes[1]):
        ax.set_xlabel("")            # the columns share an x axis
    for ax in (axes[2], axes[3]):
        # One short line rather than the analysis function's two. The second line
        # is what the panel legends underneath were landing on, and spelling the
        # direction out overhangs the right panel far enough to be cropped off
        # the page — the caption says which way is better.
        ax.set_xlabel("Evaluator LM Arena Score")
    for ax in (axes[1], axes[3]):
        # The individual panels' own label spells out the self-score averaging,
        # which takes four lines at this width and is in the caption anyway.
        ax.set_ylabel("Recognition Accuracy")
    for ax in axes:
        # Measured rather than anchored at a fixed -0.26 of the axes height: that
        # fraction was a stand-in for how tall the x-axis furniture is, and it is
        # only right at one panel height. The bottom row now carries an axis label
        # the top row does not, so no single fraction is right for both.
        legend_under_xaxis(ax, "controlled-scatter", pad=0.05, ncol=2)
    shared_legend(fig, axes[0], "controlled-scatter", ncol=4, framealpha=0.9)
    name = "figure_appx_controlled_scatter.pdf"
    return [(save_panels(fig, "controlled-scatter", name, args.output_dir), name)]


# Order here is the order the figures appear in the appendix.
FIGURES = {
    "paradigm": ("Recognition performance across paradigms (fig:paradigm_performance)",
                 build_paradigm),
    "paradigm-diff": ("Difference in performance (fig:paradigm_performance_difference)",
                      build_paradigm_diff),
    "capability": ("Model capability vs recognition (fig:capability_relationship)",
                   build_capability),
    "transfer": ("Training transfer, fully encoded (fig:training_transfer_full)",
                 build_transfer),
    "controlled-bar": ("Score-controlled performance by model (fig:app_controlled_bar)",
                       build_controlled_bar),
    "controlled-scatter": ("Capability-recognition under control (fig:app_controlled_scatter)",
                           build_controlled_scatter),
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate the COLM 2026 appendix figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Experiment config YAML (supplies training_dir and data_subsets "
                             "for the training-transfer figure)")
    parser.add_argument("--analysis_dir", default="data/alpaca_eval/analysis",
                        help="Root the AlpacaEval figures are written under")
    parser.add_argument("--output_dir", default="data/figures/appendix",
                        help="Directory the finished figures are collected into")
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES), default=sorted(FIGURES),
                        help="Build only these figures")
    parser.add_argument("--copy-to-paper", action="store_true",
                        help=f"Also copy the figures into {PAPER_FIGURES}")
    args = parser.parse_args()

    config = read_yaml(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collected = []
    # Every appendix figure names the task domains by the abbreviations the
    # captions define (WS, S-GPT, PKU, BCB) rather than in full. The standalone
    # pipeline figures keep the full names; this only covers what is built here.
    with dataset_name_style("short"):
        for key in sorted(args.only, key=list(FIGURES).index):
            description, builder = FIGURES[key]
            print(f"\n{description}")
            for path, paper_name in builder(config, args):
                if not path.exists():
                    raise RuntimeError(f"{key}: expected output at {path}, which was not written")
                destination = output_dir / paper_name
                shutil.copy2(path, destination)
                print(f"  ✓ {path}")
                print(f"  → {destination}")
                collected.append(destination)

    if args.copy_to_paper:
        PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
        print(f"\nCopying into {PAPER_FIGURES}")
        for path in collected:
            shutil.copy2(path, PAPER_FIGURES / path.name)
            print(f"  ✓ {path.name}")

    print(f"\n{len(collected)} figure(s) in {output_dir}/")


if __name__ == "__main__":
    main()
