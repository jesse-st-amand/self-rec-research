"""Copy generated figures into the COLM 2026 paper figures directory.

The figures themselves are built by make_paper_figures.py (main body) and
make_appendix_figures.py (appendix), both of which can copy into the paper
directly with --copy-to-paper. This script is the standalone version: it copies
whatever those two last wrote, without rebuilding anything.

Usage:
    uv run python scripts/figures/COLM2026/copy_figures_to_paper.py
    uv run python scripts/figures/COLM2026/copy_figures_to_paper.py --only appendix
"""

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_FIGURES = REPO_ROOT / "_external" / "COLM_2026_SGTR" / "figures"

# Where each driver collects its finished figures.
PAPER_OUTPUT = Path("data/figures/paper")
APPENDIX_OUTPUT = Path("data/figures/appendix")

# (source relative to repo root, destination filename in paper figures dir)
MAIN_BODY = [
    ("data/figures/prototypes/boxplot_with_grouped_bar.pdf", "boxplot_with_grouped_bar.pdf"),
    ("data/figures/prototypes/quality_heuristic_combined.pdf", "quality_heuristic_combined.pdf"),
    ("data/alpaca_eval/analysis/03_01-and-02/training/combined_transfer_ranking.pdf", "combined_transfer_ranking.pdf"),
]

# make_appendix_figures.py already writes these under their paper names, so the
# copy is name-for-name. Keep this list in step with its FIGURES table.
APPENDIX = [
    (APPENDIX_OUTPUT / name, name)
    for name in [
        # Multipanel figures assembled by the appendix driver.
        "figure2.pdf",                     # recognition accuracy across paradigms
        "figure2_appx.pdf",                # pairwise minus individual
        "figure3.pdf",                     # accuracy vs LM Arena Elo score
        # Fully encoded version of the main body's training-transfer figure.
        "uplift_5c_dot_plot_dual_color.pdf",
        # Score-controlled figures: pairings within +/-20 Arena Elo points.
        "figure_appx_controlled_bar.pdf",       # accuracy by evaluator
        "figure_appx_controlled_scatter.pdf",   # accuracy vs evaluator Elo score
    ]
]

GROUPS = {"main": MAIN_BODY, "appendix": APPENDIX}


def copy_group(name, figures):
    print(f"\n{name}")
    copied = 0
    for src_rel, dst_name in figures:
        src = REPO_ROOT / src_rel
        if not src.exists():
            print(f"  ⚠ Missing: {src_rel}")
            continue
        shutil.copy2(src, PAPER_FIGURES / dst_name)
        print(f"  ✓ {src_rel} → figures/{dst_name}")
        copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="+", choices=sorted(GROUPS), default=sorted(GROUPS),
                        help="Copy only these groups")
    args = parser.parse_args()

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

    total = copied = 0
    for name in sorted(args.only, key=list(GROUPS).index):
        figures = GROUPS[name]
        copied += copy_group(name, figures)
        total += len(figures)

    print(f"\nDone. {copied}/{total} figure(s) copied into {PAPER_FIGURES}/")


if __name__ == "__main__":
    main()
