"""Copy generated figures into the COLM 2026 paper figures directory.

Usage:
    uv run python scripts/figures/COLM2026/copy_figures_to_paper.py
"""

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_FIGURES = REPO_ROOT / "_external" / "COLM_2026_SGTR" / "COLM" / "figures"

# (source relative to repo root, destination filename in paper figures dir)
FIGURES = [
    ("data/figures/prototypes/boxplot_with_grouped_bar.pdf", "boxplot_with_grouped_bar.pdf"),
    ("data/figures/prototypes/quality_heuristic_combined.pdf", "quality_heuristic_combined.pdf"),
    ("data/alpaca_eval/analysis/03_01-and-02/uplift/uplift_5c_dot_plot_dual_color.pdf", "uplift_5c_dot_plot_dual_color.pdf"),
    ("data/alpaca_eval/analysis/03_01-and-02/ranking/ranking_delta_heatmap_dual_v2.pdf", "ranking_delta_heatmap_dual_v2.pdf"),
]


def main():
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

    for src_rel, dst_name in FIGURES:
        src = REPO_ROOT / src_rel
        dst = PAPER_FIGURES / dst_name

        if not src.exists():
            print(f"  ⚠ Missing: {src_rel}")
            continue

        shutil.copy2(src, dst)
        print(f"  ✓ {src_rel} → figures/{dst_name}")

    print(f"\nDone. {len(FIGURES)} figure(s) targeted, output in {PAPER_FIGURES}/")


if __name__ == "__main__":
    main()
