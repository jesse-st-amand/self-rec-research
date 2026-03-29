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
    ("data/alpaca_eval/analysis/01_final/arrow_combined_panels_real_horizontal.pdf", "training_transfer_combined.pdf"),
    ("data/figures/prototypes/boxplot_combined.pdf", "boxplot_combined.pdf"),
    ("data/figures/prototypes/quality_heuristic_combined.pdf", "quality_heuristic_combined.pdf"),
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
