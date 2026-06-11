"""Copy figures referenced by reports/msj_ica_findings.md into reports/msj_ica_figures/.

Each source path is mapped to a distinguishable filename so the report can pull
all figures from a single sibling directory.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST_DIR = REPO_ROOT / "reports" / "msj_ica_figures"

FIGURES: list[tuple[str, str]] = [
    (
        "data/msj/results/aggregate/asr_dot_arrow_by_model_by_op_notitle.png",
        "fig_01_msj_asr_by_model_op.png",
    ),
    (
        "data/ica/results/joint/all/SGTR_02_trained-OP_eval-on_self-same-OP/all/dot_arrows_notitle.png",
        "fig_02_sgtr02_full_dot_arrows.png",
    ),
    (
        "data/ica/results/joint/all/SGTR_02_trained-OP_eval-on_self-same-OP/all/per-tag-shot-avg/dot_arrows_notitle.png",
        "fig_03_sgtr02_per_tag_dot_arrows.png",
    ),
    (
        "data/ica/results/joint/all/SGTR_02_trained-OP_eval-on_self-same-OP/all/per-tag-shot-avg/dot_arrows_diff_notitle.png",
        "fig_04_sgtr02_per_tag_dot_arrows_diff.png",
    ),
    (
        "data/ica/results/joint/all/SGTR_02_trained-OP_eval-on_self-same-OP/multi-op/per-tag-shot-avg/dot_arrows_diff_notitle.png",
        "fig_05_sgtr02_multi_op_dot_arrows_diff.png",
    ),
    (
        "data/mmlu_ica/results/joint/per-tag-shot-avg/all/dot_arrows_notitle.png",
        "fig_06_mmlu_per_tag_dot_arrows.png",
    ),
    (
        "data/mmlu_ica/results/joint/per-tag-shot-avg/multi-op/dot_arrows_notitle.png",
        "fig_07_mmlu_multi_op_per_tag_dot_arrows.png",
    ),
    (
        "data/ica/results/joint/all/SGTR_02_trained-OP_eval-on_self-same-OP/randlabels/per-tag-shot-avg/dot_arrows_diff_notitle.png",
        "fig_08_sgtr02_randlabels_dot_arrows_diff.png",
    ),
    (
        "data/ica/results/joint/all/SGTR_03_trained-AT-IND_eval-on_all-OPs/all/per-tag-shot-avg/dot_arrows_diff_notitle.png",
        "fig_09_sgtr03_cross_op_dot_arrows_diff.png",
    ),
    (
        "data/ica/results/joint/all/SGTR_07_trained-OP-ShareGPT_eval-on_self-same-OP-WikiSum_ICA-ShareGPT/all/per-tag-shot-avg/dot_arrows_diff_notitle.png",
        "fig_10_sgtr07_ood_sharegpt_ice_dot_arrows_diff.png",
    ),
    (
        "data/ica/results/joint/all/SGTR_09_trained-OP-ShareGPT_eval-on_self-same-OP-WikiSum_ICA-WikiSum/all/per-tag-shot-avg/dot_arrows_diff_notitle.png",
        "fig_11_sgtr09_ood_wikisum_ice_dot_arrows_diff.png",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing.",
    )
    args = parser.parse_args()

    if not args.dry_run:
        DEST_DIR.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    copied = 0
    for src_rel, dest_name in FIGURES:
        src = REPO_ROOT / src_rel
        dest = DEST_DIR / dest_name
        if not src.is_file():
            missing.append(src_rel)
            print(f"MISSING  {src_rel}")
            continue
        if args.dry_run:
            print(f"WOULD    {src_rel} -> reports/msj_ica_figures/{dest_name}")
        else:
            shutil.copy2(src, dest)
            print(f"COPIED   {src_rel} -> reports/msj_ica_figures/{dest_name}")
            copied += 1

    print()
    print(f"{copied}/{len(FIGURES)} copied" + (" (dry run)" if args.dry_run else ""))
    if missing:
        raise SystemExit(f"{len(missing)} source files missing; aborting.")


if __name__ == "__main__":
    main()
