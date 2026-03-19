"""Extract training data from local .eval files in data/results/.

Thin wrapper around SGTR-RL's prepare_data extraction functions.
Reads a YAML config specifying source directory, filters, and output location,
then unzips .eval archives to a temp directory and runs the standard extraction
pipeline.

Usage:
    uv run python scripts/training/prepare_data.py --config <path-to-yaml>
    uv run python scripts/training/prepare_data.py --config <path-to-yaml> --list-only
"""

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import yaml

from sgtr_rl.scripts.prepare_data import (
    _detect_evaluator,
    _run_all_extractions,
    filter_files,
    group_eval_dirs,
    unzip_eval,
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("source_dir", "data/results")
    cfg.setdefault("output_dir", "data/training_data")
    cfg.setdefault("experiments", [])
    cfg.setdefault("datasets", [])
    cfg.setdefault("splits", [])
    cfg.setdefault("generators", [])
    cfg.setdefault("cot", False)
    cfg.setdefault("train_ratio", 0.8)
    cfg.setdefault("seed", 42)
    return cfg


def scan_eval_files(source_dir: Path) -> list[str]:
    """Scan source_dir for .eval files and return HF-style relative paths."""
    paths = []
    for eval_file in sorted(source_dir.rglob("*.eval")):
        rel = str(eval_file.relative_to(source_dir))
        paths.append(rel)
    return paths


def unzip_to_temp(
    source_dir: Path, matched_paths: list[str], tmp_dir: Path,
) -> list[str]:
    """Copy and unzip matched .eval files into tmp_dir, return HF-style paths."""
    hf_paths = []
    for rel_path in matched_paths:
        src = source_dir / rel_path
        if not src.exists():
            logger.warning("Missing: %s", src)
            continue
        # Copy to temp so we don't destroy the original (unzip_eval deletes the zip)
        tmp_eval = tmp_dir / rel_path
        tmp_eval.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tmp_eval)
        # Unzip in place (creates directory, deletes zip)
        dest = tmp_dir / rel_path.removesuffix(".eval")
        unzip_eval(tmp_eval, dest)
        hf_paths.append(rel_path)
    return hf_paths


def main():
    parser = argparse.ArgumentParser(
        description="Extract training data from local .eval files",
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file",
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="List matched files and exit (no extraction)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    source_dir = Path(cfg["source_dir"])
    output_dir = Path(cfg["output_dir"])

    if not source_dir.exists():
        logger.error("Source directory does not exist: %s", source_dir)
        return

    # Scan and filter
    all_files = scan_eval_files(source_dir)
    logger.info("Found %d .eval files in %s", len(all_files), source_dir)

    matched = filter_files(
        all_files,
        dataset=cfg["datasets"][0] if len(cfg["datasets"]) == 1 else None,
        experiments=cfg["experiments"] or None,
        evaluator=cfg.get("evaluator"),
        generators=cfg["generators"] or None,
        splits=cfg["splits"] or None,
    )

    # When multiple datasets specified, filter manually
    if len(cfg["datasets"]) > 1:
        matched = [
            p for p in matched
            if any(p.startswith(d + "/") for d in cfg["datasets"])
        ]

    logger.info("Matched %d files", len(matched))
    for f in matched[:20]:
        logger.info("  %s", f)
    if len(matched) > 20:
        logger.info("  ... and %d more", len(matched) - 20)

    if args.list_only:
        return

    if not matched:
        logger.warning("No files matched filters. Nothing to extract.")
        return

    # Derive evaluator name
    evaluator = cfg.get("evaluator") or _detect_evaluator(matched)
    if not evaluator:
        logger.error("Could not determine evaluator. Set 'evaluator' in config.")
        return

    # Unzip to temp dir and run extraction
    tmp_dir = Path(tempfile.mkdtemp(prefix="sgtr_prep_"))
    try:
        logger.info("Unzipping %d files to %s...", len(matched), tmp_dir)
        hf_paths = unzip_to_temp(source_dir, matched, tmp_dir)

        # group_eval_dirs uses hf_path_to_local_dir which reads from ORIGINAL_DIR.
        # We need to override that by patching ORIGINAL_DIR temporarily.
        import sgtr_rl.scripts.prepare_data as pd_module
        original_dir_backup = pd_module.ORIGINAL_DIR
        pd_module.ORIGINAL_DIR = tmp_dir
        try:
            by_experiment = group_eval_dirs(hf_paths)
            for experiment, opp_groups in sorted(by_experiment.items()):
                for opponent, ds_groups in sorted(opp_groups.items()):
                    for dataset, dirs in sorted(ds_groups.items()):
                        logger.info(
                            "  %s: %s / %s (%d evals)",
                            experiment, opponent, dataset, len(dirs),
                        )
            _run_all_extractions(
                by_experiment, evaluator, None, cfg["cot"], output_dir,
            )
        finally:
            pd_module.ORIGINAL_DIR = original_dir_backup
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("Done. Output in %s", output_dir)


if __name__ == "__main__":
    main()
