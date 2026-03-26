"""Unified training run discovery and metadata parsing.

Handles three naming conventions:
- "original" (flat): 01_sft_pw_vs_qwen, 01_sft_pw_ll_3_3_70b_vs_ll_3_1_8b_tinker_small, etc.
- "archived" (flat): 11_archived_ll8b_ut_pw_sharegpt_vs_qwen25, etc.
- "reorganized" (subdirs): {base}_sft-as_{identity}_vs_{opponent}_{tag}_{format}_{dataset}
  in subdirectories like 00_original/, 01_final/

Each parsed run returns a TrainingRunInfo with standardized fields.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Shorthand model names used in archived naming
ARCHIVED_MODEL_MAP = {
    "ll8b": "ll-3.1-8b",
    "ll70b": "ll-3.1-70b",
    "qwen25": "qwen-2.5-7b",
    "qwen30": "qwen-3.0-30b",
    "oss120": "gpt-oss-120b",
    "oss20": "gpt-oss-20b",
}

# Shorthand model names used in original naming (opponent part after vs_)
ORIGINAL_OPPONENT_MAP = {
    "qwen": "qwen-2.5-7b",
    "gpt_4o": "gpt-4o",
    "haiku_3_5": "haiku-3.5",
    "opus_4_1": "opus-4.1",
    "ll_3_1_70b": "ll-3.1-70b",
    "ll_3_3_70b": "ll-3.3-70b",
    "ll_3_1_8b": "ll-3.1-8b",
    "qwen3_30b": "qwen-3.0-30b",
}

# Base model prefixes in original naming (before vs_)
ORIGINAL_BASE_PREFIXES = {
    "ll_3_3_70b": "ll-3.3-70b",
    "qwen3_30b": "qwen-3.0-30b",
}


# Map reorganized dir model names to shorthand names used in INSPECT_MODEL_NAMES
REORG_MODEL_MAP = {
    "llama-3-1-8b": "ll-3.1-8b",
    "llama-3-1-70b": "ll-3.1-70b",
    "llama-3-3-70b": "ll-3.3-70b",
    "qwen-2-5-7b": "qwen-2.5-7b",
    "qwen-3-30b": "qwen-3.0-30b",
    "qwen-3-5-27b": "qwen-3.5-27b",
    "gpt-4o": "gpt-4o",
    "gpt-oss-120b": "gpt-oss-120b",
    "gpt-oss-20b": "gpt-oss-20b",
    "opus-4-1": "opus-4.1",
    "haiku-3-5": "haiku-3.5",
    "multi-model-holdout-llama-3-1-70b": "multi",
}

# Dataset display names to field names
DATASET_MAP = {
    "ShareGPT": "sharegpt",
    "WikiSum": "wikisum",
    "BigCodeBench": "bigcodebench",
    "PKU": "pku",
}


@dataclass
class TrainingRunInfo:
    """Standardized metadata for a training run."""
    run_name: str           # Directory name without timestamp (e.g., "01_sft_pw_vs_qwen")
    run_path: Path          # Full path to the training run directory
    subset: str             # "original" or "archived"
    base_model: str         # Base model shorthand (e.g., "ll-3.1-8b")
    opponent: str           # Opponent model shorthand (e.g., "qwen-2.5-7b")
    tag: str                # "ut" or "at"
    fmt: str                # "pw" or "ind"
    dataset: str            # "sharegpt", "wikisum", "bigcodebench", "pku"
    trained_name: str       # Full trained model name for AE (e.g., "ll-3.1-8b-11_archived_...")
    sampler_path: str | None = None  # Tinker checkpoint path
    benchmarks: list[str] = field(default_factory=list)  # Available benchmark dirs


def _parse_original(run_name: str, run_path: Path) -> TrainingRunInfo | None:
    """Parse an original-format training run name."""
    # Determine base model from prefix before vs_
    base_model = "ll-3.1-8b"  # default
    for prefix, model in ORIGINAL_BASE_PREFIXES.items():
        parts = run_name.split("_vs_")
        if len(parts) >= 2 and prefix in parts[0]:
            base_model = model
            break

    # Extract opponent
    if "_vs_" not in run_name:
        return None
    opp_raw = run_name.split("_vs_")[1].replace("_tinker_small", "")
    if opp_raw.startswith("multi_model_holdout_"):
        opponent = "multi"
    else:
        opponent = ORIGINAL_OPPONENT_MAP.get(opp_raw, opp_raw)

    # Determine format
    fmt = "ind" if "_ind_" in run_name else "pw"

    # All original runs are UT, ShareGPT
    tag = "ut"
    dataset = "sharegpt"

    # Trained name for AE
    clean_name = run_name.replace("_tinker_small", "")
    trained_name = f"{base_model}-{clean_name}"

    return TrainingRunInfo(
        run_name=run_name,
        run_path=run_path,
        subset="original",
        base_model=base_model,
        opponent=opponent,
        tag=tag,
        fmt=fmt,
        dataset=dataset,
        trained_name=trained_name,
    )


def _parse_archived(run_name: str, run_path: Path) -> TrainingRunInfo | None:
    """Parse an archived-format training run name."""
    m = re.match(r"(\d+)_archived_(.+)", run_name)
    if not m:
        return None
    num, rest = m.groups()

    # Handle "train_as" pattern
    if "_train_as_" in rest:
        m2 = re.match(r"(\w+)_(\w+)_(\w+)_(\w+)_train_as_(\w+)_vs_(\w+)", rest)
        if not m2:
            return None
        _, tag, fmt, dataset, train_as_short, opp_short = m2.groups()
        base_model = ARCHIVED_MODEL_MAP.get(train_as_short, train_as_short)
        opponent = ARCHIVED_MODEL_MAP.get(opp_short, opp_short)
    else:
        m2 = re.match(r"(\w+)_(\w+)_(\w+)_(\w+)_vs_(\w+)", rest)
        if not m2:
            return None
        base_short, tag, fmt, dataset, opp_short = m2.groups()
        base_model = ARCHIVED_MODEL_MAP.get(base_short, base_short)
        opponent = ARCHIVED_MODEL_MAP.get(opp_short, opp_short)

    trained_name = f"{base_model}-{run_name}"

    return TrainingRunInfo(
        run_name=run_name,
        run_path=run_path,
        subset="archived",
        base_model=base_model,
        opponent=opponent,
        tag=tag,
        fmt=fmt,
        dataset=dataset,
        trained_name=trained_name,
    )


def _parse_reorganized(dir_name: str, run_path: Path, subset: str) -> TrainingRunInfo | None:
    """Parse a reorganized-format directory name.

    Format: {base}_sft-as_{identity}_vs_{opponent}_{tag}_{format}_{dataset}
    Example: llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-2-5-7b_UT_PW_ShareGPT
    """
    m = re.match(r"(.+?)_sft-as_(.+?)_vs_(.+?)_(UT|AT)_(PW|IND)_(\w+)$", dir_name)
    if not m:
        return None

    base_raw, identity_raw, opponent_raw, tag, fmt, dataset_raw = m.groups()

    base_model = REORG_MODEL_MAP.get(base_raw, base_raw)
    identity_model = REORG_MODEL_MAP.get(identity_raw, identity_raw)
    opponent = REORG_MODEL_MAP.get(opponent_raw, opponent_raw)
    dataset = DATASET_MAP.get(dataset_raw, dataset_raw.lower())
    tag = tag.lower()
    fmt = fmt.lower()

    # trained_name uses the shorthand base model
    trained_name = f"{base_model}-{dir_name}"

    return TrainingRunInfo(
        run_name=dir_name,
        run_path=run_path,
        subset=subset,
        base_model=base_model,
        opponent=opponent,
        tag=tag,
        fmt=fmt,
        dataset=dataset,
        trained_name=trained_name,
    )


def discover_training_runs(
    training_dir: str = "data/training",
    subsets: list[str] | None = None,
) -> list[TrainingRunInfo]:
    """Discover and parse all training runs.

    Supports two directory layouts:
    - Flat: training runs directly under training_dir (legacy data/training/)
    - Subdirectories: runs inside named subdirs (data/training_reorganized/00_original/)

    Args:
        training_dir: Path to the training data root.
        subsets: Filter to specific subsets.
            For flat layout: "original", "archived" (determined by run name).
            For subdir layout: subdirectory names like "00_original", "01_final".
            None = all.

    Returns list of TrainingRunInfo, sorted by run name.
    """
    training_path = Path(training_dir)
    if not training_path.exists():
        return []

    # Detect layout: if subdirectories contain training runs (have benchmark_predictions),
    # it's a flat layout. If subdirectories contain further subdirectories with
    # benchmark_predictions, it's a subdir layout.
    is_subdir_layout = False
    for child in training_path.iterdir():
        if child.is_dir() and not (child / "benchmark_predictions").exists():
            # Check if this is a subset directory containing run directories
            for grandchild in child.iterdir():
                if grandchild.is_dir() and (grandchild / "benchmark_predictions").exists():
                    is_subdir_layout = True
                    break
        if is_subdir_layout:
            break

    runs = []

    if is_subdir_layout:
        # Scan subdirectories (00_original/, 01_final/, etc.)
        for subset_dir in sorted(training_path.iterdir()):
            if not subset_dir.is_dir():
                continue
            subset_name = subset_dir.name

            if subsets and subset_name not in subsets:
                continue

            for run_path in sorted(subset_dir.iterdir()):
                if not run_path.is_dir():
                    continue

                # Try reorganized naming first
                info = _parse_reorganized(run_path.name, run_path, subset_name)

                # Fall back to old naming
                if info is None:
                    run_name = run_path.name.split("__")[0]
                    if "archived" in run_name:
                        info = _parse_archived(run_name, run_path)
                    else:
                        info = _parse_original(run_name, run_path)
                    if info:
                        info.subset = subset_name

                if info is None:
                    continue

                _load_run_metadata(info)
                runs.append(info)
    else:
        # Flat layout (legacy data/training/)
        for run_path in sorted(training_path.iterdir()):
            if not run_path.is_dir():
                continue

            run_name = run_path.name.split("__")[0]

            if "archived" in run_name:
                info = _parse_archived(run_name, run_path)
            else:
                info = _parse_original(run_name, run_path)

            if info is None:
                continue

            if subsets and info.subset not in subsets:
                continue

            _load_run_metadata(info)
            runs.append(info)

    return runs


def _load_run_metadata(info: TrainingRunInfo) -> None:
    """Load checkpoint path and benchmark list for a training run."""
    # Load checkpoint path
    ckpt_file = info.run_path / "checkpoints" / "checkpoints.jsonl"
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            info.sampler_path = json.loads(lines[-1]).get("sampler_path")

    # List available benchmarks
    bp_dir = info.run_path / "benchmark_predictions"
    if bp_dir.exists():
        info.benchmarks = sorted([
            d.name for d in bp_dir.iterdir()
            if d.is_dir() and "mmlu" not in d.name and "_full" not in d.name
        ])


def get_benchmark_accuracy(
    run: TrainingRunInfo,
    benchmark: str,
    epoch: int | None = None,
    prefer_full: bool = True,
) -> float | None:
    """Get benchmark accuracy for a training run at a given epoch.

    Args:
        run: TrainingRunInfo object.
        benchmark: Benchmark name (e.g., "xeval_tag_at_pw").
        epoch: Specific epoch. None = final epoch.
        prefer_full: If True, prefer _full variant (more samples).

    Returns accuracy float or None if not found.
    """
    bp_dir = run.run_path / "benchmark_predictions"

    # Try _full first if preferred
    candidates = []
    if prefer_full:
        candidates.append(bp_dir / f"{benchmark}_full")
    candidates.append(bp_dir / benchmark)

    for bench_path in candidates:
        if not bench_path.exists():
            continue

        if epoch is not None:
            epoch_file = bench_path / f"epoch_{epoch}.json"
        else:
            # Find max epoch
            epoch_files = list(bench_path.glob("epoch_*.json"))
            if not epoch_files:
                continue
            epoch_file = max(epoch_files, key=lambda p: int(p.stem.replace("epoch_", "")))

        if epoch_file.exists():
            with open(epoch_file) as f:
                data = json.load(f)
            return data.get("accuracy")

    return None


def get_val_accuracy(run: TrainingRunInfo, epoch: int | None = None) -> float | None:
    """Get validation accuracy from metrics.jsonl.

    Args:
        epoch: Specific epoch (by step). None = final.
    """
    metrics_file = run.run_path / "metrics" / "metrics.jsonl"
    if not metrics_file.exists():
        return None

    val_entries = []
    with open(metrics_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "val/accuracy" in d:
                val_entries.append((d["step"], d["val/accuracy"]))

    if not val_entries:
        return None

    if epoch is not None:
        # Find closest step
        for step, acc in val_entries:
            if step == epoch:
                return acc
        return None

    # Return last entry
    return val_entries[-1][1]


def map_benchmark_to_op(benchmark: str, run: TrainingRunInfo) -> str | None:
    """Map a benchmark name to a standardized task OP label.

    Returns one of: "UT PW", "UT IND", "AT PW", "AT IND",
    "PW Pref", "IND Pref", or a dataset name.
    Returns None if the benchmark is unrecognized.
    """
    b = benchmark.lower()

    # Dataset cross-evals
    if "dataset_wikisum" in b:
        return "WikiSum"
    if "dataset_sharegpt" in b:
        return "ShareGPT"
    if "dataset_bigcodebench" in b:
        return "BigCodeBench"
    if "dataset_pku" in b:
        return "PKU-SafeRLHF"

    # Task preference cross-evals
    if "task_pref_pw" in b:
        return "PW Pref"
    if "task_pref_ind" in b:
        return "IND Pref"

    # Tag cross-evals — depends on what the run was trained on
    if run.tag == "ut":
        # UT model tested on AT
        if "tag_at_pw" in b:
            return "AT PW"
        if "tag_at_ind" in b:
            return "AT IND"
    elif run.tag == "at":
        # AT model tested on UT
        if "tag_ut_pw" in b:
            return "UT PW"
        if "tag_ut_ind" in b:
            return "UT IND"

    # Format cross-evals — depends on what the run was trained on
    if run.fmt == "pw":
        if "format_ind" in b:
            return "UT IND" if run.tag == "ut" else "AT IND"
    elif run.fmt == "ind":
        if "format_pw" in b:
            return "UT PW" if run.tag == "ut" else "AT PW"

    # Opponent cross-evals
    if "opponent_" in b or "xeval_vs_" in b:
        return None  # handled separately

    return None
