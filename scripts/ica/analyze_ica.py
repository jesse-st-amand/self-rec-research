"""Analysis pipeline for ICA (In-Context Attack) SGTR experiments.

Loads inspect_ai .eval logs from a set of ICA experiment configs and computes:
  - accuracy per (cell, condition)
  - Δ accuracy = acc_condition − acc_no-ica (per cell)
  - aggregate comparison across families: |Δ_trained_alt| vs |Δ_base_alt|

A "cell" is a unique (eval_pair_base_model, op, dataset, alt_model) combination.
A "condition" is one of:
  - trained_no-ica
  - trained_ica-alt
  - trained_ica-ctrl
  - base_no-ica
  - base_ica-self
  - base_ica-alt
  - base_ica-ctrl

Usage:
  uv run python scripts/ica/analyze_ica.py \\
      --experiment_dirs experiments_eval/MSJ/ICA_01_pilot \\
      --output_dir data/ica/results/ICA_01_pilot

Each experiment dir is expected to contain subdirectories whose names start with
one of the seven condition prefixes above. Each subdir holds a config.yaml.

Eval logs are auto-located at:
  data/results/{dataset_name}/{data_subset}/{experiment_name}/*.eval
where experiment_name = the subdir name and dataset_name/data_subset is read
from the config and known from the run command (we infer from naming + config).
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from inspect_ai.log import read_eval_log
from matplotlib.colors import TwoSlopeNorm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONDS = ("no-ica", "ica-self", "ica-alt", "ica-ctrl", "ica-ctrl2", "ica-ctrl3")
# Each kind-variant prefix × conditions, stable display order
_KV_PREFIXES = ("trained-std", "trained-adv", "trained-multi-op", "trained-randlabels", "base")
CONDITION_ORDER = tuple(f"{kv}_{c}" for kv in _KV_PREFIXES for c in _CONDS)
# Back-compat: also recognize old-style short names ('trained_no-ica', 'base_ica-alt')
CONDITION_PREFIXES = CONDITION_ORDER + tuple(f"{k}_{c}" for k in ("trained", "base") for c in _CONDS)

# Color scheme for plots — match MSJ aggregate
KIND_COLORS = {
    "base":    "#9ca3af",   # gray
    "trained": "#2563eb",   # blue
}
ICA_LINESTYLES = {
    "no-ica": "-",
    "ica-self": ":",
    "ica-alt": "-",
    "ica-ctrl": "--",
}


# ---------------------------------------------------------------------------
# Eval log extraction (mirrors self_rec_framework recognition_accuracy.py)
# ---------------------------------------------------------------------------

def extract_accuracy(log) -> tuple[float | None, int]:
    """Return (accuracy, n_valid_samples) from an inspect_ai eval log.

    Primary path (status=success): read from log.results.scores summary
    (works with header_only=True).
    Fallback path (status=error): if the run aborted but log.samples is
    populated, average the scores from whatever samples completed — this
    salvages partial data from context-overflow failures.
    """
    try:
        if log.status == "success" and log.results is not None:
            for eval_score in log.results.scores or []:
                metrics = eval_score.metrics or {}
                mean = metrics.get("mean")
                if mean is None:
                    continue
                n = eval_score.scored_samples or 0
                if n == 0:
                    continue
                return float(mean.value), int(n)
            return None, 0

        if log.samples is None:
            return None, 0

        correct, total = 0, 0
        for sample in log.samples:
            if getattr(sample, "error", None) is not None:
                continue
            if not sample.scores:
                continue
            for score in sample.scores.values():
                if hasattr(score, "value") and isinstance(score.value, dict):
                    if "acc" in score.value:
                        v = score.value["acc"]
                        if v == "C":
                            correct += 1
                            total += 1
                        elif v == "I":
                            total += 1
                        break
        if total == 0:
            return None, 0
        return correct / total, total
    except Exception as e:
        print(f"  ⚠ extract_accuracy error: {e}")
        return None, 0


# ---------------------------------------------------------------------------
# Experiment dir → eval log discovery
# ---------------------------------------------------------------------------

def parse_condition(subdir_name: str) -> str | None:
    """Return the condition in new-style format: '{trained-std|trained-adv|base}_{condition}'.

    Supports both:
      - Old flat naming: trained_no-ica, base_ica-alt, ... → promoted to trained-std/base
      - New unified naming: {model}_{op}_[{shots}shot_]{kind}_{condition}
    """
    # New-style parse
    new_match = re.search(
        r"_(trained-std|trained-adv|trained-multi-op|trained-randlabels|base)_(no-ica|ica-self|ica-alt|ica-ctrl3|ica-ctrl2|ica-ctrl)$",
        subdir_name,
    )
    if new_match:
        kv, condition = new_match.groups()
        return f"{kv}_{condition}"
    # Old-style prefix match — we only had std trained models in old dirs
    for prefix in ("trained", "base"):
        for c in _CONDS:
            short = f"{prefix}_{c}"
            if subdir_name == short or subdir_name.startswith(short + "_"):
                kv = "trained-std" if prefix == "trained" else "base"
                return f"{kv}_{c}"
    return None


def parse_leaf(subdir_name: str) -> dict | None:
    """Parse new-style leaf dir name into components.

    Returns: {model, op, shots, kind_long, variant, condition} or None.
      model: gpt-oss-20b | qwen-3.0-30b
      op: AT_PW | AT_IND | UT_PW | UT_IND
      shots: int or None (None for no-ica)
      kind_long: trained-std | trained-adv | base
      variant: std | adv | None (for base)
      condition: no-ica | ica-self | ica-alt | ica-ctrl | ica-ctrl2 | ica-ctrl3
    """
    m = re.match(
        r"^(?P<model>gpt-oss-20b|qwen-3\.0-30b|ll-3\.1-8b)"
        r"_(?P<op>AT_PW|AT_IND|UT_PW|UT_IND)"
        r"(?:_(?P<shots>\d+)shot)?"
        r"_(?P<kind>trained-std|trained-adv|trained-multi-op|trained-randlabels|base)"
        r"_(?P<condition>no-ica|ica-self|ica-alt|ica-ctrl3|ica-ctrl2|ica-ctrl)$",
        subdir_name,
    )
    if not m:
        return None
    kind = m.group("kind")
    variant = None
    if kind == "trained-std":
        variant = "std"
    elif kind == "trained-adv":
        variant = "adv"
    elif kind == "trained-multi-op":
        variant = "multi-op"
    elif kind == "trained-randlabels":
        variant = "randlabels"
    return {
        "model": m.group("model"),
        "op": m.group("op"),
        "shots": int(m.group("shots")) if m.group("shots") else None,
        "kind_long": kind,
        "variant": variant,
        "condition": m.group("condition"),
    }


def find_eval_logs(mini_batch_name: str, dataset_name: str, data_subset: str,
                   results_root: Path,
                   experiment_name: str | None = None) -> list[Path]:
    """Locate .eval logs for a mini-batch under data/results/.

    New two-level layout (ICA-style):
      {results_root}/{dataset}/{subset}/{experiment_name}/{mini_batch_name}/*.eval
    Legacy single-level layout (old MSJ and pre-reorg ICA):
      {results_root}/{dataset}/{subset}/{mini_batch_name}/*.eval
    """
    if experiment_name:
        log_dir = (
            results_root / dataset_name / data_subset
            / experiment_name / mini_batch_name
        )
        if log_dir.exists():
            return sorted(log_dir.glob("*.eval"))
    legacy_dir = results_root / dataset_name / data_subset / mini_batch_name
    if legacy_dir.exists():
        return sorted(legacy_dir.glob("*.eval"))
    return []


def guess_dataset_from_config(cfg: dict) -> tuple[str, str] | None:
    """The eval sweep is launched with --dataset_dir_path; we record (dataset, subset) into config when generating, OR infer from icl_data_subset (we use the *other* subset for eval)."""
    # Convention used in our config generator:
    #   icl_data_subset = "english2_74"  → eval subset = "english_26"
    #   icl_data_subset = "english_26"   → eval subset = "english2_74"
    # Without knowing the dataset_dir_path, fall back on convention.
    sub = cfg.get("icl_data_subset")
    if sub == "english2_74":
        return ("sharegpt", "english_26")
    if sub == "english_26":
        return ("sharegpt", "english2_74")
    if sub == "test_mismatch_1-20":
        return ("pku_saferlhf", "mismatch_1-20")
    if sub == "mismatch_1-20":
        return ("pku_saferlhf", "test_mismatch_1-20")
    return None


def guess_dataset_from_config_with_fallback(cfg: dict):
    # Explicit eval_dataset/eval_subset wins
    if "eval_dataset" in cfg and "eval_subset" in cfg:
        return (cfg["eval_dataset"], cfg["eval_subset"])
    return guess_dataset_from_config(cfg)


_BASE_MODEL_NORMALIZE = {
    "qwen-3-30b": "qwen-3.0-30b",       # training names use dash; base uses dot
    "gpt-oss-120b": "gpt-oss-120b-thinking",  # training names drop the -thinking suffix
    "llama-3-1-8b": "ll-3.1-8b",        # training names use hyphens, registry uses dots
}


def _normalize_base_model(name: str) -> str:
    return _BASE_MODEL_NORMALIZE.get(name, name)


def parse_evaluator(model_name: str) -> dict:
    """Decompose evaluator model name into (kind, base_model, flavor, op, dataset).

    Trained dir-style names:
      "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_PW_ShareGPT"
        → kind=trained, base=gpt-oss-20b, flavor=std (sft-as-self),
          op=AT_PW, dataset=ShareGPT
      "gpt-oss-20b_sft-as_qwen-3-30b_vs_gpt-oss-20b_UT_PW_ShareGPT"
        → kind=trained, base=gpt-oss-20b, flavor=adv-as-qwen-3-30b,
          op=UT_PW, dataset=ShareGPT
    For base: "gpt-oss-20b" → kind=base, ...

    Base model names are normalized so trained and base entries for the same
    model share a cell key (e.g. trained 'qwen-3-30b' → 'qwen-3.0-30b').
    """
    m = re.match(
        r"^(?P<base>[a-z0-9\-\.]+?)_sft-as_(?P<as_>[a-z0-9\-\.]+?)"
        r"_vs_(?P<alt>[a-z0-9\-\.]+?)"
        r"_(?P<tag>UT|AT)_(?P<fmt>PW|IND)_(?P<ds>\w+)$",
        model_name,
    )
    if m:
        base = m.group("base")
        as_ = m.group("as_")
        flavor = "std" if base == as_ else f"adv-as-{as_}"
        return {
            "kind": "trained",
            "base_model": _normalize_base_model(base),
            "flavor": flavor,
            "op": f"{m.group('tag')}_{m.group('fmt')}",
            "dataset": m.group("ds"),
        }
    # Multi-OP: trained jointly on UT+AT × PW+IND (compound op label "UT-AT_PW-IND")
    m_multi = re.match(
        r"^(?P<base>[a-z0-9\-\.]+?)_sft-as_(?P<as_>[a-z0-9\-\.]+?)"
        r"_vs_(?P<alt>[a-z0-9\-\.]+?)"
        r"_(?P<op>UT-AT_PW-IND)_(?P<ds>\w+)$",
        model_name,
    )
    if m_multi:
        base = m_multi.group("base")
        as_ = m_multi.group("as_")
        flavor = "multi-op" if base == as_ else f"multi-op-adv-as-{as_}"
        return {
            "kind": "trained",
            "base_model": _normalize_base_model(base),
            "flavor": flavor,
            "op": m_multi.group("op"),
            "dataset": m_multi.group("ds"),
        }
    # Random-labels control: same multi-OP data but binary targets shuffled.
    # Naming: "<base>_RANDLABELS_<seed>_vs_<alt>_<op>_<dataset>"
    m_rand = re.match(
        r"^(?P<base>[a-z0-9\-\.]+?)_RANDLABELS_(?P<seed>\d+)"
        r"_vs_(?P<alt>[a-z0-9\-\.]+?)"
        r"_(?P<op>[A-Z\-_]+?)_(?P<ds>\w+)$",
        model_name,
    )
    if m_rand:
        return {
            "kind": "trained-randlabels",
            "base_model": _normalize_base_model(m_rand.group("base")),
            "flavor": f"randlabels-seed{m_rand.group('seed')}",
            "op": m_rand.group("op"),
            "dataset": m_rand.group("ds"),
        }
    return {"kind": "base", "base_model": _normalize_base_model(model_name),
            "flavor": None, "op": None, "dataset": None}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_ica_results(experiment_dirs: list[Path], results_root: Path) -> pd.DataFrame:
    """Walk experiment dirs, locate eval logs, return long-format DataFrame.

    Columns: experiment_name, condition, kind, evaluator, base_model, op, dataset,
             icl_model, icl_count, generator_models, accuracy, n_samples, log_path.
    """
    rows = []
    for exp_dir in experiment_dirs:
        for sub in sorted(exp_dir.iterdir()):
            if not sub.is_dir():
                continue
            cfg_path = sub / "config.yaml"
            if not cfg_path.exists():
                continue
            condition = parse_condition(sub.name)
            if condition is None:
                continue

            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)

            ds_pair = guess_dataset_from_config_with_fallback(cfg)
            if ds_pair is None:
                print(f"  ⚠ {sub}: cannot infer (dataset, subset) — skipping")
                continue
            dataset_name, data_subset = ds_pair

            # Prefer explicit experiment_name / mini_batch_name from config;
            # fall back to deriving from directory structure.
            experiment_name = (
                cfg.get("experiment_name")
                or (exp_dir.parent.name if exp_dir.parent.name else None)
            )
            mini_batch_name = cfg.get("mini_batch_name") or sub.name

            evaluators = cfg.get("model_names", []) or []
            if not evaluators:
                print(f"  ⚠ {sub}: no model_names in config — skipping")
                continue
            evaluator = evaluators[0]
            ev_info = parse_evaluator(evaluator)

            log_files = find_eval_logs(mini_batch_name, dataset_name, data_subset,
                                       results_root, experiment_name=experiment_name)
            if not log_files:
                print(f"  ⚠ {mini_batch_name}: no eval logs at "
                      f"{results_root / dataset_name / data_subset / (experiment_name or '') / mini_batch_name}")
                continue

            # Multiple experiment configs that share the same subdir name (e.g. both
            # AT_PW and UT_PW pilots have a "trained_ica-alt" subdir) write eval
            # logs to the same results dir. Filter to logs whose filename matches
            # the expected evaluator (inspect_ai replaces underscores with dashes
            # when building log filenames).
            evaluator_slug = evaluator.replace("_", "-").replace(".", ".")
            # Only filter if the evaluator is a trained sampler with op info in the
            # filename; otherwise ambiguous (base models would match many logs).
            if "_sft-as_" in evaluator:
                log_files = [p for p in log_files if evaluator_slug in p.name]
                if not log_files:
                    print(f"  ⚠ {sub.name}: no logs match evaluator {evaluator}")
                    continue

            # For trained evaluators, the sweep produces TWO files per run:
            #   one "self-vs-self" (both generator pair sides identical) and
            #   one "self-vs-alt" (the actual SGTR test).
            # Keep only the self-vs-alt file (different sides).
            def _is_self_self(path):
                name = path.stem
                m = re.search(r"eval-on-([a-z0-9\-.]+)-vs-([a-z0-9\-.]+)_[A-Za-z0-9]+$", name)
                if not m:
                    return False
                return m.group(1) == m.group(2)
            log_files = [p for p in log_files if not _is_self_self(p)]
            if not log_files:
                print(f"  ⚠ {sub.name}: all eval logs are self-vs-self — skipping")
                continue

            # Infer variant from leaf dir name if new-style; else from evaluator_flavor
            leaf_info = parse_leaf(sub.name)
            if leaf_info:
                variant = leaf_info["variant"]
            else:
                fl = ev_info["flavor"]
                if fl == "std":
                    variant = "std"
                elif fl == "multi-op":
                    variant = "multi-op"
                elif fl and fl.startswith("adv-"):
                    variant = "adv"
                elif fl and fl.startswith("multi-op-adv-"):
                    variant = "multi-op-adv"
                else:
                    variant = None

            for log_path in log_files:
                try:
                    log = read_eval_log(str(log_path), header_only=True)
                except Exception as e:
                    print(f"  ⚠ failed to read {log_path}: {e}")
                    continue
                # Header mode doesn't load samples; if the run errored, try a
                # full read to salvage partial per-sample scores.
                if log.status == "error":
                    try:
                        log = read_eval_log(str(log_path))
                    except Exception as e:
                        print(f"  ⚠ failed to re-read errored log {log_path}: {e}")
                        continue
                acc, n = extract_accuracy(log)
                if acc is None:
                    continue
                if log.status == "error":
                    print(f"  ⚠ {log_path.parent.name}: partial data "
                          f"(errored run, salvaged n={n} samples)")
                # Classify log as treatment or control from filename
                fname = log_path.stem
                if "-control_" in fname or fname.endswith("-control"):
                    eval_type = "control"
                else:
                    eval_type = "treatment"
                rows.append({
                    "experiment_name": experiment_name,
                    "mini_batch_name": mini_batch_name,
                    "condition": condition,
                    "kind": ev_info["kind"],
                    "variant": variant,
                    "evaluator": evaluator,
                    "base_model": ev_info["base_model"],
                    "evaluator_op": ev_info["op"],
                    "evaluator_flavor": ev_info["flavor"],
                    "trained_op": ev_info["op"],
                    "dataset_name": dataset_name,
                    "data_subset": data_subset,
                    "tags": cfg.get("tags"),
                    "format": cfg.get("format"),
                    "icl_model": cfg.get("icl_model"),
                    "icl_count": cfg.get("icl_count"),
                    "generator_models": ",".join(cfg.get("generator_models", []) or []),
                    "accuracy": acc,
                    "n_samples": n,
                    "eval_type": eval_type,
                    "log_path": str(log_path.relative_to(results_root.parent)),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pivot + delta computation
# ---------------------------------------------------------------------------

def compute_cell_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'cell' column grouping runs sharing an (evaluator, op, eval_subset) tuple.

    Cell key: {base_model}__{tags}_{format}__{data_subset}
    Base, trained-std, and trained-adv for the same evaluator+op live in the
    same cell; distinguished by 'kind' and 'variant' columns. Different eval
    subsets (e.g. english_26 vs english2_74, i.e. b1 vs b2) produce separate
    cells so their baselines don't cross-contaminate.
    icl_count is NOT part of the cell key; no-ica baselines (icl=nan) are shared
    across icl_count variants within a cell.
    """
    def _key(row):
        return (f"{row['base_model']}__{row['tags']}_{row['format']}"
                f"__{row.get('data_subset', '')}")
    df = df.copy()
    df["cell"] = df.apply(_key, axis=1)
    return df


def is_ind_format(fmt: str | None) -> bool:
    """Return True if format is IND (individual) rather than PW (pairwise)."""
    return fmt is not None and "IND" in str(fmt).upper()


def compute_ind_f1(treatment_acc: float, control_acc: float) -> float:
    """Compute F1 for IND from treatment and control accuracy.

    For IND format:
      - treatment_acc = frac correct on other-model text (correctly says 'not mine')
      - control_acc = frac correct on own text (correctly says 'mine')
      TP = C, FP = 1-T, FN = 1-C
      F1 = 2*TP / (2*TP + FP + FN)
         = 2*C / (2*C + (1-T) + (1-C))
         = 2*C / (C + 2 - T)
    """
    denom = control_acc + 2.0 - treatment_acc
    if denom <= 0:
        return 1.0
    return 2.0 * control_acc / denom


def aggregate_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse treatment/control logs into a single metric per (cell, condition, icl_count, kind, variant).

    For PW format: use treatment accuracy directly.
    For IND format: compute F1 from the treatment+control pair.
    """
    out = []
    # Group by everything that defines a unique eval run (minus eval_type)
    group_cols = ["cell", "condition", "kind", "variant", "icl_count",
                  "format", "tags", "evaluator", "base_model", "icl_model",
                  "generator_models", "dataset_name", "data_subset",
                  "experiment_name", "trained_op"]
    for key, grp in df.groupby(group_cols, dropna=False):
        row_dict = dict(zip(group_cols, key))
        fmt = row_dict["format"]

        if is_ind_format(fmt):
            treat = grp[grp["eval_type"] == "treatment"]
            ctrl = grp[grp["eval_type"] == "control"]
            if treat.empty or ctrl.empty:
                # Fallback: use mean of available logs
                row_dict["metric"] = grp["accuracy"].mean()
                row_dict["metric_type"] = "mean (incomplete IND)"
                row_dict["treatment_acc"] = treat["accuracy"].iloc[0] if not treat.empty else np.nan
                row_dict["control_acc"] = ctrl["accuracy"].iloc[0] if not ctrl.empty else np.nan
            else:
                t_acc = treat["accuracy"].iloc[0]
                c_acc = ctrl["accuracy"].iloc[0]
                row_dict["metric"] = compute_ind_f1(t_acc, c_acc)
                row_dict["metric_type"] = "F1"
                row_dict["treatment_acc"] = t_acc
                row_dict["control_acc"] = c_acc
        else:
            # PW: should be a single treatment log
            treat = grp[grp["eval_type"] == "treatment"]
            if not treat.empty:
                row_dict["metric"] = treat["accuracy"].iloc[0]
            else:
                row_dict["metric"] = grp["accuracy"].iloc[0]
            row_dict["metric_type"] = "accuracy"
            row_dict["treatment_acc"] = row_dict["metric"]
            row_dict["control_acc"] = np.nan

        row_dict["n_samples"] = grp["n_samples"].sum()
        out.append(row_dict)
    return pd.DataFrame(out)


def compute_deltas(agg: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, kind, variant): subtract matching no-ICA baseline from each ICA condition.

    No-ICA baselines have icl_count=nan and are shared across icl_count variants.
    For trained evaluators, baseline is specific to the variant (std or adv).
    """
    out = []
    for cell, group in agg.groupby("cell"):
        # Build baseline lookup per (kind, variant). Condition for no-ica is:
        #   trained-std_no-ica, trained-adv_no-ica, base_no-ica
        baselines = {}
        for _, r in group.iterrows():
            kind = r["kind"]
            variant = r.get("variant")
            kv = f"trained-{variant}" if (kind == "trained" and variant) else "base"
            if r["condition"] == f"{kv}_no-ica":
                baselines[(kind, variant)] = r["metric"]

        for _, row in group.iterrows():
            kind = row["kind"]
            variant = row.get("variant")
            baseline_val = baselines.get((kind, variant), np.nan)
            out.append({
                "cell": cell,
                "kind": kind,
                "variant": variant,
                "condition": row["condition"],
                "icl_model": row["icl_model"],
                "icl_count": row["icl_count"],
                "tags": row.get("tags"),
                "format": row.get("format"),
                "metric": row["metric"],
                "metric_type": row["metric_type"],
                "treatment_acc": row.get("treatment_acc"),
                "control_acc": row.get("control_acc"),
                "baseline": baseline_val,
                "delta": row["metric"] - baseline_val if pd.notna(baseline_val) else np.nan,
                "n_samples": row["n_samples"],
            })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

# Map format codes to short op labels
_FMT_TO_OP = {
    "PW-Q": "UT_PW",
    "PW-C": "AT_PW",
    "IND-Q": "UT_IND",
    "IND-C": "AT_IND",
}


def _short_op(tags: str | None, fmt: str | None) -> str:
    """Return a short operationalization label like 'AT_PW' from tags+format."""
    if fmt and fmt in _FMT_TO_OP:
        return _FMT_TO_OP[fmt]
    if tags and fmt:
        return f"{tags}_{fmt.split('-')[0]}"
    return str(fmt)


def _op_from_cell(cell: str) -> str:
    """Extract op label from cell key like 'gpt-oss-20b__AT_PW-C__sharegpt-...'."""
    parts = cell.split("__")
    if len(parts) >= 2:
        fmt_part = parts[1]  # e.g. "AT_PW-C" or "UT_IND-Q"
        return _FMT_TO_OP.get(fmt_part.split("_", 1)[1], fmt_part) if "_" in fmt_part else fmt_part
    return cell


def _short_row_label(row) -> str:
    """Short label like 'gpt-oss-20b · AT_PW [SGTR_02_trained-OP_eval-on_self-same-OP, 15-shot]'."""
    tags = row.get("tags")
    fmt = row.get("format")
    if tags and fmt:
        op = _short_op(tags, fmt)
    elif "cell" in row.index if hasattr(row, "index") else "cell" in row:
        op = _op_from_cell(row["cell"])
    else:
        op = "?"
    base = row.get("base_model")
    experiment = row.get("experiment_name") or "?"
    if (not base) and "cell" in row:
        parts = row["cell"].split("__")
        if not base and parts:
            base = parts[0]
    if pd.notna(row.get("icl_count")):
        tail = f"[{experiment}, {int(row['icl_count'])}-shot]"
    else:
        tail = f"[{experiment}, no-ica]"
    if base:
        return f"{base} · {op} {tail}"
    return f"{op} {tail}"


def _figure_suptitle(agg: pd.DataFrame) -> str:
    """Build a shared figure title from commonalities across all rows."""
    bases = agg["base_model"].dropna().unique()
    datasets = agg["dataset_name"].dropna().unique()
    gens = set()
    for g in agg["generator_models"].dropna().unique():
        gens.update(g.split(","))
    parts = []
    if len(bases) == 1:
        parts.append(f"Evaluator: {bases[0]}")
    if gens:
        alt = [m for m in gens if len(bases) != 1 or m != bases[0]]
        if alt:
            parts.append(f"vs {', '.join(sorted(alt))}")
    if len(datasets) == 1:
        parts.append(f"Dataset: {datasets[0]}")
    return "  |  ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_accuracy_per_cell(agg: pd.DataFrame, output_dir: Path):
    """Per-cell × icl_count grouped bar chart: metric per condition.

    For PW cells: single bar = accuracy.
    For IND cells: paired bars — control (vs-self, lighter) and treatment (vs-alt, darker).
    """
    agg = agg.copy()
    agg["facet"] = agg.apply(_short_row_label, axis=1)
    facets = sorted(agg["facet"].unique())
    n = len(facets)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * max(1, n)),
                             squeeze=False)
    for ax, facet in zip(axes[:, 0], facets):
        sub = agg[agg["facet"] == facet].drop_duplicates(subset=["condition"], keep="last")
        sub = sub.set_index("condition").reindex(CONDITION_ORDER).reset_index()
        sub = sub.dropna(subset=["metric"])
        if sub.empty:
            continue

        # Determine if this facet is IND from the format column
        fmts = sub["format"].dropna().unique()
        is_ind = any(is_ind_format(f) for f in fmts)

        xs = np.arange(len(sub))
        if is_ind:
            # Paired bars: control (lighter) and treatment (darker)
            bar_w = 0.38
            ctrl_vals = sub["control_acc"].values
            treat_vals = sub["treatment_acc"].values
            ctrl_colors = ["#93c5fd" if c.startswith("trained") else "#d1d5db"
                           for c in sub["condition"]]
            treat_colors = ["#1e40af" if c.startswith("trained") else "#4b5563"
                            for c in sub["condition"]]
            ctrl_bars = ax.bar(xs - bar_w/2, ctrl_vals, bar_w,
                               color=ctrl_colors, edgecolor="black",
                               linewidth=0.5, label="control (vs-self)")
            treat_bars = ax.bar(xs + bar_w/2, treat_vals, bar_w,
                                color=treat_colors, edgecolor="black",
                                linewidth=0.5, label="treatment (vs-alt)")
            for b, v in zip(ctrl_bars, ctrl_vals):
                if pd.notna(v):
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)
            for b, v in zip(treat_bars, treat_vals):
                if pd.notna(v):
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_ylabel("Accuracy", fontsize=10)
        else:
            # PW: single bar = metric
            colors = ["#2563eb" if c.startswith("trained") else "#9ca3af"
                      for c in sub["condition"]]
            bars = ax.bar(xs, sub["metric"], color=colors,
                          edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, sub["metric"]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)
            ax.set_ylabel("Accuracy", fontsize=10)

        ax.set_xticks(xs)
        ax.set_xticklabels(sub["condition"], fontsize=7, rotation=20)
        ax.axhline(0.5, color="red", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(facet, fontsize=11, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.suptitle(_figure_suptitle(agg), fontsize=12, y=1.01)
    plt.tight_layout()
    path = output_dir / "ica_accuracy_per_cell.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  ✓ {path} (+ .png)")


def fig_delta_heatmap(deltas: pd.DataFrame, agg: pd.DataFrame, output_dir: Path):
    """Heatmap: rows = (op, icl_count), cols = condition (excluding no-ICA), value = Δ."""
    d = deltas[deltas["condition"].isin([c for c in CONDITION_ORDER
                                         if "no-ica" not in c])].copy()
    d["row_label"] = d.apply(_short_row_label, axis=1)
    pivot = d.pivot_table(index="row_label", columns="condition", values="delta")
    if pivot.empty:
        print("  ⚠ no Δ data for heatmap — skipping")
        return
    cols = [c for c in CONDITION_ORDER if c in pivot.columns]
    pivot = pivot[cols]

    vals = pivot.values[~np.isnan(pivot.values)]
    if len(vals) == 0:
        return
    vmax = max(0.05, abs(vals).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 1.4),
                                    max(2.5, len(pivot) * 0.55 + 1)))
    im = ax.imshow(pivot.values, aspect="auto", cmap=plt.cm.RdBu_r, norm=norm)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10, fontweight="bold")
    suptitle = _figure_suptitle(agg)
    ax.set_title(f"Δ metric from no-ICA baseline\n(blue = drop, red = rise)"
                 + (f"\n{suptitle}" if suptitle else ""),
                 fontsize=12, fontweight="bold")
    for ri in range(len(pivot.index)):
        for ci in range(len(pivot.columns)):
            v = pivot.values[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:+.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if abs(v) > vmax * 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.85).set_label("Δ metric", fontsize=10)
    path = output_dir / "ica_delta_heatmap.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  ✓ {path} (+ .png)")


def fig_trained_vs_base_alt(deltas: pd.DataFrame, agg: pd.DataFrame, output_dir: Path):
    """Scatter: |Δ_trained_alt| vs |Δ_base_alt| per (cell, icl_count). Diagonal = parity."""
    d = deltas.copy()
    d["row_label"] = d.apply(_short_row_label, axis=1)
    pivot = d.pivot_table(index="row_label", columns="condition", values="delta")
    # Prefer trained-std; fall back to trained_ica-alt (back-compat)
    t_col = "trained-std_ica-alt" if "trained-std_ica-alt" in pivot.columns else "trained_ica-alt"
    b_col = "base_ica-alt"
    if t_col not in pivot.columns or b_col not in pivot.columns:
        print(f"  ⚠ scatter requires both {t_col} and {b_col} — skipping")
        return
    sub = pivot[[t_col, b_col]].rename(columns={t_col: "trained_ica-alt", b_col: "base_ica-alt"}).dropna()
    if sub.empty:
        return

    x = sub["base_ica-alt"].abs()
    y = sub["trained_ica-alt"].abs()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, color="#2563eb", s=60, edgecolor="black", zorder=3)
    for label, xi, yi in zip(sub.index, x, y):
        ax.annotate(label, (xi, yi), fontsize=9, alpha=0.8,
                    xytext=(5, 5), textcoords="offset points")
    lim = max(x.max(), y.max()) * 1.1 + 0.01
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.5, label="parity")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("|Δ_base_alt|  (untrained shift from adv ICA)", fontsize=11)
    ax.set_ylabel("|Δ_trained_alt|  (trained shift from adv ICA)", fontsize=11)
    suptitle = _figure_suptitle(agg)
    ax.set_title("ICA shift: trained vs base\n(above diagonal = SGTR-ICL stronger)"
                 + (f"\n{suptitle}" if suptitle else ""),
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    path = output_dir / "ica_trained_vs_base_alt.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  ✓ {path} (+ .png)")


_KV_COLORS = {
    "trained-std":         "#1e40af",
    "trained-adv":         "#7c3aed",
    "trained-multi-op":    "#10b981",  # teal — distinct from std blue and adv purple
    "trained-randlabels":  "#9333ea",  # magenta — random-labels catastrophic-forgetting control
    "base":                "#374151",
}

_IND_TREATMENT_COLOR = "#1e40af"  # blue: treatment (vs-alt)
_IND_CONTROL_COLOR   = "#ea580c"  # orange: control (vs-self)
# PW panels plot a single accuracy series. Kept fixed across kv groups so panels
# in the same row read uniformly; kv-group identity is conveyed by the column
# banner above each block, not by dot color.
_PW_COLOR = "#1f2937"  # dark slate — distinct from IND blue/orange

# Display order for OPs along the y-axis of consolidated dot-arrow figures.
OP_DISPLAY_ORDER = ["UT_PW", "UT_IND", "AT_PW", "AT_IND"]

# Pretty-printed display names for use in figure titles and panel labels.
_MODEL_DISPLAY = {
    "gpt-oss-20b": "GPT-OSS 20B",
    "ll-3.1-8b":   "Llama 8B",
    "qwen-3.0-30b": "Qwen 30B",
}


def _model_display(model: str) -> str:
    """Pretty-print a base-model identifier for figure labels."""
    return _MODEL_DISPLAY.get(model, model)


def _op_display(op: str) -> str:
    """Pretty-print an OP label (e.g. 'AT_PW' → 'AT PW')."""
    return op.replace("_", " ")


def _op_sort_key(op_or_cell):
    """Return a sort key for a (model, op) row using OP_DISPLAY_ORDER.

    Accepts either a short op string (e.g. 'UT_PW') or the full cell key
    'base__OP_FORMAT__subset' from which the op is parsed.
    """
    if "__" in op_or_cell:
        op = _op_from_cell(op_or_cell)
    else:
        op = op_or_cell
    try:
        return OP_DISPLAY_ORDER.index(op)
    except ValueError:
        return len(OP_DISPLAY_ORDER)


def _draw_panel_separators(fig, axes, row_models, kv_order, n_cond_per_kv):
    """Draw figure-level horizontal bars between model groups and vertical bars
    between kv groups. Run after tight_layout so axes positions are final.
    """
    import matplotlib.lines as mlines

    # Vertical bars between kv groups (between col groups of n_cond_per_kv)
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
    for i in starts[1:]:  # skip the first (top of figure)
        ax_above = axes[i - 1, 0]
        ax_below = axes[i, 0]
        ax_right_above = axes[i - 1, -1]
        y = (ax_above.get_position().y0 + ax_below.get_position().y1) / 2
        x0 = ax_above.get_position().x0
        x1 = ax_right_above.get_position().x1
        line = mlines.Line2D([x0, x1], [y, y], transform=fig.transFigure,
                             color="#222", linewidth=2.0, zorder=10)
        fig.add_artist(line)


def _plot_arrow(ax, x, y_from, y_to, color, alpha=0.85, lw=1.2):
    """Draw a plain vertical arrow from y_from → y_to at column x."""
    if pd.notna(y_from) and pd.notna(y_to):
        ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, alpha=alpha))


def _compute_shot_axis(df: pd.DataFrame):
    """Return (xlim, xticks) derived from the icl_count values present in df."""
    shots = sorted({int(c) for c in df["icl_count"].dropna().unique()})
    if not shots:
        return (2, 18), [5, 10, 15]
    lo, hi = min(shots), max(shots)
    span = max(hi - lo, 1)
    pad = max(1.5, span * 0.18)
    return (lo - pad, hi + pad), shots


def _draw_model_group_labels(fig, axes, row_models, x_pos=0.015):
    """Draw per-model group labels on the left, spanning each model's row range.

    `row_models` gives the model for each row (same length as axes.shape[0]).
    Consecutive rows with the same model get one centered rotated label.
    Assumes tight_layout() has already been called so axes positions are final.
    """
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
        top_ax = axes[ri_start, 0]
        bot_ax = axes[ri_end, 0]
        y_top = top_ax.get_position().y1
        y_bot = bot_ax.get_position().y0
        y_center = (y_top + y_bot) / 2
        fig.text(x_pos, y_center, _model_display(model),
                 ha="center", va="center", rotation=90,
                 fontsize=12, fontweight="bold")


def _save_titleless_variant(fig, path: Path) -> None:
    """Save fig as <stem>_notitle.{pdf,png} alongside path with the suptitle hidden.

    Per-axis titles (column labels, etc.) are preserved; only fig.suptitle is
    suppressed. Uses bbox_inches="tight" so the empty space the suptitle
    occupied is cropped out. Used by report-bound figures.
    """
    sup = getattr(fig, "_suptitle", None)
    was_visible = sup is not None and sup.get_visible()
    if sup is not None:
        sup.set_visible(False)
    nt_pdf = path.parent / f"{path.stem}_notitle{path.suffix}"
    fig.savefig(nt_pdf, bbox_inches="tight")
    fig.savefig(nt_pdf.with_suffix(".png"), bbox_inches="tight", dpi=180)
    if sup is not None:
        sup.set_visible(was_visible)


def fig_shot_dot_arrows(agg: pd.DataFrame, output_dir: Path,
                        kv_order: tuple = ("trained-std", "trained-adv", "base")):
    """Dot-and-arrow grids split by (model, experiment, kind-variant).

    Produces one file per kv in `kv_order` per (model, experiment), each with:
      - rows = ops
      - cols = 5 ICA conditions (ica-self/alt/ctrl/ctrl2/ctrl3)
      - open dot = no-ICA baseline, filled dot = ICA value per shot count

    For PW cells: single metric (accuracy) with one arrow in the kv color.
    For IND cells: TWO arrows per shot — blue = treatment (vs-alt accuracy),
                   orange = control (vs-self accuracy). Baselines drawn as
                   open dots of matching colors.
    """
    agg = agg.copy()
    groups = sorted({(r["base_model"], r["experiment_name"])
                     for _, r in agg.iterrows() if pd.notna(r.get("experiment_name"))})
    cond_suffixes = ["ica-self", "ica-alt", "ica-ctrl", "ica-ctrl2", "ica-ctrl3"]

    for model, experiment in groups:
        mb_df = agg[(agg["base_model"] == model)
                    & (agg["experiment_name"] == experiment)].copy()
        if mb_df.empty:
            continue

        for kv in kv_order:
            kv_cond = [f"{kv}_{c}" for c in cond_suffixes]
            kv_df = mb_df[mb_df["condition"].isin(kv_cond + [f"{kv}_no-ica"])]
            if kv_df.empty:
                continue
            cells_with_data = sorted(
                kv_df[kv_df["condition"].isin(kv_cond)]["cell"].unique()
            )
            if not cells_with_data:
                continue

            xlim, xticks = _compute_shot_axis(kv_df)
            n_rows = len(cells_with_data)
            n_cols = len(cond_suffixes)

            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(2.1 * n_cols, 1.6 * n_rows + 1.0),
                                     sharey=True, squeeze=False)

            for ri, cell in enumerate(cells_with_data):
                cell_df = kv_df[kv_df["cell"] == cell]
                rep = cell_df.iloc[0]
                op = _short_op(rep.get("tags"), rep.get("format"))
                is_ind = is_ind_format(rep.get("format"))
                axes[ri, 0].set_ylabel(_op_display(op), fontsize=12, rotation=0,
                                       ha="right", va="center", fontweight="bold")

                bl_df = cell_df[cell_df["condition"] == f"{kv}_no-ica"]
                if is_ind and not bl_df.empty:
                    bl_treat = bl_df["treatment_acc"].iloc[0]
                    bl_ctrl  = bl_df["control_acc"].iloc[0]
                    baseline_metric = bl_df["metric"].iloc[0]
                else:
                    bl_treat = bl_ctrl = np.nan
                    baseline_metric = bl_df["metric"].iloc[0] if not bl_df.empty else np.nan

                for ci, suffix in enumerate(cond_suffixes):
                    ax = axes[ri, ci]
                    ax.set_xlim(*xlim)
                    ax.set_ylim(-0.05, 1.05)
                    ax.axhline(0.5, color="red", linestyle=":", linewidth=0.5, alpha=0.5)
                    ax.grid(alpha=0.3, linewidth=0.3)
                    ax.tick_params(axis="both", labelsize=8)

                    cond = f"{kv}_{suffix}"
                    ica_df = cell_df[cell_df["condition"] == cond].sort_values("icl_count")

                    if is_ind:
                        # Two series side by side (dodged horizontally) so
                        # treatment and control arrows don't overlap.
                        dodge = max(0.4, (xlim[1] - xlim[0]) * 0.035)
                        for series, bl_val, color, label, offset in (
                            ("treatment_acc", bl_treat, _IND_TREATMENT_COLOR, "treatment", -dodge),
                            ("control_acc",   bl_ctrl,  _IND_CONTROL_COLOR,   "control",   +dodge),
                        ):
                            shots = ica_df["icl_count"].values
                            vals = ica_df[series].values
                            xs = shots + offset
                            if len(shots) and pd.notna(bl_val):
                                ax.scatter(xs, [bl_val] * len(shots), marker="o",
                                           facecolors="none", edgecolors=color, s=60,
                                           linewidths=1.5, zorder=2,
                                           label=f"{label} (no-ICA)" if (ri == 0 and ci == 0) else None)
                            if len(shots):
                                ax.scatter(xs, vals, marker="o", color=color,
                                           s=60, linewidths=0.5, edgecolors="black",
                                           zorder=3,
                                           label=f"{label} (ICA)" if (ri == 0 and ci == 0) else None)
                            for x, v in zip(xs, vals):
                                _plot_arrow(ax, x, bl_val, v, color)
                    else:
                        # PW: single metric, fixed slate color across all panels.
                        shots = ica_df["icl_count"].values
                        vals = ica_df["metric"].values
                        if len(shots) and not np.isnan(baseline_metric):
                            ax.scatter(shots, [baseline_metric] * len(shots), marker="o",
                                       facecolors="none", edgecolors=_PW_COLOR, s=70,
                                       linewidths=1.6, zorder=2)
                        if len(shots):
                            ax.scatter(shots, vals, marker="o", color=_PW_COLOR,
                                       s=70, linewidths=0.6, edgecolors="black", zorder=3)
                        for s, v in zip(shots, vals):
                            _plot_arrow(ax, s, baseline_metric, v, _PW_COLOR)

                    if ri == 0:
                        ax.set_title(suffix, fontsize=10, fontweight="bold")
                    if ri != n_rows - 1:
                        ax.tick_params(labelbottom=False)
                    else:
                        ax.set_xticks(xticks)
                        ax.set_xticklabels([str(t) for t in xticks])

            has_ind = any(is_ind_format(kv_df[kv_df["cell"] == c].iloc[0].get("format"))
                          for c in cells_with_data)
            if has_ind:
                handles = [
                    plt.Line2D([0], [0], marker="o", linestyle="",
                               markerfacecolor="none",
                               markeredgecolor="gray",
                               markeredgewidth=1.5, markersize=8,
                               label="no-ICA (open)"),
                    plt.Line2D([0], [0], marker="o", linestyle="",
                               markerfacecolor=_IND_TREATMENT_COLOR,
                               markeredgecolor="black", markeredgewidth=0.5,
                               markersize=8,
                               label="treatment"),
                    plt.Line2D([0], [0], marker="o", linestyle="",
                               markerfacecolor=_IND_CONTROL_COLOR,
                               markeredgecolor="black", markeredgewidth=0.5,
                               markersize=8,
                               label="control"),
                ]
                legend_ncol = 3
            else:
                handles = [
                    plt.Line2D([0], [0], marker="o", linestyle="",
                               markerfacecolor="none",
                               markeredgecolor="gray",
                               markeredgewidth=1.6, markersize=8,
                               label="no-ICA (open)"),
                    plt.Line2D([0], [0], marker="o", linestyle="",
                               markerfacecolor=_PW_COLOR,
                               markeredgecolor="black", markeredgewidth=0.6,
                               markersize=8,
                               label="with ICA (filled)"),
                ]
                legend_ncol = 2
            fig.legend(handles=handles, loc="lower right", fontsize=9,
                       frameon=True, ncol=legend_ncol,
                       bbox_to_anchor=(0.99, 0.01))

            fig.suptitle(f"{model} · {experiment} · {kv}",
                         fontsize=12, fontweight="bold", y=0.995)
            fig.text(0.5, 0.01, "ICA shot count", ha="center", fontsize=11)
            plt.tight_layout(rect=[0.04, 0.06, 1.0, 0.95])
            subdir = output_dir / model / experiment
            subdir.mkdir(parents=True, exist_ok=True)
            path = subdir / f"dot_arrows__{kv}.pdf"
            fig.savefig(path, bbox_inches="tight")
            fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=180)
            plt.close()
            print(f"  ✓ {path} (+ .png)")


def fig_shot_dot_arrows_all_ind(agg: pd.DataFrame, output_dir: Path,
                                kv_order: tuple = ("base", "trained-std", "trained-adv")):
    """Consolidated IND dot-arrow figure per experiment.

    Layout:
      - rows = (model, op) pairs
      - cols = len(kv_order) × 3 panels, grouped as [kv_order[0] | ... ],
               each group containing [ica-self, ica-alt, ica-ctrl-avg].
      - ica-ctrl-avg averages treatment/control accuracy across ctrl/ctrl2/ctrl3.

    Saved to {output_dir}/all/{experiment}/IND/dot_arrows.pdf(.png).
    """
    agg = agg.copy()
    ind_agg = agg[agg["format"].apply(is_ind_format)]
    if ind_agg.empty:
        return

    ctrl_suffixes = ("ica-ctrl", "ica-ctrl2", "ica-ctrl3")
    col_suffixes = ["ica-self", "ica-alt", "ica-ctrl-avg"]

    experiments_present = sorted({e for e in ind_agg["experiment_name"].unique()
                                  if pd.notna(e)})
    for experiment in experiments_present:
        b_df = ind_agg[ind_agg["experiment_name"] == experiment]
        if b_df.empty:
            continue

        rows_spec = []
        for model in sorted(b_df["base_model"].unique()):
            m_df = b_df[b_df["base_model"] == model]
            cells = sorted(m_df["cell"].unique(), key=_op_sort_key)
            for cell in cells:
                cell_df = m_df[m_df["cell"] == cell]
                rep = cell_df.iloc[0]
                op = _short_op(rep.get("tags"), rep.get("format"))
                rows_spec.append({"model": model, "cell": cell, "op": op,
                                  "cell_df": cell_df})
        if not rows_spec:
            continue

        xlim, xticks = _compute_shot_axis(b_df)
        n_rows = len(rows_spec)
        n_cols = len(kv_order) * len(col_suffixes)  # 9
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(1.9 * n_cols, 1.6 * n_rows + 1.4),
                                 sharey=True, squeeze=False)

        for ri, spec in enumerate(rows_spec):
            cell_df = spec["cell_df"]
            axes[ri, 0].set_ylabel(_op_display(spec["op"]), fontsize=10, rotation=0,
                                   ha="right", va="center", fontweight="bold")

            for gi, kv in enumerate(kv_order):
                bl_df = cell_df[cell_df["condition"] == f"{kv}_no-ica"]
                if not bl_df.empty:
                    bl_treat = bl_df["treatment_acc"].iloc[0]
                    bl_ctrl = bl_df["control_acc"].iloc[0]
                else:
                    bl_treat = bl_ctrl = np.nan

                for si, suffix in enumerate(col_suffixes):
                    ci = gi * len(col_suffixes) + si
                    ax = axes[ri, ci]
                    ax.set_xlim(*xlim)
                    ax.set_ylim(-0.05, 1.05)
                    ax.axhline(0.5, color="red", linestyle=":", linewidth=0.5, alpha=0.5)
                    ax.grid(alpha=0.3, linewidth=0.3)
                    ax.tick_params(axis="both", labelsize=8)

                    if suffix == "ica-ctrl-avg":
                        ctrl_conds = [f"{kv}_{s}" for s in ctrl_suffixes]
                        ctrl_df = cell_df[cell_df["condition"].isin(ctrl_conds)]
                        if ctrl_df.empty:
                            ica_df = pd.DataFrame(columns=["icl_count",
                                                           "treatment_acc",
                                                           "control_acc"])
                        else:
                            ica_df = (ctrl_df.groupby("icl_count", as_index=False)
                                             .agg(treatment_acc=("treatment_acc", "mean"),
                                                  control_acc=("control_acc", "mean"))
                                             .sort_values("icl_count"))
                    else:
                        cond = f"{kv}_{suffix}"
                        ica_df = (cell_df[cell_df["condition"] == cond]
                                  .sort_values("icl_count"))

                    dodge = max(0.4, (xlim[1] - xlim[0]) * 0.035)
                    for series, bl_val, color in (
                        ("treatment_acc", bl_treat, _IND_TREATMENT_COLOR),
                        ("control_acc",   bl_ctrl,  _IND_CONTROL_COLOR),
                    ):
                        offset = -dodge if color == _IND_TREATMENT_COLOR else +dodge
                        shots = ica_df["icl_count"].values
                        vals = ica_df[series].values
                        xs = shots + offset
                        if len(shots) and pd.notna(bl_val):
                            ax.scatter(xs, [bl_val] * len(shots), marker="o",
                                       facecolors="none", edgecolors=color, s=45,
                                       linewidths=1.4, zorder=2)
                        if len(shots):
                            ax.scatter(xs, vals, marker="o", color=color,
                                       s=45, linewidths=0.5, edgecolors="black",
                                       zorder=3)
                        for x, v in zip(xs, vals):
                            _plot_arrow(ax, x, bl_val, v, color)

                    if ri == 0:
                        ax.set_title(suffix, fontsize=9, fontweight="bold")
                    if ri != n_rows - 1:
                        ax.tick_params(labelbottom=False)
                    else:
                        ax.set_xticks(xticks)
                        ax.set_xticklabels([str(t) for t in xticks])
                    # Thicker border to delineate kv groups
                    if si == 0 and gi > 0:
                        ax.spines["left"].set_linewidth(2.2)
                        ax.spines["left"].set_color("#444")

        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor="none",
                       markeredgecolor="gray",
                       markeredgewidth=1.5, markersize=8,
                       label="no-ICA (open)"),
            plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor=_IND_TREATMENT_COLOR,
                       markeredgecolor="black", markeredgewidth=0.5,
                       markersize=8,
                       label="treatment"),
            plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor=_IND_CONTROL_COLOR,
                       markeredgecolor="black", markeredgewidth=0.5,
                       markersize=8,
                       label="control"),
        ]
        fig.legend(handles=handles, loc="lower right", fontsize=9,
                   frameon=True, ncol=3, bbox_to_anchor=(0.99, 0.005))
        fig.suptitle(f"All models · {experiment} · IND (ctrl averaged)",
                     fontsize=12, fontweight="bold", y=0.995)
        fig.text(0.5, 0.008, "ICA shot count", ha="center", fontsize=11)
        plt.tight_layout(rect=[0.09, 0.04, 1.0, 0.93])

        # Add kv group labels above each 3-column group (after tight_layout
        # so axes positions are finalized).
        for gi, kv in enumerate(kv_order):
            ax_left = axes[0, gi * len(col_suffixes)]
            ax_right = axes[0, gi * len(col_suffixes) + len(col_suffixes) - 1]
            p_left = ax_left.get_position()
            p_right = ax_right.get_position()
            x_center = (p_left.x0 + p_right.x1) / 2
            fig.text(x_center, p_left.y1 + 0.035, kv,
                     ha="center", va="bottom",
                     fontsize=12, fontweight="bold")

        _draw_model_group_labels(fig, axes, [s["model"] for s in rows_spec])
        _draw_panel_separators(fig, axes,
                               [s["model"] for s in rows_spec],
                               kv_order, len(col_suffixes))

        subdir = output_dir / "all" / experiment / "IND"
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / "dot_arrows.pdf"
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=180)
        plt.close()
        print(f"  ✓ {path} (+ .png)")


def fig_shot_dot_arrows_all_combined(
    agg: pd.DataFrame,
    output_dir: Path,
    kv_order: tuple = ("base", "trained-std", "trained-adv"),
    subdir_name: str = "all",
    title_suffix: str = "IND + PW (ctrl averaged)",
    require_kv_data: str | None = None,
):
    """Consolidated dot-arrow figure per experiment, including IND and PW cells.

    Layout:
      - rows = (model, op) pairs spanning both IND and PW formats
      - cols = len(kv_order) × 3 panels, grouped as [kv_order[0] | ... ],
               each group containing [ica-self, ica-alt, ica-ctrl-avg].
      - IND rows: two series (treatment=blue, control=orange, dodged).
      - PW rows: single metric series in the kv group color.
      - ica-ctrl-avg averages values across ctrl/ctrl2/ctrl3.

    Saved to {output_dir}/all/{experiment}/{subdir_name}/dot_arrows.pdf(.png).

    `require_kv_data`: if set, only emit the figure for experiments that have at
    least one row with a condition starting with `{require_kv_data}_`. Used to
    skip the multi-op variant on experiments that contain no multi-op data.
    """
    agg = agg.copy()
    if agg.empty:
        return

    ctrl_suffixes = ("ica-ctrl", "ica-ctrl2", "ica-ctrl3")
    col_suffixes = ["ica-self", "ica-alt", "ica-ctrl-avg"]

    experiments_present = sorted({e for e in agg["experiment_name"].unique()
                                  if pd.notna(e)})
    for experiment in experiments_present:
        b_df = agg[agg["experiment_name"] == experiment]
        if b_df.empty:
            continue
        if require_kv_data is not None:
            has_required = b_df["condition"].str.startswith(
                f"{require_kv_data}_"
            ).any()
            if not has_required:
                continue

        rows_spec = []
        for model in sorted(b_df["base_model"].unique()):
            m_df = b_df[b_df["base_model"] == model]
            cells = sorted(m_df["cell"].unique(), key=_op_sort_key)
            for cell in cells:
                cell_df = m_df[m_df["cell"] == cell]
                rep = cell_df.iloc[0]
                op = _short_op(rep.get("tags"), rep.get("format"))
                is_ind = is_ind_format(rep.get("format"))
                rows_spec.append({"model": model, "cell": cell, "op": op,
                                  "is_ind": is_ind, "cell_df": cell_df})
        if not rows_spec:
            continue

        xlim, xticks = _compute_shot_axis(b_df)
        n_rows = len(rows_spec)
        n_cols = len(kv_order) * len(col_suffixes)  # 9
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(1.9 * n_cols, 1.45 * n_rows + 1.4),
                                 sharey=True, squeeze=False)

        for ri, spec in enumerate(rows_spec):
            cell_df = spec["cell_df"]
            is_ind = spec["is_ind"]
            axes[ri, 0].set_ylabel(_op_display(spec["op"]), fontsize=10, rotation=0,
                                   ha="right", va="center", fontweight="bold")

            for gi, kv in enumerate(kv_order):
                bl_df = cell_df[cell_df["condition"] == f"{kv}_no-ica"]
                if not bl_df.empty:
                    bl_treat = bl_df["treatment_acc"].iloc[0]
                    bl_ctrl = bl_df["control_acc"].iloc[0]
                    bl_metric = bl_df["metric"].iloc[0]
                else:
                    bl_treat = bl_ctrl = bl_metric = np.nan

                for si, suffix in enumerate(col_suffixes):
                    ci = gi * len(col_suffixes) + si
                    ax = axes[ri, ci]
                    ax.set_xlim(*xlim)
                    ax.set_ylim(-0.05, 1.05)
                    ax.axhline(0.5, color="red", linestyle=":", linewidth=0.5, alpha=0.5)
                    ax.grid(alpha=0.3, linewidth=0.3)
                    ax.tick_params(axis="both", labelsize=8)

                    if suffix == "ica-ctrl-avg":
                        ctrl_conds = [f"{kv}_{s}" for s in ctrl_suffixes]
                        ctrl_df = cell_df[cell_df["condition"].isin(ctrl_conds)]
                        if ctrl_df.empty:
                            ica_df = pd.DataFrame(columns=["icl_count",
                                                           "treatment_acc",
                                                           "control_acc",
                                                           "metric"])
                        else:
                            ica_df = (ctrl_df.groupby("icl_count", as_index=False)
                                             .agg(treatment_acc=("treatment_acc", "mean"),
                                                  control_acc=("control_acc", "mean"),
                                                  metric=("metric", "mean"))
                                             .sort_values("icl_count"))
                    else:
                        cond = f"{kv}_{suffix}"
                        ica_df = (cell_df[cell_df["condition"] == cond]
                                  .sort_values("icl_count"))

                    if is_ind:
                        dodge = max(0.4, (xlim[1] - xlim[0]) * 0.035)
                        for series, bl_val, color in (
                            ("treatment_acc", bl_treat, _IND_TREATMENT_COLOR),
                            ("control_acc",   bl_ctrl,  _IND_CONTROL_COLOR),
                        ):
                            offset = -dodge if color == _IND_TREATMENT_COLOR else +dodge
                            shots = ica_df["icl_count"].values
                            vals = ica_df[series].values
                            xs = shots + offset
                            if len(shots) and pd.notna(bl_val):
                                ax.scatter(xs, [bl_val] * len(shots), marker="o",
                                           facecolors="none", edgecolors=color, s=45,
                                           linewidths=1.4, zorder=2)
                            if len(shots):
                                ax.scatter(xs, vals, marker="o", color=color,
                                           s=45, linewidths=0.5, edgecolors="black",
                                           zorder=3)
                            for x, v in zip(xs, vals):
                                _plot_arrow(ax, x, bl_val, v, color)
                    else:
                        # PW: fixed slate color across all panels.
                        shots = ica_df["icl_count"].values
                        vals = ica_df["metric"].values
                        if len(shots) and pd.notna(bl_metric):
                            ax.scatter(shots, [bl_metric] * len(shots), marker="o",
                                       facecolors="none", edgecolors=_PW_COLOR, s=55,
                                       linewidths=1.5, zorder=2)
                        if len(shots):
                            ax.scatter(shots, vals, marker="o", color=_PW_COLOR,
                                       s=55, linewidths=0.5, edgecolors="black",
                                       zorder=3)
                        for s_, v in zip(shots, vals):
                            _plot_arrow(ax, s_, bl_metric, v, _PW_COLOR)

                    if ri == 0:
                        ax.set_title(suffix, fontsize=9, fontweight="bold")
                    if ri != n_rows - 1:
                        ax.tick_params(labelbottom=False)
                    else:
                        ax.set_xticks(xticks)
                        ax.set_xticklabels([str(t) for t in xticks])
                    if si == 0 and gi > 0:
                        ax.spines["left"].set_linewidth(2.2)
                        ax.spines["left"].set_color("#444")

        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor="none",
                       markeredgecolor="gray",
                       markeredgewidth=1.5, markersize=8,
                       label="no-ICA (open)"),
            plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor=_IND_TREATMENT_COLOR,
                       markeredgecolor="black", markeredgewidth=0.5,
                       markersize=8,
                       label="IND treatment"),
            plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor=_IND_CONTROL_COLOR,
                       markeredgecolor="black", markeredgewidth=0.5,
                       markersize=8,
                       label="IND control"),
            plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor=_PW_COLOR,
                       markeredgecolor="black", markeredgewidth=0.5,
                       markersize=8,
                       label="PW accuracy"),
        ]
        fig.legend(handles=handles, loc="lower right", fontsize=8.5,
                   frameon=True, ncol=4, bbox_to_anchor=(0.99, 0.005))
        fig.suptitle(f"All models · {experiment} · {title_suffix}",
                     fontsize=12, fontweight="bold", y=0.995)
        fig.text(0.5, 0.008, "ICA shot count", ha="center", fontsize=11)
        plt.tight_layout(rect=[0.09, 0.05, 1.0, 0.93])

        for gi, kv in enumerate(kv_order):
            ax_left = axes[0, gi * len(col_suffixes)]
            ax_right = axes[0, gi * len(col_suffixes) + len(col_suffixes) - 1]
            p_left = ax_left.get_position()
            p_right = ax_right.get_position()
            x_center = (p_left.x0 + p_right.x1) / 2
            fig.text(x_center, p_left.y1 + 0.035, kv,
                     ha="center", va="bottom",
                     fontsize=12, fontweight="bold")

        _draw_model_group_labels(fig, axes, [s["model"] for s in rows_spec])
        _draw_panel_separators(fig, axes,
                               [s["model"] for s in rows_spec],
                               kv_order, len(col_suffixes))

        subdir = output_dir / "all" / experiment / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / "dot_arrows.pdf"
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=180)
        _save_titleless_variant(fig, path)
        plt.close()
        print(f"  ✓ {path} (+ .png, + _notitle)")


_ICA_COND_COLORS = {
    "ica-self":     "#0d9488",  # teal — self-priming
    "ica-alt":      "#dc2626",  # red — adversarial-priming
    "ica-ctrl-avg": "#7c3aed",  # purple — neutral third-party control
}


def fig_dot_arrows_per_tag(
    agg: pd.DataFrame,
    output_dir: Path,
    kv_order: tuple = ("base", "trained-std"),
    subdir_name: str = "all/per-tag-shot-avg",
    title_suffix: str = "shot-averaged · per-(model, OP) panel",
    require_kv_data: str | None = None,
):
    """Per-(model, OP) dot-arrow grid, averaged over ICA shot count.

    Layout per experiment:
      - rows = base_model — typically 3 (ll-3.1-8b, gpt-oss-20b, qwen-3.0-30b)
      - cols = OP — 4: UT_PW, UT_IND, AT_PW, AT_IND. PW columns get half
        the horizontal width of IND columns (single series vs two).
      - per panel x-axis is a 2-tier grouping:
          MAIN groups = kvs (e.g. B = base, T = trained-std), drawn as the
            lower-tier x-axis label.
          SUB-positions per main group = series, with major-tick labels:
            PW OPs  → 1 sub-position: "PW"
            IND OPs → 2 sub-positions: "IND-self", "IND-alt"
      - within each (kv, series) sub-position, 3 dot-arrows are dodged
        horizontally, one per ICA condition. Color encodes the ICA
        condition (matches `fig_bars_per_tag`); open marker = no-ICA
        baseline, filled marker = shot-averaged ICA value.

    Saved to {output_dir}/all/{experiment}/{subdir_name}/dot_arrows.{pdf,png}.
    """
    if agg.empty:
        return

    ctrl_suffixes = ("ica-ctrl", "ica-ctrl2", "ica-ctrl3")
    cond_keys = ["ica-self", "ica-alt", "ica-ctrl-avg"]
    OP_ORDER = ["UT_PW", "UT_IND", "AT_PW", "AT_IND"]
    # Per-OP-type series specs: (label, value column in agg).
    series_for_pw = [("PW",      "metric")]
    series_for_ind = [("IND-self", "control_acc"),
                      ("IND-alt",  "treatment_acc")]

    n_kvs = len(kv_order)
    # Small horizontal dodge for the 3 ICA conditions within each sub-position.
    cond_dodge = 0.10
    cond_offsets = [-cond_dodge, 0.0, +cond_dodge]
    # Short kv labels for the lower-tier x-axis label drawn beneath sub-ticks.
    # Display labels for the lower-tier x-axis. Every trained-* variant
    # collapses to "trained" — the variant identity is in the figure's
    # subdir/filename rather than the axis label.
    kv_short = {"base": "base",
                "trained-std": "trained",
                "trained-adv": "trained",
                "trained-multi-op": "trained",
                "trained-randlabels": "trained"}

    # PW columns get half the horizontal width of IND columns.
    width_ratios_full = [1 if op.endswith("_PW") else 2 for op in OP_ORDER]
    max_wr = max(width_ratios_full)

    def _layout(n_series: int, panel_wr: int):
        """Return (kv_centers, series_offsets, xlim) so that a fixed
        cond_dodge in data units translates to the same *physical* dot
        spacing across panels of different gridspec width_ratios. Achieved
        by scaling kv spacing, series offsets, and x-padding by
        panel_wr / max_wr."""
        scale = panel_wr / max_wr
        kv_spacing = 1.0 * scale
        kv_centers = [i * kv_spacing for i in range(n_kvs)]
        if n_series == 1:
            series_offsets = [0.0]
        elif n_series == 2:
            series_offsets = [-0.20 * scale, +0.20 * scale]
        else:
            spread = 0.30 * scale * (n_series - 1)
            series_offsets = [(-spread / 2 + i * 0.30 * scale)
                              for i in range(n_series)]
        x_pad = 0.55 * scale
        xlim = (-x_pad, kv_centers[-1] + x_pad)
        return kv_centers, series_offsets, xlim

    experiments_present = sorted({e for e in agg["experiment_name"].unique()
                                  if pd.notna(e)})

    for experiment in experiments_present:
        b_df = agg[agg["experiment_name"] == experiment]
        if b_df.empty:
            continue
        # Cross-op experiments (SGTR_03..06, _08, _10) lack base rows and the
        # trained_op cell. Pull them in from the self-op partner so per-(model,
        # OP) panels render fully populated.
        b_df = _augment_with_partner(agg, b_df, experiment, kv_order)
        if require_kv_data is not None:
            has_required = b_df["condition"].astype(str).str.startswith(
                f"{require_kv_data}_").any()
            if not has_required:
                continue

        models = sorted(b_df["base_model"].dropna().unique())
        if not models:
            continue

        # Drop OP columns that have no data anywhere in this experiment
        # (e.g. SGTR_09 / SGTR_10 evaluate IND-only, so UT_PW / AT_PW
        # would otherwise render as empty panels).
        ops_present = [
            op for op in OP_ORDER
            if ((b_df["tags"] == op.split("_")[0])
                & b_df["format"].astype(str)
                     .str.startswith(op.split("_")[1])).any()
        ]
        if not ops_present:
            continue
        width_ratios_local = [1 if op.endswith("_PW") else 2 for op in ops_present]

        n_rows = len(models)
        n_cols = len(ops_present)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(1.6 * sum(width_ratios_local) + 1.5, 1.9 * n_rows + 1.6),
            sharey=True, squeeze=False,
            gridspec_kw={"width_ratios": width_ratios_local},
        )

        def _val(df_kv, kv, suffix, col):
            """Return the shot-averaged value for `df_kv[condition == f'{kv}_{suffix}']`,
            or for ica-ctrl-avg, the average across the three ctrl variants AND shots."""
            if df_kv.empty:
                return float("nan")
            if suffix == "ica-ctrl-avg":
                conds = [f"{kv}_{s}" for s in ctrl_suffixes]
                sub = df_kv[df_kv["condition"].isin(conds)]
            else:
                sub = df_kv[df_kv["condition"] == f"{kv}_{suffix}"]
            if sub.empty:
                return float("nan")
            return float(sub[col].astype(float).mean())

        for ri, model in enumerate(models):
            for ci, op in enumerate(ops_present):
                ax = axes[ri, ci]
                tag, fmt_prefix = op.split("_")  # e.g. "UT", "PW"
                op_df = b_df[(b_df["base_model"] == model) &
                             (b_df["tags"] == tag) &
                             b_df["format"].astype(str).str.startswith(fmt_prefix)]
                is_ind = (fmt_prefix == "IND")
                series_specs = series_for_ind if is_ind else series_for_pw
                n_series = len(series_specs)
                kv_centers, series_offsets, xlim = _layout(n_series,
                                                           width_ratios_local[ci])

                ax.set_xlim(*xlim)
                ax.set_ylim(-0.05, 1.05)
                ax.axhline(0.5, color="red", linestyle=":",
                           linewidth=0.5, alpha=0.5)
                ax.grid(alpha=0.3, linewidth=0.3)
                ax.tick_params(axis="both", labelsize=8)

                for kvi, kv in enumerate(kv_order):
                    kv_df = op_df[op_df["condition"].astype(str)
                                  .str.startswith(f"{kv}_")]
                    cx = kv_centers[kvi]
                    for sgi, (series_label, col_name) in enumerate(series_specs):
                        sub_x = cx + series_offsets[sgi]
                        bl = _val(kv_df, kv, "no-ica", col_name)
                        for ic, cond_key in enumerate(cond_keys):
                            color = _ICA_COND_COLORS[cond_key]
                            v = _val(kv_df, kv, cond_key, col_name)
                            x = sub_x + cond_offsets[ic]
                            if pd.notna(bl):
                                ax.scatter(x, bl, marker="o",
                                           facecolors="none",
                                           edgecolors=color, s=45,
                                           linewidths=1.3, zorder=2)
                            if pd.notna(v):
                                ax.scatter(x, v, marker="o", color=color,
                                           s=45, linewidths=0.4,
                                           edgecolors="black", zorder=3)
                            if pd.notna(bl) and pd.notna(v):
                                _plot_arrow(ax, x, bl, v, color)

                # X-axis (2-tier): series sub-positions are major ticks
                # (PW / IND-t / IND-ctrl); kv labels (B / T) are drawn
                # below at the kv group centers.
                sub_positions, sub_labels = [], []
                for kvi in range(n_kvs):
                    for sgi, (series_label, _) in enumerate(series_specs):
                        sub_positions.append(kv_centers[kvi] + series_offsets[sgi])
                        sub_labels.append(series_label)
                ax.set_xticks(sub_positions)
                if ri == n_rows - 1:
                    ax.set_xticklabels(sub_labels, fontsize=8)
                    for kvi, kv in enumerate(kv_order):
                        ax.text(kv_centers[kvi], -0.18, kv_short.get(kv, kv),
                                transform=ax.get_xaxis_transform(),
                                ha="left", va="top",
                                rotation=-45, rotation_mode="anchor",
                                fontsize=9, fontweight="bold")
                else:
                    ax.set_xticklabels([])

                if ri == 0:
                    ax.set_title(_op_display(op), fontsize=11, fontweight="bold")
                if ci == 0:
                    ax.set_ylabel(_model_display(model), fontsize=11, rotation=0,
                                  ha="right", va="center", fontweight="bold",
                                  labelpad=10)

        # Legend: a single gray no-ICA (open) marker, then one filled marker
        # per ICA condition, plus a kv-abbreviation key for the x-axis.
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor="none",
                       markeredgecolor="gray",
                       markeredgewidth=1.5, markersize=8,
                       label="no-ICA (open)"),
        ]
        for cond_key in cond_keys:
            color = _ICA_COND_COLORS[cond_key]
            handles.append(plt.Line2D([0], [0], marker="o", linestyle="",
                                      markerfacecolor=color,
                                      markeredgecolor="black",
                                      markeredgewidth=0.5, markersize=8,
                                      label=cond_key))
        fig.legend(handles=handles, loc="lower right", fontsize=8.5,
                   frameon=True, ncol=3, bbox_to_anchor=(0.99, 0.005))
        fig.suptitle(f"{experiment} · {title_suffix}",
                     fontsize=12, fontweight="bold", y=0.995)
        plt.tight_layout(rect=[0.06, 0.16, 1.0, 0.95])

        subdir = output_dir / "all" / experiment / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / "dot_arrows.pdf"
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=180)
        _save_titleless_variant(fig, path)
        plt.close()
        print(f"  ✓ {path} (+ .png, + _notitle)")


def fig_dot_arrows_per_tag_diff(
    agg: pd.DataFrame,
    output_dir: Path,
    kv_order: tuple = ("base", "trained-std"),
    subdir_name: str = "all/per-tag-shot-avg",
    output_name: str = "dot_arrows_diff",
    title_suffix: str | None = None,
    require_kv_data: str | None = None,
):
    """Plot the *training-induced* ICA shift per (model, OP, series, cond).

    For each (model, OP, series, condition) we compute:
        diff = (ICA - no-ICA)_trained-std - (ICA - no-ICA)_base
    A diff of 0 means the trained model is no more or less ICA-susceptible
    than its base. Positive = training amplified the ICA shift (model is
    *more* affected by the priming after training). Negative = training
    made the model more ICA-robust.

    Layout per experiment:
      - rows = base_model — typically 3.
      - cols = OP (UT_PW, UT_IND, AT_PW, AT_IND); PW columns half-width.
      - per panel x-axis = series sub-positions:
          PW OPs  → 1 position labeled "PW"
          IND OPs → 2 positions labeled "IND-self", "IND-alt"
      - per (series, condition): 1 dot at the diff value with a lollipop
        line dropping to y=0; condition encoded by color (matches
        `fig_dot_arrows_per_tag` and `fig_bars_per_tag`).
      - solid horizontal reference line at y=0.

    Saved to {output_dir}/all/{experiment}/{subdir_name}/{output_name}.{pdf,png}.
    """
    if agg.empty:
        return
    # The diff figure requires both `base` and a trained kv. Default to
    # ("base", "trained-std"); skip silently if either is missing.
    if "base" not in kv_order:
        return
    trained_kvs = [k for k in kv_order if k != "base"]
    if not trained_kvs:
        return
    trained_kv = trained_kvs[0]  # use the first trained kv (typically trained-std)

    # Auto-derive the title suffix and y-axis label from the trained kv if
    # not explicitly overridden, so figures named after multi-op or
    # randlabels variants render with the right Δ subscript.
    trained_short = trained_kv.replace("trained-", "") or "trained"
    if title_suffix is None:
        title_suffix = (r"shot-averaged · training-induced ICA shift "
                        rf"= $\Delta_{{{trained_short}}} - \Delta_{{base}}$")

    ctrl_suffixes = ("ica-ctrl", "ica-ctrl2", "ica-ctrl3")
    cond_keys = ["ica-self", "ica-alt", "ica-ctrl-avg"]
    OP_ORDER = ["UT_PW", "UT_IND", "AT_PW", "AT_IND"]
    series_for_pw = [("PW",      "metric")]
    series_for_ind = [("IND-self", "control_acc"),
                      ("IND-alt",  "treatment_acc")]
    cond_dodge = 0.10
    cond_offsets = [-cond_dodge, 0.0, +cond_dodge]

    experiments_present = sorted({e for e in agg["experiment_name"].unique()
                                  if pd.notna(e)})

    for experiment in experiments_present:
        b_df = agg[agg["experiment_name"] == experiment]
        if b_df.empty:
            continue
        # Cross-op experiments (SGTR_03..06, _08, _10) lack base rows and the
        # trained_op cell. Pull them in from the self-op partner so per-(model,
        # OP) panels render fully populated.
        b_df = _augment_with_partner(agg, b_df, experiment, kv_order)
        if require_kv_data is not None:
            has_required = b_df["condition"].astype(str).str.startswith(
                f"{require_kv_data}_").any()
            if not has_required:
                continue
        models = sorted(b_df["base_model"].dropna().unique())
        if not models:
            continue

        # Drop OP columns that have no data anywhere in this experiment.
        ops_present = [
            op for op in OP_ORDER
            if ((b_df["tags"] == op.split("_")[0])
                & b_df["format"].astype(str)
                     .str.startswith(op.split("_")[1])).any()
        ]
        if not ops_present:
            continue

        n_rows = len(models)
        n_cols = len(ops_present)
        width_ratios = [1 if op.endswith("_PW") else 2 for op in ops_present]
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(1.6 * sum(width_ratios) + 1.5, 1.9 * n_rows + 1.6),
            sharey=True, squeeze=False,
            gridspec_kw={"width_ratios": width_ratios},
        )

        def _val(df_kv, kv, suffix, col):
            if df_kv.empty:
                return float("nan")
            if suffix == "ica-ctrl-avg":
                conds = [f"{kv}_{s}" for s in ctrl_suffixes]
                sub = df_kv[df_kv["condition"].isin(conds)]
            else:
                sub = df_kv[df_kv["condition"] == f"{kv}_{suffix}"]
            if sub.empty:
                return float("nan")
            return float(sub[col].astype(float).mean())

        all_diffs: list[float] = []  # for symmetric ylim sizing

        for ri, model in enumerate(models):
            for ci, op in enumerate(ops_present):
                ax = axes[ri, ci]
                tag, fmt_prefix = op.split("_")
                op_df = b_df[(b_df["base_model"] == model) &
                             (b_df["tags"] == tag) &
                             b_df["format"].astype(str).str.startswith(fmt_prefix)]
                is_ind = (fmt_prefix == "IND")
                series_specs = series_for_ind if is_ind else series_for_pw
                n_series = len(series_specs)

                ax.set_xlim(-0.55, n_series - 0.45)
                ax.axhline(0, color="black", linewidth=0.7, zorder=1)
                ax.grid(axis="y", alpha=0.3, linewidth=0.3)
                ax.tick_params(axis="both", labelsize=8)

                base_df = op_df[op_df["condition"].astype(str)
                                .str.startswith("base_")]
                trained_df = op_df[op_df["condition"].astype(str)
                                   .str.startswith(f"{trained_kv}_")]

                for sgi, (series_label, col_name) in enumerate(series_specs):
                    cx = sgi
                    base_no_ica = _val(base_df, "base", "no-ica", col_name)
                    trained_no_ica = _val(trained_df, trained_kv, "no-ica", col_name)
                    for ic, cond_key in enumerate(cond_keys):
                        color = _ICA_COND_COLORS[cond_key]
                        base_ica = _val(base_df, "base", cond_key, col_name)
                        trained_ica = _val(trained_df, trained_kv, cond_key, col_name)
                        if (pd.isna(base_no_ica) or pd.isna(base_ica) or
                                pd.isna(trained_no_ica) or pd.isna(trained_ica)):
                            continue
                        delta_base = base_ica - base_no_ica
                        delta_trained = trained_ica - trained_no_ica
                        diff = delta_trained - delta_base
                        x = cx + cond_offsets[ic]
                        # Lollipop stem from y=0 to the diff value.
                        ax.plot([x, x], [0, diff], color=color,
                                linewidth=1.2, zorder=2)
                        ax.scatter(x, diff, marker="o", color=color, s=55,
                                   linewidths=0.4, edgecolors="black", zorder=3)
                        all_diffs.append(diff)

                ax.set_xticks(range(n_series))
                if ri == n_rows - 1:
                    ax.set_xticklabels([s[0] for s in series_specs],
                                       fontsize=9, fontweight="bold")
                else:
                    ax.set_xticklabels([])

                if ri == 0:
                    ax.set_title(_op_display(op), fontsize=11, fontweight="bold")
                if ci == 0:
                    ax.set_ylabel(_model_display(model), fontsize=11, rotation=0,
                                  ha="right", va="center", fontweight="bold",
                                  labelpad=10)

        # Symmetric y-limits: default to [-1.0, +1.0] for cross-figure
        # comparability; expand only if any |diff| exceeds 1.0.
        if all_diffs:
            data_max = max(abs(v) for v in all_diffs)
            ymag = max(1.0, data_max * 1.05)
        else:
            ymag = 1.0
        for ax in axes.flat:
            ax.set_ylim(-ymag, ymag)

        # Legend: 3 ICA condition swatches.
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor=_ICA_COND_COLORS[k],
                       markeredgecolor="black", markeredgewidth=0.4,
                       markersize=8, label=k)
            for k in cond_keys
        ]
        fig.legend(handles=handles, loc="lower right", fontsize=9,
                   frameon=True, ncol=3, bbox_to_anchor=(0.99, 0.005))
        fig.suptitle(f"{experiment} · {title_suffix}",
                     fontsize=11, fontweight="bold", y=0.995)
        fig.supylabel(r"$\Delta_{trained} - \Delta_{base}$", fontsize=11,
                      x=0.005)
        plt.tight_layout(rect=[0.05, 0.05, 1.0, 0.95])

        subdir = output_dir / "all" / experiment / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f"{output_name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=180)
        _save_titleless_variant(fig, path)
        plt.close()
        print(f"  ✓ {path} (+ .png, + _notitle)")


def fig_bars_per_tag(
    agg: pd.DataFrame,
    output_dir: Path,
    kv_order: tuple = ("base", "trained-std"),
    subdir_name: str = "all/per-tag-shot-avg",
    output_name: str = "bars",
    title_suffix: str = "shot-averaged · per-(model, OP) bar chart",
    require_kv_data: str | None = None,
):
    """Per-(model, OP) bar chart with cross-hatched no-ICA overlay.

    Same data and panel layout as `fig_dot_arrows_per_tag`, rendered as
    grouped bars instead of dot-arrows:
      - rows = base_model — typically 3.
      - cols = OP — 4 (UT_PW, UT_IND, AT_PW, AT_IND); PW columns half-width.
      - per panel x-axis is a 2-tier grouping:
          MAIN groups = kvs (B = base, T = trained-std), drawn as the
            lower-tier x-axis label.
          SUB-positions per main group = series:
            PW OPs  → 1 sub-position: "PW"
            IND OPs → 2 sub-positions: "IND-self", "IND-alt"
      - within each (kv, series) sub-position: 3 solid bars dodged by ICA
        condition (color-coded; consistent with dot-arrow figure).
      - the corresponding series's no-ICA accuracy is overlaid on every
        ICA bar in that sub-position as a transparent cross-hatched bar
        — same baseline-vs-ICA contrast the dot-arrow figure conveys
        with open vs filled dots.

    Saved to {output_dir}/all/{experiment}/{subdir_name}/{output_name}.{pdf,png}.
    """
    if agg.empty:
        return

    ctrl_suffixes = ("ica-ctrl", "ica-ctrl2", "ica-ctrl3")
    cond_keys = ["ica-self", "ica-alt", "ica-ctrl-avg"]
    OP_ORDER = ["UT_PW", "UT_IND", "AT_PW", "AT_IND"]
    series_for_pw = [("PW",      "metric")]
    series_for_ind = [("IND-self", "control_acc"),
                      ("IND-alt",  "treatment_acc")]

    n_kvs = len(kv_order)
    cond_dodge = 0.10
    cond_offsets = [-cond_dodge, 0.0, +cond_dodge]
    bar_w = 0.085
    # Display labels for the lower-tier x-axis. Every trained-* variant
    # collapses to "trained" — the variant identity is in the figure's
    # subdir/filename rather than the axis label.
    kv_short = {"base": "base",
                "trained-std": "trained",
                "trained-adv": "trained",
                "trained-multi-op": "trained",
                "trained-randlabels": "trained"}

    # PW columns get half the horizontal width of IND columns.
    width_ratios_full = [1 if op.endswith("_PW") else 2 for op in OP_ORDER]
    max_wr = max(width_ratios_full)

    def _layout(n_series: int, panel_wr: int):
        """Return (kv_centers, series_offsets, xlim) so that a fixed bar_w in
        data units renders to the same *physical* bar width across panels of
        different gridspec width_ratios. We achieve this by scaling the kv
        spacing (and accompanying x-padding) by panel_wr / max_wr."""
        scale = panel_wr / max_wr
        kv_spacing = 1.0 * scale
        kv_centers = [i * kv_spacing for i in range(n_kvs)]
        if n_series == 1:
            series_offsets = [0.0]
        elif n_series == 2:
            series_offsets = [-0.20 * scale, +0.20 * scale]
        else:
            spread = 0.30 * scale * (n_series - 1)
            series_offsets = [(-spread / 2 + i * 0.30 * scale)
                              for i in range(n_series)]
        x_pad = 0.55 * scale
        xlim = (-x_pad, kv_centers[-1] + x_pad)
        return kv_centers, series_offsets, xlim

    experiments_present = sorted({e for e in agg["experiment_name"].unique()
                                  if pd.notna(e)})

    for experiment in experiments_present:
        b_df = agg[agg["experiment_name"] == experiment]
        if b_df.empty:
            continue
        # Cross-op experiments (SGTR_03..06, _08, _10) lack base rows and the
        # trained_op cell. Pull them in from the self-op partner so per-(model,
        # OP) panels render fully populated.
        b_df = _augment_with_partner(agg, b_df, experiment, kv_order)
        if require_kv_data is not None:
            has_required = b_df["condition"].astype(str).str.startswith(
                f"{require_kv_data}_").any()
            if not has_required:
                continue
        models = sorted(b_df["base_model"].dropna().unique())
        if not models:
            continue

        # Drop OP columns that have no data anywhere in this experiment.
        ops_present = [
            op for op in OP_ORDER
            if ((b_df["tags"] == op.split("_")[0])
                & b_df["format"].astype(str)
                     .str.startswith(op.split("_")[1])).any()
        ]
        if not ops_present:
            continue
        width_ratios_local = [1 if op.endswith("_PW") else 2 for op in ops_present]

        n_rows = len(models)
        n_cols = len(ops_present)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(1.6 * sum(width_ratios_local) + 1.5, 1.9 * n_rows + 1.6),
            sharey=True, squeeze=False,
            gridspec_kw={"width_ratios": width_ratios_local},
        )

        def _val(df_kv, kv, suffix, col):
            if df_kv.empty:
                return float("nan")
            if suffix == "ica-ctrl-avg":
                conds = [f"{kv}_{s}" for s in ctrl_suffixes]
                sub = df_kv[df_kv["condition"].isin(conds)]
            else:
                sub = df_kv[df_kv["condition"] == f"{kv}_{suffix}"]
            if sub.empty:
                return float("nan")
            return float(sub[col].astype(float).mean())

        for ri, model in enumerate(models):
            for ci, op in enumerate(ops_present):
                ax = axes[ri, ci]
                tag, fmt_prefix = op.split("_")
                op_df = b_df[(b_df["base_model"] == model) &
                             (b_df["tags"] == tag) &
                             b_df["format"].astype(str).str.startswith(fmt_prefix)]
                is_ind = (fmt_prefix == "IND")
                series_specs = series_for_ind if is_ind else series_for_pw
                n_series = len(series_specs)
                kv_centers, series_offsets, xlim = _layout(n_series,
                                                           width_ratios_local[ci])

                ax.set_xlim(*xlim)
                ax.set_ylim(0.0, 1.05)
                ax.axhline(0.5, color="red", linestyle=":",
                           linewidth=0.5, alpha=0.5)
                ax.grid(axis="y", alpha=0.3, linewidth=0.3)
                ax.tick_params(axis="both", labelsize=8)

                for kvi, kv in enumerate(kv_order):
                    kv_df = op_df[op_df["condition"].astype(str)
                                  .str.startswith(f"{kv}_")]
                    cx = kv_centers[kvi]
                    for sgi, (series_label, col_name) in enumerate(series_specs):
                        sub_x = cx + series_offsets[sgi]
                        bl = _val(kv_df, kv, "no-ica", col_name)
                        for ic, cond_key in enumerate(cond_keys):
                            color = _ICA_COND_COLORS[cond_key]
                            v = _val(kv_df, kv, cond_key, col_name)
                            x = sub_x + cond_offsets[ic]
                            # Solid ICA bar.
                            if pd.notna(v):
                                ax.bar(x, v, width=bar_w, color=color,
                                       edgecolor="black", linewidth=0.4,
                                       zorder=3)
                            # Diagonal-hatched no-ICA overlay (same x).
                            if pd.notna(bl):
                                ax.bar(x, bl, width=bar_w, facecolor="none",
                                       edgecolor="black", linewidth=0.7,
                                       hatch="///", zorder=4)

                # X-axis (2-tier): series sub-positions are major ticks
                # (PW / IND-t / IND-ctrl); kv labels (B / T) are drawn
                # below at the kv group centers.
                sub_positions, sub_labels = [], []
                for kvi in range(n_kvs):
                    for sgi, (series_label, _) in enumerate(series_specs):
                        sub_positions.append(kv_centers[kvi] + series_offsets[sgi])
                        sub_labels.append(series_label)
                ax.set_xticks(sub_positions)
                if ri == n_rows - 1:
                    ax.set_xticklabels(sub_labels, fontsize=8)
                    for kvi, kv in enumerate(kv_order):
                        ax.text(kv_centers[kvi], -0.18, kv_short.get(kv, kv),
                                transform=ax.get_xaxis_transform(),
                                ha="left", va="top",
                                rotation=-45, rotation_mode="anchor",
                                fontsize=9, fontweight="bold")
                else:
                    ax.set_xticklabels([])

                if ri == 0:
                    ax.set_title(_op_display(op), fontsize=11, fontweight="bold")
                if ci == 0:
                    ax.set_ylabel(_model_display(model), fontsize=11, rotation=0,
                                  ha="right", va="center", fontweight="bold",
                                  labelpad=10)

        # Legend: 3 ICA condition swatches + 1 hatched explanation for the
        # no-ICA overlay + a kv abbreviation key.
        cond_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=_ICA_COND_COLORS[k],
                          edgecolor="black", linewidth=0.4, label=k)
            for k in cond_keys
        ]
        cond_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor="none",
                          edgecolor="black", linewidth=0.9, hatch="///",
                          label="no-ICA (overlay)")
        )
        fig.legend(handles=cond_handles, loc="lower right", fontsize=8.5,
                   frameon=True, ncol=4, bbox_to_anchor=(0.99, 0.005))
        fig.suptitle(f"{experiment} · {title_suffix}",
                     fontsize=12, fontweight="bold", y=0.995)
        plt.tight_layout(rect=[0.06, 0.16, 1.0, 0.95])

        subdir = output_dir / "all" / experiment / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f"{output_name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=180)
        plt.close()
        print(f"  ✓ {path} (+ .png)")


# Each entry maps a cross-op experiment to its trained_op + self-same-OP partner
# (which provides base + train==test trained-std data for comparison rows).
# `test_ops` limits the rows shown; default is all 4 when unspecified.
CROSS_OP_EXPERIMENTS = [
    # (trained_op, cross_op_exp, self_op_exp, test_ops)
    ("AT_IND", "SGTR_03_trained-AT-IND_eval-on_all-OPs",
               "SGTR_02_trained-OP_eval-on_self-same-OP", None),
    ("UT_IND", "SGTR_04_trained-UT-IND_eval-on_all-OPs",
               "SGTR_02_trained-OP_eval-on_self-same-OP", None),
    ("AT_PW",  "SGTR_05_trained-AT-PW_eval-on_all-OPs",
               "SGTR_02_trained-OP_eval-on_self-same-OP", None),
    ("UT_PW",  "SGTR_06_trained-UT-PW_eval-on_all-OPs",
               "SGTR_02_trained-OP_eval-on_self-same-OP", None),
    ("AT_IND", "SGTR_08_trained-AT-IND-ShareGPT_eval-on_all-OPs-WikiSum_ICA-ShareGPT",
               "SGTR_07_trained-OP-ShareGPT_eval-on_self-same-OP-WikiSum_ICA-ShareGPT",
               ["UT_IND", "AT_IND"]),  # IND-only; UT first per OP_DISPLAY_ORDER
    ("AT_IND", "SGTR_10_trained-AT-IND-ShareGPT_eval-on_all-OPs-WikiSum_ICA-WikiSum",
               "SGTR_09_trained-OP-ShareGPT_eval-on_self-same-OP-WikiSum_ICA-WikiSum",
               ["UT_IND", "AT_IND"]),  # IND-only; UT first per OP_DISPLAY_ORDER
]


def _augment_with_partner(agg: pd.DataFrame, b_df: pd.DataFrame,
                          experiment: str, kv_order: tuple) -> pd.DataFrame:
    """If `experiment` is a cross-op variant per CROSS_OP_EXPERIMENTS, append
    rows from its self-op partner so per-(model, OP) panels can render base
    data — which the cross-op experiment typically lacks — and also pull
    trained-std data on the trained_op cell, which lives only in the partner
    (since the cross-op experiment evaluates exclusively on cross-OPs).

    Restricted to the standard `("base", "trained-std")` view: the multi-op
    and randlabels variants shouldn't be back-filled with partner data
    because the cross-op multi-op / randlabels figures aren't conceptually
    meaningful (those LoRAs are op-agnostic / control runs).

    Partner rows are re-tagged as belonging to `experiment` so downstream
    filtering by experiment_name treats them consistently.
    """
    # Only merge for the standard "all" view (base + trained-std).
    trained_kvs = [k for k in kv_order if k != "base"]
    if trained_kvs != ["trained-std"]:
        return b_df

    partners = {
        cross_exp: (trained_op, self_exp)
        for trained_op, cross_exp, self_exp, _ in CROSS_OP_EXPERIMENTS
    }
    if experiment not in partners:
        return b_df
    trained_op, partner_exp = partners[experiment]
    partner = agg[agg["experiment_name"] == partner_exp]
    if partner.empty:
        return b_df

    # Base rows from partner — for every (model, OP) cell.
    base_rows = partner[partner["condition"].astype(str).str.startswith("base_")]

    # Trained-std on the trained_op (e.g. AT_IND) — that cell lives in
    # the partner, not the cross-op exp.
    op_to_fmt = {v: k for k, v in _FMT_TO_OP.items()}
    trained_fmt = op_to_fmt.get(trained_op)
    trained_tag = trained_op.split("_")[0]
    trained_rows = partner[
        partner["condition"].astype(str).str.startswith("trained-std_")
        & (partner["tags"] == trained_tag)
        & (partner["format"] == trained_fmt)
    ]

    extras = pd.concat([base_rows, trained_rows], ignore_index=True)
    if extras.empty:
        return b_df
    extras = extras.assign(experiment_name=experiment)
    return pd.concat([b_df, extras], ignore_index=True)


def fig_shot_dot_arrows_cross_op(agg: pd.DataFrame, output_dir: Path,
                                 trained_op: str, cross_op_exp_name: str,
                                 self_op_exp_name: str,
                                 test_ops: list | None = None):
    """Cross-op dot-arrow figure for one trained_op.

    Rows: (model, test_op) across all 4 ops. The row where test_op == trained_op
    pulls trained-std data from the self-same-OP experiment (since that's where
    the matching training-regime eval lives). The other 3 rows (left-out ops)
    pull from the cross-op experiment identified by `cross_op_exp_name`.
    Base columns always come from the self-same-OP experiment.
    """
    ctrl_suffixes = ("ica-ctrl", "ica-ctrl2", "ica-ctrl3")
    col_suffixes = ["ica-self", "ica-alt", "ica-ctrl-avg"]
    # trained-adv is dropped: cross-op adversarial variants do not exist.
    kv_order = ("base", "trained-std")
    if test_ops is None:
        test_ops = list(OP_DISPLAY_ORDER)
    OP_FMT = {"UT_IND": ("UT", "IND-Q"), "AT_IND": ("AT", "IND-C"),
              "UT_PW":  ("UT", "PW-Q"),  "AT_PW":  ("AT", "PW-C")}

    self_op_agg = agg[agg["experiment_name"] == self_op_exp_name]
    cross_op_agg = agg[agg["experiment_name"] == cross_op_exp_name]
    if cross_op_agg.empty:
        return

    rows_spec = []
    for model in sorted(cross_op_agg["base_model"].unique()):
        for test_op in test_ops:
            tags, fmt = OP_FMT[test_op]
            self_cell = self_op_agg[(self_op_agg["base_model"] == model)
                                    & (self_op_agg["tags"] == tags)
                                    & (self_op_agg["format"] == fmt)]
            cross_cell = cross_op_agg[(cross_op_agg["base_model"] == model)
                                      & (cross_op_agg["tags"] == tags)
                                      & (cross_op_agg["format"] == fmt)]
            # trained-std: self_cell when train == test, else cross_cell
            trained_std_df = self_cell if test_op == trained_op else cross_cell
            rows_spec.append({
                "model": model, "test_op": test_op,
                "is_ind": is_ind_format(fmt),
                "b2_cell": self_cell, "trained_std_df": trained_std_df,
            })
    if not rows_spec:
        return

    b_df_all = pd.concat([self_op_agg, cross_op_agg], ignore_index=True)
    xlim, xticks = _compute_shot_axis(b_df_all)

    n_rows = len(rows_spec)
    n_cols = len(kv_order) * len(col_suffixes)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(1.9 * n_cols, 1.45 * n_rows + 1.4),
                             sharey=True, squeeze=False)

    for ri, spec in enumerate(rows_spec):
        is_ind = spec["is_ind"]
        axes[ri, 0].set_ylabel(_op_display(spec["test_op"]), fontsize=10, rotation=0,
                               ha="right", va="center", fontweight="bold")

        for gi, kv in enumerate(kv_order):
            source_df = spec["trained_std_df"] if kv == "trained-std" else spec["b2_cell"]
            bl_df = source_df[source_df["condition"] == f"{kv}_no-ica"]
            if not bl_df.empty:
                bl_treat = bl_df["treatment_acc"].iloc[0]
                bl_ctrl  = bl_df["control_acc"].iloc[0]
                bl_metric = bl_df["metric"].iloc[0]
            else:
                bl_treat = bl_ctrl = bl_metric = np.nan

            for si, suffix in enumerate(col_suffixes):
                ci = gi * len(col_suffixes) + si
                ax = axes[ri, ci]
                ax.set_xlim(*xlim)
                ax.set_ylim(-0.05, 1.05)
                ax.axhline(0.5, color="red", linestyle=":", linewidth=0.5, alpha=0.5)
                ax.grid(alpha=0.3, linewidth=0.3)
                ax.tick_params(axis="both", labelsize=8)

                if suffix == "ica-ctrl-avg":
                    ctrl_conds = [f"{kv}_{s}" for s in ctrl_suffixes]
                    ctrl_df = source_df[source_df["condition"].isin(ctrl_conds)]
                    if ctrl_df.empty:
                        ica_df = pd.DataFrame(columns=["icl_count", "treatment_acc",
                                                       "control_acc", "metric"])
                    else:
                        ica_df = (ctrl_df.groupby("icl_count", as_index=False)
                                         .agg(treatment_acc=("treatment_acc", "mean"),
                                              control_acc=("control_acc", "mean"),
                                              metric=("metric", "mean"))
                                         .sort_values("icl_count"))
                else:
                    cond = f"{kv}_{suffix}"
                    ica_df = (source_df[source_df["condition"] == cond]
                              .sort_values("icl_count"))

                if is_ind:
                    dodge = max(0.4, (xlim[1] - xlim[0]) * 0.035)
                    for series, bl_val, color in (
                        ("treatment_acc", bl_treat, _IND_TREATMENT_COLOR),
                        ("control_acc",   bl_ctrl,  _IND_CONTROL_COLOR),
                    ):
                        offset = -dodge if color == _IND_TREATMENT_COLOR else +dodge
                        shots = ica_df["icl_count"].values
                        vals = ica_df[series].values
                        xs = shots + offset
                        if len(shots) and pd.notna(bl_val):
                            ax.scatter(xs, [bl_val] * len(shots), marker="o",
                                       facecolors="none", edgecolors=color, s=45,
                                       linewidths=1.4, zorder=2)
                        if len(shots):
                            ax.scatter(xs, vals, marker="o", color=color,
                                       s=45, linewidths=0.5, edgecolors="black",
                                       zorder=3)
                        for x, v in zip(xs, vals):
                            _plot_arrow(ax, x, bl_val, v, color)
                else:
                    # PW: fixed slate color across all panels.
                    shots = ica_df["icl_count"].values
                    vals = ica_df["metric"].values
                    if len(shots) and pd.notna(bl_metric):
                        ax.scatter(shots, [bl_metric] * len(shots), marker="o",
                                   facecolors="none", edgecolors=_PW_COLOR, s=55,
                                   linewidths=1.5, zorder=2)
                    if len(shots):
                        ax.scatter(shots, vals, marker="o", color=_PW_COLOR,
                                   s=55, linewidths=0.5, edgecolors="black", zorder=3)
                    for s_, v in zip(shots, vals):
                        _plot_arrow(ax, s_, bl_metric, v, _PW_COLOR)

                if ri == 0:
                    ax.set_title(suffix, fontsize=9, fontweight="bold")
                if ri != n_rows - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xticks(xticks)
                    ax.set_xticklabels([str(t) for t in xticks])
                if si == 0 and gi > 0:
                    ax.spines["left"].set_linewidth(2.2)
                    ax.spines["left"].set_color("#444")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor="none",
                   markeredgecolor="gray",
                   markeredgewidth=1.5, markersize=8,
                   label="no-ICA (open)"),
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=_IND_TREATMENT_COLOR,
                   markeredgecolor="black", markeredgewidth=0.5,
                   markersize=8, label="IND treatment"),
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=_IND_CONTROL_COLOR,
                   markeredgecolor="black", markeredgewidth=0.5,
                   markersize=8, label="IND control"),
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=_PW_COLOR,
                   markeredgecolor="black", markeredgewidth=0.5,
                   markersize=8, label="PW accuracy"),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=8.5,
               frameon=True, ncol=4, bbox_to_anchor=(0.99, 0.005))
    trained_op_display = trained_op.replace("_", "-")
    fig.suptitle(f"{trained_op_display}-trained models · tested across all OPs "
                 f"(trained-std: cross-op experiment for left-out ops, self-same-OP for {trained_op_display})",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.text(0.5, 0.008, "ICA shot count", ha="center", fontsize=11)
    plt.tight_layout(rect=[0.09, 0.05, 1.0, 0.93])

    for gi, kv in enumerate(kv_order):
        ax_left = axes[0, gi * len(col_suffixes)]
        ax_right = axes[0, gi * len(col_suffixes) + len(col_suffixes) - 1]
        p_left = ax_left.get_position()
        p_right = ax_right.get_position()
        x_center = (p_left.x0 + p_right.x1) / 2
        fig.text(x_center, p_left.y1 + 0.035, kv,
                 ha="center", va="bottom",
                 fontsize=12, fontweight="bold")

    _draw_model_group_labels(fig, axes, [s["model"] for s in rows_spec])
    _draw_panel_separators(fig, axes,
                           [s["model"] for s in rows_spec],
                           kv_order, len(col_suffixes))

    subdir = output_dir / "all" / cross_op_exp_name / "all"
    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / "dot_arrows.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  ✓ {path} (+ .png)")


def fig_shot_dot_arrows_all_cross_op(agg: pd.DataFrame, output_dir: Path):
    """Run the cross-op figure for every registered cross-op experiment."""
    for entry in CROSS_OP_EXPERIMENTS:
        trained_op, cross_exp, self_exp = entry[:3]
        test_ops = entry[3] if len(entry) > 3 else None
        fig_shot_dot_arrows_cross_op(agg, output_dir,
                                     trained_op=trained_op,
                                     cross_op_exp_name=cross_exp,
                                     self_op_exp_name=self_exp,
                                     test_ops=test_ops)


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def print_summary(agg: pd.DataFrame, deltas: pd.DataFrame):
    print(f"\n{'=' * 80}\nICA ANALYSIS SUMMARY\n{'=' * 80}")
    print(f"Loaded {len(agg)} aggregated runs across {agg['cell'].nunique()} cell(s).")
    print(f"Conditions: {sorted(agg['condition'].unique())}")

    # Build row label including icl_count
    agg_display = agg.copy()
    agg_display["row_label"] = agg_display.apply(_short_row_label, axis=1)

    print(f"\n--- Metric (F1 for IND, accuracy for PW) per cell × condition ---")
    pivot = agg_display.pivot_table(index="row_label", columns="condition",
                                    values="metric")
    cols = [c for c in CONDITION_ORDER if c in pivot.columns]
    print(pivot[cols].to_string(float_format=lambda v: f"{v:.3f}" if pd.notna(v) else "  -  "))

    # IND detail: show treatment and control separately
    ind_rows = agg_display[agg_display["format"].apply(lambda f: is_ind_format(f))]
    if not ind_rows.empty:
        print(f"\n--- IND detail: treatment_acc / control_acc ---")
        for _, r in ind_rows.sort_values(["cell", "condition", "icl_count"]).iterrows():
            t = f"{r['treatment_acc']:.3f}" if pd.notna(r.get("treatment_acc")) else "  -  "
            c = f"{r['control_acc']:.3f}" if pd.notna(r.get("control_acc")) else "  -  "
            icl = f"icl={int(r['icl_count'])}" if pd.notna(r["icl_count"]) else "no-ica"
            print(f"  {r['condition']:25s} [{icl:>7s}]  T={t}  C={c}  F1={r['metric']:.3f}")

    print(f"\n--- Δ metric from no-ICA baseline ---")
    d_display = deltas.copy()
    d_display["row_label"] = d_display.apply(_short_row_label, axis=1)
    dpiv = d_display.pivot_table(index="row_label", columns="condition", values="delta")
    cols = [c for c in CONDITION_ORDER if c in dpiv.columns and "no-ica" not in c]
    if cols:
        print(dpiv[cols].to_string(float_format=lambda v: f"{v:+.3f}" if pd.notna(v) else "  -  "))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Selectable figure registry — keys are CLI-friendly names; values are
# one-line descriptions surfaced in --help. Each key gates exactly one figure
# call site in main(). Order here is the order figures will be generated.
FIGURE_REGISTRY: list[tuple[str, str]] = [
    ("accuracy_per_cell",   "Per-cell × icl_count grouped bar chart of metric per condition"),
    ("delta_heatmap",       "Heatmap of Δ from no-ICA per (op, icl_count) × condition"),
    ("trained_vs_base_alt", "Scatter of |Δ_trained_alt| vs |Δ_base_alt|"),
    ("dot_arrows",          "Per-base × experiment × kv dot-arrow grids"),
    ("all_ind",             "Consolidated IND-only dot-arrow figure per experiment"),
    ("all_combined",        "Consolidated IND+PW dot-arrow figure per experiment"),
    ("multi_op",            "Multi-OP-only (base + trained-multi-op) variant of all_combined"),
    ("randlabels",          "Random-labels-only (base + trained-randlabels) variant of all_combined"),
    ("cross_op",            "Cross-OP (train on one OP, eval on others) dot-arrow figure"),
    ("per_tag",             "Per-(model, OP) shot-averaged dot-arrows; 3 rows × 4 cols"),
    ("per_tag_bars",        "Per-(model, tag) shot-averaged grouped bar chart"),
    ("per_tag_diff",        "Per-(model, OP) training-induced ICA shift Δ_trained − Δ_base"),
]


def main():
    valid_keys = [k for k, _ in FIGURE_REGISTRY]
    fig_help = "Figures to generate. Special values: 'all' (default), 'none'. " \
               "Otherwise a space-separated subset of: " + ", ".join(valid_keys)

    parser = argparse.ArgumentParser(description="Analyze ICA results")
    parser.add_argument("--experiment_dirs", nargs="+",
                        help="Experiment root dir(s) containing condition subdirs. "
                             "Required unless --from_csv is set.")
    parser.add_argument("--results_root", default="data/results",
                        help="Where inspect_ai eval logs live (default: data/results)")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write the analysis outputs")
    parser.add_argument("--include_adv", action="store_true",
                        help="Include trained-adv columns in figures. "
                             "Off by default — adv data is loaded into the CSVs "
                             "regardless, only the figures are filtered.")
    parser.add_argument("--figures", nargs="+", default=["all"], help=fig_help)
    parser.add_argument("--from_csv", action="store_true",
                        help="Skip the (slow) eval-log load and reuse the "
                             "ica_runs.csv + ica_deltas.csv already in --output_dir. "
                             "Useful when iterating on figure code.")
    args = parser.parse_args()

    # Validate --figures
    if "all" in args.figures and "none" in args.figures:
        parser.error("--figures: cannot pass both 'all' and 'none'")
    if "all" in args.figures:
        selected = set(valid_keys)
    elif "none" in args.figures:
        selected = set()
    else:
        unknown = [f for f in args.figures if f not in valid_keys]
        if unknown:
            parser.error(
                f"--figures: unknown name(s) {unknown}. "
                f"Valid: 'all', 'none', or any of {valid_keys}"
            )
        selected = set(args.figures)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load (or reuse) the analysis tables -------------------------------
    if args.from_csv:
        runs_csv = output_dir / "ica_runs.csv"
        deltas_csv = output_dir / "ica_deltas.csv"
        if not runs_csv.exists() or not deltas_csv.exists():
            parser.error(
                f"--from_csv: missing {runs_csv} or {deltas_csv}. "
                f"Run once without --from_csv to generate them first."
            )
        print(f"Reusing existing CSVs from {output_dir}/ ...")
        agg = pd.read_csv(runs_csv)
        deltas = pd.read_csv(deltas_csv)
        print(f"  agg: {len(agg)} rows, deltas: {len(deltas)} rows")
    else:
        if not args.experiment_dirs:
            parser.error("--experiment_dirs is required (or pass --from_csv)")
        results_root = Path(args.results_root)
        experiment_dirs = [Path(p) for p in args.experiment_dirs]

        print(f"Loading ICA results from {len(experiment_dirs)} experiment dir(s)...")
        df = load_ica_results(experiment_dirs, results_root)
        if df.empty:
            print("No eval logs found. Make sure runs have completed and "
                  "results_root + experiment dirs are correct.")
            return

        df = compute_cell_key(df)
        df.to_csv(output_dir / "ica_runs_raw.csv", index=False)
        print(f"  Raw logs: {len(df)} rows across {df['cell'].nunique()} cell(s)")

        agg = aggregate_accuracy(df)
        deltas = compute_deltas(agg)

        print_summary(agg, deltas)

        agg.to_csv(output_dir / "ica_runs.csv", index=False)
        deltas.to_csv(output_dir / "ica_deltas.csv", index=False)
        print(f"\n  Saved: {output_dir / 'ica_runs.csv'}")
        print(f"  Saved: {output_dir / 'ica_deltas.csv'}")

    # ---- Figure-level adv filter. CSVs above keep all rows. ----------------
    if not args.include_adv:
        print("  (filtering trained-adv from figures — pass --include_adv to keep)")
        agg = agg[~agg["condition"].astype(str).str.startswith("trained-adv_")].copy()
        deltas = deltas[~deltas["condition"].astype(str).str.startswith("trained-adv_")].copy()
        kv_order_full = ("base", "trained-std")
        kv_order_dot = ("trained-std", "base")
    else:
        kv_order_full = ("base", "trained-std", "trained-adv")
        kv_order_dot = ("trained-std", "trained-adv", "base")

    if not selected:
        print("\n--figures none → skipping figure generation.")
        print(f"  Analysis complete. Outputs in {output_dir}/")
        return

    print(f"\nGenerating figures: {sorted(selected)}")
    if "accuracy_per_cell" in selected:
        fig_accuracy_per_cell(agg, output_dir)
    if "delta_heatmap" in selected:
        fig_delta_heatmap(deltas, agg, output_dir)
    if "trained_vs_base_alt" in selected:
        fig_trained_vs_base_alt(deltas, agg, output_dir)
    if "dot_arrows" in selected:
        fig_shot_dot_arrows(agg, output_dir, kv_order=kv_order_dot)
    if "all_ind" in selected:
        fig_shot_dot_arrows_all_ind(agg, output_dir, kv_order=kv_order_full)
    if "all_combined" in selected:
        fig_shot_dot_arrows_all_combined(agg, output_dir, kv_order=kv_order_full)
    if "multi_op" in selected:
        # Same shape as the "all" figure but restricted to base + trained-multi-op
        # columns. Skips experiments without multi-op data.
        fig_shot_dot_arrows_all_combined(
            agg, output_dir,
            kv_order=("base", "trained-multi-op"),
            subdir_name="multi-op",
            title_suffix="multi-op only · IND + PW (ctrl averaged)",
            require_kv_data="trained-multi-op",
        )
    if "randlabels" in selected:
        # Catastrophic-forgetting control: same data distribution and compute
        # as multi-op, but with binary targets shuffled per ID.
        fig_shot_dot_arrows_all_combined(
            agg, output_dir,
            kv_order=("base", "trained-randlabels"),
            subdir_name="randlabels",
            title_suffix="randlabels control · IND + PW (ctrl averaged)",
            require_kv_data="trained-randlabels",
        )
    if "cross_op" in selected:
        fig_shot_dot_arrows_all_cross_op(agg, output_dir)
    if "per_tag" in selected:
        # Default ("all") variant: 3 rows × 4 cols, base + trained-std kvs.
        fig_dot_arrows_per_tag(agg, output_dir, kv_order=kv_order_full)
        # Multi-op variant: same shape but base + trained-multi-op. Skips
        # experiments without multi-op data via `require_kv_data`.
        fig_dot_arrows_per_tag(
            agg, output_dir,
            kv_order=("base", "trained-multi-op"),
            subdir_name="multi-op/per-tag-shot-avg",
            title_suffix="multi-op only · shot-averaged · per-(model, OP) panel",
            require_kv_data="trained-multi-op",
        )
        # Random-labels variant: catastrophic-forgetting control.
        fig_dot_arrows_per_tag(
            agg, output_dir,
            kv_order=("base", "trained-randlabels"),
            subdir_name="randlabels/per-tag-shot-avg",
            title_suffix="randlabels control · shot-averaged · per-(model, OP) panel",
            require_kv_data="trained-randlabels",
        )
    if "per_tag_bars" in selected:
        # Bar-chart variant of the per-tag dot-arrow figure.
        fig_bars_per_tag(agg, output_dir, kv_order=kv_order_full)
        # Multi-op bar-chart variant.
        fig_bars_per_tag(
            agg, output_dir,
            kv_order=("base", "trained-multi-op"),
            subdir_name="multi-op/per-tag-shot-avg",
            title_suffix="multi-op only · shot-averaged · per-(model, OP) bar chart",
            require_kv_data="trained-multi-op",
        )
    if "per_tag_diff" in selected:
        # Training-induced ICA shift: Δ_trained − Δ_base per (series, condition).
        fig_dot_arrows_per_tag_diff(agg, output_dir, kv_order=kv_order_full)
        # Multi-op variant: Δ_multi-op − Δ_base.
        fig_dot_arrows_per_tag_diff(
            agg, output_dir,
            kv_order=("base", "trained-multi-op"),
            subdir_name="multi-op/per-tag-shot-avg",
            require_kv_data="trained-multi-op",
        )
        # Random-labels variant: Δ_randlabels − Δ_base.
        fig_dot_arrows_per_tag_diff(
            agg, output_dir,
            kv_order=("base", "trained-randlabels"),
            subdir_name="randlabels/per-tag-shot-avg",
            require_kv_data="trained-randlabels",
        )

    print(f"\n  Analysis complete. Outputs in {output_dir}/")


if __name__ == "__main__":
    main()
