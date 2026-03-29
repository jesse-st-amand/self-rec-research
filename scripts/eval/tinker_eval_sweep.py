"""Run SGTR evaluations via Tinker's sampling API.

Replaces inspect_ai for models that need Tinker (trained models with
checkpoints, or hf/ models routed through Tinker). Produces .eval-compatible
output files in the same directory structure as the standard sweep.

Usage:
    uv run python scripts/eval/tinker_eval_sweep.py \
        --experiment_config experiments_eval/COLM/.../config.yaml \
        --dataset_dir data/input/wikisum/training_set_1-20

    Or via bash:
    bash experiments_eval/COLM/.../bash/sweep/00_wikisum_training_set_1-20.sh
"""

import argparse
import json
import os
import sys
from datetime import datetime, UTC
from pathlib import Path

import tinker
from dotenv import load_dotenv
from tinker_cookbook import renderers as r
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer

from self_rec_framework.src.inspect.config import load_experiment_config
from self_rec_framework.src.inspect.data import load_dataset_pairwise
from self_rec_framework.src.helpers.model_names import (
    INSPECT_MODEL_NAMES, get_data_model_name, inspect_model_name, temp_suffix,
)
from self_rec_framework.src.helpers.utils import data_dir
from self_rec_framework.scripts.utils import expand_model_names
from scripts.alpaca_eval.generate_outputs import resolve_tinker_checkpoint
from scripts.alpaca_eval.run_self_preference import resolve_base_model


def _resolve_renderer(hf_model_id: str):
    """Get the renderer and tokenizer for a model."""
    tokenizer = get_tokenizer(hf_model_id)
    try:
        renderer_name = get_recommended_renderer_name(hf_model_id)
    except (KeyError, ValueError):
        # Fallback: try common renderer names
        model_lower = hf_model_id.lower()
        if "llama" in model_lower:
            renderer_name = "llama3"
        elif "qwen" in model_lower:
            renderer_name = "qwen3"
        elif "gpt-oss" in model_lower or "oss" in model_lower:
            renderer_name = "gpt_oss_no_sysprompt"
        else:
            raise RuntimeError(f"Cannot determine renderer for {hf_model_id}")
    renderer = r.get_renderer(renderer_name, tokenizer)
    return renderer, tokenizer


def _extract_answer(completion: str) -> str | None:
    """Extract 1 or 2 from model completion."""
    text = r.get_text_content if hasattr(r, 'get_text_content') else None
    if text:
        # Already have text content
        pass

    clean = completion.strip()

    # Direct answer
    if clean in ("1", "2"):
        return clean

    # "Answer: 1" or "Answer: 2"
    import re
    m = re.search(r"(?:answer|choice|response)[\s:]*([12])\b", clean, re.IGNORECASE)
    if m:
        return m.group(1)

    # Last standalone 1 or 2
    matches = re.findall(r"\b([12])\b", clean)
    if matches:
        return matches[-1]

    return None


def run_tinker_eval(
    evaluator_name: str,
    control_name: str,
    treatment_name: str,
    dataset_name: str,
    data_subset: str,
    exp_config,
    output_dir: Path,
    max_tokens: int = 512,
):
    """Run a single pairwise evaluation via Tinker.

    Args:
        evaluator_name: The evaluator model (may be trained or base)
        control_name: Control treatment dir name (evaluator's own data)
        treatment_name: Treatment dir name (other model's data)
        dataset_name: e.g., "wikisum"
        data_subset: e.g., "training_set_1-20"
        exp_config: ExperimentConfig
        output_dir: Where to save the eval log
        max_tokens: Max generation tokens
    """
    # Build output filename
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S+00-00")
    task_name = f"{evaluator_name}-eval-on-{control_name}-vs-{treatment_name}"
    log_filename = f"{timestamp}_{task_name}.json"
    log_path = output_dir / log_filename

    # Check if already exists (look for matching task name)
    existing = list(output_dir.glob(f"*{evaluator_name}-eval-on-{control_name}-vs-{treatment_name}*"))
    if existing:
        print(f"  ⊘ {evaluator_name} vs {treatment_name}: already exists, skipping")
        return

    # Resolve Tinker model
    hf_model_id, sampler_path = resolve_tinker_checkpoint(evaluator_name)

    # Load dataset
    dataset_samples = load_dataset_pairwise(
        control_name, treatment_name, dataset_name, data_subset
    )

    if not dataset_samples:
        print(f"  ⚠ No samples for {control_name} vs {treatment_name}")
        return

    # Set up Tinker client
    client = tinker.ServiceClient()
    if sampler_path:
        sampling_client = client.create_sampling_client(model_path=sampler_path)
        print(f"  Loading trained: {sampler_path}")
    else:
        sampling_client = client.create_sampling_client(base_model=hf_model_id)
        print(f"  Loading base: {hf_model_id}")

    renderer, tokenizer = _resolve_renderer(hf_model_id)
    stop_sequences = renderer.get_stop_sequences()

    # Determine if thinking model needs more tokens
    model_lower = hf_model_id.lower()
    is_thinking = (
        "oss" in model_lower
        or "qwen3.5" in model_lower
        or evaluator_name.endswith("-thinking")
    )
    if is_thinking:
        max_tokens = max(max_tokens, 4096)

    sampling_params = tinker.types.SamplingParams(
        max_tokens=max_tokens,
        temperature=exp_config.temperature or 0,
        stop=stop_sequences,
    )

    # Build prompts and submit
    print(f"  Running {evaluator_name}: {control_name} vs {treatment_name} ({len(dataset_samples)} samples)...")

    futures = []
    for sample in dataset_samples:
        if exp_config.tags == "UT":
            generation_prompt = exp_config.generation_prompt.format(
                content=sample["content"]
            )
            reasoning1 = sample.get("cot1") or ""
            reasoning2 = sample.get("cot2") or ""
            prompt = exp_config.SR_task_prompt.format(
                generation_prompt=generation_prompt,
                output1=sample["output1"],
                output2=sample["output2"],
                reasoning1=reasoning1,
                reasoning2=reasoning2,
            )
        else:
            # AT format
            prompt = exp_config.SR_task_prompt.format(
                output1=sample["output1"],
                output2=sample["output2"],
            )

        convo = [
            {"role": "system", "content": exp_config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        model_input = renderer.build_generation_prompt(convo)
        futures.append(sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=sampling_params,
        ))

    # Collect results
    results = []
    correct = 0
    total = 0

    for i, (future, sample) in enumerate(zip(futures, dataset_samples)):
        if (i + 1) % 10 == 0 or i == len(futures) - 1:
            print(f"    [{i+1}/{len(futures)}]", flush=True)

        result = future.result()
        seq = result.sequences[0]
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        completion = r.get_text_content(parsed_msg)

        answer = _extract_answer(completion)
        target = sample["metadata"]["correct_answer"]
        is_correct = answer == target

        if answer:
            total += 1
            if is_correct:
                correct += 1

        results.append({
            "metadata": sample["metadata"],
            "completion": completion[:500],
            "prediction": answer,
            "target": target,
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0
    parse_failures = len(results) - total

    print(f"  ✓ {evaluator_name} vs {treatment_name}: "
          f"acc={accuracy:.3f} ({correct}/{total}), "
          f"parse_failures={parse_failures}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_result = {
        "timestamp": timestamp,
        "evaluator": evaluator_name,
        "control": control_name,
        "treatment": treatment_name,
        "dataset": dataset_name,
        "data_subset": data_subset,
        "model": hf_model_id,
        "sampler_path": sampler_path,
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "parse_failures": parse_failures,
        "n_samples": len(results),
        "results": results,
    }

    with open(log_path, "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"    Saved to {log_path}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run SGTR evaluations via Tinker")
    parser.add_argument("--experiment_config", required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--dataset_dir", required=True,
                        help="Path to dataset dir (e.g., data/input/wikisum/training_set_1-20)")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max generation tokens (default: 512)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    parts = dataset_path.parts
    if "input" in parts:
        idx = parts.index("input")
        dataset_name = parts[idx + 1]
        data_subset = parts[idx + 2]
    else:
        dataset_name = parts[-2]
        data_subset = parts[-1]

    exp_config = load_experiment_config(args.experiment_config, dataset_name=dataset_name)
    experiment_name = Path(args.experiment_config).resolve().parent.name

    # Read raw config for model lists and training_dir
    import yaml
    with open(args.experiment_config) as f:
        raw_config = yaml.safe_load(f)

    training_dir = raw_config.get("training_dir", "data/training")
    data_subsets = raw_config.get("data_subsets")

    # Set TRAINING_DIR on generate_outputs so resolve_tinker_checkpoint finds runs
    import scripts.alpaca_eval.generate_outputs as _gen_mod
    _gen_mod.TRAINING_DIR = training_dir
    gen_temp = raw_config.get("generator_temperature")
    gen_temp_sfx = temp_suffix(gen_temp)

    # Expand model names
    raw_eval = raw_config.get("model_names", [])
    raw_gen = raw_config.get("generator_models", [])
    evaluators = expand_model_names(raw_eval, training_dir=training_dir, data_subsets=data_subsets)
    generators = expand_model_names(raw_gen)

    # Output directory
    output_dir = data_dir() / "results" / dataset_name / data_subset / experiment_name

    print(f"\n{'='*70}")
    print("TINKER EVAL SWEEP")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_name}/{data_subset}")
    print(f"Experiment: {experiment_name}")
    print(f"Evaluators: {len(evaluators)}")
    for e in evaluators:
        print(f"  {e}")
    print(f"Generators: {len(generators)}")
    for g in generators:
        print(f"  {g}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Build evaluation pairs
    for evaluator in evaluators:
        data_model = get_data_model_name(evaluator)
        control_name = data_model + gen_temp_sfx

        # Check control data exists
        control_path = data_dir() / "input" / dataset_name / data_subset / control_name / "data.json"
        if not control_path.exists():
            # Try without temp suffix
            if gen_temp_sfx and (data_dir() / "input" / dataset_name / data_subset / data_model / "data.json").exists():
                control_name = data_model
            else:
                print(f"  ⚠ No control data for {evaluator} (looked for {control_name}), skipping")
                continue

        base_model = resolve_base_model(evaluator)

        for generator in generators:
            gen_data_model = get_data_model_name(generator)
            treatment_name = gen_data_model + gen_temp_sfx

            # Check treatment data exists
            treatment_path = data_dir() / "input" / dataset_name / data_subset / treatment_name / "data.json"
            if not treatment_path.exists():
                if gen_temp_sfx and (data_dir() / "input" / dataset_name / data_subset / gen_data_model / "data.json").exists():
                    treatment_name = gen_data_model
                else:
                    continue

            # Skip if control == treatment (same model data)
            if control_name == treatment_name:
                continue

            try:
                run_tinker_eval(
                    evaluator_name=evaluator,
                    control_name=control_name,
                    treatment_name=treatment_name,
                    dataset_name=dataset_name,
                    data_subset=data_subset,
                    exp_config=exp_config,
                    output_dir=output_dir,
                    max_tokens=args.max_tokens,
                )
            except Exception as e:
                print(f"  ⚠ {evaluator} vs {treatment_name}: {e}")

    print(f"\n✓ Tinker eval sweep complete. Results in {output_dir}/")


if __name__ == "__main__":
    main()
