"""Run pairwise self-preference evaluation.

For each judge model, compare its own outputs against every other model's outputs.
Supports both API-based judges (via alpaca_eval PairwiseAnnotator) and local
HuggingFace models (hf/ prefix, dispatched to RunPod GPU).

Usage:
    uv run python scripts/alpaca_eval/run_self_preference.py \
        --judges ll-3.1-8b \
        --outputs_dir data/alpaca_eval/outputs \
        --output_dir data/alpaca_eval/results
"""

import argparse
import ast
import json
import os
import random
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from self_rec_framework.src.helpers.model_names import INSPECT_MODEL_NAMES
from self_rec_framework.scripts.utils import expand_model_names


def resolve_base_model(short_name: str) -> str:
    """Resolve the base model for a potentially trained model name.

    'gpt-oss-20b_sft-as_gpt-oss-20b_vs_...' -> 'gpt-oss-20b'
    'll-3.1-8b' -> 'll-3.1-8b'
    """
    # Strip -thinking suffix for resolution, but preserve it in the result
    has_thinking = short_name.endswith("-thinking")
    clean = short_name.removesuffix("-thinking") if has_thinking else short_name

    if clean in INSPECT_MODEL_NAMES:
        return short_name if has_thinking else clean

    # Reorganized naming: {base}_sft-as_{identity}_vs_{opponent}_{tag}_{fmt}_{dataset}
    if "_sft-as_" in clean:
        base_part = clean.split("_sft-as_")[0]
        try:
            from scripts.alpaca_eval.training_runs import REORG_MODEL_MAP
            resolved = REORG_MODEL_MAP.get(base_part, base_part)
        except ImportError:
            resolved = base_part
        return resolved + "-thinking" if has_thinking else resolved

    # Legacy: try to match {base_model}-{training_run} pattern
    for base_name in sorted(INSPECT_MODEL_NAMES.keys(), key=len, reverse=True):
        if clean.startswith(base_name + "-"):
            return base_name + "-thinking" if has_thinking else base_name
    return short_name


def discover_trained_models(base_models: list[str],
                            training_dir: str = "data/training",
                            subsets: list[str] | None = None) -> list[str]:
    """Discover all trained model names for a set of base models.

    Uses the unified training_runs module to handle both original and
    archived naming conventions.

    Returns the list of trained model names (not including base models).
    """
    from scripts.alpaca_eval.training_runs import discover_training_runs

    base_set = set(base_models)
    runs = discover_training_runs(training_dir, subsets=subsets)
    trained = []
    for run in runs:
        if run.base_model in base_set:
            trained.append(run.trained_name)
    return trained


def expand_evaluators_with_trained(evaluator_models: list[str],
                                   training_dir: str = "data/training",
                                   subsets: list[str] | None = None) -> list[str]:
    """Expand a list of evaluator models by adding all trained versions.

    Given ['ll-3.1-8b', 'qwen-3.0-30b'], returns:
    ['ll-3.1-8b', 'qwen-3.0-30b',
     'll-3.1-8b-01_sft_pw_vs_qwen', 'll-3.1-8b-11_archived_ll8b_ut_pw_sharegpt_vs_qwen25', ...]
    """
    trained = discover_trained_models(evaluator_models, training_dir, subsets=subsets)
    # Combine: base models first, then trained versions, deduplicated
    seen = set()
    result = []
    for m in evaluator_models + trained:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


# ---------------------------------------------------------------------------
# Prompt template (same as alpaca_eval_gpt4/alpaca_eval.txt)
# ---------------------------------------------------------------------------

# Standard alpaca_eval prompt (works well with GPT-4 class models)
JUDGE_SYSTEM_PROMPT = (
    "You are a helpful assistant, that ranks models by the quality of their answers."
)

JUDGE_USER_TEMPLATE = """\
I want you to create a leaderboard of different of large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

Here is the prompt:
{{
    "instruction": \"\"\"{instruction}\"\"\",
}}

Here are the outputs of the models:
[
    {{
        "model": "model_1",
        "answer": \"\"\"{output_1}\"\"\"
    }},
    {{
        "model": "model_2",
        "answer": \"\"\"{output_2}\"\"\"
    }}
]

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[
    {{'model': <model-name>, 'rank': <model-rank>}},
    {{'model': <model-name>, 'rank': <model-rank>}}
]

Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give."""

# Simple prompt for smaller models that struggle with the structured format
JUDGE_SIMPLE_SYSTEM = "You are a helpful and precise assistant for checking the quality of answers."

JUDGE_SIMPLE_USER = """\
[Question]
{instruction}

[Response A]
{output_1}

[Response B]
{output_2}

[Task]
Which response better answers the question? Reply with ONLY "A" or "B" (nothing else)."""


# ---------------------------------------------------------------------------
# Ranking mode prompts (multi-model, all outputs at once)
# ---------------------------------------------------------------------------

RANKING_SYSTEM_PROMPT = (
    "You are a helpful assistant that ranks model outputs by quality."
)

RANKING_USER_TEMPLATE = """\
I want you to rank the following model outputs by quality. I will give you an instruction and the responses from {n_models} models. Please rank them from best (rank 1) to worst (rank {n_models}).

Here is the instruction:
{{
    "instruction": \"\"\"{instruction}\"\"\"
}}

Here are the model outputs:
{model_outputs_block}

Now rank all models by the quality of their answers, so that rank 1 is the best. Return a Python list of dicts:
[
    {{'model': '<model-name>', 'rank': <rank>}},
    ...
]

Your response must be a valid Python list and should contain nothing else because we will directly execute it in Python. Provide the ranking that the majority of humans would give."""

RANKING_COT_USER_TEMPLATE = """\
I want you to rank the following model outputs by quality. I will give you an instruction and the responses from {n_models} models. Please rank them from best (rank 1) to worst (rank {n_models}).

Here is the instruction:
{{
    "instruction": \"\"\"{instruction}\"\"\"
}}

Here are the model outputs:
{model_outputs_block}

First, briefly explain your reasoning for the ranking. Then, on a new line, provide your final ranking as a Python list of dicts:
[
    {{'model': '<model-name>', 'rank': <rank>}},
    ...
]

The final line of your response must be the valid Python list and nothing else."""


def is_local_model(short_name: str) -> bool:
    """Check if a model requires local GPU inference."""
    inspect_name = INSPECT_MODEL_NAMES.get(short_name, "")
    return inspect_name.startswith("hf/")


def parse_provider(short_name: str) -> tuple[str, str, dict]:
    """Map shorthand model name to (fn_completions, api_model_name, extra_kwargs)."""
    inspect_name = INSPECT_MODEL_NAMES[short_name]
    if inspect_name.startswith("together/"):
        api_model = inspect_name.removeprefix("together/")
        return "openai_completions", api_model, {
            "openai_api_base": "https://api.together.xyz/v1",
        }
    elif inspect_name.startswith("anthropic/"):
        api_model = inspect_name.removeprefix("anthropic/")
        return "anthropic_completions", api_model, {}
    elif inspect_name.startswith("google/"):
        api_model = inspect_name.removeprefix("google/")
        return "google_completions", api_model, {}
    elif inspect_name.startswith("openai/"):
        api_model = inspect_name.removeprefix("openai/")
        return "openai_completions", api_model, {}
    else:
        raise ValueError(f"Unknown provider for {short_name}: {inspect_name}")


def _strip_thinking(text: str) -> str:
    """Strip thinking/reasoning content from model output.

    Handles:
    - gpt-oss: <|channel|>analysis<|message|>...<|channel|>response<|message|>
    - Qwen3.5: <think>...</think>
    - DeepSeek: <think>...</think>
    """
    import re
    # gpt-oss channel tokens: extract content after last <|channel|>response<|message|>
    if "<|channel|>" in text:
        parts = text.split("<|channel|>response<|message|>")
        if len(parts) > 1:
            return parts[-1].strip()
        # If no response channel, strip all channel tokens and take last segment
        parts = text.split("<|message|>")
        if len(parts) > 1:
            return parts[-1].strip()

    # <think>...</think> blocks (Qwen3.5, DeepSeek)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned:
        return cleaned

    return text


def parse_ranking(completion: str) -> float:
    """Parse model ranking from completion text.

    Handles multiple formats:
    - Python list of dicts: [{'model': 'model_1', 'rank': 1}, ...]
    - Simple "A" or "B" response (maps A->1, B->2)
    - "1" or "2" response
    - "model_1" or "model_2" mentioned first

    For thinking models, strips reasoning content first and looks at the
    final answer. Also checks the last line as a fallback.

    Returns 1.0 if model_1/A preferred, 2.0 if model_2/B preferred, nan on failure.
    """
    import re
    import numpy as np

    # Strip thinking content first
    text = _strip_thinking(completion).strip()

    # Try structured Python dict format (alpaca_eval standard)
    try:
        parsed = ast.literal_eval(text)
        rank = [c for c in parsed if c["model"] == "model_1"][0]["rank"]
        if rank in (1, 2):
            return float(rank)
    except Exception:
        pass

    # Try simple A/B format (exact match)
    clean = text.strip().upper()
    if clean in ("A", "RESPONSE A", "A.", "A)"):
        return 1.0
    if clean in ("B", "RESPONSE B", "B.", "B)"):
        return 2.0

    # Try leading A or B in longer responses
    m = re.match(r"^(?:response\s+)?([AB])\b", text.strip(), re.IGNORECASE)
    if m:
        return 1.0 if m.group(1).upper() == "A" else 2.0

    # Try "1" or "2"
    if clean in ("1", "1."):
        return 1.0
    if clean in ("2", "2."):
        return 2.0

    # Fallback: check the LAST line of the response (thinking models often
    # reason first and give the answer at the end)
    last_line = text.strip().split("\n")[-1].strip().upper()
    if last_line in ("A", "RESPONSE A", "A.", "A)", "**A**", "**RESPONSE A**"):
        return 1.0
    if last_line in ("B", "RESPONSE B", "B.", "B)", "**B**", "**RESPONSE B**"):
        return 2.0
    m = re.match(r"^(?:\*\*)?(?:response\s+)?([AB])(?:\*\*)?\s*\.?$", last_line, re.IGNORECASE)
    if m:
        return 1.0 if m.group(1).upper() == "A" else 2.0

    # Last resort: find the last standalone A or B in the text
    matches = re.findall(r"\b(?:response\s+)?([AB])\b", text, re.IGNORECASE)
    if matches:
        last = matches[-1].upper()
        return 1.0 if last == "A" else 2.0

    return float(np.nan)


def parse_multi_ranking(completion: str, model_labels: list[str]) -> dict[str, int] | None:
    """Parse a multi-model ranking from completion text.

    Expected format: [{'model': 'model_1', 'rank': 1}, {'model': 'model_2', 'rank': 2}, ...]

    Returns dict mapping model_label -> rank, or None on parse failure.
    """
    import re

    text = _strip_thinking(completion).strip()

    # Try to find a Python list in the text (may be preceded by reasoning)
    # Look for the last [...] block
    list_matches = re.findall(r'\[.*?\]', text, re.DOTALL)
    for candidate in reversed(list_matches):
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, list) and len(parsed) >= 2:
                # Validate structure
                ranking = {}
                for item in parsed:
                    if isinstance(item, dict) and 'model' in item and 'rank' in item:
                        ranking[item['model']] = int(item['rank'])
                if len(ranking) >= 2:
                    return ranking
        except Exception:
            continue

    # Try the whole text as a literal
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            ranking = {}
            for item in parsed:
                if isinstance(item, dict) and 'model' in item and 'rank' in item:
                    ranking[item['model']] = int(item['rank'])
            if len(ranking) >= 2:
                return ranking
    except Exception:
        pass

    return None


def build_ranking_prompt(
    instruction: str,
    model_outputs: dict[str, str],
    cot: bool = False,
) -> str:
    """Build the ranking prompt with all model outputs.

    Args:
        instruction: The original instruction/question.
        model_outputs: Dict mapping model label (e.g., 'model_1') to output text.
        cot: Whether to use the CoT variant.

    Returns the formatted user message.
    """
    entries = []
    for label, output in model_outputs.items():
        entries.append(
            f'    {{\n'
            f'        "model": "{label}",\n'
            f'        "answer": \"\"\"{output}\"\"\"\n'
            f'    }}'
        )
    model_outputs_block = "[\n" + ",\n".join(entries) + "\n]"

    template = RANKING_COT_USER_TEMPLATE if cot else RANKING_USER_TEMPLATE
    return template.format(
        instruction=instruction,
        model_outputs_block=model_outputs_block,
        n_models=len(model_outputs),
    )


# ---------------------------------------------------------------------------
# Local HF judging
# ---------------------------------------------------------------------------

def judge_local_hf(
    judge_outputs: list[dict],
    opponent_outputs: list[dict],
    hf_model_id: str,
) -> pd.DataFrame:
    """Run pairwise judging using a local HuggingFace model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading judge model {hf_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    print(f"  Judge loaded on {device} ({dtype})")

    results = []
    for i, (j_out, o_out) in enumerate(zip(judge_outputs, opponent_outputs)):
        # Randomize order to reduce position bias
        flip = random.random() < 0.5
        if flip:
            out_1, out_2 = o_out["output"], j_out["output"]
        else:
            out_1, out_2 = j_out["output"], o_out["output"]

        user_msg = JUDGE_SIMPLE_USER.format(
            instruction=j_out["instruction"],
            output_1=out_1,
            output_2=out_2,
        )

        messages = [
            {"role": "system", "content": JUDGE_SIMPLE_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt_text = f"SYSTEM: {JUDGE_SIMPLE_SYSTEM}\n\nUSER: {user_msg}\n\nASSISTANT:"

        encoded = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=4096,
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_width = encoded["input_ids"].shape[1]
        completion = tokenizer.decode(generated[0][input_width:], skip_special_tokens=True)

        raw_pref = parse_ranking(completion)
        # If we flipped the order, invert the preference
        if flip and raw_pref in (1.0, 2.0):
            pref = 3.0 - raw_pref  # 1->2, 2->1
        else:
            pref = raw_pref

        print(f"  [{i+1}/{len(judge_outputs)}] pref={pref} (flipped={flip})")

        results.append({
            "instruction": j_out["instruction"],
            "output_1": j_out["output"],
            "generator_1": j_out["generator"],
            "dataset": j_out.get("dataset", "helpful_base"),
            "output_2": o_out["output"],
            "generator_2": o_out["generator"],
            "preference": pref,
            "preference_raw_completion": completion,
            "order_flipped": flip,
        })

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Tinker-based judging
# ---------------------------------------------------------------------------

def judge_tinker(
    self_outputs: list[dict],
    opponent_outputs: list[dict],
    hf_model_id: str,
    sampler_path: str | None = None,
) -> pd.DataFrame:
    """Run pairwise judging via Tinker's sampling API."""
    import tinker
    from tinker_cookbook import renderers as r
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    print(f"  Connecting to Tinker for judging...")
    client = tinker.ServiceClient()

    if sampler_path:
        print(f"  Loading trained judge from: {sampler_path}")
        sampling_client = client.create_sampling_client(model_path=sampler_path)
    else:
        print(f"  Loading base judge: {hf_model_id}")
        sampling_client = client.create_sampling_client(base_model=hf_model_id)

    tokenizer = get_tokenizer(hf_model_id)
    try:
        renderer_name = get_recommended_renderer_name(hf_model_id)
    except (KeyError, ValueError):
        raise RuntimeError(
            f"Model '{hf_model_id}' not recognized by tinker_cookbook. "
            f"Check the HuggingFace model ID or update tinker_cookbook."
        )
    renderer = r.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    # Thinking/reasoning models need more tokens for their reasoning trace + answer
    model_lower = hf_model_id.lower()
    is_thinking = (
        "oss" in model_lower  # gpt-oss models
        or "thinking" in renderer_name.lower()
        or "qwen3.5" in model_lower  # Qwen 3.5 always thinks
        or "qwen-3.5" in model_lower
    )
    judge_max_tokens = 4096 if is_thinking else 100

    sampling_params = tinker.types.SamplingParams(
        max_tokens=judge_max_tokens,
        temperature=0,
        stop=stop_sequences,
    )

    # Fire all requests asynchronously
    print(f"  Submitting {len(self_outputs)} judge requests...")
    futures = []
    flip_flags = []
    for s_out, o_out in zip(self_outputs, opponent_outputs):
        flip = random.random() < 0.5
        flip_flags.append(flip)

        if flip:
            out_1, out_2 = o_out["output"], s_out["output"]
        else:
            out_1, out_2 = s_out["output"], o_out["output"]

        user_msg = JUDGE_SIMPLE_USER.format(
            instruction=s_out["instruction"],
            output_1=out_1,
            output_2=out_2,
        )

        convo = [
            {"role": "system", "content": JUDGE_SIMPLE_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        model_input = renderer.build_generation_prompt(convo)
        futures.append(sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=sampling_params,
        ))

    # Collect results
    results = []
    for i, (future, flip) in enumerate(zip(futures, flip_flags)):
        if (i + 1) % 50 == 0 or i == len(futures) - 1:
            print(f"  [{i+1}/{len(futures)}] Collecting judge results...", flush=True)

        result = future.result()
        seq = result.sequences[0]
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        completion = r.get_text_content(parsed_msg)

        raw_pref = parse_ranking(completion)
        if flip and raw_pref in (1.0, 2.0):
            pref = 3.0 - raw_pref
        else:
            pref = raw_pref

        s_out = self_outputs[i]
        o_out = opponent_outputs[i]
        results.append({
            "instruction": s_out["instruction"],
            "output_1": s_out["output"],
            "generator_1": s_out["generator"],
            "dataset": s_out.get("dataset", "helpful_base"),
            "output_2": o_out["output"],
            "generator_2": o_out["generator"],
            "preference": pref,
            "preference_raw_completion": completion,
            "order_flipped": flip,
        })

    print(f"  ✓ Completed {len(results)} judgments via Tinker")
    return pd.DataFrame(results)


def judge_tinker_ranking(
    all_model_outputs: dict[str, list[dict]],
    evaluator_label: str,
    hf_model_id: str,
    sampler_path: str | None = None,
    cot: bool = False,
) -> pd.DataFrame:
    """Run multi-model ranking evaluation via Tinker.

    Presents ALL model outputs at once for each instruction.
    Returns DataFrame with one row per instruction, containing full ranking.
    """
    import tinker
    from tinker_cookbook import renderers as r
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    print(f"  Connecting to Tinker for ranking evaluation...")
    client = tinker.ServiceClient()

    if sampler_path:
        print(f"  Loading trained judge from: {sampler_path}")
        sampling_client = client.create_sampling_client(model_path=sampler_path)
    else:
        print(f"  Loading base judge: {hf_model_id}")
        sampling_client = client.create_sampling_client(base_model=hf_model_id)

    tokenizer = get_tokenizer(hf_model_id)
    try:
        renderer_name = get_recommended_renderer_name(hf_model_id)
    except (KeyError, ValueError):
        raise RuntimeError(
            f"Model '{hf_model_id}' not recognized by tinker_cookbook."
        )
    renderer = r.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    # More tokens for ranking (more models to list + optional CoT)
    model_lower = hf_model_id.lower()
    is_thinking = (
        "oss" in model_lower
        or "thinking" in renderer_name.lower()
        or "qwen3.5" in model_lower
        or "qwen-3.5" in model_lower
    )
    judge_max_tokens = 8192 if is_thinking else (2048 if cot else 512)

    sampling_params = tinker.types.SamplingParams(
        max_tokens=judge_max_tokens,
        temperature=0,
        stop=stop_sequences,
    )

    # Get model labels and number of instructions
    model_labels = sorted(all_model_outputs.keys())
    n_instructions = len(next(iter(all_model_outputs.values())))

    # Randomize model presentation order per instruction to reduce position bias
    print(f"  Submitting {n_instructions} ranking requests ({len(model_labels)} models each)...")
    futures = []
    shuffle_orders = []

    for i in range(n_instructions):
        instruction = all_model_outputs[model_labels[0]][i]["instruction"]

        # Shuffle model order for this instruction
        shuffled = model_labels.copy()
        random.shuffle(shuffled)
        shuffle_orders.append(shuffled)

        # Build output dict with anonymized labels
        model_outputs_anon = {}
        for idx, label in enumerate(shuffled, 1):
            model_outputs_anon[f"model_{idx}"] = all_model_outputs[label][i]["output"]

        user_msg = build_ranking_prompt(instruction, model_outputs_anon, cot=cot)
        convo = [
            {"role": "system", "content": RANKING_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        model_input = renderer.build_generation_prompt(convo)
        futures.append(sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=sampling_params,
        ))

    # Collect results
    results = []
    parse_failures = 0
    for i, (future, shuffled) in enumerate(zip(futures, shuffle_orders)):
        if (i + 1) % 50 == 0 or i == len(futures) - 1:
            print(f"  [{i+1}/{len(futures)}] Collecting ranking results...", flush=True)

        result = future.result()
        seq = result.sequences[0]
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        completion = r.get_text_content(parsed_msg)

        # Parse ranking and map back to original labels
        anon_labels = [f"model_{idx}" for idx in range(1, len(shuffled) + 1)]
        ranking = parse_multi_ranking(completion, anon_labels)

        row = {
            "instruction": all_model_outputs[model_labels[0]][i]["instruction"],
            "raw_completion": completion,
        }

        if ranking:
            # Map anonymous labels back to real model names
            for idx, real_label in enumerate(shuffled, 1):
                anon_key = f"model_{idx}"
                rank = ranking.get(anon_key)
                row[f"rank_{real_label}"] = rank
        else:
            parse_failures += 1
            for label in model_labels:
                row[f"rank_{label}"] = None

        results.append(row)

    if parse_failures:
        print(f"  ⚠ {parse_failures}/{len(futures)} ranking parse failures")
    print(f"  ✓ Completed {len(results)} ranking evaluations via Tinker")
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# API-based judging (via alpaca_eval PairwiseAnnotator)
# ---------------------------------------------------------------------------

def build_annotator_config(judge_short_name: str, config_dir: Path) -> str:
    """Build an alpaca_eval annotator config YAML for the given judge model."""
    fn_completions, api_model, extra_kwargs = parse_provider(judge_short_name)

    completions_kwargs = {
        "model_name": api_model,
        "max_tokens": 100,
        "temperature": 0,
    }
    if "openai_api_base" in extra_kwargs:
        completions_kwargs["openai_api_base"] = extra_kwargs["openai_api_base"]

    import alpaca_eval.constants as ae_constants
    prompt_template = str(
        ae_constants.EVALUATORS_CONFIG_DIR
        / "alpaca_eval_gpt4"
        / "alpaca_eval.txt"
    )

    config = {
        f"self_preference_{judge_short_name}": {
            "prompt_template": prompt_template,
            "fn_completions": fn_completions,
            "completions_kwargs": completions_kwargs,
            "fn_completion_parser": "ranking_parser",
            "batch_size": 1,
        }
    }

    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = (config_dir / f"{judge_short_name}_config.yaml").resolve()
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return str(config_path)


def judge_api(
    judge_name: str,
    judge_outputs: list[dict],
    opponent_outputs: list[dict],
    config_dir: Path,
) -> pd.DataFrame:
    """Run pairwise judging using alpaca_eval's PairwiseAnnotator (API models)."""
    config_path = build_annotator_config(judge_name, config_dir)

    # Set Together API key if needed
    inspect_name = INSPECT_MODEL_NAMES[judge_name]
    if inspect_name.startswith("together/"):
        together_key = os.environ.get("TOGETHER_API_KEY", "")
        os.environ["OPENAI_API_KEY"] = together_key

    from alpaca_eval.annotators.pairwise_evaluator import PairwiseAnnotator

    annotator = PairwiseAnnotator(annotators_config=config_path)
    annotations = annotator.annotate_head2head(
        outputs_1=judge_outputs,
        outputs_2=opponent_outputs,
    )

    if isinstance(annotations, list):
        annotations = pd.DataFrame(annotations)

    return annotations


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_ranking_evaluation(
    judge_name: str,
    generator_models: list[str],
    outputs_dir: Path,
    output_dir: Path,
    gpu_dispatch: str = "runpod",
    cot: bool = False,
):
    """Run multi-model ranking evaluation: judge ranks ALL generator outputs + its own.

    For trained models, the evaluator's "self" outputs come from the base model.
    All generator outputs are presented at once alongside the evaluator's base outputs.
    """
    base_model = resolve_base_model(judge_name)
    result_path = output_dir / judge_name / "ranking.json"
    if result_path.exists():
        print(f"  ⊘ {judge_name} ranking: already exists, skipping")
        return

    # Collect all model outputs (base model + generators)
    # The evaluator's own outputs come from the base model
    all_models = []
    if base_model not in generator_models:
        all_models.append(base_model)
    all_models.extend(generator_models)
    # Deduplicate while preserving order
    seen = set()
    unique_models = []
    for m in all_models:
        if m not in seen:
            seen.add(m)
            unique_models.append(m)

    # Load outputs
    all_model_outputs = {}
    for model in unique_models:
        path = outputs_dir / f"{model}.json"
        if not path.exists():
            print(f"  ⚠ Missing outputs for {model}, skipping ranking eval for {judge_name}")
            return
        with open(path) as f:
            all_model_outputs[model] = json.load(f)

    is_trained = judge_name != base_model
    label = f"{judge_name} (trained)" if is_trained else judge_name
    n_inst = len(next(iter(all_model_outputs.values())))
    print(f"  Running {label} ranking: {len(unique_models)} models × {n_inst} instructions...")

    # Route to backend
    if gpu_dispatch == "tinker" and (is_trained or is_local_model(judge_name)):
        from scripts.alpaca_eval.generate_outputs import resolve_tinker_checkpoint
        hf_model, sampler_path = resolve_tinker_checkpoint(judge_name)
        rankings = judge_tinker_ranking(
            all_model_outputs, judge_name, hf_model, sampler_path, cot=cot,
        )
    else:
        # For API models, we'd need to implement an API-based ranking judge
        # For now, fall back to Tinker if available, or error
        raise NotImplementedError(
            f"Ranking mode not yet implemented for API-only models ({judge_name}). "
            f"Use gpu_dispatch: tinker or judge_mode: simple."
        )

    # Save results
    result_path.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_json(result_path, orient="records", indent=2)

    # Compute self-preference: average rank of the evaluator's own model
    rank_col = f"rank_{base_model}"
    if rank_col in rankings.columns:
        valid = rankings[rank_col].dropna()
        avg_self_rank = valid.mean()
        n_first = (valid == 1).sum()
        print(f"  ✓ {judge_name}: avg self-rank={avg_self_rank:.2f}, "
              f"ranked #1 in {n_first}/{len(valid)} cases ({n_first/len(valid)*100:.1f}%)")
    print(f"    Saved to {result_path}")


def run_pairwise_evaluation(
    judge_name: str,
    opponent_name: str,
    outputs_dir: Path,
    output_dir: Path,
    config_dir: Path,
    gpu_dispatch: str = "runpod",
):
    """Run pairwise evaluation: judge compares base model outputs vs opponent outputs.

    For trained models (e.g., ll-3.1-8b-01_sft_pw_vs_qwen), the "self" outputs
    come from the base model (ll-3.1-8b), not the trained model.
    """
    base_model = resolve_base_model(judge_name)
    result_path = output_dir / judge_name / f"vs_{opponent_name}.json"
    if result_path.exists():
        print(f"  ⊘ {judge_name} vs {opponent_name}: already exists, skipping")
        return

    # "Self" outputs come from the base model
    self_outputs_path = outputs_dir / f"{base_model}.json"
    opponent_outputs_path = outputs_dir / f"{opponent_name}.json"

    if not self_outputs_path.exists():
        print(f"  ⚠ Missing outputs for base model {base_model}, skipping")
        return
    if not opponent_outputs_path.exists():
        print(f"  ⚠ Missing outputs for opponent {opponent_name}, skipping")
        return

    with open(self_outputs_path) as f:
        self_outputs = json.load(f)
    with open(opponent_outputs_path) as f:
        opponent_outputs = json.load(f)

    is_trained = judge_name != base_model
    label = f"{judge_name} (trained)" if is_trained else judge_name
    print(f"  Running {label} judging: {base_model} vs {opponent_name} ({len(self_outputs)} pairs)...")

    # Route to appropriate judge backend
    if gpu_dispatch == "tinker" and (is_trained or is_local_model(judge_name)):
        from scripts.alpaca_eval.generate_outputs import resolve_tinker_checkpoint
        hf_model, sampler_path = resolve_tinker_checkpoint(judge_name)
        annotations = judge_tinker(self_outputs, opponent_outputs, hf_model, sampler_path)
    elif is_local_model(judge_name):
        hf_model_id = INSPECT_MODEL_NAMES[judge_name].removeprefix("hf/")
        annotations = judge_local_hf(self_outputs, opponent_outputs, hf_model_id)
    else:
        annotations = judge_api(judge_name, self_outputs, opponent_outputs, config_dir)

    # Save results
    result_path.parent.mkdir(parents=True, exist_ok=True)
    annotations.to_json(result_path, orient="records", indent=2)

    # Compute win rate (drop NaN preferences from parsing failures)
    prefs = annotations["preference"].dropna().values
    judge_wins = (prefs == 1).sum()
    opponent_wins = (prefs == 2).sum()
    ties = (prefs == 1.5).sum()
    total = len(prefs)
    win_rate = judge_wins / total if total > 0 else 0

    print(f"  ✓ {judge_name} vs {opponent_name}: "
          f"win={judge_wins}, loss={opponent_wins}, tie={ties}, "
          f"win_rate={win_rate:.3f}")
    print(f"    Saved to {result_path}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run self-preference pairwise evaluation")
    parser.add_argument("--judges", nargs="+", default=None,
                        help="Judge model names (each judges its own outputs vs opponents)")
    parser.add_argument("--opponents", nargs="+", default=None,
                        help="Opponent models (default: same as judges)")
    parser.add_argument("--config", default=None,
                        help="Path to experiment config YAML (reads model_names for both judges and opponents)")
    parser.add_argument("--outputs_dir", default="data/alpaca_eval/outputs",
                        help="Directory with generated model outputs")
    parser.add_argument("--output_dir", default="data/alpaca_eval/results",
                        help="Directory to save evaluation results")
    parser.add_argument("--runtime", default=None,
                        help="Path to RunPod runtime YAML (for hf/ judges that need GPU)")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Number of API judges to run in parallel (default: 1)")
    parser.add_argument("--run_mode", default="both", choices=["both", "api", "runpod"],
                        help="Which judges to run: 'both' (default), 'api' only, or 'runpod' only")
    parser.add_argument("--local", action="store_true",
                        help="Run hf/ models locally instead of dispatching to RunPod (use when already on a GPU pod)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print RunPod payload without launching")
    args = parser.parse_args()

    # Defaults (may be overridden by config)
    judge_mode = "simple"
    judge_cot = False
    gpu_dispatch = "runpod"
    data_subsets = None
    config = {}

    # Resolve model names from --config or --judges/--opponents
    if args.config:
        import yaml as _yaml
        with open(args.config) as f:
            config = _yaml.safe_load(f)
        judge_mode = config.get("judge_mode", judge_mode)
        judge_cot = config.get("judge_cot", judge_cot)
        gpu_dispatch = config.get("gpu_dispatch", gpu_dispatch)
        # evaluator_models = judges (auto-expand with trained versions)
        raw_eval = config.get("evaluator_models", config.get("model_names", []))
        raw_gen = config.get("generator_models", config.get("model_names", []))
        data_subsets = config.get("data_subsets", None)
        training_dir = config.get("training_dir", "data/training")
        # Set module-level training dir so resolve_tinker_checkpoint finds the right runs
        import scripts.alpaca_eval.generate_outputs as _gen_mod
        _gen_mod.TRAINING_DIR = training_dir
        judges = expand_model_names(raw_eval, training_dir=training_dir, data_subsets=data_subsets)
        opponents = expand_model_names(args.opponents) if args.opponents else expand_model_names(raw_gen)
        if args.max_workers == 1:
            args.max_workers = config.get("max_workers", args.max_workers)
        if args.run_mode == "both":
            args.run_mode = config.get("run_mode", args.run_mode)
        # Resolve gpu_dispatch runtime: look for {dispatch}.yaml next to config
        if not args.runtime and config.get("gpu_dispatch"):
            dispatch_name = config["gpu_dispatch"]
            config_dir = Path(args.config).resolve().parent
            runtime_path = config_dir / f"{dispatch_name}.yaml"
            if runtime_path.exists():
                args.runtime = str(runtime_path)
                print(f"GPU dispatch: {dispatch_name} (from {runtime_path})")
    elif args.judges:
        judges = expand_model_names(args.judges)
        opponents = expand_model_names(args.opponents) if args.opponents else judges
    else:
        parser.error("Provide either --config or --judges")

    outputs_dir = Path(args.outputs_dir)
    config_dir = Path("data/alpaca_eval/configs")

    # Normalize judge_mode to list and use first mode for eval
    # (eval runs are mode-specific; use the first mode in the list)
    judge_modes = judge_mode if isinstance(judge_mode, list) else [judge_mode]
    active_mode = judge_modes[0]
    if len(judge_modes) > 1:
        print(f"Note: config has multiple judge_modes {judge_modes}. Running eval for: {active_mode}")
    judge_mode = active_mode
    judge_cot = judge_cot if not isinstance(judge_cot, list) else judge_cot[0]

    # Include data subset in output path if specified
    _output_base = Path(args.output_dir)
    if data_subsets:
        active_subset = data_subsets[0] if isinstance(data_subsets, list) else data_subsets
        if len(data_subsets) > 1:
            print(f"Note: config has multiple data_subsets {data_subsets}. Saving results to: {active_subset}")
        _output_base = _output_base / active_subset
    output_dir = _output_base / judge_mode

    print(f"Evaluator models (judges): {', '.join(judges)}")
    print(f"Generator models (opponents): {', '.join(opponents)}")
    print(f"Outputs dir: {outputs_dir}")
    print(f"Results dir: {output_dir}")
    print(f"Setup: Each judge compares its own outputs vs each generator's outputs.")
    print(f"       Skipping when judge == generator (all three models identical).")

    # Split judges into API/Tinker (inline) and RunPod (dispatch to GPU pod)
    # With tinker dispatch, all hf/ and trained models run inline via Tinker API
    api_judges = []
    runpod_judges = []
    for j in judges:
        is_trained = j not in INSPECT_MODEL_NAMES
        if gpu_dispatch == "tinker" and (is_trained or is_local_model(j)):
            api_judges.append(j)
        elif is_local_model(j) and not args.local:
            runpod_judges.append(j)
        else:
            api_judges.append(j)

    # Filter based on run_mode
    if args.run_mode == "api":
        runpod_judges = []
    elif args.run_mode == "runpod":
        api_judges = []

    if api_judges:
        print(f"\nAPI judges (inline): {', '.join(api_judges)}")
    if runpod_judges:
        print(f"RunPod judges (GPU dispatch): {', '.join(runpod_judges)}")
    if not api_judges and not runpod_judges:
        print("\nNo judges to run (all filtered by run_mode).")

    # Dispatch RunPod judges first (run in background)
    runpod_pod_ids = {}
    if runpod_judges:
        from scripts.alpaca_eval.runpod_dispatch import launch_runpod_job, get_runtime_for_model

        dispatch_type = "runpod"
        if args.config:
            dispatch_type = config.get("gpu_dispatch", dispatch_type)

        hf_repo_id = "SGTR-Geodesic/self-rec-results"

        for judge in runpod_judges:
            runtime_yaml = args.runtime or get_runtime_for_model(judge, dispatch_type)
            if not runtime_yaml:
                print(f"  ⚠ No GPU config found for {judge}. Skipping.")
                continue

            from sgtr_rl.runtime_config import load_runtime_config as _load_rt
            _rt = _load_rt(str(runtime_yaml))
            pod_result_dir = f"{_rt.runpod.volume_mount_path}/alpaca_eval/results"

            from self_rec_framework.src.helpers.model_names import get_gpu_tier
            tier = get_gpu_tier(judge) or "unknown"

            opponent_str = " ".join(opponents)
            local_results = "data/alpaca_eval/results"
            hf_dir = f"alpaca_eval/results/{judge}"
            cmd = (
                f"python scripts/alpaca_eval/run_self_preference.py "
                f"--judges {judge} --opponents {opponent_str} "
                f"--outputs_dir data/alpaca_eval/outputs "
                f"--output_dir {local_results} "
                f"--local "
                f"&& mkdir -p {pod_result_dir} "
                f"&& cp -r {local_results}/{judge} {pod_result_dir}/ "
                f"&& python -c \""
                f"from huggingface_hub import upload_folder; "
                f"upload_folder("
                f"folder_path='{local_results}/{judge}', "
                f"path_in_repo='{hf_dir}', "
                f"repo_id='{hf_repo_id}', "
                f"repo_type='dataset'"
                f")\""
            )

            print(f"\n{'='*70}")
            print(f"Dispatching judge {judge} to RunPod (tier: {tier}, config: {Path(runtime_yaml).name})")
            print(f"{'='*70}")

            try:
                pod_id = launch_runpod_job(
                    command=cmd,
                    runtime_yaml=runtime_yaml,
                    pod_name_prefix=f"ae-judge-{judge}",
                    no_wait=True,
                    dry_run=args.dry_run,
                )
                if pod_id:
                    runpod_pod_ids[judge] = (pod_id, runtime_yaml)
            except Exception as e:
                print(f"  ⚠ Failed to launch pod for {judge}: {e}")
                print(f"    Skipping — try again later or check GPU availability.")

    print(f"Judge mode: {judge_mode}" + (f" (with CoT)" if judge_cot else ""))

    # Run API judges (parallel if max_workers > 1)
    def _run_judge(judge):
        base_model = resolve_base_model(judge)
        label = f"{judge} (trained, base={base_model})" if judge != base_model else judge

        if judge_mode == "ranking":
            print(f"\n{'='*70}")
            print(f"Judge: {label} [RANKING]")
            print(f"  Ranking all {len(opponents)} generator models + own outputs")
            print(f"{'='*70}")
            run_ranking_evaluation(
                judge, opponents, outputs_dir, output_dir,
                gpu_dispatch=gpu_dispatch, cot=judge_cot,
            )
        else:
            # Simple pairwise mode
            print(f"\n{'='*70}")
            print(f"Judge: {label} [SIMPLE]")
            print(f"  Comparing {base_model}'s outputs vs each generator's outputs")
            print(f"{'='*70}")
            for generator in opponents:
                if generator == base_model:
                    print(f"  ⊘ {judge} vs {generator}: skipping (base model == generator)")
                    continue
                run_pairwise_evaluation(judge, generator, outputs_dir, output_dir, config_dir, gpu_dispatch)

    if api_judges and args.max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"\nRunning {len(api_judges)} API judges with {args.max_workers} workers...")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(_run_judge, judge): judge
                for judge in api_judges
            }
            for future in as_completed(futures):
                judge = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"⚠ {judge}: judging failed: {e}")
    else:
        for judge in api_judges:
            _run_judge(judge)

    # Wait for RunPod judges
    if runpod_pod_ids:
        print(f"\n{'='*70}")
        print(f"Waiting for {len(runpod_pod_ids)} RunPod judge(s)...")
        print(f"{'='*70}")

        from sgtr_rl.scripts.runpod_utils import RunPodClient
        from sgtr_rl.runtime_config import load_runtime_config as _load_rt2
        client = RunPodClient(os.environ["RUNPOD_API_KEY"])

        hf_repo_id = "SGTR-Geodesic/self-rec-results"

        for judge, (pod_id, runtime_yaml) in runpod_pod_ids.items():
            _rt2 = _load_rt2(str(runtime_yaml))
            print(f"  Waiting for {judge} (pod {pod_id})...")
            final_pod = client.wait_for_exit(
                pod_id, poll_interval_seconds=_rt2.runpod.poll_interval_seconds
            )
            status = final_pod.get("desiredStatus", "unknown")
            print(f"  ✓ {judge}: pod {pod_id} finished ({status})")
            if _rt2.runpod.terminate_on_exit:
                client.delete_pod(pod_id)
                print(f"    Deleted pod {pod_id}")

            # Download judge results from HF
            local_judge_dir = output_dir / judge
            if local_judge_dir.exists() and any(local_judge_dir.iterdir()):
                print(f"    ✓ Results already available locally at {local_judge_dir}")
            else:
                print(f"    Downloading results from HF...")
                try:
                    from huggingface_hub import snapshot_download
                    local_judge_dir.mkdir(parents=True, exist_ok=True)
                    snapshot_download(
                        repo_id=hf_repo_id,
                        repo_type="dataset",
                        allow_patterns=f"alpaca_eval/results/{judge}/*",
                        local_dir=str(output_dir.parent.parent),  # data/
                    )
                    print(f"    ✓ Downloaded to {local_judge_dir}")
                except Exception as e:
                    print(f"    ⚠ Failed to download from HF: {e}")
                    nv_path = f"{_rt2.runpod.volume_mount_path}/alpaca_eval/results/{judge}"
                    print(f"    Results are on network volume: {nv_path}")

    print(f"\n✓ All evaluations complete. Results in {output_dir}/")


if __name__ == "__main__":
    main()
