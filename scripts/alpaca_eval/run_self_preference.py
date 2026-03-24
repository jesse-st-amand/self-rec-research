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

    'll-3.1-8b-01_sft_pw_vs_qwen' -> 'll-3.1-8b'
    'll-3.1-8b' -> 'll-3.1-8b'
    """
    if short_name in INSPECT_MODEL_NAMES:
        return short_name
    # Try to match {base_model}-{training_run} pattern
    for base_name in sorted(INSPECT_MODEL_NAMES.keys(), key=len, reverse=True):
        if short_name.startswith(base_name + "-"):
            return base_name
    return short_name


# ---------------------------------------------------------------------------
# Prompt template (same as alpaca_eval_gpt4/alpaca_eval.txt)
# ---------------------------------------------------------------------------

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


def parse_ranking(completion: str) -> float:
    """Parse model ranking from completion text.

    Returns 1 if model_1 preferred, 2 if model_2 preferred, nan on failure.
    """
    import numpy as np
    try:
        parsed = ast.literal_eval(completion.strip())
        rank = [c for c in parsed if c["model"] == "model_1"][0]["rank"]
        if rank in (1, 2):
            return float(rank)
    except Exception:
        pass
    return float(np.nan)


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

        user_msg = JUDGE_USER_TEMPLATE.format(
            instruction=j_out["instruction"],
            output_1=out_1,
            output_2=out_2,
        )

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt_text = f"SYSTEM: {JUDGE_SYSTEM_PROMPT}\n\nUSER: {user_msg}\n\nASSISTANT:"

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

    renderer_name = get_recommended_renderer_name(hf_model_id)
    tokenizer = get_tokenizer(hf_model_id)
    renderer = r.get_renderer(renderer_name, tokenizer)

    sampling_params = tinker.types.SamplingParams(
        max_tokens=100,
        temperature=0,
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

        user_msg = JUDGE_USER_TEMPLATE.format(
            instruction=s_out["instruction"],
            output_1=out_1,
            output_2=out_2,
        )

        convo = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
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

    # Resolve model names from --config or --judges/--opponents
    if args.config:
        import yaml as _yaml
        with open(args.config) as f:
            config = _yaml.safe_load(f)
        # evaluator_models = judges, generator_models = opponents
        raw_eval = config.get("evaluator_models", config.get("model_names", []))
        raw_gen = config.get("generator_models", config.get("model_names", []))
        judges = expand_model_names(raw_eval)
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
    output_dir = Path(args.output_dir)
    config_dir = Path("data/alpaca_eval/configs")

    print(f"Evaluator models (judges): {', '.join(judges)}")
    print(f"Generator models (opponents): {', '.join(opponents)}")
    print(f"Outputs dir: {outputs_dir}")
    print(f"Results dir: {output_dir}")
    print(f"Setup: Each judge compares its own outputs vs each generator's outputs.")
    print(f"       Skipping when judge == generator (all three models identical).")

    # Resolve gpu_dispatch for splitting judges
    _gpu_dispatch = "runpod"
    if args.config:
        _gpu_dispatch = config.get("gpu_dispatch", _gpu_dispatch)

    # Split judges into API/Tinker (inline) and RunPod (dispatch to GPU pod)
    # With tinker dispatch, all hf/ and trained models run inline via Tinker API
    api_judges = []
    runpod_judges = []
    for j in judges:
        is_trained = j not in INSPECT_MODEL_NAMES
        if _gpu_dispatch == "tinker" and (is_trained or is_local_model(j)):
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

    # Resolve gpu_dispatch from config
    gpu_dispatch = "runpod"
    if args.config:
        gpu_dispatch = config.get("gpu_dispatch", gpu_dispatch)

    # Run API judges (parallel if max_workers > 1)
    def _run_judge(judge):
        base_model = resolve_base_model(judge)
        label = f"{judge} (trained, base={base_model})" if judge != base_model else judge
        print(f"\n{'='*70}")
        print(f"Judge: {label} [API]")
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
