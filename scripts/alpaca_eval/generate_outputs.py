"""Generate model outputs on AlpacaEval 805 instructions.

Usage:
    uv run python scripts/alpaca_eval/generate_outputs.py \
        --model_names ll-3.1-8b qwen-2.5-7b \
        --output_dir data/alpaca_eval/outputs \
        --max_tokens 2048 --temperature 0.7
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from self_rec_framework.src.helpers.model_names import INSPECT_MODEL_NAMES
from self_rec_framework.scripts.utils import expand_model_names


def load_alpaca_eval_instructions() -> list[dict]:
    """Load the 805 AlpacaEval instructions from HuggingFace."""
    path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="alpaca_eval.json",
        repo_type="dataset",
    )
    with open(path) as f:
        return json.load(f)


def resolve_tinker_checkpoint(short_name: str) -> tuple[str, str | None]:
    """Resolve a model name that may include a training run suffix.

    Names like 'll-3.1-8b-01_sft_pw_vs_qwen' split into:
      base model 'll-3.1-8b' + training run '01_sft_pw_vs_qwen'

    Returns (hf_base_model_id, tinker_sampler_path_or_None).
    """
    from pathlib import Path
    import glob

    training_dir = Path("data/training")

    # Try to match {base_model}-{training_run} pattern
    if training_dir.exists():
        for base_name in sorted(INSPECT_MODEL_NAMES.keys(), key=len, reverse=True):
            if not short_name.startswith(base_name + "-"):
                continue
            run_suffix = short_name[len(base_name) + 1:]
            matches = glob.glob(str(training_dir / f"{run_suffix}__*"))
            if not matches:
                print(f"  ⚠ No training run matching '{run_suffix}' in data/training/")
                break
            run_dir = sorted(matches)[-1]  # latest timestamp
            ckpt_file = Path(run_dir) / "checkpoints" / "checkpoints.jsonl"
            if not ckpt_file.exists():
                print(f"  ⚠ No checkpoints.jsonl in {run_dir}")
                break
            # Read last checkpoint entry
            with open(ckpt_file) as f:
                last_line = [l.strip() for l in f if l.strip()][-1]
            ckpt = json.loads(last_line)
            sampler_path = ckpt.get("sampler_path")
            if not sampler_path:
                print(f"  ⚠ No sampler_path in checkpoint for {run_suffix}")
                break
            # Resolve base model HF ID
            inspect_name = INSPECT_MODEL_NAMES[base_name]
            hf_model = inspect_name.removeprefix("hf/").removeprefix("together/").removeprefix("openai/")
            return hf_model, sampler_path

    # No training suffix — base model only
    if short_name in INSPECT_MODEL_NAMES:
        inspect_name = INSPECT_MODEL_NAMES[short_name]
        # Strip any provider prefix to get the HF model ID
        for prefix in ("hf/", "together/", "openai/", "anthropic/", "google/"):
            if inspect_name.startswith(prefix):
                return inspect_name.removeprefix(prefix), None
    raise ValueError(f"Cannot resolve Tinker model for '{short_name}'")


def parse_provider(short_name: str, gpu_dispatch: str = "runpod") -> tuple[str, str, dict]:
    """Map shorthand model name to (provider, api_model_name, extra_kwargs).

    When gpu_dispatch='tinker', hf/ models and trained model names route
    through Tinker's sampling API instead of RunPod/local.

    Returns:
        provider: "openai", "anthropic", "google", "together", "local_hf", or "tinker"
        api_model: The model ID to pass to the API
        extra_kwargs: Additional kwargs
    """
    # Tinker dispatch: handles both trained models (ll-3.1-8b-01_sft_pw_vs_qwen)
    # and base hf/ models (ll-3.1-8b) via Tinker's sampling API
    if gpu_dispatch == "tinker":
        # Check if it's a trained model name or a base hf/ model
        is_trained = short_name not in INSPECT_MODEL_NAMES
        is_hf = (not is_trained and INSPECT_MODEL_NAMES[short_name].startswith("hf/"))
        if is_trained or is_hf:
            hf_model, sampler_path = resolve_tinker_checkpoint(short_name)
            return "tinker", hf_model, {"sampler_path": sampler_path}

    if short_name not in INSPECT_MODEL_NAMES:
        raise ValueError(f"Unknown model: {short_name}")

    inspect_name = INSPECT_MODEL_NAMES[short_name]
    if inspect_name.startswith("hf/"):
        hf_model = inspect_name.removeprefix("hf/")
        return "local_hf", hf_model, {}
    elif inspect_name.startswith("together/"):
        api_model = inspect_name.removeprefix("together/")
        return "together", api_model, {"base_url": "https://api.together.xyz/v1"}
    elif inspect_name.startswith("anthropic/"):
        api_model = inspect_name.removeprefix("anthropic/")
        return "anthropic", api_model, {}
    elif inspect_name.startswith("google/"):
        api_model = inspect_name.removeprefix("google/")
        return "google", api_model, {}
    elif inspect_name.startswith("openai/"):
        api_model = inspect_name.removeprefix("openai/")
        return "openai", api_model, {}
    else:
        raise ValueError(f"Unknown provider for {short_name}: {inspect_name}")


def generate_openai(instructions, model, max_tokens, temperature, **kwargs):
    """Generate outputs using OpenAI-compatible API (OpenAI or Together)."""
    from openai import OpenAI

    base_url = kwargs.get("base_url")
    if base_url:
        api_key = os.environ.get("TOGETHER_API_KEY")
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI()

    outputs = []
    for i, inst in enumerate(instructions):
        print(f"  [{i+1}/{len(instructions)}] {inst['instruction'][:60]}...", end="", flush=True)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": inst["instruction"]}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            print(f" ERROR: {e}")
            text = f"[ERROR: {e}]"
        else:
            print(f" ({len(text)} chars)")
        outputs.append(text)
    return outputs


def generate_anthropic(instructions, model, max_tokens, temperature, **kwargs):
    """Generate outputs using Anthropic API."""
    from anthropic import Anthropic

    client = Anthropic()
    outputs = []
    for i, inst in enumerate(instructions):
        print(f"  [{i+1}/{len(instructions)}] {inst['instruction'][:60]}...", end="", flush=True)
        try:
            resp = client.messages.create(
                model=model,
                messages=[{"role": "user", "content": inst["instruction"]}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = resp.content[0].text
        except Exception as e:
            print(f" ERROR: {e}")
            text = f"[ERROR: {e}]"
        else:
            print(f" ({len(text)} chars)")
        outputs.append(text)
    return outputs


def generate_google(instructions, model, max_tokens, temperature, **kwargs):
    """Generate outputs using Google GenAI API."""
    from google import genai

    client = genai.Client()
    outputs = []
    for i, inst in enumerate(instructions):
        print(f"  [{i+1}/{len(instructions)}] {inst['instruction'][:60]}...", end="", flush=True)
        try:
            resp = client.models.generate_content(
                model=model,
                contents=inst["instruction"],
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            text = resp.text or ""
        except Exception as e:
            print(f" ERROR: {e}")
            text = f"[ERROR: {e}]"
        else:
            print(f" ({len(text)} chars)")
        outputs.append(text)
    return outputs


def generate_local_hf(instructions, model, max_tokens, temperature, **kwargs):
    """Generate outputs using a local HuggingFace model on GPU.

    Requires torch + transformers. Model weights are downloaded from HuggingFace Hub.
    Suitable for models that are no longer available via serverless API (dedicated-only).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading model {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=dtype,
        device_map="auto",
    )
    model_obj.eval()
    print(f"  Model loaded on {device} ({dtype})")

    batch_size = 8
    outputs = []
    tokenizer.padding_side = "left"

    for start in range(0, len(instructions), batch_size):
        batch = instructions[start:start + batch_size]
        end = min(start + batch_size, len(instructions))
        print(f"  [{start+1}-{end}/{len(instructions)}] Generating batch...", end="", flush=True)

        prompts = []
        for inst in batch:
            messages = [{"role": "user", "content": inst["instruction"]}]
            if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = f"User: {inst['instruction']}\nAssistant:"
            prompts.append(text)

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        encoded = {k: v.to(model_obj.device) for k, v in encoded.items()}

        with torch.inference_mode():
            generated = model_obj.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_width = encoded["input_ids"].shape[1]
        for row in generated:
            completion_ids = row[input_width:]
            text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            outputs.append(text)

        print(f" ({sum(len(o) for o in outputs[start:end])} chars)")

    # Clean up GPU memory
    del model_obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return outputs


def generate_tinker(instructions, model, max_tokens, temperature, **kwargs):
    """Generate outputs using Tinker's sampling API.

    Args:
        model: HuggingFace model ID (e.g., 'meta-llama/Llama-3.1-8B-Instruct')
        kwargs['sampler_path']: Tinker checkpoint path for trained models, or None for base model
    """
    import tinker
    from tinker_cookbook import renderers as r
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    sampler_path = kwargs.get("sampler_path")

    print(f"  Connecting to Tinker...")
    client = tinker.ServiceClient()

    if sampler_path:
        print(f"  Loading trained model from: {sampler_path}")
        sampling_client = client.create_sampling_client(model_path=sampler_path)
    else:
        print(f"  Loading base model: {model}")
        sampling_client = client.create_sampling_client(base_model=model)

    # Set up renderer for chat template
    renderer_name = get_recommended_renderer_name(model)
    tokenizer = get_tokenizer(model)
    renderer = r.get_renderer(renderer_name, tokenizer)

    sampling_params = tinker.types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Fire all requests asynchronously
    print(f"  Submitting {len(instructions)} generation requests...")
    futures = []
    for inst in instructions:
        convo = [{"role": "user", "content": inst["instruction"]}]
        model_input = renderer.build_generation_prompt(convo)
        futures.append(sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=sampling_params,
        ))

    # Collect results
    outputs = []
    for i, future in enumerate(futures):
        if (i + 1) % 50 == 0 or i == len(futures) - 1:
            print(f"  [{i+1}/{len(futures)}] Collecting results...", flush=True)
        result = future.result()
        seq = result.sequences[0]
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        content = r.get_text_content(parsed_msg)
        outputs.append(content)

    print(f"  ✓ Generated {len(outputs)} outputs via Tinker")
    return outputs


GENERATORS = {
    "openai": generate_openai,
    "together": generate_openai,  # Together uses OpenAI-compatible API
    "anthropic": generate_anthropic,
    "google": generate_google,
    "local_hf": generate_local_hf,
    "tinker": generate_tinker,
}


def generate_for_model(
    short_name: str,
    instructions: list[dict],
    output_dir: Path,
    max_tokens: int,
    temperature: float,
    gpu_dispatch: str = "runpod",
):
    """Generate outputs for a single model and save to JSON."""
    output_path = output_dir / f"{short_name}.json"
    if output_path.exists():
        print(f"⊘ {short_name}: already exists at {output_path}, skipping")
        return

    provider, api_model, extra_kwargs = parse_provider(short_name, gpu_dispatch=gpu_dispatch)
    gen_fn = GENERATORS[provider]

    print(f"\n{'='*70}")
    print(f"Generating outputs for {short_name}")
    print(f"  Provider: {provider}, API model: {api_model}")
    print(f"  Instructions: {len(instructions)}, max_tokens: {max_tokens}, temp: {temperature}")
    print(f"{'='*70}")

    raw_outputs = gen_fn(instructions, api_model, max_tokens, temperature, **extra_kwargs)

    # Format as alpaca_eval expects
    results = []
    for inst, output in zip(instructions, raw_outputs):
        results.append({
            "dataset": inst.get("dataset", "helpful_base"),
            "instruction": inst["instruction"],
            "output": output,
            "generator": short_name,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved {len(results)} outputs to {output_path}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate model outputs on AlpacaEval instructions")
    parser.add_argument("--model_names", nargs="+", default=None,
                        help="Model names (e.g., gpt-4o-mini qwen-2.5-7b)")
    parser.add_argument("--config", default=None,
                        help="Path to experiment config YAML (reads model_names from it)")
    parser.add_argument("--output_dir", default="data/alpaca_eval/outputs",
                        help="Directory to save outputs")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--runtime", default=None,
                        help="Path to RunPod runtime YAML (for hf/ models that need GPU)")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Number of API models to generate in parallel (default: 1)")
    parser.add_argument("--run_mode", default="both", choices=["both", "api", "runpod"],
                        help="Which models to run: 'both' (default), 'api' only, or 'runpod' only")
    parser.add_argument("--local", action="store_true",
                        help="Run hf/ models locally instead of dispatching to RunPod (use when already on a GPU pod)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print RunPod payload without launching")
    args = parser.parse_args()

    # Resolve model names and settings from --config or CLI args
    if args.config:
        import yaml as _yaml
        with open(args.config) as f:
            config = _yaml.safe_load(f)
        # Generation needs outputs for:
        # 1. All generator_models (opponents in pairwise eval)
        # 2. Base models of evaluator_models (the "self" outputs judges compare against)
        # Trained evaluators (ll-3.1-8b-01_sft_pw_vs_qwen) use their base model's
        # outputs (ll-3.1-8b), so we resolve base models and deduplicate.
        from scripts.alpaca_eval.run_self_preference import resolve_base_model
        raw_gen = config.get("generator_models", [])
        raw_eval = config.get("evaluator_models", [])
        raw_legacy = config.get("model_names", [])

        gen_models = expand_model_names(raw_gen or raw_legacy)
        # Add base models of evaluators (for "self" outputs)
        eval_base_models = []
        if raw_eval:
            for m in expand_model_names(raw_eval):
                base = resolve_base_model(m)
                if base not in eval_base_models:
                    eval_base_models.append(base)

        # Merge: generators + evaluator base models, deduplicated
        seen = set()
        models = []
        for m in gen_models + eval_base_models:
            if m not in seen:
                seen.add(m)
                models.append(m)
        if args.max_tokens == 2048:
            args.max_tokens = config.get("max_tokens", args.max_tokens)
        if args.temperature == 0.7:
            args.temperature = config.get("temperature", args.temperature)
        if args.max_workers == 1:
            args.max_workers = config.get("max_workers", args.max_workers)
        if args.run_mode == "both":
            args.run_mode = config.get("run_mode", args.run_mode)
        # Resolve gpu_dispatch runtime: look for {dispatch}.yaml next to config
        if not args.runtime and config.get("gpu_dispatch"):
            dispatch_name = config["gpu_dispatch"]  # e.g., "runpod"
            config_dir = Path(args.config).resolve().parent
            runtime_path = config_dir / f"{dispatch_name}.yaml"
            if runtime_path.exists():
                args.runtime = str(runtime_path)
                print(f"GPU dispatch: {dispatch_name} (from {runtime_path})")
    elif args.model_names:
        models = expand_model_names(args.model_names)
    else:
        parser.error("Provide either --config or --model_names")

    print(f"Models to generate: {', '.join(models)}")

    instructions = load_alpaca_eval_instructions()
    print(f"Loaded {len(instructions)} AlpacaEval instructions")

    output_dir = Path(args.output_dir)

    # Resolve gpu_dispatch from config
    gpu_dispatch = "runpod"
    if args.config:
        gpu_dispatch = config.get("gpu_dispatch", gpu_dispatch)

    # Split models into API/Tinker (run inline) and RunPod (dispatch to GPU pod)
    api_models = []
    runpod_models = []
    for model in models:
        if output_dir.joinpath(f"{model}.json").exists():
            print(f"⊘ {model}: already exists, skipping")
            continue

        # Tinker dispatch: all hf/ and trained models run inline via Tinker API
        if gpu_dispatch == "tinker":
            api_models.append(model)
            continue

        # Check if it's a trained model name (not in INSPECT_MODEL_NAMES)
        if model not in INSPECT_MODEL_NAMES:
            print(f"⚠ {model}: not found in INSPECT_MODEL_NAMES, skipping")
            continue

        inspect_name = INSPECT_MODEL_NAMES[model]
        if inspect_name.startswith("hf/") and not args.local:
            runpod_models.append(model)
        else:
            api_models.append(model)

    # Filter based on run_mode
    if args.run_mode == "api":
        runpod_models = []
    elif args.run_mode == "runpod":
        api_models = []

    if api_models:
        print(f"\nAPI models (inline): {', '.join(api_models)}")
    if runpod_models:
        print(f"RunPod models (GPU dispatch): {', '.join(runpod_models)}")
    if not api_models and not runpod_models:
        print("\nNo models to generate (all exist or filtered by run_mode).")

    # ── RunPod workers: each model gets an independent thread with retry ──
    runpod_results = {}  # model -> "completed" | "failed"
    runpod_futures = {}

    if runpod_models:
        from scripts.alpaca_eval.runpod_dispatch import launch_runpod_job, get_runtime_for_model
        from sgtr_rl.scripts.runpod_utils import RunPodClient
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time, threading

        dispatch_type = "runpod"
        if args.config:
            dispatch_type = config.get("gpu_dispatch", dispatch_type)

        hf_repo_id = "SGTR-Geodesic/self-rec-results"
        api_key = os.environ.get("RUNPOD_API_KEY")

        STARTUP_TIMEOUT = 600   # 10 min to get a GPU
        JOB_TIMEOUT = 7200      # 2 hours for the job
        RETRY_INTERVAL = 30     # seconds between launch retries
        MAX_RETRIES = 120       # max launch attempts (~1 hour at 30s)
        POLL_INTERVAL = 30
        print_lock = threading.Lock()

        def _log(msg):
            with print_lock:
                print(msg, flush=True)

        def _run_runpod_model(model):
            """Independent worker: retry launch → wait for GPU → wait for completion → download."""
            runtime_yaml = args.runtime or get_runtime_for_model(model, dispatch_type)
            if not runtime_yaml:
                _log(f"  ⚠ {model}: No GPU config found. Skipping.")
                return "failed"

            from self_rec_framework.src.helpers.model_names import get_gpu_tier
            tier = get_gpu_tier(model) or "unknown"

            local_out = "data/alpaca_eval/outputs"
            hf_path = f"alpaca_eval/outputs/{model}.json"
            cmd = (
                f"python scripts/alpaca_eval/generate_outputs.py "
                f"--model_names {model} "
                f"--output_dir {local_out} "
                f"--max_tokens {args.max_tokens} "
                f"--temperature {args.temperature} "
                f"--local "
                f"&& python -c \""
                f"from huggingface_hub import upload_file; "
                f"upload_file("
                f"path_or_fileobj='{local_out}/{model}.json', "
                f"path_in_repo='{hf_path}', "
                f"repo_id='{hf_repo_id}', "
                f"repo_type='dataset'"
                f")\""
            )

            client = RunPodClient(api_key)

            for attempt in range(1, MAX_RETRIES + 1):
                # Phase 0: Launch pod
                _log(f"  {model}: Launch attempt {attempt} (tier: {tier})")
                try:
                    pod_id = launch_runpod_job(
                        command=cmd,
                        runtime_yaml=runtime_yaml,
                        pod_name_prefix=f"ae-gen-{model}",
                        no_wait=True,
                        dry_run=args.dry_run,
                    )
                    if not pod_id:
                        _log(f"  {model}: Dry run, no pod created.")
                        return "completed"
                except Exception as e:
                    err_msg = str(e)
                    if "no instances" in err_msg.lower() or "currently available" in err_msg.lower():
                        _log(f"  {model}: No GPUs available (attempt {attempt}). Retrying in {RETRY_INTERVAL}s...")
                        time.sleep(RETRY_INTERVAL)
                        continue
                    _log(f"  ⚠ {model}: Launch error: {e}")
                    time.sleep(RETRY_INTERVAL)
                    continue

                _log(f"  {model}: Pod {pod_id} created. Waiting for GPU...")

                # Phase 1: Wait for GPU assignment
                start = time.time()
                gpu_assigned = False
                while time.time() - start < STARTUP_TIMEOUT:
                    try:
                        pod = client.get_pod(pod_id)
                    except Exception:
                        time.sleep(POLL_INTERVAL)
                        continue

                    status = pod.get("desiredStatus", "")
                    if status in ("EXITED", "TERMINATED"):
                        gpu_assigned = True  # it ran and finished fast
                        break
                    # REST API uses machineId (not runtime) to indicate GPU assignment
                    if pod.get("machineId") or pod.get("runtime") is not None:
                        gpu_assigned = True
                        elapsed = int(time.time() - start)
                        gpu_name = (pod.get("machine") or {}).get("gpuDisplayName", "unknown")
                        _log(f"  {model}: GPU assigned after {elapsed}s ({gpu_name})")
                        break
                    time.sleep(POLL_INTERVAL)

                if not gpu_assigned:
                    elapsed = int(time.time() - start)
                    _log(f"  {model}: No GPU after {elapsed}s. Deleting pod and retrying...")
                    try:
                        client.delete_pod(pod_id)
                    except Exception:
                        pass
                    time.sleep(RETRY_INTERVAL)
                    continue  # retry launch

                # Phase 2: Wait for job completion
                start_job = time.time()
                finished = False
                while time.time() - start_job < JOB_TIMEOUT:
                    try:
                        pod = client.get_pod(pod_id)
                    except Exception:
                        time.sleep(POLL_INTERVAL)
                        continue

                    status = pod.get("desiredStatus", "")
                    if status in ("EXITED", "TERMINATED"):
                        elapsed = int(time.time() - start_job)
                        _log(f"  ✓ {model}: Pod finished ({status}) [{elapsed}s]")
                        finished = True
                        break
                    time.sleep(POLL_INTERVAL)

                if not finished:
                    _log(f"  ⚠ {model}: Job timed out after {JOB_TIMEOUT}s. Deleting pod.")
                    try:
                        client.delete_pod(pod_id)
                    except Exception:
                        pass
                    return "failed"

                # Cleanup pod
                try:
                    client.delete_pod(pod_id)
                except Exception:
                    pass

                # Phase 3: Download from HF
                local_output = output_dir / f"{model}.json"
                if local_output.exists():
                    _log(f"  ✓ {model}: Output already available locally")
                else:
                    _log(f"  {model}: Downloading from HF...")
                    try:
                        from huggingface_hub import hf_hub_download
                        import shutil
                        downloaded = hf_hub_download(
                            repo_id=hf_repo_id,
                            filename=hf_path,
                            repo_type="dataset",
                        )
                        output_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(downloaded, local_output)
                        _log(f"  ✓ {model}: Downloaded to {local_output}")
                    except Exception as e:
                        _log(f"  ⚠ {model}: Failed to download from HF: {e}")
                        return "failed"

                return "completed"

            _log(f"  ⚠ {model}: Exhausted {MAX_RETRIES} launch attempts.")
            return "failed"

        # Launch all RunPod workers in parallel
        runpod_executor = ThreadPoolExecutor(max_workers=len(runpod_models))
        for model in runpod_models:
            runpod_futures[model] = runpod_executor.submit(_run_runpod_model, model)

    # ── API models: run in parallel while RunPod workers are active ──
    if api_models and args.max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"\nGenerating {len(api_models)} API models with {args.max_workers} workers...")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    generate_for_model, model, instructions, output_dir,
                    args.max_tokens, args.temperature, gpu_dispatch
                ): model
                for model in api_models
            }
            for future in as_completed(futures):
                model = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"⚠ {model}: generation failed: {e}")
    else:
        for model in api_models:
            generate_for_model(model, instructions, output_dir, args.max_tokens, args.temperature, gpu_dispatch)

    # ── Collect RunPod results ──
    if runpod_futures:
        print(f"\nWaiting for {len(runpod_futures)} RunPod worker(s)...")
        for model, future in runpod_futures.items():
            runpod_results[model] = future.result()
        runpod_executor.shutdown(wait=True)

        failed = [m for m, r in runpod_results.items() if r != "completed"]
        if failed:
            print(f"\n⚠ Failed models ({len(failed)}): {', '.join(sorted(failed))}")
            print(f"  Re-run to retry (completed outputs are preserved).")
            sys.exit(1)

    print(f"\n✓ Generation complete. Outputs in {output_dir}/")


if __name__ == "__main__":
    main()
