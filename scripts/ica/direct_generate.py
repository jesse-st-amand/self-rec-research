"""Direct-OAI generator for ICA pool data — bypasses inspect_ai sweep.

Used when `srf-generate-sweep` hangs (observed with gpt-oss-20b on Tinker).
Calls Tinker's OAI-compatible endpoint directly, writes results in the
`data/input/{dataset}/{subset}/{model}/data.json` format expected by the
ICA loader.

Usage:
  uv run python scripts/ica/direct_generate.py \\
      --models qwen-3.0-30b gpt-oss-20b \\
      --subsets sharegpt/english_26 sharegpt/english2_74
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from openai import AsyncOpenAI


MODEL_ID_MAP = {
    "qwen-3.0-30b": "openai/Qwen/Qwen3-30B-A3B-Instruct-2507",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
}


def load_dotenv_minimal():
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def strip_harmony(text: str) -> str:
    """Extract final-channel content from a gpt-oss Harmony-style response.

    Tinker's OAI proxy returns text like:
        analysis<reasoning>...<more>assistantfinal<actual_answer>
    We want <actual_answer>. If the pattern isn't present, return as-is.
    """
    m = re.search(r"assistantfinal(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


async def generate_one(client, model_name, tinker_id, uuid, prompt, system_prompt,
                       max_tokens, temperature, semaphore):
    async with semaphore:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=tinker_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=120.0,
                )
                content = r.choices[0].message.content or ""
                if model_name.startswith("gpt-oss-"):
                    content = strip_harmony(content)
                return uuid, content
            except Exception as e:
                if attempt == 2:
                    print(f"    ✗ {uuid[:8]}... failed after 3 retries: {e}")
                    return uuid, None
                print(f"    ⚠ {uuid[:8]}... retry {attempt+1}: {type(e).__name__}")


async def generate_for_subset(client, model_name, tinker_id, dataset_subset,
                               system_prompt, max_tokens, temperature, concurrency):
    dataset, subset = dataset_subset.split("/")
    input_path = Path(f"data/input/{dataset}/{subset}/input.json")
    out_dir = Path(f"data/input/{dataset}/{subset}/{model_name}")
    out_path = out_dir / "data.json"

    with open(input_path) as f:
        prompts = json.load(f)

    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        todo = {k: v for k, v in prompts.items() if k not in existing}
        results = dict(existing)
        print(f"  [{model_name}/{dataset_subset}] {len(existing)} existing, {len(todo)} to generate")
    else:
        todo = prompts
        results = {}
        print(f"  [{model_name}/{dataset_subset}] {len(todo)} to generate")

    if not todo:
        return

    sem = asyncio.Semaphore(concurrency)
    tasks = [
        generate_one(client, model_name, tinker_id, uid, prm,
                     system_prompt, max_tokens, temperature, sem)
        for uid, prm in todo.items()
    ]
    done_count = 0
    for coro in asyncio.as_completed(tasks):
        uid, content = await coro
        if content is not None:
            results[uid] = content
        done_count += 1
        if done_count % 10 == 0 or done_count == len(tasks):
            print(f"    {done_count}/{len(tasks)} done")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Wrote {out_path} ({len(results)} entries)")


async def main_async(models, subsets, max_tokens, temperature, concurrency):
    load_dotenv_minimal()
    client = AsyncOpenAI(
        api_key=os.environ["TINKER_API_KEY"],
        base_url="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
        timeout=120.0,
    )

    # Load system prompt per dataset family
    # Same system prompt for sharegpt + pku; just read sharegpt version
    system_prompt_path = Path("data/input/sharegpt/prompts.yaml")
    import yaml
    with open(system_prompt_path) as f:
        sys_cfg = yaml.safe_load(f)
    system_prompt = sys_cfg["system_prompt"].replace("{priming}", "").strip()

    for model in models:
        tinker_id = MODEL_ID_MAP.get(model)
        if not tinker_id:
            print(f"⚠ Unknown model: {model}, skipping")
            continue
        print(f"\n=== {model} → {tinker_id} ===")
        for subset in subsets:
            await generate_for_subset(
                client, model, tinker_id, subset, system_prompt,
                max_tokens, temperature, concurrency,
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--subsets", nargs="+", required=True,
                   help="Dataset/subset pairs like sharegpt/english_26")
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    asyncio.run(main_async(args.models, args.subsets, args.max_tokens,
                           args.temperature, args.concurrency))


if __name__ == "__main__":
    main()
