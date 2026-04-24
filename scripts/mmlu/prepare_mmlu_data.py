"""Prep MMLU eval data for the MMLU-ICA experiment.

Downloads cais/mmlu (test split) from HF and writes a 50-question
subset-stratified sample to data/input/mmlu/mmlu_50/input.json in the
self-rec-framework dataset layout.

Layout written:
  data/input/mmlu/mmlu_50/input.json
    { "<uuid16>": {
        "question": "...",
        "choices": ["...", "...", "...", "..."],
        "answer": "A" | "B" | "C" | "D",
        "subject": "..."
      }, ...
    }

UUIDs are sha256(f"{subject}|{question}")[:16] for cross-run stability.
Stratification: one question per subject (57 subjects) until we hit N,
then random top-up with seed=42. With N=50, we pick 50 of 57 subjects.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def make_uuid(subject: str, question: str) -> str:
    h = hashlib.sha256(f"{subject}|{question}".encode("utf-8")).hexdigest()
    return h[:16]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", type=str, default="data/input/mmlu")
    parser.add_argument("--subset-name", type=str, default=None,
                        help="Defaults to mmlu_<n>")
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    letter = {0: "A", 1: "B", 2: "C", 3: "D"}

    by_subject: dict[str, list[dict]] = {}
    for row in ds:
        by_subject.setdefault(row["subject"], []).append({
            "question": row["question"],
            "subject": row["subject"],
            "choices": list(row["choices"]),
            "answer": letter[row["answer"]],
        })

    rng = random.Random(args.seed)
    for items in by_subject.values():
        rng.shuffle(items)

    subjects = sorted(by_subject.keys())
    rng.shuffle(subjects)

    picked: list[dict] = []
    cursors = {s: 0 for s in subjects}
    while len(picked) < args.n:
        progressed = False
        for s in subjects:
            if len(picked) >= args.n:
                break
            if cursors[s] < len(by_subject[s]):
                picked.append(by_subject[s][cursors[s]])
                cursors[s] += 1
                progressed = True
        if not progressed:
            break

    out = {}
    for item in picked:
        uuid = make_uuid(item["subject"], item["question"])
        out[uuid] = item

    subset_name = args.subset_name or f"mmlu_{args.n}"
    out_dir = Path(args.out_root) / subset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "input.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    subjects_used = sorted({v["subject"] for v in out.values()})
    print(f"Wrote {len(out)} questions across {len(subjects_used)} subjects to {out_path}")


if __name__ == "__main__":
    main()
