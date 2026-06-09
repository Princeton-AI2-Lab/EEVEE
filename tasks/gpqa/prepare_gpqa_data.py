#!/usr/bin/env python3
"""
Prepare GPQA (closed-book) JSONL for EEVEE.

This script:
  - Downloads `gpqa_{main,diamond,extended}.csv` from Hugging Face, or reads it from a local GPQA directory
  - Converts rows into EEVEE-friendly JSONL format (same schema used by tasks/gpqa/data_worker.py)
  - Does NOT create any train/val/test split (GPQA has no official split here); outputs a single JSONL per subset.

We intentionally do NOT prepare any Bing/open-book retrieval dataset here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from typing import Dict, List


def _stable_rng(seed: int, example_id: str) -> random.Random:
    h = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).digest()
    # Use 64 bits to seed Random deterministically across runs/platforms
    seed_int = int.from_bytes(h[:8], byteorder="big", signed=False)
    return random.Random(seed_int)


def _load_gpqa_csv(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"GPQA csv not found: {csv_path}")
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _resolve_csv_path(csv_root: str, subset: str, hf_repo: str) -> str:
    filename = f"gpqa_{subset}.csv"
    if csv_root:
        return os.path.join(csv_root, filename)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=hf_repo,
        repo_type="dataset",
        filename=filename,
    )


def _row_to_jsonl(row: Dict[str, str], seed: int, idx: int) -> Dict:
    q = (row.get("Question") or "").strip()
    correct = (row.get("Correct Answer") or "").strip()
    inc1 = (row.get("Incorrect Answer 1") or "").strip()
    inc2 = (row.get("Incorrect Answer 2") or "").strip()
    inc3 = (row.get("Incorrect Answer 3") or "").strip()

    if not q or not correct or not inc1 or not inc2 or not inc3:
        raise ValueError(f"Missing required fields in row idx={idx}")

    example_id = (row.get("Record ID") or str(idx)).strip()
    rng = _stable_rng(seed, example_id)

    options = [correct, inc1, inc2, inc3]
    rng.shuffle(options)
    correct_index = options.index(correct)
    letters = ["A", "B", "C", "D"]
    answer_letter = letters[correct_index]

    choices = {letters[i]: options[i] for i in range(4)}
    meta = {
        "record_id": example_id,
        "high_level_domain": row.get("High-level domain"),
        "subdomain": row.get("Subdomain"),
        "question_writer": row.get("Question Writer"),
        "difficulty": row.get("Writer's Difficulty Estimate"),
        "explanation": row.get("Explanation"),
        "canary": row.get("Canary String"),
    }
    # Drop Nones to keep files smaller
    meta = {k: v for k, v in meta.items() if v not in [None, ""]}

    return {
        "id": example_id,
        "question": q,
        "choices": choices,
        "answer": answer_letter,
        "meta": meta,
    }


def _write_jsonl(path: str, items: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_root",
        type=str,
        default=None,
        help="Optional directory containing gpqa_main.csv, gpqa_diamond.csv, gpqa_extended.csv",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="Idavidrein/gpqa",
        help="Hugging Face dataset repo used when --csv_root is not provided",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="main",
        choices=["main", "diamond", "extended"],
        help="Which GPQA subset CSV to use",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "data")),
        help="Output directory for JSONL files",
    )
    args = parser.parse_args()

    csv_path = _resolve_csv_path(args.csv_root, args.subset, args.hf_repo)
    rows = _load_gpqa_csv(csv_path)

    items: List[Dict] = []
    for i, row in enumerate(rows):
        items.append(_row_to_jsonl(row, seed=args.seed, idx=i))

    out_path = os.path.join(args.out_dir, f"gpqa_{args.subset}.jsonl")
    _write_jsonl(out_path, items)

    print(f"Wrote GPQA {args.subset}:")
    print(f"  test: {out_path} ({len(items)})")


if __name__ == "__main__":
    main()

