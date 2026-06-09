#!/usr/bin/env python3
"""
Prepare MMLU-Pro (TIGER-Lab/MMLU-Pro) JSONL for EEVEE.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List

_HASH_DIR_RE = re.compile(r"^[0-9a-f]{16}$")
_CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

CATEGORIES = [
    "biology", "business", "chemistry", "computer_science", 
    "economics", "engineering", "health", "history", 
    "law", "math", "other", "philosophy", "physics", "psychology"
]


def _write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _convert_category(category: str, dataset) -> List[Dict[str, Any]]:
    """Convert one category into EEVEE JSONL records."""
    filtered = dataset.filter(lambda x: x["category"] == category)
    
    out: List[Dict[str, Any]] = []
    for i in range(len(filtered)):
        row = filtered[i]
        q = str(row.get("question", "")).strip()
        options = row.get("options", [])
        ans_idx = row.get("answer_index", None)
        cot = row.get("cot_content", "")
        
        if not q or not isinstance(options, list) or ans_idx is None:
            continue
        
        try:
            ans_idx = int(ans_idx)
        except Exception:
            continue
        
        if ans_idx < 0 or ans_idx >= len(options):
            continue
        
        # Build the options dictionary.
        choice_map = {}
        for j, opt in enumerate(options):
            if j >= len(_CHOICES):
                break
            choice_map[_CHOICES[j]] = str(opt)
        
        item = {
            "id": f"mmlu_pro-{category}-test-{i}",
            "question": q,
            "choices": choice_map,
            "answer": _CHOICES[ans_idx],
            "meta": {
                "category": category,
                "split": "test",
            }
        }
        
        if cot:
            item["meta"]["cot_content"] = str(cot)
        
        out.append(item)
    
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "data")),
        help="Output directory for JSONL files",
    )
    args = p.parse_args()

    from datasets import load_dataset

    print("Loading MMLU-Pro dataset...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    
    stats: Dict[str, Any] = {"categories": [], "counts": {}}
    all_items: List[Dict[str, Any]] = []

    for category in CATEGORIES:
        print(f"Processing category: {category}")
        items = _convert_category(category, dataset)
        
        # Write one category file.
        out_name = f"mmlu_pro-{category}.jsonl"
        out_path = os.path.join(args.out_dir, out_name)
        _write_jsonl(out_path, items)
        
        task_name = f"mmlu_pro-{category}"
        stats["categories"].append(task_name)
        stats["counts"][task_name] = len(items)
        all_items.extend(items)
        
        print(f"  Wrote {task_name}: {out_path} ({len(items)} items)")

    # Write the merged all-category data.
    all_path = os.path.join(args.out_dir, "mmlu_pro-all.jsonl")
    _write_jsonl(all_path, all_items)
    stats["counts"]["mmlu_pro-all"] = len(all_items)
    print(f"\nWrote mmlu_pro-all: {all_path} ({len(all_items)} items)")

    # Write metadata.
    meta_path = os.path.join(args.out_dir, "meta.json")
    _write_json(meta_path, stats)
    print(f"Wrote meta: {meta_path}")


if __name__ == "__main__":
    main()
