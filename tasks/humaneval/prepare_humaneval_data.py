#!/usr/bin/env python3
"""
Prepare HumanEval (openai/openai_humaneval) JSONL for EEVEE.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

def _write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _convert_humaneval(dataset) -> List[Dict[str, Any]]:
    """Convert HumanEval records into EEVEE JSONL records."""
    out: List[Dict[str, Any]] = []
    
    for i in range(len(dataset)):
        row = dataset[i]
        task_id = str(row.get("task_id", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        canonical_solution = str(row.get("canonical_solution", "")).strip()
        test = str(row.get("test", "")).strip()
        entry_point = str(row.get("entry_point", "")).strip()
        
        if not task_id or not prompt:
            continue
        
        out.append(
            {
                "id": f"humaneval-{task_id}",
                "task_id": task_id,
                "prompt": prompt,
                "canonical_solution": canonical_solution,
                "test": test,
                "entry_point": entry_point,
                "meta": {
                    "split": "test",
                    "task_type": "code_generation"
                },
            }
        )
    
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

    print("Loading HumanEval dataset...")
    dataset = load_dataset(
        "openai/openai_humaneval",
        split="test"
    )
    
    print("Processing HumanEval...")
    items = _convert_humaneval(dataset)
    
    # Write data files.
    out_path = os.path.join(args.out_dir, "humaneval.jsonl")
    _write_jsonl(out_path, items)
    print(f"Wrote humaneval: {out_path} ({len(items)} items)")
    
    # Write metadata.
    stats = {
        "task": "humaneval",
        "count": len(items),
        "task_type": "code_generation"
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    _write_json(meta_path, stats)
    print(f"Wrote meta: {meta_path}")


if __name__ == "__main__":
    main()
