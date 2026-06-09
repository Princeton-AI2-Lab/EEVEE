#!/usr/bin/env python3
"""
Prepare MBPP (google-research-datasets/mbpp) JSONL for EEVEE.
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


def _convert_mbpp(split: str, dataset) -> List[Dict[str, Any]]:
    """Convert MBPP records into EEVEE JSONL records."""
    out: List[Dict[str, Any]] = []
    
    for i in range(len(dataset)):
        row = dataset[i]
        task_id = row.get("task_id", i)
        text = str(row.get("text", "")).strip()
        code = str(row.get("code", "")).strip()
        test_list = row.get("test_list", [])
        test_setup_code = str(row.get("test_setup_code", "")).strip()
        challenge_test_list = row.get("challenge_test_list", [])
        
        if not text:
            continue
        
        out.append(
            {
                "id": f"mbpp-{split}-{task_id}",
                "task_id": task_id,
                "text": text,
                "code": code,
                "test_list": test_list,
                "test_setup_code": test_setup_code,
                "challenge_test_list": challenge_test_list,
                "meta": {
                    "split": split,
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

    print("Loading MBPP dataset...")
    
    # Load train, validation, and test splits.
    splits_to_process = {
        "train": "train",
        "validation": "validation",
        "test": "test"
    }
    
    all_stats = {"splits": {}}
    
    for split_name, split_key in splits_to_process.items():
        print(f"\nProcessing {split_name} split...")
        
        try:
            dataset = load_dataset(
                "google-research-datasets/mbpp",
                "full",
                split=split_key
            )
        except Exception as e:
            print(f"  Warning: Failed to load {split_name}: {e}")
            continue
        
        items = _convert_mbpp(split_name, dataset)
        
        # Write data files.
        out_name = f"mbpp-{split_name}.jsonl"
        out_path = os.path.join(args.out_dir, out_name)
        _write_jsonl(out_path, items)
        
        all_stats["splits"][split_name] = len(items)
        print(f"  Wrote mbpp-{split_name}: {out_path} ({len(items)} items)")
    
    # Write metadata.
    all_stats["task"] = "mbpp"
    all_stats["task_type"] = "code_generation"
    meta_path = os.path.join(args.out_dir, "meta.json")
    _write_json(meta_path, all_stats)
    print(f"\nWrote meta: {meta_path}")


if __name__ == "__main__":
    main()
