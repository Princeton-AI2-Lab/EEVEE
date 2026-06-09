#!/usr/bin/env python3
"""
Prepare TheoremQA JSONL for EEVEE.

Default input:
  Hugging Face dataset TIGER-Lab/TheoremQA

Optional local cache input:
  <source_root>/default/0.0.0/*/theorem_qa-test.arrow

Output:
  tasks/theorem_qa/data/theorem_qa_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List

_HASH_DIR_RE = re.compile(r"^[0-9a-f]{16}$")


def _find_test_arrow(root: str) -> str:
    base = os.path.join(root, "default", "0.0.0")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Unexpected theorem_qa cache layout (missing): {base}")
    candidates: List[str] = []
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if os.path.isdir(p) and _HASH_DIR_RE.match(name):
            f = os.path.join(p, "theorem_qa-test.arrow")
            if os.path.exists(f):
                candidates.append(f)
    if not candidates:
        raise FileNotFoundError(f"No theorem_qa-test.arrow found under {base}")
    return sorted(candidates)[0]


def _write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source_root",
        type=str,
        default=None,
        help="Optional root directory of a cached theorem_qa dataset.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "data")),
        help="Output directory for JSONL files.",
    )
    args = p.parse_args()

    if args.source_root:
        from datasets import Dataset

        test_arrow = _find_test_arrow(args.source_root)
        ds = Dataset.from_file(test_arrow)
    else:
        from datasets import load_dataset

        ds = load_dataset("TIGER-Lab/TheoremQA", split="test")

    out: List[Dict[str, Any]] = []
    for i in range(len(ds)):
        row = ds[i]
        q = str(row.get("Question", "")).strip()
        a = str(row.get("Answer", "")).strip()
        if not q or not a:
            continue
        out.append(
            {
                "id": f"test-{i}",
                "question": q,
                "answer": a,
                "answer_type": row.get("Answer_type"),
                "meta": {"split": "test"},
            }
        )

    out_path = os.path.join(args.out_dir, "theorem_qa_test.jsonl")
    _write_jsonl(out_path, out)

    print("Wrote TheoremQA:")
    print(f"  test: {out_path} ({len(out)})")


if __name__ == "__main__":
    main()

