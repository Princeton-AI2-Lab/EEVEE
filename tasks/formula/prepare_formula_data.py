#!/usr/bin/env python3
"""Prepare FinLoRA Formula JSONL files for EEVEE."""
from __future__ import annotations

import argparse
import os
import urllib.request


FILES = {
    "formula_train.jsonl": "https://raw.githubusercontent.com/Open-Finance-Lab/FinLoRA/main/data/train/formula_train.jsonl",
    "formula_test.jsonl": "https://raw.githubusercontent.com/Open-Finance-Lab/FinLoRA/main/data/test/formula_test.jsonl",
}


def _download(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "data")),
        help="Output directory for JSONL files",
    )
    args = parser.parse_args()

    for filename, url in FILES.items():
        out_path = os.path.join(args.out_dir, filename)
        _download(url, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
