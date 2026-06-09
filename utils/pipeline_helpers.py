from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Sequence, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def make_qid(task_name: str, idx: int, question: str) -> str:
    """Stable question id from task name + index + hash of question text."""
    digest = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    return f"{task_name}_{idx}_{digest}"


def build_correctness_cache(per_question: Sequence[Dict[str, Any]]) -> Dict[str, bool]:
    """Build a qid -> correctness cache from per-question eval results."""
    cache: Dict[str, bool] = {}
    for item in per_question:
        qid = str(item.get("qid", "")).strip()
        if qid:
            cache[qid] = bool(item.get("is_correct", False))
    return cache


def split_train_val_examples(
    samples: List[Dict[str, Any]],
    val_frac: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a sample list into train/val using the same rule as run_gepa_evolve.py."""
    if not samples:
        return [], []

    n = len(samples)
    if n == 1:
        return list(samples), [dict(samples[0])]

    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)

    if val_frac > 0:
        n_val = max(1, int(round(n * val_frac)))
    else:
        n_val = max(1, min(5, n // 4))

    if n_val >= n:
        n_val = max(1, n // 5)

    val_idx = idxs[:n_val]
    train_idx = idxs[n_val:]
    if not train_idx:
        train_idx = val_idx[:-1]
        val_idx = val_idx[-1:]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    return train_samples, val_samples
