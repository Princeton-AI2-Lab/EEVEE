"""
TheoremQA evaluation utilities.

TheoremQA's official scorer handles answer types differently:
- exact string / boolean / multiple-choice matching
- integer exact after rounding
- float within 4% relative tolerance
- numeric lists with order-insensitive element comparison

The task worker uses ``theoremqa_answer_is_correct`` for answer matching.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Optional

_CHOICE_RE = re.compile(r"\(([a-f])\)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _clean_units(text: str) -> str:
    value = str(text).strip()
    value = value.replace("\\pi", "pi").replace("\u03c0", "pi")
    value = re.sub(r"(?<![\d}])pi", str(math.pi), value)
    value = re.sub(r"(\d)\s*pi", rf"\1*{math.pi}", value)
    for unit in ("$", "\u00a5", "\u00b0C", " C", "\u00b0"):
        value = value.replace(unit, "")
    return value.strip()


def _extract_direct_answer(text: Any) -> str:
    value = str("" if text is None else text).strip().rstrip(".").rstrip("/").strip()
    if not value:
        return ""
    lower = value.lower()
    if any(token in lower for token in ("yes", "true")):
        return "True"
    if any(token in lower for token in ("no", "false")):
        return "False"
    if _CHOICE_RE.search(value):
        return value

    value = value.split("=")[-1].strip()
    value = _clean_units(value)
    if re.match(r"-?[\d.]+\s+\D+$", value) or re.match(r"-?[\d.]+\s+[^\s]+$", value):
        return value.split()[0]

    matches = _NUMBER_RE.findall(value)
    if matches:
        return matches[-1]
    return value


def _to_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _clean_units(str(value))
    try:
        return float(text)
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
            return float(parsed)
    except Exception:
        pass
    return None


def _numbers_equal(predicted: Any, ground_truth: Any, *, integer: bool) -> bool:
    pred_num = _to_number(_extract_direct_answer(predicted))
    gold_num = _to_number(ground_truth)
    if pred_num is None or gold_num is None or math.isnan(pred_num) or math.isnan(gold_num):
        return False
    if integer:
        return round(pred_num) == round(gold_num)
    eps = abs(gold_num) * 0.04
    return (gold_num - eps) <= pred_num <= (gold_num + eps)


def _to_list(value: Any) -> Optional[list[Any]]:
    if isinstance(value, list):
        return value
    text = str(value).strip()
    bracket_match = re.search(r"(\[[^\]]+\]|\([^)]+\))", text)
    if bracket_match:
        text = bracket_match.group(1)
    if text.startswith("(") and text.endswith(")"):
        text = f"[{text[1:-1]}]"
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(parsed, tuple):
        parsed = list(parsed)
    return parsed if isinstance(parsed, list) else None


def _lists_equal(predicted: Any, ground_truth: Any) -> bool:
    pred_list = _to_list(str(predicted).strip())
    gold_list = _to_list(ground_truth)
    if pred_list is None or gold_list is None or len(pred_list) != len(gold_list):
        return False
    pred_nums = [_to_number(item) for item in pred_list]
    gold_nums = [_to_number(item) for item in gold_list]
    if any(item is None for item in pred_nums + gold_nums):
        return False
    pred_sorted = sorted(float(item) for item in pred_nums if item is not None)
    gold_sorted = sorted(float(item) for item in gold_nums if item is not None)
    return all(_numbers_equal(pred, gold, integer=False) for pred, gold in zip(pred_sorted, gold_sorted))


def theoremqa_answer_is_correct(predicted: Any, ground_truth: Any, answer_type: str = "") -> bool:
    pred = str("" if predicted is None else predicted).strip()
    gold = str("" if ground_truth is None else ground_truth).strip()
    at = str(answer_type or "").strip().lower()

    if gold.lower() in {"(a)", "(b)", "(c)", "(d)", "(e)", "(f)"}:
        return gold.lower() in pred.lower()
    if at in {"bool", "boolean"}:
        return _extract_direct_answer(pred).lower() == _extract_direct_answer(gold).lower()
    if at in {"integer", "int"}:
        return _numbers_equal(pred, gold, integer=True)
    if at in {"float", "number"}:
        return _numbers_equal(pred, gold, integer=False)
    if at.startswith("list"):
        return _lists_equal(pred, gold)

    if pred.lower() == gold.lower():
        return True
    return _extract_direct_answer(pred).lower() == gold.lower()
