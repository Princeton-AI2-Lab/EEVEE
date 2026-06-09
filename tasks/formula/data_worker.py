"""
Formula DataWorker for EEVEE.

This task follows the EEVEE task layout:
  - prompt template lives in ``prompt.jinja``
  - the processor lives in ``data_worker.py``
  - raw items are mapped into ``context/question/target/others``

The upstream FinLoRA / FinAgents data already stores prompt text in ``context``
and answers in ``target`` for both FiNER and Formula. We therefore preserve the
official prompt wording instead of reverse-parsing and reconstructing it.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import BaseLoader, Environment, StrictUndefined

_env = Environment(
    loader=BaseLoader(),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)

_template_path = Path(__file__).parent / "prompt.jinja"
with open(_template_path, "r", encoding="utf-8") as f:
    _prompt_template = _env.from_string(f.read())

_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")
_LEADING_ENUM_RE = re.compile(r"^\s*\d+\s*[\)\].:\-]\s*")


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _finer_raw_prediction(text: Any) -> str:
    return _to_text(text).strip()


def _formula_raw_prediction(text: Any) -> str:
    t = _to_text(text).strip()
    if not t:
        return ""
    lowered = t.lower()
    marker = "answer:"
    idx = lowered.rfind(marker)
    if idx != -1:
        t = t[idx + len(marker) :].strip()
    return t


def _normalize_finer_parts(text: Any) -> List[str]:
    cleaned = _finer_raw_prediction(text).replace("\n", ",")
    if not cleaned:
        return []

    parts = []
    for raw_part in cleaned.split(","):
        part = _LEADING_ENUM_RE.sub("", raw_part.strip().strip("`").strip("\"'"))
        if part:
            parts.append(part.lower())
    return parts


def _normalize_formula_value(text: Any) -> str:
    cleaned = _formula_raw_prediction(text)
    if not cleaned:
        return ""

    match = _NUMBER_RE.search(cleaned)
    if match:
        return match.group(0).replace(",", "")
    return cleaned.replace(",", "").strip()


class DataProcessor:
    """Processor for the FinLoRA formula calculation task."""

    def __init__(self, task_name: str = "formula"):
        self.task_name = task_name
        if self.task_name != "formula":
            raise ValueError(f"Unknown Formula task: {self.task_name}")

    def process_task_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_data):
            prompt_text = _to_text(item.get("question", item.get("context", ""))).strip()
            target = _to_text(item.get("answer", item.get("target", ""))).strip()
            if not prompt_text or not target:
                raise ValueError(f"Formula item missing prompt/target: idx={idx}")

            processed.append(
                {
                    "context": "",
                    "question": _prompt_template.render(prompt_text=prompt_text),
                    "target": target,
                    "others": {
                        "id": item.get("id", idx),
                        "meta": item.get("meta", {}),
                        "raw_context": _to_text(item.get("context", "")),
                        "task": self.task_name,
                        "data_source": "finlora",
                    },
                }
            )
        return processed

    def parse_prediction_for_evaluation(
        self, predicted: str, task_dict: Dict[str, Any] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        if self.task_name == "finer":
            parts = _normalize_finer_parts(predicted)
            if not parts:
                return False, None, "empty_prediction"
            return True, ",".join(parts), None

        normalized = _normalize_formula_value(predicted)
        if not normalized:
            return False, None, "empty_prediction"
        return True, normalized, None

    def answer_is_correct(self, predicted: str, ground_truth: str, task_dict: Dict[str, Any] = None) -> bool:
        if self.task_name == "finer":
            return _normalize_finer_parts(predicted) == _normalize_finer_parts(ground_truth)

        pred = _normalize_formula_value(predicted)
        gold = _normalize_formula_value(ground_truth)
        if not pred or not gold:
            return False

        try:
            return math.isclose(float(pred), float(gold), rel_tol=0.0, abs_tol=1e-6)
        except ValueError:
            return pred == gold

    def evaluate_accuracy(
        self, predictions: List[str], ground_truths: List[str], processed_data: List[Dict[str, Any]] = None
    ) -> float:
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have the same length")
        if not predictions:
            return 0.0

        correct = 0
        for predicted, ground_truth in zip(predictions, ground_truths):
            if self.answer_is_correct(predicted, ground_truth):
                correct += 1
        return correct / len(predictions)
