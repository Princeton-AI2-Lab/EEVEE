
"""
GPQA DataWorker for EEVEE.

EEVEE expects a "data processor" object that provides:
  - process_task_data(raw_data) -> standardized dicts with keys: context, question, target, others
  - answer_is_correct(predicted, ground_truth, task_dict=None) -> bool
  - evaluate_accuracy(predictions, ground_truths, processed_data=None) -> float (or tuple)

This file is named data_worker.py to match the user's request, but it implements the
DataProcessor interface used by the rest of EEVEE.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import BaseLoader, Environment, StrictUndefined

# Initialize Jinja2 environment
_env = Environment(
    loader=BaseLoader(),
    undefined=StrictUndefined,
    keep_trailing_newline=True
)

# Load prompt template
_template_path = Path(__file__).parent / "prompt.jinja"
with open(_template_path, "r", encoding="utf-8") as f:
    _prompt_template = _env.from_string(f.read())


_LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def _normalize_choice_letter(text: str) -> Optional[str]:
    """
    Extract a multiple-choice option letter A/B/C/D from model output.

    Mirrors the official lm-evaluation-harness GPQA generative flexible
    extraction path, which selects the last parenthesized option.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    matches = re.findall(r"\(\s*([ABCD])\s*\)", s.upper(), flags=re.DOTALL)
    if matches:
        return matches[-1]

    return None


def _format_question_with_choices(question: str, choices: Dict[str, str]) -> str:
    """
    Create the "question" string passed to the generator. Keep context empty.
    """
    return _prompt_template.render(question=question, choices=choices)


class DataProcessor:
    """
    GPQA (closed-book) DataProcessor.

    Expects raw JSONL items like:
      {
        "id": "...",
        "question": "...",
        "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
        "answer": "A",
        "meta": {...}
      }
    """

    def __init__(self, task_name: str = "gpqa"):
        self.task_name = task_name

    def process_task_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        for item in raw_data:
            q = item.get("question", "")
            choices = item.get("choices", {})
            if not isinstance(choices, dict):
                raise ValueError("GPQA JSONL item must contain dict field 'choices'")

            # Normalize keys to 'A'...'D'
            norm_choices: Dict[str, str] = {}
            for k, v in choices.items():
                if k is None:
                    continue
                kk = str(k).strip().upper()
                if kk in _LETTER_TO_INDEX:
                    norm_choices[kk] = str(v)

            missing = [k for k in "ABCD" if k not in norm_choices]
            if missing:
                raise ValueError(f"Missing choice keys {missing} in GPQA item id={item.get('id')}")

            answer = item.get("answer", item.get("target", ""))
            answer_text = str(answer).strip().upper()
            answer_letter = answer_text if answer_text in _LETTER_TO_INDEX else _normalize_choice_letter(answer_text)
            if answer_letter is None:
                raise ValueError(f"Invalid answer in GPQA item id={item.get('id')}: {answer}")

            processed.append(
                {
                    "context": "",  # GPQA is closed-book here
                    "question": _format_question_with_choices(str(q), norm_choices),
                    "target": f"({answer_letter})",
                    "others": {
                        "id": item.get("id"),
                        "choices": norm_choices,
                        "meta": item.get("meta", {}),
                        "task": self.task_name,
                        "data_source": "gpqa",
                    },
                }
            )
        return processed

    def answer_is_correct(self, predicted: str, ground_truth: str, task_dict: Dict[str, Any] = None) -> bool:
        pred_letter = _normalize_choice_letter(predicted)
        gold_letter = _normalize_choice_letter(ground_truth)
        if pred_letter is None or gold_letter is None:
            return False
        return pred_letter == gold_letter

    def evaluate_accuracy(
        self, predictions: List[str], ground_truths: List[str], processed_data: List[Dict[str, Any]] = None
    ) -> float:
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have the same length")
        if not predictions:
            return 0.0
        correct = 0
        for p, g in zip(predictions, ground_truths):
            if self.answer_is_correct(p, g):
                correct += 1
        return correct / len(predictions)
