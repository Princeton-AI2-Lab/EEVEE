"""
MMLU-Pro DataWorker for EEVEE.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import BaseLoader, Environment, StrictUndefined

_env = Environment(
    loader=BaseLoader(),
    undefined=StrictUndefined,
    keep_trailing_newline=True
)

_template_path = Path(__file__).parent / "prompt_header.jinja"
with open(_template_path, "r", encoding="utf-8") as f:
    _prompt_header_template = _env.from_string(f.read())

_CHOICE_MAP = "ABCDEFGHIJ"


def _extract_mmlu_pro_letter(text: Any) -> Optional[str]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    m = re.search(r"answer is \(?([A-J])\)?", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r".*[aA]nswer:\s*([A-J])", s)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", s, flags=re.DOTALL)
    if m:
        return m.group(0).upper()
    return None


def _format_example(question: str, options: List[str], cot_content: str = "", including_answer: bool = True) -> str:
    prompt = f"Question: {str(question).strip()}\n"
    prompt += "Options: "
    for i, opt in enumerate(options):
        if i >= len(_CHOICE_MAP):
            break
        if opt and opt != "N/A":
            prompt += f"{_CHOICE_MAP[i]}. {opt}\n"

    if including_answer:
        cot = str(cot_content or "")
        cot = cot.replace("A: Let's think step by step.", "Answer: Let's think step by step.")
        prompt += cot.strip() + "\n\n"
    else:
        prompt += "Answer: Let's think step by step.\n\n"
    return prompt


class DataProcessor:
    def __init__(self, task_name: str):
        self.task_name = task_name

    def process_task_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        for item in raw_data:
            question = str(item.get("question", "")).strip()
            choices = item.get("choices", {})
            
            if isinstance(choices, dict):
                options = []
                for i in range(len(_CHOICE_MAP)):
                    opt = choices.get(_CHOICE_MAP[i], "")
                    if opt and opt != "N/A":
                        options.append(str(opt))
            else:
                options = [str(opt) for opt in (item.get("options", []) or []) if opt and opt != "N/A"]
            
            answer = item.get("answer", "")
            gold = _extract_mmlu_pro_letter(answer)
            
            meta = item.get("meta", {})
            category = str(meta.get("category", "")).strip()
            
            prompt = ""
            if category:
                prompt = _prompt_header_template.render(subject=category) + "\n"
            prompt += _format_example(question, options, cot_content="", including_answer=False)

            processed.append(
                {
                    "context": "",
                    "question": prompt,
                    "target": gold or "",
                    "others": {
                        "id": item.get("id"),
                        "category": category,
                        "options": options,
                        "task": self.task_name,
                        "data_source": "mmlu_pro",
                    },
                }
            )
        return processed

    def answer_is_correct(self, predicted: str, ground_truth: str, task_dict: Dict[str, Any] = None) -> bool:
        pred = _extract_mmlu_pro_letter(predicted)
        gold = _extract_mmlu_pro_letter(ground_truth)
        if pred is None or gold is None:
            return False
        return pred == gold

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
