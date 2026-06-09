"""
TheoremQA DataWorker for EEVEE.

Expected raw JSONL item format:
{
  "id": "...",
  "question": "...",
  "answer": "...",
  "answer_type": "integer" | "float" | "string" | ...,
  "meta": {...}
}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from jinja2 import BaseLoader, Environment, StrictUndefined

from . import utils

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


class DataProcessor:
    def __init__(self, task_name: str = "theorem_qa"):
        self.task_name = task_name

    def process_task_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        for item in raw_data:
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", item.get("target", ""))).strip()
            if not q or not a:
                raise ValueError(f"TheoremQA item missing question/answer: id={item.get('id')}")

            prompt = _prompt_template.render(question=q)

            processed.append(
                {
                    "context": "",
                    "question": prompt,
                    "target": a,
                    "others": {
                        "id": item.get("id"),
                        "answer_type": item.get("answer_type"),
                        "meta": item.get("meta", {}),
                        "task": self.task_name,
                        "data_source": "theorem_qa",
                    },
                }
            )
        return processed

    def answer_is_correct(self, predicted: str, ground_truth: str, task_dict: Dict[str, Any] = None) -> bool:
        """
        Check if predicted answer matches ground truth using TheoremQA rules.
        
        This uses the same evaluation logic as lm-evaluation-harness, which handles:
        - Fraction format differences (1/2 vs \\frac{1}{2})
        - Decimal vs fraction (4.5 vs \\frac{9}{2})
        - Symbolic expression equivalence
        - And many other mathematical equivalence cases
        
        For non-numeric answers (string type), falls back to string comparison.
        """
        at = ""
        if task_dict is not None:
            at = str(task_dict.get("others", {}).get("answer_type", "") or "").strip().lower()

        return utils.theoremqa_answer_is_correct(predicted, ground_truth, at)

    def evaluate_accuracy(
        self, predictions: List[str], ground_truths: List[str], processed_data: List[Dict[str, Any]] = None
    ) -> float:
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have the same length")
        if not predictions:
            return 0.0
        correct = 0
        for i, (p, g) in enumerate(zip(predictions, ground_truths)):
            td = processed_data[i] if processed_data is not None and i < len(processed_data) else None
            if self.answer_is_correct(p, g, task_dict=td):
                correct += 1
        return correct / len(predictions)

