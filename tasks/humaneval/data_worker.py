"""
HumanEval DataWorker for EEVEE.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from jinja2 import BaseLoader, Environment, StrictUndefined

from utils.text_cleaning import strip_hidden_thinking
from tasks.code_eval_utils import CodeEvalBackendError, CodeEvalDependencyError, compute_pass_at_k

_env = Environment(
    loader=BaseLoader(),
    undefined=StrictUndefined,
    keep_trailing_newline=True
)

_template_path = Path(__file__).parent / "prompt.jinja"
with open(_template_path, "r", encoding="utf-8") as f:
    _prompt_template = _env.from_string(f.read())


def _compose_continuation_code(prompt: str, predicted: str) -> str:
    prompt = prompt or ""
    predicted = strip_hidden_thinking(predicted).strip("\n")

    if not predicted.strip():
        return prompt

    # HumanEval official-style continuation: directly append completion text.
    if prompt and not prompt.endswith("\n"):
        prompt += "\n"
    return prompt + predicted


class DataProcessor:
    def __init__(self, task_name: str = "humaneval"):
        self.task_name = task_name

    def process_task_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        for item in raw_data:
            prompt = str(item.get("prompt", ""))
            if not prompt.strip():
                raise ValueError(f"HumanEval item missing prompt: id={item.get('id')}")

            test = str(item.get("test", "")).strip()
            entry_point = str(item.get("entry_point", "")).strip()
            
            target = f"{test}\ncheck({entry_point})"
            
            processed.append(
                {
                    "context": "",
                    "question": _prompt_template.render(prompt=prompt),
                    "target": target,
                    "others": {
                        "id": item.get("id"),
                        "task_id": item.get("task_id"),
                        "prompt": prompt,
                        "test": test,
                        "entry_point": entry_point,
                        "canonical_solution": item.get("canonical_solution", ""),
                        "meta": item.get("meta", {}),
                        "task": self.task_name,
                        "data_source": "humaneval",
                    },
                }
            )
        return processed

    def answer_is_correct(self, predicted: str, ground_truth: str, task_dict: Dict[str, Any] = None) -> bool:
        if task_dict is None:
            return False
        
        prompt = task_dict.get("others", {}).get("prompt", "")
        full_code = _compose_continuation_code(prompt, predicted)
        
        references = [ground_truth]
        predictions = [[full_code]]
        
        try:
            result = compute_pass_at_k.compute(
                references=references,
                predictions=predictions,
                k=[1]
            )
        except CodeEvalDependencyError:
            raise
        except Exception as exc:
            raise CodeEvalBackendError(
                f"HumanEval code evaluation failed: {type(exc).__name__}: {exc}"
            ) from exc

        return bool(result[0]["pass@1"] > 0)

    def evaluate_accuracy(
        self, predictions: List[str], ground_truths: List[str], processed_data: List[Dict[str, Any]] = None
    ) -> float:
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have the same length")
        if not predictions:
            return 0.0
        
        references = ground_truths
        full_predictions = []
        
        for i, pred in enumerate(predictions):
            prompt = processed_data[i]["others"]["prompt"] if processed_data else ""
            full_predictions.append([_compose_continuation_code(prompt, pred)])
        
        try:
            result = compute_pass_at_k.compute(
                references=references,
                predictions=full_predictions,
                k=[1]
            )
        except CodeEvalDependencyError:
            raise
        except Exception as exc:
            raise CodeEvalBackendError(
                f"HumanEval batch code evaluation failed: {type(exc).__name__}: {exc}"
            ) from exc

        return float(result[0]["pass@1"])
