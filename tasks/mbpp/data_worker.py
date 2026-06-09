"""
MBPP DataWorker for EEVEE.
"""
from __future__ import annotations

import re
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


def _extract_code_blocks(text: str) -> str:
    s = strip_hidden_thinking(text)

    # Extract first complete fenced code block if present.
    m = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        code = m.group(1).strip()
        if code:
            return code

    # Otherwise fall back to raw text (official-style direct execution path).
    return s


_FEWSHOT_SAMPLES = [
    {
        "task_id": 2,
        "text": "Write a function to find the similar elements from the given two tuple lists.",
        "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
        "test_list": [
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
            "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
        ],
    },
    {
        "task_id": 3,
        "text": "Write a python function to identify non-prime numbers.",
        "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
        "test_list": [
            "assert is_not_prime(2) == False",
            "assert is_not_prime(10) == True",
            "assert is_not_prime(35) == True",
        ],
    },
    {
        "task_id": 4,
        "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
        "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
        "test_list": [
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
        ],
    },
]


class DataProcessor:
    def __init__(self, task_name: str = "mbpp", num_fewshot: int = 3):
        self.task_name = task_name
        self.num_fewshot = num_fewshot

    def process_task_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        for item in raw_data:
            text = str(item.get("text", "")).strip()
            test_list = item.get("test_list", [])
            
            if not text:
                raise ValueError(f"MBPP item missing text: id={item.get('id')}")
            
            if len(test_list) < 3:
                raise ValueError(f"MBPP item needs at least 3 tests: id={item.get('id')}")
            
            fewshot_examples = []
            for ex in _FEWSHOT_SAMPLES[:self.num_fewshot]:
                fewshot_examples.append({
                    "text": ex["text"],
                    "test_list": ex["test_list"],
                    "code": ex["code"],
                })
            
            processed.append(
                {
                    "context": "",
                    "question": _prompt_template.render(
                        text=text,
                        test_list=test_list,
                        fewshot_examples=fewshot_examples
                    ),
                    "target": "\n".join(test_list[:3]),
                    "others": {
                        "id": item.get("id"),
                        "task_id": item.get("task_id"),
                        "text": text,
                        "code": item.get("code", ""),
                        "test_list": test_list,
                        "test_setup_code": item.get("test_setup_code", ""),
                        "meta": item.get("meta", {}),
                        "task": self.task_name,
                        "data_source": "mbpp",
                    },
                }
            )
        return processed

    def answer_is_correct(self, predicted: str, ground_truth: str, task_dict: Dict[str, Any] = None) -> bool:
        if task_dict is None:
            return False
        
        code = _extract_code_blocks(predicted)
        references = [ground_truth]
        predictions = [[code]]
        
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
                f"MBPP code evaluation failed: {type(exc).__name__}: {exc}"
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
        
        for pred in predictions:
            code = _extract_code_blocks(pred)
            full_predictions.append([code])
        
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
                f"MBPP batch code evaluation failed: {type(exc).__name__}: {exc}"
            ) from exc

        return float(result[0]["pass@1"])
