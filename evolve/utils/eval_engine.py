"""
Evaluation Engine.

Runs the generator with a given prompt on a set of samples,
computes eval_score, and returns per-question correctness for
difficulty tracking plus all fail cases.
"""
import json
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from utils import safe_filename_fragment


def evaluate_prompt(
    generator,
    prompt_text: str,
    samples: List[Dict[str, Any]],
    evaluate_fn,
    max_workers: int = 20,
    results_dir: Optional[str] = None,
    candidate_id: Optional[int] = None,
    log_dir: Optional[str] = None,
    label: str = "eval",
) -> Dict[str, Any]:
    """
    Evaluate a prompt on a list of samples.

    Args:
        generator: Generator agent instance
        prompt_text: the candidate prompt to evaluate
        samples: list of sample dicts (must have question, target; optionally context, _qid, _task_name)
        evaluate_fn: callable(predicted, ground_truth, task_dict, **kwargs) -> bool
        max_workers: parallel workers
        results_dir: directory to save results/npy (optional)
        candidate_id: candidate ID for file naming (optional)
        log_dir: LLM call log directory (optional)
        label: label prefix for call_ids

    Returns:
        Dict with:
            eval_score: float
            correct: int
            total: int
            fail_cases: list of ALL fail case dicts
            per_question: list of {qid, is_correct} for difficulty updates
            experiment_log: str
            judge_scores_path: str or None
    """
    if not samples:
        return {
            "eval_score": 0.0, "correct": 0, "total": 0,
            "fail_cases": [], "per_question": [],
            "experiment_log": "No samples.", "judge_scores_path": None,
        }

    def eval_single(args: Tuple[int, Dict]) -> Dict[str, Any]:
        i, sample = args
        try:
            context = sample.get("context", "")
            question = sample["question"]
            target = sample["target"]

            task_runner = getattr(evaluate_fn, "run_sample", None)
            if callable(task_runner):
                task_result = task_runner(
                    generator=generator,
                    prompt_text=prompt_text,
                    sample=sample,
                    log_dir=log_dir,
                    call_id=f"{label}_{i}",
                )
                response = str(task_result.get("answer", "") or "")
                is_correct = bool(task_result.get("is_correct", False))
                score = float(task_result.get("score", 1.0 if is_correct else 0.0))
                metadata = task_result.get("metadata", {})
            else:
                response, _ = generator.generate(
                    question=question,
                    prompt_text=prompt_text,
                    context=context,
                    call_id=f"{label}_{i}",
                    log_dir=log_dir,
                )

                is_correct = evaluate_fn(
                    predicted=response,
                    ground_truth=target,
                    task_dict=sample,
                    question=question,
                    context=context,
                    log_dir=log_dir,
                    call_id=f"{label}_{i}",
                )
                score = 1.0 if is_correct else 0.0
                metadata = {}

            row = {
                "index": i,
                "is_correct": bool(is_correct),
                "answer": response,
                "target": target,
                "question": question[:500],
                "qid": sample.get("_qid", ""),
                "score": score,
            }
            if metadata:
                row["metadata"] = metadata
            return row
        except Exception as e:
            try:
                from tasks.code_eval_utils import CodeEvalError
            except Exception:
                CodeEvalError = ()  # type: ignore[assignment]

            if isinstance(e, CodeEvalError):
                raise

            return {
                "index": i,
                "is_correct": False,
                "answer": f"ERROR: {e}",
                "target": sample.get("target", ""),
                "question": sample.get("question", "")[:500],
                "qid": sample.get("_qid", ""),
                "score": 0.0,
            }

    results_list: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(eval_single, (i, s)): i
            for i, s in enumerate(samples)
        }
        for future in as_completed(futures):
            results_list.append(future.result())

    results_list.sort(key=lambda r: r["index"])

    correct = sum(1 for r in results_list if r["is_correct"])
    total = len(results_list)
    eval_score = correct / total if total > 0 else 0.0

    judge_scores = np.array([r["score"] for r in results_list], dtype=np.float32)

    judge_scores_path = None
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        safe_label = safe_filename_fragment(label, fallback="eval")
        npy_name = f"judge_scores_{safe_label}_{candidate_id}.npy" if candidate_id is not None else f"judge_scores_{safe_label}.npy"
        judge_scores_path = os.path.join(results_dir, npy_name)
        os.makedirs(os.path.dirname(judge_scores_path), exist_ok=True)
        np.save(judge_scores_path, judge_scores)

    fail_cases = [r for r in results_list if not r["is_correct"]]

    per_question = [{"qid": r["qid"], "is_correct": r["is_correct"]} for r in results_list]

    log_lines = [
        f"Eval Score: {eval_score:.4f} ({correct}/{total})",
        "",
    ]
    if fail_cases:
        log_lines.append(f"=== Fail Cases ({len(fail_cases)} failures) ===")
        for fc in fail_cases:
            log_lines.append(f"\n--- Fail Case #{fc['index']} ---")
            log_lines.append(f"Question: {fc['question']}")
            log_lines.append(f"Expected: {fc['target']}")
            log_lines.append(f"Got: {fc['answer']}")
    experiment_log = "\n".join(log_lines)

    if results_dir:
        results_json = {
            "eval_score": eval_score,
            "correct": correct,
            "total": total,
            "judge_scores_path": judge_scores_path,
        }
        results_path = os.path.join(
            results_dir,
            f"results_{safe_label}_{candidate_id}.json" if candidate_id is not None else f"results_{safe_label}.json",
        )
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=2)

    return {
        "eval_score": eval_score,
        "correct": correct,
        "total": total,
        "fail_cases": fail_cases,
        "all_results": results_list,
        "per_question": per_question,
        "experiment_log": experiment_log,
        "judge_scores_path": judge_scores_path,
    }
