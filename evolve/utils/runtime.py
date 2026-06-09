from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .structures import DatasetEval


def sample_qids(qids: Sequence[str], size: int, rng: random.Random) -> List[str]:
    ordered = sorted(set(str(qid) for qid in qids))
    if size <= 0 or len(ordered) <= size:
        return ordered
    return sorted(rng.sample(ordered, size))


def dataset_eval_from_maps(
    *,
    split_name: str,
    correctness_by_qid: Mapping[str, bool],
    answers_by_qid: Optional[Mapping[str, str]] = None,
) -> DatasetEval:
    ordered = {str(qid): bool(ok) for qid, ok in correctness_by_qid.items()}
    answers = {str(qid): str((answers_by_qid or {}).get(qid, "")) for qid in ordered}
    total = len(ordered)
    correct = sum(1 for ok in ordered.values() if ok)
    score = (correct / total) if total else 0.0
    return DatasetEval(
        split_name=split_name,
        score=score,
        correct=correct,
        total=total,
        correctness_by_qid=dict(ordered),
        answers_by_qid=answers,
    )


def slice_eval(base_eval: DatasetEval, qids: Iterable[str]) -> DatasetEval:
    selected = [str(qid) for qid in qids if str(qid) in base_eval.correctness_by_qid]
    return dataset_eval_from_maps(
        split_name=base_eval.split_name,
        correctness_by_qid={qid: base_eval.correctness_by_qid[qid] for qid in selected},
        answers_by_qid={qid: base_eval.answers_by_qid.get(qid, "") for qid in selected},
    )


def build_slot_labels(slot_count: int) -> List[str]:
    labels: List[str] = []
    for slot_id in range(max(0, slot_count)):
        n = slot_id
        chars: List[str] = []
        while True:
            n, rem = divmod(n, 26)
            chars.append(chr(ord("A") + rem))
            if n == 0:
                break
            n -= 1
        labels.append("".join(reversed(chars)))
    return labels


def build_visible_text(sample: Dict[str, Any]) -> str:
    context = str(sample.get("context", "") or "").strip()
    question = str(sample.get("question", "") or "").strip()
    if context and question:
        return f"{context}\n\n{question}"
    return context or question


def build_prompt_examples(samples: Sequence[Dict[str, Any]], *, max_examples: int) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for sample in list(samples)[: max(0, max_examples)]:
        examples.append(
            {
                "task_name": str(sample.get("_task_name", "")),
                "question": str(sample.get("question", "") or ""),
                "context": str(sample.get("context", "") or ""),
            }
        )
    return examples


def build_router_support_examples(
    samples_by_qid: Mapping[str, Dict[str, Any]],
    route_targets: Mapping[str, int],
    *,
    display_slot_labels: Sequence[str],
    max_examples: int,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for qid in sorted(route_targets)[: max(0, max_examples)]:
        sample = samples_by_qid.get(qid)
        if sample is None:
            continue
        examples.append(
            {
                "task_name": str(sample.get("_task_name", "")),
                "visible_text": build_visible_text(sample),
                "best_label": str(display_slot_labels[int(route_targets[qid])]),
            }
        )
    return examples


def group_samples_by_slot(route_by_qid: Mapping[str, int], samples: Sequence[Dict[str, Any]], slot_count: int) -> Dict[int, List[Dict[str, Any]]]:
    groups: Dict[int, List[Dict[str, Any]]] = {slot_id: [] for slot_id in range(slot_count)}
    for sample in samples:
        qid = str(sample.get("_qid", ""))
        slot_id = int(route_by_qid[qid])
        if 0 <= slot_id < slot_count:
            groups[slot_id].append(sample)
    return groups


def prompt_scores_by_qid(
    prompt_evals: Mapping[int, DatasetEval],
    qids: Iterable[str],
) -> Dict[str, Dict[int, bool]]:
    out: Dict[str, Dict[int, bool]] = {}
    for qid in qids:
        qid_text = str(qid)
        out[qid_text] = {
            slot_id: bool(dataset_eval.correctness_by_qid.get(qid_text, False))
            for slot_id, dataset_eval in prompt_evals.items()
        }
    return out


@dataclass
class RouterAnalysisCase:
    qid: str
    task_name: str
    visible_text: str
    ground_truth: str
    wrong_slot_id: int
    better_slot_id: int
    wrong_prompt_text: str
    better_prompt_text: str
    wrong_answer: str
    better_answer: str
    suggestion: str = ""


def choose_better_slot(
    *,
    wrong_slot_id: int,
    qid: str,
    prompt_val_scores: Mapping[int, float],
    prompt_train_evals: Mapping[int, DatasetEval],
) -> Optional[int]:
    candidates = [
        slot_id
        for slot_id, train_eval in prompt_train_evals.items()
        if slot_id != wrong_slot_id and train_eval.correctness_by_qid.get(qid, False)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda slot_id: (prompt_val_scores.get(slot_id, 0.0), -slot_id))


def build_router_analysis_cases(
    *,
    train_samples: Sequence[Dict[str, Any]],
    route_by_qid: Mapping[str, int],
    prompt_set: Sequence[str],
    prompt_train_evals: Mapping[int, DatasetEval],
    prompt_val_scores: Mapping[int, float],
    max_analysis_cases: int,
) -> List[RouterAnalysisCase]:
    cases: List[RouterAnalysisCase] = []
    for sample in train_samples:
        qid = str(sample["_qid"])
        wrong_slot_id = int(route_by_qid[qid])
        if wrong_slot_id not in prompt_train_evals:
            continue
        if prompt_train_evals[wrong_slot_id].correctness_by_qid.get(qid, False):
            continue
        better_slot_id = choose_better_slot(
            wrong_slot_id=wrong_slot_id,
            qid=qid,
            prompt_val_scores=prompt_val_scores,
            prompt_train_evals=prompt_train_evals,
        )
        if better_slot_id is None:
            continue
        wrong_eval = prompt_train_evals[wrong_slot_id]
        better_eval = prompt_train_evals[better_slot_id]
        cases.append(
            RouterAnalysisCase(
                qid=qid,
                task_name=str(sample.get("_task_name", "")),
                visible_text=build_visible_text(sample),
                ground_truth=str(sample.get("target", "") or ""),
                wrong_slot_id=wrong_slot_id,
                better_slot_id=better_slot_id,
                wrong_prompt_text=str(prompt_set[wrong_slot_id]),
                better_prompt_text=str(prompt_set[better_slot_id]),
                wrong_answer=str(wrong_eval.answers_by_qid.get(qid, "")),
                better_answer=str(better_eval.answers_by_qid.get(qid, "")),
            )
        )
        if len(cases) >= max_analysis_cases:
            break
    return cases


def split_samples_by_task(samples: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample.get("_task_name", ""))].append(sample)
    return grouped
