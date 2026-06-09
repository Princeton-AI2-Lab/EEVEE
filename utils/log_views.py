from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence

from evolve.utils.runtime import build_slot_labels


def format_score(score: float, correct: int, total: int) -> str:
    return f"{float(score):.4f} ({int(correct)}/{int(total)})"


def format_eval(dataset_eval: Any) -> str:
    if dataset_eval is None:
        return format_score(0.0, 0, 0)
    return format_score(
        float(getattr(dataset_eval, "score", 0.0)),
        int(getattr(dataset_eval, "correct", 0)),
        int(getattr(dataset_eval, "total", 0)),
    )


def build_task_slot_count_table(
    *,
    routed_train: Mapping[int, Sequence[Dict[str, Any]]],
    routed_val: Mapping[int, Sequence[Dict[str, Any]]],
    slot_count: int,
) -> List[str]:
    rows: Dict[str, Dict[str, List[int]]] = defaultdict(
        lambda: {
            "train": [0] * slot_count,
            "val": [0] * slot_count,
        }
    )

    for split_name, groups in (("train", routed_train), ("val", routed_val)):
        for slot_id in range(slot_count):
            for sample in groups.get(slot_id, []):
                task_name = str(sample.get("_task_name", "") or "")
                rows[task_name][split_name][slot_id] += 1

    if not rows:
        return []

    task_width = max(len("benchmark"), max(len(task_name) for task_name in rows))
    slot_labels = build_slot_labels(slot_count)
    cell_width = max(11, max((len(label) for label in slot_labels), default=1))

    header = [f"{'benchmark':<{task_width}}"]
    header.extend(f"{slot_labels[slot_id]:>{cell_width}}" for slot_id in range(slot_count))
    header.append(f"{'total':>{cell_width}}")
    divider = "-" * (task_width + (slot_count + 1) * (cell_width + 1))

    lines = ["cell = train/val", " ".join(header), divider]

    total_train = [0] * slot_count
    total_val = [0] * slot_count
    for task_name in sorted(rows):
        train_counts = rows[task_name]["train"]
        val_counts = rows[task_name]["val"]
        for slot_id in range(slot_count):
            total_train[slot_id] += train_counts[slot_id]
            total_val[slot_id] += val_counts[slot_id]
        row = [f"{task_name:<{task_width}}"]
        row.extend(f"{train_counts[slot_id]}/{val_counts[slot_id]}".rjust(cell_width) for slot_id in range(slot_count))
        row.append(f"{sum(train_counts)}/{sum(val_counts)}".rjust(cell_width))
        lines.append(" ".join(row))

    overall = [f"{'OVERALL':<{task_width}}"]
    overall.extend(f"{total_train[slot_id]}/{total_val[slot_id]}".rjust(cell_width) for slot_id in range(slot_count))
    overall.append(f"{sum(total_train)}/{sum(total_val)}".rjust(cell_width))
    lines.append(divider)
    lines.append(" ".join(overall))
    return lines


def build_task_slot_test_count_table(
    *,
    routed_test: Mapping[int, Sequence[Dict[str, Any]]],
    slot_count: int,
) -> List[str]:
    rows: Dict[str, List[int]] = defaultdict(lambda: [0] * slot_count)

    for slot_id in range(slot_count):
        for sample in routed_test.get(slot_id, []):
            task_name = str(sample.get("_task_name", "") or "")
            rows[task_name][slot_id] += 1

    if not rows:
        return []

    task_width = max(len("benchmark"), max(len(task_name) for task_name in rows))
    slot_labels = build_slot_labels(slot_count)
    cell_width = max(11, max((len(label) for label in slot_labels), default=1))

    header = [f"{'benchmark':<{task_width}}"]
    header.extend(f"{slot_labels[slot_id]:>{cell_width}}" for slot_id in range(slot_count))
    header.append(f"{'total':>{cell_width}}")
    divider = "-" * (task_width + (slot_count + 1) * (cell_width + 1))

    lines = ["cell = test", " ".join(header), divider]

    total_counts = [0] * slot_count
    for task_name in sorted(rows):
        counts = rows[task_name]
        for slot_id in range(slot_count):
            total_counts[slot_id] += counts[slot_id]
        row = [f"{task_name:<{task_width}}"]
        row.extend(f"{counts[slot_id]}".rjust(cell_width) for slot_id in range(slot_count))
        row.append(f"{sum(counts)}".rjust(cell_width))
        lines.append(" ".join(row))

    overall = [f"{'OVERALL':<{task_width}}"]
    overall.extend(f"{total_counts[slot_id]}".rjust(cell_width) for slot_id in range(slot_count))
    overall.append(f"{sum(total_counts)}".rjust(cell_width))
    lines.append(divider)
    lines.append(" ".join(overall))
    return lines
