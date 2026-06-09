from __future__ import annotations

from datetime import datetime
from pathlib import Path
import threading
from typing import Any, Dict

from .log_views import build_task_slot_count_table, build_task_slot_test_count_table, format_eval, format_score


class RunLogger:
    """Writes a compact human-readable run log."""

    def __init__(self, log_path: str):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._write(f"EEVEE Run Log  |  started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write("=" * 80)

    def _write(self, text: str) -> None:
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(text + "\n")

    def section(self, title: str) -> None:
        self._write("")
        self._write("=" * 80)
        self._write(f"  {title}")
        self._write("=" * 80)

    def kv(self, key: str, value: Any) -> None:
        self._write(f"  {key}: {value}")

    def line(self, text: str) -> None:
        self._write(f"  {text}")

    def eval_text(self, dataset_eval: Any) -> str:
        return format_eval(dataset_eval)

    def log_data_loaded(
        self,
        train_counts: Dict[str, int],
        val_counts: Dict[str, int],
        test_counts: Dict[str, int],
    ) -> None:
        self.section("Data Loaded")
        self.kv("train total", sum(train_counts.values()))
        self.kv("val total", sum(val_counts.values()))
        self.kv("test total", sum(test_counts.values()))
        all_tasks = sorted(set(train_counts) | set(val_counts) | set(test_counts))
        for task_name in all_tasks:
            self._write(
                f"    {task_name:30s}  train={train_counts.get(task_name, 0):>5d}  "
                f"val={val_counts.get(task_name, 0):>5d}  test={test_counts.get(task_name, 0):>5d}"
            )

    def log_phase_results(self, tag: str, per_task: Dict[str, Dict[str, Any]]) -> None:
        self._write(f"  {tag}")
        total_correct = 0
        total_count = 0
        for task_name, result in sorted(per_task.items()):
            accuracy = float(result.get("accuracy", 0.0))
            correct = int(result.get("correct", 0))
            total = int(result.get("total", 0))
            repeats = int(result.get("repeats", 1))
            std = float(result.get("accuracy_std", 0.0))
            suffix = f"  repeats={repeats} std={std:.4f}" if repeats > 1 else ""
            self._write(f"    {task_name:30s}  {format_score(accuracy, correct, total)}{suffix}")
            total_correct += correct
            total_count += total
        overall = total_correct / total_count if total_count else 0.0
        self._write(f"    {'OVERALL':30s}  {format_score(overall, total_correct, total_count)}")

    def log_empty_baseline(
        self,
        *,
        train_score: float,
        train_correct: int,
        train_total: int,
        val_score: float,
        val_correct: int,
        val_total: int,
    ) -> None:
        self.section("Empty Prompt Baseline")
        self._write(f"    train  {format_score(train_score, train_correct, train_total)}")
        self._write(f"    val    {format_score(val_score, val_correct, val_total)}")

    def log_candidate(self, tag: str, candidate: Any) -> None:
        self._write(f"  {tag}")
        self._write(f"    candidate_id={getattr(candidate, 'candidate_id', None)}")
        self._write(f"    val={format_eval(getattr(candidate, 'val_eval', None))}")
        self._write(f"    train={format_eval(getattr(candidate, 'train_eval', None))}")
        self._write(f"    prompt_set_size={len(getattr(candidate, 'prompt_set', []))}")

    def log_prompt_part_table(
        self,
        *,
        routed_train: Dict[int, Any],
        routed_val: Dict[int, Any],
        slot_count: int,
    ) -> None:
        self._write("  Grouped Training Data")
        for line in build_task_slot_count_table(
            routed_train=routed_train,
            routed_val=routed_val,
            slot_count=slot_count,
        ):
            self._write(f"    {line}")

    def log_final_test_route_table(
        self,
        *,
        routed_test: Dict[int, Any],
        slot_count: int,
    ) -> None:
        self._write("  Final Test Routing")
        for line in build_task_slot_test_count_table(
            routed_test=routed_test,
            slot_count=slot_count,
        ):
            self._write(f"    {line}")
