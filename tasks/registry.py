"""
Task registry for EEVEE.

Discovers task families under tasks/*/task.yaml.
Supports simple two-way train/test split (no per-category valid split).
"""
from __future__ import annotations

import importlib
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class TaskInfo:
    """Resolved per-task metadata from a task plugin YAML."""
    task_name: str
    family: str
    processor_path: str  # "some.module:ClassName"
    splits: Dict[str, str]  # keys: train_data / test_data


class TaskRegistry:
    """Discovers task families under tasks/*/task.yaml."""

    def __init__(self, repo_root: str):
        self.repo_root = os.path.abspath(repo_root)
        self.tasks_root = os.path.join(self.repo_root, "tasks")
        self._by_task: Dict[str, TaskInfo] = {}

    def discover(self) -> "TaskRegistry":
        if not os.path.isdir(self.tasks_root):
            raise FileNotFoundError(f"Tasks root not found: {self.tasks_root}")

        for family in sorted(os.listdir(self.tasks_root)):
            family_dir = os.path.join(self.tasks_root, family)
            if not os.path.isdir(family_dir):
                continue
            yaml_path = os.path.join(family_dir, "task.yaml")
            if not os.path.exists(yaml_path):
                continue
            with open(yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            self._ingest_family_yaml(family=family, yaml_path=yaml_path, cfg=cfg)
        return self

    def _ingest_family_yaml(self, family: str, yaml_path: str, cfg: Dict[str, Any]) -> None:
        declared_family = str(cfg.get("family") or family).strip()
        if declared_family != family:
            raise ValueError(
                f"Task plugin family mismatch in {yaml_path}: folder={family!r} yaml.family={declared_family!r}"
            )
        processor = str(cfg.get("processor") or "").strip()
        if ":" not in processor:
            raise ValueError(f"Invalid 'processor' in {yaml_path}. Expected 'module:ClassName', got {processor!r}")

        tasks = cfg.get("tasks")
        if not isinstance(tasks, dict) or not tasks:
            raise ValueError(f"Invalid 'tasks' mapping in {yaml_path}.")

        for task_name, task_cfg in tasks.items():
            if not isinstance(task_name, str) or not task_name.strip():
                raise ValueError(f"Invalid task name in {yaml_path}: {task_name!r}")
            if not isinstance(task_cfg, dict):
                raise ValueError(f"Invalid task config for {task_name!r} in {yaml_path}.")

            splits = {}
            for k in ("train_data", "test_data"):
                v = task_cfg.get(k)
                if isinstance(v, str) and v.strip():
                    splits[k] = v.strip()
            if not splits:
                raise ValueError(f"No split paths for task {task_name!r} in {yaml_path}.")

            info = TaskInfo(task_name=task_name.strip(), family=family, processor_path=processor, splits=splits)
            if info.task_name in self._by_task:
                raise ValueError(f"Duplicate task_name {info.task_name!r}")
            self._by_task[info.task_name] = info

    def list_tasks(self) -> List[str]:
        return sorted(self._by_task.keys())

    def get(self, task_name: str) -> TaskInfo:
        if task_name in self._by_task:
            return self._by_task[task_name]
        raise KeyError(f"Unknown task {task_name!r}. Available: {', '.join(self.list_tasks())}")

    @staticmethod
    def load_processor(processor_path: str):
        mod_path, cls_name = processor_path.split(":", 1)
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)

    @staticmethod
    def resolve_repo_path(repo_root: str, p: str) -> str:
        if not isinstance(p, str):
            raise TypeError(f"Path must be string, got: {type(p)}")
        if p.startswith("./"):
            return os.path.abspath(os.path.join(repo_root, p[2:]))
        if not os.path.isabs(p):
            return os.path.abspath(os.path.join(repo_root, p))
        return p

    def _has_builtin_split(self, info: TaskInfo) -> bool:
        if "train_data" not in info.splits or "test_data" not in info.splits:
            return False
        train_path = self.resolve_repo_path(self.repo_root, info.splits["train_data"])
        test_path = self.resolve_repo_path(self.repo_root, info.splits["test_data"])
        return train_path != test_path

    def load_splits(
        self,
        task_name: str,
        load_jsonl_fn: Callable[[str], List[Dict[str, Any]]],
        max_benchmark_items: int,
        split_ratio: List[float],
        seed: int = 42,
        benchmark_mode: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Any]:
        """
        Two-way data split: train and test.

        split_ratio = [train_frac, test_frac] (e.g. [0.8, 0.2])

        If the task has separate train_data and test_data files, use those.
        Otherwise, split the single dataset by ratio.

        benchmark_mode: if True, test set is the full dataset (no sampling).

        Returns:
            (train_samples, test_samples, processor)
        """
        assert len(split_ratio) == 2, f"split_ratio must have 2 elements, got {len(split_ratio)}"
        assert abs(sum(split_ratio) - 1.0) < 1e-6, f"split_ratio must sum to 1.0, got {sum(split_ratio)}"

        train_frac, test_frac = split_ratio

        info = self.get(task_name)
        processor_cls = self.load_processor(info.processor_path)
        processor = processor_cls(task_name=task_name)
        rng = random.Random(seed)

        def load_split(key: str) -> List[Dict[str, Any]]:
            if key not in info.splits:
                raise KeyError(f"Task {task_name!r} missing {key!r}")
            path = self.resolve_repo_path(self.repo_root, info.splits[key])
            return processor.process_task_data(load_jsonl_fn(path))

        train_cap = int(train_frac * max_benchmark_items)
        test_cap = max_benchmark_items - train_cap

        has_builtin = self._has_builtin_split(info)

        if has_builtin:
            train_samples = load_split("train_data")
            test_samples = load_split("test_data")
            if len(train_samples) > train_cap:
                train_samples = rng.sample(train_samples, train_cap)
            if len(test_samples) > test_cap:
                test_samples = rng.sample(test_samples, test_cap)
        else:
            load_key = "train_data" if "train_data" in info.splits else "test_data"
            raw = load_split(load_key)
            if len(raw) > max_benchmark_items:
                raw = rng.sample(raw, max_benchmark_items)
            n = len(raw)
            n_train = int(n * train_frac)
            rng.shuffle(raw)
            train_samples = raw[:n_train]
            test_samples = raw[n_train:]
            if len(train_samples) > train_cap:
                train_samples = rng.sample(train_samples, train_cap)

        if benchmark_mode:
            if has_builtin:
                test_samples = load_split("test_data")
            else:
                load_key = "train_data" if "train_data" in info.splits else "test_data"
                test_samples = load_split(load_key)

        return train_samples, test_samples, processor
