"""
Shared code_eval loading and runtime checks for code-generation tasks.

MBPP and HumanEval use HuggingFace's ``evaluate`` ``code_eval`` metric. That
metric internally combines threads and multiprocessing; when the upstream
``check_correctness`` implementation creates a ``multiprocessing.Manager`` from a
worker thread it can deadlock. We patch that code path to use a plain queue
instead.

This module also raises explicit dependency/configuration errors instead of
letting callers silently convert backend failures into wrong-answer judgments.
"""
from __future__ import annotations

import importlib.util
import multiprocessing
import os
import sys
import threading
import uuid
from typing import Any


class CodeEvalError(RuntimeError):
    """Base class for code_eval backend failures that must not be swallowed."""


class CodeEvalDependencyError(CodeEvalError):
    """Raised when the code_eval backend cannot be imported or initialized."""


class CodeEvalBackendError(CodeEvalError):
    """Raised when code_eval runtime execution fails unexpectedly."""


_COMMON_DEPENDENCIES = (
    ("evaluate", "evaluate"),
    ("sentence_transformers", "sentence-transformers"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
)


def _patch_execute_module(execute_mod: Any) -> None:
    """Replace Manager-based result passing with Queue-based passing."""
    unsafe_execute = execute_mod.unsafe_execute

    def check_correctness_fixed(check_program, timeout, task_id, completion_id):
        result_queue = multiprocessing.Queue()

        def wrapper():
            result_list = []
            try:
                unsafe_execute(check_program, result_list, timeout)
            except Exception:
                result_list.append("timed out")
            result_queue.put(result_list[0] if result_list else "timed out")

        process = multiprocessing.Process(target=wrapper, args=())
        process.start()
        process.join(timeout=timeout + 1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)

        try:
            result = result_queue.get(timeout=2)
        except Exception:
            result = "timed out"

        return {
            "task_id": task_id,
            "passed": result == "passed",
            "result": result,
            "completion_id": completion_id,
        }

    execute_mod.check_correctness = check_correctness_fixed


def _find_code_eval_cache_dir() -> str | None:
    """Return the cached code_eval directory containing ``code_eval.py``."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    base = os.path.join(
        hf_home, "modules", "evaluate_modules", "metrics", "evaluate-metric--code_eval"
    )
    if not os.path.isdir(base):
        return None

    for name in os.listdir(base):
        if name.startswith(".") or len(name) < 10:
            continue
        path = os.path.join(base, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "code_eval.py")):
            return path
    return None


def _missing_dependency_names() -> list[str]:
    return [
        display_name
        for module_name, display_name in _COMMON_DEPENDENCIES
        if importlib.util.find_spec(module_name) is None
    ]


def _build_backend_error_message(exc: BaseException) -> str:
    return (
        "Could not initialize HuggingFace evaluate/code_eval. "
        "This usually means the code judge dependencies are missing or incompatible "
        "(for example evaluate, sentence-transformers, transformers, datasets, or "
        f"huggingface_hub). Original error: {type(exc).__name__}: {exc}"
    )


def load_code_eval():
    """Load the code_eval metric and apply the thread-safety patch."""
    missing_dependencies = _missing_dependency_names()
    if missing_dependencies:
        raise CodeEvalDependencyError(
            "The code_eval judge requires additional packages that are not installed: "
            + ", ".join(missing_dependencies)
        )

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    hub_offline = os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            import evaluate as hf_evaluate
        except Exception as exc:  # noqa: BLE001 - convert env issues to a clear dependency error
            raise CodeEvalDependencyError(_build_backend_error_message(exc)) from exc

        local_dir = _find_code_eval_cache_dir()
        # Avoid shared-cache collisions on default_experiment-1-0.arrow (multi-user / multi-job).
        _experiment_id = f"eevee_code_eval_{os.getpid()}_{uuid.uuid4().hex}"
        try:
            if local_dir:
                script_path = os.path.join(local_dir, "code_eval.py")
                metric = hf_evaluate.load(script_path, experiment_id=_experiment_id)
            else:
                metric = hf_evaluate.load("code_eval", experiment_id=_experiment_id)
        except Exception as exc:  # noqa: BLE001 - surface dependency/version issues clearly
            raise CodeEvalDependencyError(_build_backend_error_message(exc)) from exc
    finally:
        if hub_offline is not None:
            os.environ["HF_HUB_OFFLINE"] = hub_offline
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

    execute_mod = None
    for module_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if "code_eval" not in str(module_name) and "evaluate_modules" not in str(module_name):
            continue
        if getattr(mod, "check_correctness", None) is not None and getattr(mod, "unsafe_execute", None) is not None:
            execute_mod = mod
            break
    if execute_mod is not None:
        _patch_execute_module(execute_mod)

    return metric


class _LazyCodeEval:
    """Lazy wrapper so importing task processors does not immediately import evaluate."""

    def __init__(self):
        self._metric = None
        self._compute_lock = threading.Lock()

    def _get_metric(self):
        if self._metric is None:
            self._metric = load_code_eval()
        return self._metric

    def compute(self, *args, **kwargs):
        with self._compute_lock:
            return self._get_metric().compute(*args, **kwargs)


def ensure_code_eval_ready() -> Any:
    """Eagerly validate that the code_eval backend can be loaded."""
    return compute_pass_at_k._get_metric()


compute_pass_at_k = _LazyCodeEval()
