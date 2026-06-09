from __future__ import annotations

import json
import math
import random
import yaml
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

from infer import Generator, Router
from utils.artifacts import write_dev_manifest, write_jsonl
from evolve.agents import PromptAgents, RouterAgents, build_evolve_agents
from evolve.stage import ConvergenceStage, ExplorationStage, InitializationStage
from evolve.utils.cache import EvaluationCache
from evolve.utils.eval_engine import evaluate_prompt
from evolve.utils.runtime import (
    build_slot_labels,
    build_visible_text,
    dataset_eval_from_maps,
    group_samples_by_slot,
)
from evolve.unit import PromptEvolve, RouterEvolve
from evolve.utils.structures import DatasetEval, EEVEECandidate, stable_hash_text
from tasks.registry import TaskRegistry
from utils import (
    RunLogger,
    create_client,
    get_api_params_for_model,
    load_api_params_config,
    load_jsonl,
    make_qid,
    parse_model_spec,
    split_train_val_examples,
)


DEFAULT_ROUTER_PROMPT = (
    "Choose the slot whose prompt is most likely to help solve the task. "
    "Use prompt intent and output constraints, not keyword overlap."
)


@dataclass
class PipelineDependencies:
    generator: Generator
    router: Router
    prompt_agents: PromptAgents
    router_agents: RouterAgents
    llm_judge: Any = None


class EEVEEPipeline:
    def __init__(self, config: Dict[str, Any], dependencies: Optional[PipelineDependencies] = None):
        self.config = config
        self.repo_root = Path(__file__).parents[1].resolve()
        base_exp_name = str(config.get("experiment_name", "eevee_run"))
        timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"{timestamp_prefix}_{base_exp_name}"
        self.save_path = Path(config.get("save_path", "results"))
        self.exp_dir = self.save_path / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self._persist_run_config()

        api_cfg_path = config.get("api_params_config")
        self.api_params_config = load_api_params_config(api_cfg_path) if api_cfg_path else load_api_params_config()

        self.initialization_cfg = dict(config.get("initialization", {}) or {})
        self.router_cfg = dict(config.get("router", {}) or {})
        self.prompt_cfg = dict(config.get("prompt", {}) or {})
        self.random_cfg = dict(config.get("random", {}) or {})
        self.execution_cfg = dict(config.get("execution", {}) or {})
        self.exploration_cfg = dict(config.get("exploration", {}) or {})
        self.convergence_cfg = dict(config.get("convergence", {}) or {})

        self.slot_count = int(config.get("prompt_set_size", 4))
        self.seed = int(self.random_cfg.get("seed", self.config.get("data", {}).get("seed", 42)))
        self.eval_workers = int(self.execution_cfg.get("eval_workers", 20))
        self.router_workers = int(self.execution_cfg.get("router_workers", self.eval_workers))
        self.test_workers = int(self.execution_cfg.get("test_workers", self.eval_workers))
        self.test_repeats = max(1, int(self.execution_cfg.get("test_repeats", 1)))
        self.max_parallel_slots = int(self.prompt_cfg.get("max_parallel_slots", self.slot_count))
        self.prompt_minibatch_size = int(self.prompt_cfg.get("minibatch_size", 16))
        self.initialization_budget = int(self.initialization_cfg.get("prompt_evolve_budget", 8))
        self.router_max_analysis_cases = int(self.router_cfg.get("max_analysis_cases", 8))
        self.prompt_temporary_pool_max_size = self.prompt_cfg.get("temporary_pool_max_size")
        self.router_temporary_pool_max_size = self.router_cfg.get("temporary_pool_max_size")
        self.max_prompt_length = int(self.execution_cfg.get("max_prompt_length", 10000))
        self.example_limit = int(self.execution_cfg.get("max_examples", 6))
        self.total_mini_step_budget = int(self.exploration_cfg.get("total_mini_step_budget", 40))
        legacy_window_size = max(1, int(self.exploration_cfg.get("phase_window_size", 5)))
        self.router_window_size = max(1, int(self.exploration_cfg.get("router_window_size", legacy_window_size)))
        self.prompt_window_size = max(1, int(self.exploration_cfg.get("prompt_window_size", legacy_window_size)))
        self.phase_switch_epsilon = float(self.exploration_cfg.get("phase_switch_epsilon", 0.005))
        self.convergence_budget_per_slot = int(
            self.convergence_cfg.get("prompt_evolve_budget_per_slot", 0)
        )
        score_weights_cfg = dict(self.router_cfg.get("score_weights", {}) or {})
        self.router_score_weights = {
            "accuracy": float(score_weights_cfg.get("accuracy", 0.6)),
            "consistency": float(score_weights_cfg.get("consistency", 0.2)),
            "balance": float(score_weights_cfg.get("balance", 0.2)),
        }
        final_score_weights_cfg = dict(self.router_cfg.get("final_score_weights", {}) or {})
        self.router_final_score_weights = {
            "accuracy": float(final_score_weights_cfg.get("accuracy", 1.0)),
            "consistency": float(final_score_weights_cfg.get("consistency", 0.0)),
            "balance": float(final_score_weights_cfg.get("balance", 0.0)),
        }
        consistency_cfg = dict(self.router_cfg.get("consistency", {}) or {})
        self.consistency_component_weights = {
            "compact": float(consistency_cfg.get("compact_weight", 0.5)),
            "separate": float(consistency_cfg.get("separate_weight", 0.5)),
        }
        balance_cfg = dict(self.router_cfg.get("balance", {}) or {})
        self.balance_component_weights = {
            "use": float(balance_cfg.get("use_weight", 0.5)),
            "distribution": float(balance_cfg.get("distribution_weight", 0.5)),
        }
        self.eval_cache = EvaluationCache(self.exp_dir / "eval_cache.json")
        self._eval_cache_lock = Lock()
        self._rng = random.Random(self.seed)
        self.rlog: Optional[RunLogger] = None

        self.dependencies = dependencies or self._init_dependencies(dict(config.get("models", {}) or {}))
        self.generator = self.dependencies.generator
        self.router = self.dependencies.router
        self.prompt_agents = self.dependencies.prompt_agents
        self.router_agents = self.dependencies.router_agents
        self.llm_judge = self.dependencies.llm_judge

    def _persist_run_config(self) -> None:
        config_path = self.exp_dir / "run_config.yaml"
        config_path.write_text(
            yaml.safe_dump(self.config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def _slot_label(self, slot_id: int) -> str:
        labels = build_slot_labels(self.slot_count)
        idx = int(slot_id)
        if not (0 <= idx < len(labels)):
            raise ValueError(f"slot_id_out_of_range:{slot_id}")
        return labels[idx]

    def _log_line(self, text: str) -> None:
        if self.rlog is not None:
            self.rlog.line(text)

    def _init_dependencies(self, models: Dict[str, str]) -> PipelineDependencies:
        def make_client(*keys: str, default: str = "openrouter:deepseek/deepseek-chat-v3.1"):
            spec = default
            for key in keys:
                if key and models.get(key):
                    spec = str(models[key])
                    break
            provider, raw_model = parse_model_spec(spec)
            client, client_model = create_client(provider, raw_model)
            params = get_api_params_for_model(provider, raw_model, self.api_params_config)
            return client, provider, client_model, params

        generator_client, generator_provider, generator_model, generator_params = make_client("generator")
        router_client, router_provider, router_model, router_params = make_client(
            "router",
            "generator",
        )

        generator = Generator(
            generator_client,
            generator_provider,
            generator_model,
            generator_params,
            max_retry=int(self.execution_cfg.get("max_retry", 3)),
        )
        router = Router(
            api_client=router_client,
            api_provider=router_provider,
            model=router_model,
            api_params=router_params,
            max_retry=int(self.execution_cfg.get("max_retry", 3)),
        )
        prompt_agents, router_agents = build_evolve_agents(
            client_factory=make_client,
            prompts_dir=self.repo_root / "prompts",
            max_prompt_length=self.max_prompt_length,
            max_retry=int(self.execution_cfg.get("max_retry", 3)),
        )

        llm_judge = None
        if "judge" in models:
            judge_client, judge_provider, judge_model, judge_params = make_client("judge")
            from tasks.judge_utils import LLMJudge

            llm_judge = LLMJudge(
                api_provider=judge_provider,
                model=judge_model,
                api_params=judge_params,
                client=judge_client,
                client_model=judge_model,
            )

        return PipelineDependencies(
            generator=generator,
            router=router,
            prompt_agents=prompt_agents,
            router_agents=router_agents,
            llm_judge=llm_judge,
        )

    def run(self) -> None:
        log_dir = None
        if bool(self.config.get("logging", {}).get("llm_call_log", False)):
            logs_dir = self.exp_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_dir = str(logs_dir)

        self.rlog = RunLogger(str(self.exp_dir / "run_log.txt"))
        events_path = self.exp_dir / "events.jsonl"
        wandb_run = self._init_wandb()

        def log_event(event: Dict[str, Any]) -> None:
            with open(events_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

        train_pool, test_by_task, evaluators = self._load_data(log_event)
        if not train_pool:
            self._log_line("No train data loaded; exiting.")
            return

        train_samples, val_samples = self._split_train_val_from_pool(train_pool)
        if bool(self.config.get("data", {}).get("exclude_train_val_from_test", False)):
            test_by_task, removed_by_task = self._exclude_train_val_from_test(
                train_samples=train_samples,
                val_samples=val_samples,
                test_by_task=test_by_task,
            )
            removed_total = sum(removed_by_task.values())
            self._log_line(
                f"Excluded train/val overlap from test: total_removed={removed_total} "
                f"removed_by_task={removed_by_task}"
            )
            log_event(
                {
                    "type": "test_overlap_filter_done",
                    "removed_total": removed_total,
                    "removed_by_task": removed_by_task,
                    "test_total_after_filter": sum(len(samples) for samples in test_by_task.values()),
                }
            )
        self.train_by_qid = {str(sample["_qid"]): sample for sample in train_samples}
        self.val_by_qid = {str(sample["_qid"]): sample for sample in val_samples}
        self.train_counts = dict(Counter(sample["_task_name"] for sample in train_samples))
        self.val_counts = dict(Counter(sample["_task_name"] for sample in val_samples))
        self.test_counts = {task_name: len(samples) for task_name, samples in test_by_task.items()}
        self.rlog.log_data_loaded(self.train_counts, self.val_counts, self.test_counts)

        eval_fn = self._make_eval_fn(evaluators)
        self.rlog.section("Initial Test")
        initial_test = self._evaluate_single_prompt_on_test(
            prompt_text="",
            test_by_task=test_by_task,
            evaluators=evaluators,
            label_prefix="initial_empty",
            log_dir=log_dir,
        )
        self.rlog.log_phase_results("Initial Empty", initial_test)
        (self.exp_dir / "initial_test_results.json").write_text(
            json.dumps(initial_test, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        empty_train_eval = self._evaluate_prompt_cached(
            prompt_text="",
            samples=train_samples,
            eval_fn=eval_fn,
            split_name="train",
            label="empty_train",
            log_dir=log_dir,
            results_dir=self.exp_dir / "empty_eval" / "train",
        )
        empty_val_eval = self._evaluate_prompt_cached(
            prompt_text="",
            samples=val_samples,
            eval_fn=eval_fn,
            split_name="val",
            label="empty_val",
            log_dir=log_dir,
            results_dir=self.exp_dir / "empty_eval" / "val",
        )
        self.rlog.log_empty_baseline(
            train_score=empty_train_eval.score,
            train_correct=empty_train_eval.correct,
            train_total=empty_train_eval.total,
            val_score=empty_val_eval.score,
            val_correct=empty_val_eval.correct,
            val_total=empty_val_eval.total,
        )

        prompt_evolve = PromptEvolve(self)
        router_evolve = RouterEvolve(self)

        initialization_candidates = InitializationStage(self, prompt_evolve).run(
            train_samples=train_samples,
            val_samples=val_samples,
            eval_fn=eval_fn,
            log_dir=log_dir,
            log_event=log_event,
        )
        current_router_prompt = DEFAULT_ROUTER_PROMPT
        current_prompt_set = [candidate.prompt_text for candidate in initialization_candidates]

        current_candidate = self._build_candidate_from_router_and_prompt_set(
            router_prompt=current_router_prompt,
            prompt_set=current_prompt_set,
            train_samples=train_samples,
            val_samples=val_samples,
            eval_fn=eval_fn,
            log_dir=log_dir,
            label_prefix="initial_prompt_set",
        )
        self._persist_best_candidate(current_candidate)
        self.rlog.log_candidate("initialization candidate", current_candidate)

        current_router_prompt, current_prompt_set, best_candidate = ExplorationStage(
            self,
            router_evolve,
            prompt_evolve,
        ).run(
            router_prompt=current_router_prompt,
            prompt_set=current_prompt_set,
            train_samples=train_samples,
            val_samples=val_samples,
            empty_val_eval=empty_val_eval,
            eval_fn=eval_fn,
            log_dir=log_dir,
        )

        if self.convergence_budget_per_slot > 0:
            self.rlog.section("convergence")
            convergence_prompt_set = ConvergenceStage(self, prompt_evolve).run(
                router_prompt=current_router_prompt,
                prompt_set=current_prompt_set,
                train_samples=train_samples,
                val_samples=val_samples,
                eval_fn=eval_fn,
                log_dir=log_dir,
            )
            best_candidate = self._build_candidate_from_router_and_prompt_set(
                router_prompt=current_router_prompt,
                prompt_set=convergence_prompt_set,
                train_samples=train_samples,
                val_samples=val_samples,
                eval_fn=eval_fn,
                log_dir=log_dir,
                label_prefix="convergence_prompt_set",
            )
            self._persist_best_candidate(best_candidate)
            self.rlog.log_candidate("convergence candidate", best_candidate)

        final_results = self._evaluate_best_candidate_on_test(
            best_candidate=best_candidate,
            test_by_task=test_by_task,
            evaluators=evaluators,
            log_dir=log_dir,
        )
        self.rlog.section("Final Test")
        self.rlog.log_phase_results("Best Candidate", final_results)
        (self.exp_dir / "final_test_results.json").write_text(
            json.dumps(final_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_dev_manifest(
            exp_dir=self.exp_dir,
            best_candidate_path=self.exp_dir / "best_candidate.json",
        )
        self.eval_cache.save()

        if wandb_run:
            try:
                wandb_run.finish()
            except Exception:
                pass

    def _make_eval_fn(self, evaluators: Dict[str, Any]):
        def eval_fn(predicted, ground_truth, task_dict, **kwargs):
            task_name = str(task_dict["_task_name"])
            evaluator = evaluators[task_name]
            return evaluator(
                predicted=predicted,
                ground_truth=ground_truth,
                task_dict=task_dict,
                question=kwargs.get("question"),
                context=kwargs.get("context"),
                log_dir=kwargs.get("log_dir"),
                call_id=kwargs.get("call_id"),
            )

        return eval_fn

    @staticmethod
    def _sample_content_key(sample: Dict[str, Any]) -> str:
        payload = {
            "task_name": str(sample.get("_task_name", "")),
            "context": str(sample.get("context", "") or ""),
            "question": str(sample.get("question", "") or ""),
            "target": str(sample.get("target", "") or ""),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _exclude_train_val_from_test(
        self,
        *,
        train_samples: Sequence[Dict[str, Any]],
        val_samples: Sequence[Dict[str, Any]],
        test_by_task: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
        seen = {
            self._sample_content_key(sample)
            for sample in list(train_samples) + list(val_samples)
        }
        filtered: Dict[str, List[Dict[str, Any]]] = {}
        removed_by_task: Dict[str, int] = {}
        for task_name, task_samples in test_by_task.items():
            kept = [
                sample
                for sample in task_samples
                if self._sample_content_key(sample) not in seen
            ]
            filtered[task_name] = kept
            removed_by_task[task_name] = len(task_samples) - len(kept)
        return filtered, removed_by_task

    def _load_data(self, log_event) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        from tasks.judge_utils import create_evaluator

        data_cfg = self.config.get("data", {})
        max_items = int(data_cfg.get("max_benchmark_items", 500))
        split_ratio = data_cfg.get("split_ratio", [0.8, 0.2])
        seed = int(data_cfg.get("seed", 42))
        benchmark_mode = bool(data_cfg.get("benchmark_mode", False))

        reg = TaskRegistry(str(self.repo_root)).discover()
        tasks_cfg = self.config.get("tasks", [])
        if not tasks_cfg:
            return [], {}, {}

        all_train: List[Dict[str, Any]] = []
        test_by_task: Dict[str, List[Dict[str, Any]]] = {}
        evaluators: Dict[str, Any] = {}
        task_cfg_by_name = {task.get("name", ""): task for task in tasks_cfg if task.get("name")}

        for task_spec in tasks_cfg:
            task_name = str(task_spec.get("name", "")).strip()
            if not task_name:
                continue
            task_mode = str(task_spec.get("mode", "train")).strip().lower()
            if task_mode == "eval":
                task_train, task_test, task_processor = reg.load_splits(
                    task_name,
                    load_jsonl,
                    max_items,
                    [0.0, 1.0],
                    seed,
                    benchmark_mode=benchmark_mode,
                )
            else:
                task_train, task_test, task_processor = reg.load_splits(
                    task_name,
                    load_jsonl,
                    max_items,
                    split_ratio,
                    seed,
                    benchmark_mode=benchmark_mode,
                )
            configure_processor = getattr(task_processor, "configure", None)
            if callable(configure_processor):
                processor_config_key = str(getattr(task_processor, "task_name", task_name))
                configure_processor(
                    global_config=dict(self.config.get(processor_config_key, {}) or self.config.get("appworld", {}) or {}),
                    task_config=task_spec,
                )
            for idx, sample in enumerate(task_train):
                sample["_task_name"] = task_name
                sample["_qid"] = make_qid(task_name, idx, sample.get("question", ""))
            for idx, sample in enumerate(task_test):
                sample["_task_name"] = task_name
                sample["_qid"] = make_qid(task_name, idx, sample.get("question", ""))

            all_train.extend(task_train)
            test_by_task[task_name] = task_test

            judge_mode = str(task_cfg_by_name.get(task_name, {}).get("judge_mode", "rule")).strip().lower()
            if judge_mode not in ("rule", "llm"):
                judge_mode = "rule"
            evaluators[task_name] = create_evaluator(
                data_processor=task_processor,
                judge_mode=judge_mode,
                llm_judge=self.llm_judge if judge_mode == "llm" else None,
            )

        log_event(
            {
                "type": "phase0_done",
                "train_pool_total": len(all_train),
                "test_total": sum(len(samples) for samples in test_by_task.values()),
                "tasks": list(test_by_task.keys()),
            }
        )
        return all_train, test_by_task, evaluators

    def _split_train_val_from_pool(self, train_pool: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        val_ratio = float(self.config.get("data", {}).get("val_ratio_from_train", 0.25))
        seed = int(self.config.get("data", {}).get("seed", 42))
        by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sample in train_pool:
            by_task[str(sample["_task_name"])].append(sample)

        train_samples: List[Dict[str, Any]] = []
        val_samples: List[Dict[str, Any]] = []
        for task_name in sorted(by_task):
            task_train, task_val = split_train_val_examples(by_task[task_name], val_ratio, seed)
            train_samples.extend(task_train)
            val_samples.extend(task_val)
        train_samples.sort(key=lambda item: str(item["_qid"]))
        val_samples.sort(key=lambda item: str(item["_qid"]))
        return train_samples, val_samples

    def _partition_samples_evenly(
        self,
        samples: Sequence[Dict[str, Any]],
        *,
        seed_offset: int,
    ) -> Dict[int, List[Dict[str, Any]]]:
        groups: Dict[int, List[Dict[str, Any]]] = {slot_id: [] for slot_id in range(self.slot_count)}
        ordered = list(samples)
        ordered.sort(key=lambda item: str(item.get("_qid", "")))
        rng = random.Random(self.seed + seed_offset)
        rng.shuffle(ordered)
        for idx, sample in enumerate(ordered):
            groups[idx % self.slot_count].append(sample)
        return groups

    def _evaluate_prompt_cached(
        self,
        *,
        prompt_text: str,
        samples: Sequence[Dict[str, Any]],
        eval_fn,
        split_name: str,
        label: str,
        log_dir: Optional[str],
        results_dir: Optional[Path] = None,
    ) -> DatasetEval:
        if not samples:
            return dataset_eval_from_maps(split_name=split_name, correctness_by_qid={}, answers_by_qid={})

        artifact_hash = stable_hash_text(prompt_text)
        qids = [str(sample["_qid"]) for sample in samples]
        with self._eval_cache_lock:
            cached = self.eval_cache.get_many(artifact_hash, split_name, qids)
        missing_samples = [sample for sample in samples if str(sample["_qid"]) not in cached]
        if missing_samples:
            raw_eval = evaluate_prompt(
                self.generator,
                prompt_text,
                list(missing_samples),
                eval_fn,
                max_workers=self.eval_workers,
                results_dir=str(results_dir) if results_dir is not None else None,
                candidate_id=None,
                log_dir=log_dir,
                label=label,
            )
            with self._eval_cache_lock:
                for row in raw_eval.get("all_results", []):
                    self.eval_cache.put(
                        artifact_hash,
                        split_name,
                        str(row.get("qid", "")),
                        answer=str(row.get("answer", "") or ""),
                        is_correct=bool(row.get("is_correct", False)),
                    )

        correctness_by_qid: Dict[str, bool] = {}
        answers_by_qid: Dict[str, str] = {}
        with self._eval_cache_lock:
            for sample in samples:
                qid = str(sample["_qid"])
                cached_eval = self.eval_cache.get(artifact_hash, split_name, qid)
                if cached_eval is None:
                    raise KeyError(f"Missing cached eval for {split_name}:{qid}")
                correctness_by_qid[qid] = bool(cached_eval.is_correct)
                answers_by_qid[qid] = str(cached_eval.answer)
        return dataset_eval_from_maps(
            split_name=split_name,
            correctness_by_qid=correctness_by_qid,
            answers_by_qid=answers_by_qid,
        )

    @staticmethod
    def _score_std(scores: Sequence[float]) -> float:
        if len(scores) <= 1:
            return 0.0
        mean = sum(scores) / len(scores)
        return math.sqrt(sum((score - mean) ** 2 for score in scores) / len(scores))

    @staticmethod
    def _repeat_label(label: str, repeat_idx: int, repeats: int) -> str:
        if repeats <= 1:
            return label
        return f"{label}_repeat_{repeat_idx:02d}"

    def _repeated_test_result(self, *, corrects: List[int], totals: List[int]) -> Dict[str, Any]:
        scores = [
            (correct / total) if total else 0.0
            for correct, total in zip(corrects, totals)
        ]
        return {
            "accuracy": sum(scores) / len(scores) if scores else 0.0,
            "correct": sum(corrects),
            "total": sum(totals),
            "example_total": totals[0] if totals else 0,
            "accuracy_std": self._score_std(scores),
            "repeats": len(scores),
            "repeat_scores": scores,
            "repeat_correct": corrects,
            "repeat_total": totals,
        }

    def _evaluate_single_prompt_on_test(
        self,
        *,
        prompt_text: str,
        test_by_task: Dict[str, List[Dict[str, Any]]],
        evaluators: Dict[str, Any],
        label_prefix: str,
        log_dir: Optional[str],
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for task_name, task_samples in sorted(test_by_task.items()):
            if not task_samples:
                continue
            repeat_corrects: List[int] = []
            repeat_totals: List[int] = []
            for repeat_idx in range(1, self.test_repeats + 1):
                raw_eval = evaluate_prompt(
                    self.generator,
                    prompt_text,
                    list(task_samples),
                    evaluators[task_name],
                    max_workers=self.test_workers,
                    results_dir=str(self.exp_dir / "tests" / label_prefix / task_name),
                    candidate_id=0,
                    log_dir=log_dir,
                    label=self._repeat_label(f"{label_prefix}_{task_name}", repeat_idx, self.test_repeats),
                )
                repeat_corrects.append(int(raw_eval["correct"]))
                repeat_totals.append(int(raw_eval["total"]))
            out[task_name] = self._repeated_test_result(corrects=repeat_corrects, totals=repeat_totals)
        return out

    def _route_samples(
        self,
        *,
        router_prompt: str,
        prompt_set: Sequence[str],
        samples: Sequence[Dict[str, Any]],
        call_prefix: str,
        log_dir: Optional[str],
        output_path: Optional[Path] = None,
    ) -> Dict[str, int]:
        display_slot_labels = build_slot_labels(len(prompt_set))
        route_by_qid: Dict[str, int] = {}
        rows: List[Dict[str, Any]] = []

        def work(sample: Dict[str, Any]) -> Tuple[str, int, str]:
            qid = str(sample["_qid"])
            decision = self.router.route(
                router_prompt=router_prompt,
                display_slot_labels=display_slot_labels,
                prompt_set=prompt_set,
                visible_text=build_visible_text(sample),
                call_id=f"{call_prefix}_{qid}",
                log_dir=log_dir,
            )
            return qid, int(decision.slot_id), str(decision.reason)

        with ThreadPoolExecutor(max_workers=max(1, self.router_workers)) as executor:
            futures = [executor.submit(work, sample) for sample in samples]
            for future in as_completed(futures):
                qid, slot_id, reason = future.result()
                route_by_qid[qid] = slot_id
                rows.append(
                    {
                        "qid": qid,
                        "slot_id": slot_id,
                        "label": display_slot_labels[slot_id],
                        "reason": reason,
                    }
                )

        if output_path is not None:
            write_jsonl(output_path, rows)
        return route_by_qid

    def _compute_full_prompt_evals(
        self,
        *,
        prompt_set: Sequence[str],
        train_samples: Sequence[Dict[str, Any]],
        val_samples: Sequence[Dict[str, Any]],
        eval_fn,
        log_dir: Optional[str],
        label_prefix: str,
        output_dir: Path,
    ) -> Tuple[Dict[int, DatasetEval], Dict[int, DatasetEval]]:
        train_evals: Dict[int, DatasetEval] = {}
        val_evals: Dict[int, DatasetEval] = {}
        for slot_id, prompt_text in enumerate(prompt_set):
            slot_label = self._slot_label(slot_id)
            train_evals[slot_id] = self._evaluate_prompt_cached(
                prompt_text=prompt_text,
                samples=train_samples,
                eval_fn=eval_fn,
                split_name="train",
                label=f"{label_prefix}_{slot_label}_full_train",
                log_dir=log_dir,
                results_dir=output_dir / "prompt_full" / slot_label / "train",
            )
            val_evals[slot_id] = self._evaluate_prompt_cached(
                prompt_text=prompt_text,
                samples=val_samples,
                eval_fn=eval_fn,
                split_name="val",
                label=f"{label_prefix}_{slot_label}_full_val",
                log_dir=log_dir,
                results_dir=output_dir / "prompt_full" / slot_label / "val",
            )
        return train_evals, val_evals

    def _build_candidate_from_router_and_prompt_set(
        self,
        *,
        router_prompt: str,
        prompt_set: Sequence[str],
        train_samples: Sequence[Dict[str, Any]],
        val_samples: Sequence[Dict[str, Any]],
        eval_fn,
        log_dir: Optional[str],
        label_prefix: str,
    ) -> EEVEECandidate:
        train_route_by_qid = self._route_samples(
            router_prompt=router_prompt,
            prompt_set=prompt_set,
            samples=train_samples,
            call_prefix=f"{label_prefix}_train_route",
            log_dir=log_dir,
            output_path=self.exp_dir / "artifacts" / f"{label_prefix}_train_routes.jsonl",
        )
        val_route_by_qid = self._route_samples(
            router_prompt=router_prompt,
            prompt_set=prompt_set,
            samples=val_samples,
            call_prefix=f"{label_prefix}_val_route",
            log_dir=log_dir,
            output_path=self.exp_dir / "artifacts" / f"{label_prefix}_val_routes.jsonl",
        )
        train_groups = group_samples_by_slot(train_route_by_qid, train_samples, self.slot_count)
        val_groups = group_samples_by_slot(val_route_by_qid, val_samples, self.slot_count)
        train_correctness_by_qid: Dict[str, bool] = {str(sample["_qid"]): False for sample in train_samples}
        train_answers_by_qid: Dict[str, str] = {str(sample["_qid"]): "" for sample in train_samples}
        val_correctness_by_qid: Dict[str, bool] = {str(sample["_qid"]): False for sample in val_samples}
        val_answers_by_qid: Dict[str, str] = {str(sample["_qid"]): "" for sample in val_samples}
        for slot_id, prompt_text in enumerate(prompt_set):
            slot_label = self._slot_label(slot_id)
            if train_groups.get(slot_id):
                slot_train_eval = self._evaluate_prompt_cached(
                    prompt_text=prompt_text,
                    samples=train_groups[slot_id],
                    eval_fn=eval_fn,
                    split_name="train",
                    label=f"{label_prefix}_{slot_label}_train",
                    log_dir=log_dir,
                    results_dir=self.exp_dir / "artifacts" / label_prefix / slot_label / "train",
                )
                train_correctness_by_qid.update(slot_train_eval.correctness_by_qid)
                train_answers_by_qid.update(slot_train_eval.answers_by_qid)
            if val_groups.get(slot_id):
                slot_val_eval = self._evaluate_prompt_cached(
                    prompt_text=prompt_text,
                    samples=val_groups[slot_id],
                    eval_fn=eval_fn,
                    split_name="val",
                    label=f"{label_prefix}_{slot_label}_val",
                    log_dir=log_dir,
                    results_dir=self.exp_dir / "artifacts" / label_prefix / slot_label / "val",
                )
                val_correctness_by_qid.update(slot_val_eval.correctness_by_qid)
                val_answers_by_qid.update(slot_val_eval.answers_by_qid)
        train_eval = dataset_eval_from_maps(
            split_name="train",
            correctness_by_qid=train_correctness_by_qid,
            answers_by_qid=train_answers_by_qid,
        )
        val_eval = dataset_eval_from_maps(
            split_name="val",
            correctness_by_qid=val_correctness_by_qid,
            answers_by_qid=val_answers_by_qid,
        )
        return EEVEECandidate(
            router_prompt=router_prompt,
            prompt_set=list(prompt_set),
            prompt_set_ids=[f"{self._slot_label(slot_id)}:{stable_hash_text(prompt_text)[:8]}" for slot_id, prompt_text in enumerate(prompt_set)],
            train_eval=train_eval,
            val_eval=val_eval,
            train_route_by_qid=train_route_by_qid,
            val_route_by_qid=val_route_by_qid,
            parent_candidate_ids=[],
            metadata={},
        )

    def _evaluate_best_candidate_on_test(
        self,
        *,
        best_candidate: EEVEECandidate,
        test_by_task: Dict[str, List[Dict[str, Any]]],
        evaluators: Dict[str, Any],
        log_dir: Optional[str],
    ) -> Dict[str, Dict[str, Any]]:
        per_task: Dict[str, Dict[str, Any]] = {}
        final_routes: List[Dict[str, Any]] = []
        slot_labels = build_slot_labels(len(best_candidate.prompt_set))
        final_routed_test: Dict[int, List[Dict[str, Any]]] = {slot_id: [] for slot_id in range(self.slot_count)}

        for task_name, task_samples in sorted(test_by_task.items()):
            if not task_samples:
                continue
            repeat_corrects: List[int] = []
            repeat_totals: List[int] = []
            for repeat_idx in range(1, self.test_repeats + 1):
                repeat_suffix = "" if self.test_repeats <= 1 else f"_repeat_{repeat_idx:02d}"
                route_by_qid = self._route_samples(
                    router_prompt=best_candidate.router_prompt,
                    prompt_set=best_candidate.prompt_set,
                    samples=task_samples,
                    call_prefix=f"final_test{repeat_suffix}_{task_name}",
                    log_dir=log_dir,
                )
                for qid, slot_id in route_by_qid.items():
                    route_row = {
                        "task_name": task_name,
                        "qid": qid,
                        "label": slot_labels[slot_id],
                    }
                    if self.test_repeats > 1:
                        route_row["repeat"] = repeat_idx
                    final_routes.append(route_row)

                grouped = group_samples_by_slot(route_by_qid, task_samples, self.slot_count)
                for slot_id, slot_samples in grouped.items():
                    if slot_samples:
                        final_routed_test[slot_id].extend(slot_samples)
                task_correct = 0
                task_total = len(task_samples)
                for slot_id, slot_samples in grouped.items():
                    if not slot_samples:
                        continue
                    slot_label = self._slot_label(slot_id)
                    raw_eval = evaluate_prompt(
                        self.generator,
                        best_candidate.prompt_set[slot_id],
                        list(slot_samples),
                        evaluators[task_name],
                        max_workers=self.test_workers,
                        results_dir=str(self.exp_dir / "tests" / "final_best_candidate" / task_name / slot_label),
                        candidate_id=0,
                        log_dir=log_dir,
                        label=self._repeat_label(
                            f"final_best_candidate_{task_name}_{slot_label}",
                            repeat_idx,
                            self.test_repeats,
                        ),
                    )
                    task_correct += int(raw_eval["correct"])
                repeat_corrects.append(task_correct)
                repeat_totals.append(task_total)
            per_task[task_name] = self._repeated_test_result(
                corrects=repeat_corrects,
                totals=repeat_totals,
            )

        self.rlog.log_final_test_route_table(
            routed_test=final_routed_test,
            slot_count=self.slot_count,
        )
        write_jsonl(self.exp_dir / "final_test_routes.jsonl", final_routes)
        return per_task

    def _save_pool_snapshot(self, pool, path: Path) -> None:
        pool.save(path, serializer=lambda item: item.to_dict())

    def _persist_best_candidate(self, candidate: EEVEECandidate) -> None:
        (self.exp_dir / "best_candidate.json").write_text(
            json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _init_wandb(self):
        wandb_run = None
        if self.config.get("logging", {}).get("wandb", {}).get("enabled"):
            try:
                import wandb

                wandb_cfg = self.config["logging"]["wandb"]
                offline = wandb_cfg.get("offline", False)
                wandb_mode = "offline" if offline else "online"
                wandb_run = wandb.init(
                    project=wandb_cfg.get("project", "eevee"),
                    entity=wandb_cfg.get("entity") or None,
                    name=self.exp_name,
                    config=self.config,
                    mode=wandb_mode,
                )
            except Exception as exc:
                print(f"wandb init failed: {exc}")
        return wandb_run
