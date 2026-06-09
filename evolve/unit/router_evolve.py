from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from evolve.pool.router_pool import RouterPool
from evolve.utils.evolution_utils import pick_mutation_step_winner
from evolve.utils.runtime import (
    build_router_analysis_cases,
    build_router_support_examples,
    build_slot_labels,
    sample_qids,
    slice_eval,
)
from evolve.utils.structures import DatasetEval


@dataclass
class RouterEvolveState:
    router_pool: RouterPool
    prompt_train_evals: Dict[int, DatasetEval]
    prompt_val_evals: Dict[int, DatasetEval]
    best_slot_targets: Dict[str, int]
    history: List[float] = field(default_factory=list)
    steps_in_phase: int = 0
    checkpoint_best: float = 0.0

    @property
    def baseline_router_score(self) -> float:
        return self.router_pool.baseline_score

    @property
    def best_candidate(self):
        return self.router_pool.best()


class RouterEvolve:
    """Single-router RouterEvolve unit."""

    def __init__(self, runtime: Any):
        self.runtime = runtime

    def score_weights_for_remaining(self, remaining_budget: int) -> Dict[str, float]:
        router_pool = RouterPool(self.runtime)
        return router_pool.score_weights_for_remaining(remaining_budget)

    def start(
        self,
        *,
        router_prompt: str,
        prompt_set: Sequence[str],
        train_samples: Sequence[Dict[str, Any]],
        val_samples: Sequence[Dict[str, Any]],
        eval_fn,
        log_dir: Optional[str],
        phase_tag: str,
        score_weights: Dict[str, float],
    ) -> RouterEvolveState:
        router_pool = RouterPool(
            self.runtime,
            max_size=(
                int(self.runtime.router_temporary_pool_max_size)
                if self.runtime.router_temporary_pool_max_size is not None
                else None
            ),
        )
        output_dir = self.runtime.exp_dir / "exploration" / phase_tag
        prompt_train_evals, prompt_val_evals = self.runtime._compute_full_prompt_evals(
            prompt_set=prompt_set,
            train_samples=train_samples,
            val_samples=val_samples,
            eval_fn=eval_fn,
            log_dir=log_dir,
            label_prefix=phase_tag,
            output_dir=output_dir,
        )
        seed_candidate = router_pool.evaluate_candidate(
            router_prompt=router_prompt,
            prompt_set=prompt_set,
            prompt_train_evals=prompt_train_evals,
            prompt_val_evals=prompt_val_evals,
            train_samples=train_samples,
            val_samples=val_samples,
            log_dir=log_dir,
            call_prefix=f"{phase_tag}_seed",
            source="seed",
            parent_candidate_ids=[],
            score_weights=score_weights,
        )
        router_pool.add_seed(seed_candidate)
        return RouterEvolveState(
            router_pool=router_pool,
            prompt_train_evals=prompt_train_evals,
            prompt_val_evals=prompt_val_evals,
            best_slot_targets=RouterPool.build_best_slot_targets(
                train_samples=train_samples,
                prompt_train_evals=prompt_train_evals,
                prompt_val_evals=prompt_val_evals,
            ),
            history=[seed_candidate.router_score],
            checkpoint_best=seed_candidate.router_score,
        )

    def run_step(
        self,
        *,
        state: RouterEvolveState,
        prompt_set: Sequence[str],
        train_samples: Sequence[Dict[str, Any]],
        val_samples: Sequence[Dict[str, Any]],
        log_dir: Optional[str],
        phase_tag: str,
        score_weights: Dict[str, float],
    ) -> RouterEvolveState:
        step_idx = state.steps_in_phase
        state.router_pool.apply_score_weights(score_weights)
        parent_candidate = state.router_pool.sample()
        labeled_qids = sorted(state.best_slot_targets)
        if labeled_qids:
            minibatch_qids = sample_qids(
                labeled_qids,
                self.runtime.prompt_minibatch_size,
                random.Random(self.runtime.seed + 1700 + step_idx),
            )
            samples_by_qid = {str(sample["_qid"]): sample for sample in train_samples}
            minibatch_samples = [samples_by_qid[qid] for qid in minibatch_qids if qid in samples_by_qid]
        else:
            minibatch_samples = []
            minibatch_qids = []

        preview_prompt_train_evals = {
            slot_id: slice_eval(dataset_eval, minibatch_qids)
            for slot_id, dataset_eval in state.prompt_train_evals.items()
        }
        ref_candidates = state.router_pool.reference_candidates(
            parent_candidate=parent_candidate,
            limit=2,
        )
        display_slot_labels = build_slot_labels(len(prompt_set))
        support_examples = build_router_support_examples(
            samples_by_qid={str(sample["_qid"]): sample for sample in train_samples},
            route_targets={qid: state.best_slot_targets[qid] for qid in minibatch_qids if qid in state.best_slot_targets},
            display_slot_labels=display_slot_labels,
            max_examples=self.runtime.example_limit,
        )

        mutation_router = self.runtime.router_agents.mutation.run(
            call_id=f"{phase_tag}_mutation_{step_idx}",
            log_dir=log_dir,
            router_a=ref_candidates[0].router_prompt if len(ref_candidates) >= 1 else "",
            router_b=ref_candidates[1].router_prompt if len(ref_candidates) >= 2 else "",
            slot_entries=[
                {
                    "slot_label": label,
                    "prompt_text": str(prompt_set[slot_id]),
                }
                for slot_id, label in enumerate(display_slot_labels)
            ],
            support_examples=support_examples,
        )
        mutation_preview = state.router_pool.evaluate_candidate(
            router_prompt=mutation_router,
            prompt_set=prompt_set,
            prompt_train_evals=preview_prompt_train_evals,
            prompt_val_evals=preview_prompt_train_evals,
            train_samples=minibatch_samples,
            val_samples=minibatch_samples,
            log_dir=log_dir,
            call_prefix=f"{phase_tag}_mutation_preview_{step_idx}",
            source="mutation",
            parent_candidate_ids=[
                candidate_id
                for candidate_id in [parent_candidate.candidate_id, *[candidate.candidate_id for candidate in ref_candidates]]
                if candidate_id is not None
            ],
            score_weights=score_weights,
        )

        prompt_val_scores = {slot_id: dataset_eval.score for slot_id, dataset_eval in state.prompt_val_evals.items()}
        analysis_cases = build_router_analysis_cases(
            train_samples=minibatch_samples,
            route_by_qid=mutation_preview.train_route_by_qid,
            prompt_set=prompt_set,
            prompt_train_evals=state.prompt_train_evals,
            prompt_val_scores=prompt_val_scores,
            max_analysis_cases=self.runtime.router_max_analysis_cases,
        )
        reflection_router = mutation_router
        reflection_preview = mutation_preview
        if analysis_cases:
            formatted_analysis_cases = []
            for analysis_case in analysis_cases:
                suggestion = self.runtime.router_agents.analysis.run(
                    call_id=f"{phase_tag}_analysis_{step_idx}_{analysis_case.qid}",
                    log_dir=log_dir,
                    visible_text=analysis_case.visible_text,
                    ground_truth=analysis_case.ground_truth,
                    wrong_slot_label=display_slot_labels[analysis_case.wrong_slot_id],
                    wrong_prompt_text=analysis_case.wrong_prompt_text,
                    wrong_answer=analysis_case.wrong_answer,
                    better_slot_label=display_slot_labels[analysis_case.better_slot_id],
                    better_prompt_text=analysis_case.better_prompt_text,
                    better_answer=analysis_case.better_answer,
                )
                formatted_analysis_cases.append(
                    {
                        "task_name": analysis_case.task_name,
                        "visible_text": analysis_case.visible_text,
                        "wrong_slot_label": display_slot_labels[analysis_case.wrong_slot_id],
                        "better_slot_label": display_slot_labels[analysis_case.better_slot_id],
                        "suggestion": suggestion,
                    }
                )
            reflection_router = self.runtime.router_agents.reflection.run(
                call_id=f"{phase_tag}_reflection_{step_idx}",
                log_dir=log_dir,
                router_prompt=mutation_router,
                slot_entries=[
                    {
                        "slot_label": label,
                        "prompt_text": str(prompt_set[slot_id]),
                    }
                    for slot_id, label in enumerate(display_slot_labels)
                ],
                analysis_cases=formatted_analysis_cases,
            )
            reflection_preview = state.router_pool.evaluate_candidate(
                router_prompt=reflection_router,
                prompt_set=prompt_set,
                prompt_train_evals=preview_prompt_train_evals,
                prompt_val_evals=preview_prompt_train_evals,
                train_samples=minibatch_samples,
                val_samples=minibatch_samples,
                log_dir=log_dir,
                call_prefix=f"{phase_tag}_reflection_preview_{step_idx}",
                source="reflection",
                parent_candidate_ids=[
                    candidate_id
                    for candidate_id in [parent_candidate.candidate_id, *[candidate.candidate_id for candidate in ref_candidates]]
                    if candidate_id is not None
                ],
                score_weights=score_weights,
            )

        winner_source, winner_prompt, _ = pick_mutation_step_winner(
            mutation_text=mutation_router,
            mutation_score=mutation_preview.train_eval.score,
            reflection_text=reflection_router,
            reflection_score=reflection_preview.train_eval.score,
        )
        candidate = state.router_pool.evaluate_candidate(
            router_prompt=winner_prompt,
            prompt_set=prompt_set,
            prompt_train_evals=state.prompt_train_evals,
            prompt_val_evals=state.prompt_val_evals,
            train_samples=train_samples,
            val_samples=val_samples,
            log_dir=log_dir,
            call_prefix=f"{phase_tag}_candidate_{step_idx}",
            source=winner_source,
            parent_candidate_ids=[
                candidate_id
                for candidate_id in [parent_candidate.candidate_id, *[candidate.candidate_id for candidate in ref_candidates]]
                if candidate_id is not None
            ],
            score_weights=score_weights,
        )
        admitted = state.router_pool.admit_if_improves_baseline(candidate)
        state.steps_in_phase += 1
        state.history.append(state.best_candidate.router_score)
        self.runtime._log_line(
            f"RouterEvolve step={step_idx} winner={winner_source} "
            f"candidate_score={candidate.router_score:.4f} admitted={admitted}"
        )
        return state
