from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from evolve.pool import PromptPool
from evolve.utils.evolution_utils import pick_mutation_step_winner
from evolve.utils.runtime import build_prompt_examples, dataset_eval_from_maps, sample_qids, slice_eval
from evolve.utils.structures import PromptCandidate


@dataclass
class PromptEvolveState:
    slot_id: int
    best_candidate: PromptCandidate
    prompt_pool: PromptPool[PromptCandidate]
    baseline_score: float
    active: bool


class PromptEvolve:
    """Single-slot PromptEvolve unit.

    The stage modules decide grouping, parallelism, and budgets. This unit only
    evolves one prompt slot against the data routed to that slot.
    """

    def __init__(self, runtime: Any):
        self.runtime = runtime

    def create_state(
        self,
        *,
        slot_id: int,
        seed_candidate: PromptCandidate,
        baseline_score: float,
        active: bool,
        seed_offset: int,
    ) -> PromptEvolveState:
        prompt_pool = PromptPool[PromptCandidate](
            rng=random.Random(self.runtime.seed + seed_offset + slot_id),
            max_size=(
                int(self.runtime.prompt_temporary_pool_max_size)
                if self.runtime.prompt_temporary_pool_max_size is not None
                else None
            ),
        )
        prompt_pool.add_seed(
            seed_candidate,
            correct_qids=seed_candidate.val_correct_qids,
            score=seed_candidate.val_score,
        )
        return PromptEvolveState(
            slot_id=slot_id,
            best_candidate=seed_candidate,
            prompt_pool=prompt_pool,
            baseline_score=float(baseline_score),
            active=bool(active),
        )

    def run_step(
        self,
        *,
        state: PromptEvolveState,
        routed_train: Dict[int, Sequence[Dict[str, Any]]],
        routed_val: Dict[int, Sequence[Dict[str, Any]]],
        train_samples: Sequence[Dict[str, Any]],
        val_samples: Sequence[Dict[str, Any]],
        eval_fn,
        log_dir: Optional[str],
        phase_tag: str,
        full_eval: bool,
        step_idx: int,
    ) -> Tuple[PromptEvolveState, str]:
        slot_id = state.slot_id
        grouped_train_samples = list(routed_train.get(slot_id, []))
        grouped_val_samples = list(routed_val.get(slot_id, []))
        parent_candidate = state.prompt_pool.sample()
        minibatch_qids = sample_qids(
            [str(sample["_qid"]) for sample in grouped_train_samples],
            self.runtime.prompt_minibatch_size,
            random.Random(self.runtime.seed + 5000 + slot_id * 31 + step_idx),
        )
        grouped_train_by_qid = {str(sample["_qid"]): sample for sample in grouped_train_samples}
        minibatch_samples = [grouped_train_by_qid[qid] for qid in minibatch_qids]
        ref_candidates = self._sample_reference_candidates(
            state.prompt_pool,
            exclude_candidate_id=parent_candidate.candidate_id,
            limit=2,
        )

        slot_label = self.runtime._slot_label(slot_id)
        mutation_prompt = self.runtime.prompt_agents.mutation.run(
            call_id=f"{phase_tag}_{slot_label}_mutation_{step_idx}",
            log_dir=log_dir,
            examples=build_prompt_examples(minibatch_samples, max_examples=self.runtime.example_limit),
            context_prompts=[{"prompt_text": candidate.prompt_text} for candidate in ref_candidates],
        )
        mutation_minibatch_eval = self.runtime._evaluate_prompt_cached(
            prompt_text=mutation_prompt,
            samples=minibatch_samples,
            eval_fn=eval_fn,
            split_name="train",
            label=f"{phase_tag}_{slot_label}_mutation_minibatch_{step_idx}",
            log_dir=log_dir,
            results_dir=self.runtime.exp_dir / "exploration" / phase_tag / slot_label / "minibatch_mutation",
        )
        reflection_prompt = self.runtime.prompt_agents.reflection.run(
            call_id=f"{phase_tag}_{slot_label}_reflection_{step_idx}",
            log_dir=log_dir,
            prompt_text=mutation_prompt,
            all_cases=[
                {
                    "is_correct": mutation_minibatch_eval.correctness_by_qid.get(str(sample["_qid"]), False),
                    "question": str(sample.get("question", "") or ""),
                    "target": str(sample.get("target", "") or ""),
                    "answer": mutation_minibatch_eval.answers_by_qid.get(str(sample["_qid"]), ""),
                }
                for sample in minibatch_samples
            ],
            eval_score=f"{mutation_minibatch_eval.score:.4f}",
            correct=mutation_minibatch_eval.correct,
            total=mutation_minibatch_eval.total,
        )
        reflection_minibatch_eval = self.runtime._evaluate_prompt_cached(
            prompt_text=reflection_prompt,
            samples=minibatch_samples,
            eval_fn=eval_fn,
            split_name="train",
            label=f"{phase_tag}_{slot_label}_reflection_minibatch_{step_idx}",
            log_dir=log_dir,
            results_dir=self.runtime.exp_dir / "exploration" / phase_tag / slot_label / "minibatch_reflection",
        )
        winner_source, winner_prompt, _ = pick_mutation_step_winner(
            mutation_text=mutation_prompt,
            mutation_score=mutation_minibatch_eval.score,
            reflection_text=reflection_prompt,
            reflection_score=reflection_minibatch_eval.score,
        )
        candidate = self.evaluate_candidate(
            slot_id=slot_id,
            prompt_text=winner_prompt,
            grouped_train_samples=grouped_train_samples,
            grouped_val_samples=grouped_val_samples,
            all_train_samples=train_samples,
            all_val_samples=val_samples,
            eval_fn=eval_fn,
            log_dir=log_dir,
            label_prefix=f"{phase_tag}_{slot_label}_{step_idx}",
            full_eval=full_eval,
            source=winner_source,
            parent_candidate_ids=[
                candidate_id
                for candidate_id in [parent_candidate.candidate_id, *[item.candidate_id for item in ref_candidates]]
                if candidate_id is not None
            ],
        )
        admitted = False
        if candidate.val_score > state.baseline_score:
            admitted = state.prompt_pool.try_add(
                candidate,
                correct_qids=candidate.val_correct_qids,
                score=candidate.val_score,
            )
        state.best_candidate = state.prompt_pool.best()
        return state, (
            f"PromptEvolve slot={slot_label} step={step_idx} "
            f"mutation_train={mutation_minibatch_eval.score:.4f} "
            f"reflection_train={reflection_minibatch_eval.score:.4f} winner={winner_source} "
            f"candidate_val={candidate.val_score:.4f} empty_val_floor={state.baseline_score:.4f} admitted={admitted}"
        )

    def evaluate_candidate(
        self,
        *,
        slot_id: int,
        prompt_text: str,
        grouped_train_samples: Sequence[Dict[str, Any]],
        grouped_val_samples: Sequence[Dict[str, Any]],
        all_train_samples: Sequence[Dict[str, Any]],
        all_val_samples: Sequence[Dict[str, Any]],
        eval_fn,
        log_dir: Optional[str],
        label_prefix: str,
        full_eval: bool,
        source: str,
        parent_candidate_ids: Sequence[int],
    ) -> PromptCandidate:
        slot_label = self.runtime._slot_label(slot_id)
        if full_eval:
            full_val_eval = self.runtime._evaluate_prompt_cached(
                prompt_text=prompt_text,
                samples=all_val_samples,
                eval_fn=eval_fn,
                split_name="val",
                label=f"{label_prefix}_{slot_label}_full_val",
                log_dir=log_dir,
                results_dir=self.runtime.exp_dir / "artifacts" / label_prefix / slot_label / "full_val",
            )
            train_eval = dataset_eval_from_maps(split_name="train", correctness_by_qid={}, answers_by_qid={})
            val_eval = slice_eval(full_val_eval, [sample["_qid"] for sample in grouped_val_samples])
        else:
            train_eval = dataset_eval_from_maps(split_name="train", correctness_by_qid={}, answers_by_qid={})
            val_eval = self.runtime._evaluate_prompt_cached(
                prompt_text=prompt_text,
                samples=grouped_val_samples,
                eval_fn=eval_fn,
                split_name="val",
                label=f"{label_prefix}_{slot_label}_val",
                log_dir=log_dir,
                results_dir=self.runtime.exp_dir / "artifacts" / label_prefix / slot_label / "val",
            )
        return PromptCandidate(
            slot_id=slot_id,
            prompt_text=prompt_text,
            source=source,
            train_eval=train_eval,
            val_eval=val_eval,
            parent_candidate_ids=[int(v) for v in parent_candidate_ids],
        )

    @staticmethod
    def _sample_reference_candidates(
        pool: PromptPool[Any],
        *,
        exclude_candidate_id: Optional[int],
        limit: int,
    ) -> Sequence[Any]:
        if limit <= 0:
            return []
        ordered = [
            candidate
            for candidate in pool.all_items()
            if getattr(candidate, "candidate_id", None) != exclude_candidate_id
        ]
        if len(ordered) <= limit:
            return ordered
        chosen: list[Any] = []
        chosen_ids = set()
        for _ in range(len(ordered) * 3):
            candidate = pool.sample()
            candidate_id = getattr(candidate, "candidate_id", None)
            if candidate_id == exclude_candidate_id or candidate_id in chosen_ids:
                continue
            chosen.append(candidate)
            chosen_ids.add(candidate_id)
            if len(chosen) >= limit:
                return chosen
        for candidate in ordered:
            candidate_id = getattr(candidate, "candidate_id", None)
            if candidate_id in chosen_ids:
                continue
            chosen.append(candidate)
            chosen_ids.add(candidate_id)
            if len(chosen) >= limit:
                break
        return chosen
