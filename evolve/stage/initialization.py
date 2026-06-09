from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evolve.pool import PromptPool
from evolve.unit import PromptEvolve
from evolve.utils.runtime import slice_eval
from evolve.utils.structures import DatasetEval, PromptCandidate


class InitializationStage:
    """Initialization stage: run PromptEvolve on all data, then retain prompt set P."""

    def __init__(self, runtime: Any, prompt_evolve: PromptEvolve):
        self.runtime = runtime
        self.prompt_evolve = prompt_evolve

    def run(
        self,
        *,
        train_samples: Sequence[Dict[str, Any]],
        val_samples: Sequence[Dict[str, Any]],
        eval_fn,
        log_dir: Optional[str],
        log_event,
    ) -> List[PromptCandidate]:
        self.runtime.rlog.section("initialization")
        initialization_train = {slot_id: [] for slot_id in range(self.runtime.slot_count)}
        initialization_val = {slot_id: [] for slot_id in range(self.runtime.slot_count)}
        initialization_train[0] = list(train_samples)
        initialization_val[0] = list(val_samples)
        self.runtime.rlog.log_prompt_part_table(
            routed_train=initialization_train,
            routed_val=initialization_val,
            slot_count=self.runtime.slot_count,
        )
        empty_train_eval = self.runtime._evaluate_prompt_cached(
            prompt_text="",
            samples=train_samples,
            eval_fn=eval_fn,
            split_name="train",
            label="initialization_empty_full_train",
            log_dir=log_dir,
            results_dir=self.runtime.exp_dir / "artifacts" / "initialization_empty" / "full_train",
        )
        empty_val_eval = self.runtime._evaluate_prompt_cached(
            prompt_text="",
            samples=val_samples,
            eval_fn=eval_fn,
            split_name="val",
            label="initialization_empty_full_val",
            log_dir=log_dir,
            results_dir=self.runtime.exp_dir / "artifacts" / "initialization_empty" / "full_val",
        )

        _, best_candidate, prompt_pool, train_count, val_count = self._run_prompt_evolve_on_all_data(
            slot_id=0,
            grouped_train=train_samples,
            grouped_val=val_samples,
            train_samples=train_samples,
            val_samples=val_samples,
            empty_train_eval=empty_train_eval,
            empty_val_eval=empty_val_eval,
            eval_fn=eval_fn,
            log_dir=log_dir,
        )
        retained_candidates = self._select_retained_candidates(prompt_pool)
        self.runtime._save_pool_snapshot(
            prompt_pool,
            self.runtime.exp_dir / "initialization" / "initialization_prompt_pool.json",
        )
        self.runtime._log_line(
            f"initialization prompt pool best val={self.runtime.rlog.eval_text(best_candidate.val_eval)} "
            f"retained={len(retained_candidates)}"
        )
        for slot_id, candidate in enumerate(retained_candidates):
            self.runtime._log_line(
                f"initialization retained {self.runtime._slot_label(slot_id)} "
                f"source_id={candidate.metadata.get('initialization_source_candidate_id')} "
                f"coverage={candidate.metadata.get('initialization_prompt_pool_coverage')} "
                f"val={self.runtime.rlog.eval_text(candidate.val_eval)}"
            )
        log_event(
            {
                "type": "initialization_prompt_pool_done",
                "train_count": train_count,
                "val_count": val_count,
                "best_val_score": best_candidate.val_score,
                "retained": [
                    {
                        "slot_id": slot_id,
                        "source_candidate_id": candidate.metadata.get("initialization_source_candidate_id"),
                        "prompt_pool_coverage": candidate.metadata.get("initialization_prompt_pool_coverage"),
                        "val_score": candidate.val_score,
                    }
                    for slot_id, candidate in enumerate(retained_candidates)
                ],
            }
        )
        return retained_candidates

    def _run_prompt_evolve_on_all_data(
        self,
        *,
        slot_id: int,
        grouped_train: Sequence[Dict[str, Any]],
        grouped_val: Sequence[Dict[str, Any]],
        train_samples: Sequence[Dict[str, Any]],
        val_samples: Sequence[Dict[str, Any]],
        empty_train_eval: DatasetEval,
        empty_val_eval: DatasetEval,
        eval_fn,
        log_dir: Optional[str],
    ) -> Tuple[int, PromptCandidate, PromptPool[PromptCandidate], int, int]:
        seed_candidate = PromptCandidate(
            slot_id=slot_id,
            prompt_text="",
            train_eval=slice_eval(empty_train_eval, [sample["_qid"] for sample in grouped_train]),
            val_eval=slice_eval(empty_val_eval, [sample["_qid"] for sample in grouped_val]),
            source="initialization_seed",
            parent_candidate_ids=[],
        )
        empty_val_on_slot = slice_eval(empty_val_eval, [sample["_qid"] for sample in grouped_val])
        state = self.prompt_evolve.create_state(
            slot_id=slot_id,
            seed_candidate=seed_candidate,
            baseline_score=float(empty_val_on_slot.score),
            active=bool(grouped_train) and bool(grouped_val),
            seed_offset=2400,
        )
        if state.active:
            routed_train = {slot_id: list(grouped_train)}
            routed_val = {slot_id: list(grouped_val)}
            for round_idx in range(self.runtime.initialization_budget):
                state, log_line = self.prompt_evolve.run_step(
                    state=state,
                    routed_train=routed_train,
                    routed_val=routed_val,
                    train_samples=train_samples,
                    val_samples=val_samples,
                    eval_fn=eval_fn,
                    log_dir=log_dir,
                    phase_tag=f"initialization_{self.runtime._slot_label(slot_id)}",
                    full_eval=True,
                    step_idx=round_idx,
                )
                self.runtime._log_line(log_line)
        return slot_id, state.prompt_pool.best(), state.prompt_pool, len(grouped_train), len(grouped_val)

    def _select_retained_candidates(
        self,
        prompt_pool: PromptPool[PromptCandidate],
    ) -> List[PromptCandidate]:
        retained: List[PromptCandidate] = []
        for slot_id, (candidate_id, candidate, coverage) in enumerate(prompt_pool.select_prompt_set(self.runtime.slot_count)):
            retained.append(
                replace(
                    candidate,
                    slot_id=slot_id,
                    metadata={
                        **dict(candidate.metadata),
                        "initialization_source_candidate_id": candidate_id,
                        "initialization_prompt_pool_coverage": coverage,
                    },
                )
            )
        return retained
