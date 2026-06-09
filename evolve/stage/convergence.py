from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evolve.pool import PromptPool
from evolve.unit import PromptEvolve
from evolve.utils.runtime import group_samples_by_slot
from evolve.utils.structures import PromptCandidate


class ConvergenceStage:
    """Convergence stage: run parallel single-slot PromptEvolve with larger budget."""

    def __init__(self, runtime: Any, prompt_evolve: PromptEvolve):
        self.runtime = runtime
        self.prompt_evolve = prompt_evolve

    def run(
        self,
        *,
        router_prompt: str,
        prompt_set: Sequence[str],
        train_samples: Sequence[Dict[str, Any]],
        val_samples: Sequence[Dict[str, Any]],
        eval_fn,
        log_dir: Optional[str],
    ) -> List[str]:
        routed_train_by_qid = self.runtime._route_samples(
            router_prompt=router_prompt,
            prompt_set=prompt_set,
            samples=train_samples,
            call_prefix="convergence_train_route",
            log_dir=log_dir,
            output_path=self.runtime.exp_dir / "convergence" / "train_routes.jsonl",
        )
        routed_val_by_qid = self.runtime._route_samples(
            router_prompt=router_prompt,
            prompt_set=prompt_set,
            samples=val_samples,
            call_prefix="convergence_val_route",
            log_dir=log_dir,
            output_path=self.runtime.exp_dir / "convergence" / "val_routes.jsonl",
        )
        routed_train = group_samples_by_slot(routed_train_by_qid, train_samples, self.runtime.slot_count)
        routed_val = group_samples_by_slot(routed_val_by_qid, val_samples, self.runtime.slot_count)
        self.runtime.rlog.log_prompt_part_table(
            routed_train=routed_train,
            routed_val=routed_val,
            slot_count=self.runtime.slot_count,
        )

        convergence_prompt_set = list(prompt_set)
        active_slot_ids = [
            slot_id
            for slot_id in range(self.runtime.slot_count)
            if routed_train.get(slot_id) and routed_val.get(slot_id)
        ]
        if not active_slot_ids:
            return convergence_prompt_set

        worker_count = max(1, min(len(active_slot_ids), self.runtime.max_parallel_slots))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    self._run_slot_prompt_evolve,
                    slot_id=slot_id,
                    prompt_text=convergence_prompt_set[slot_id],
                    grouped_train=routed_train[slot_id],
                    grouped_val=routed_val[slot_id],
                    eval_fn=eval_fn,
                    log_dir=log_dir,
                ): slot_id
                for slot_id in active_slot_ids
            }
            for future in as_completed(futures):
                slot_id, best_prompt_text, prompt_pool = future.result()
                convergence_prompt_set[slot_id] = best_prompt_text
                if prompt_pool is not None:
                    self.runtime._save_pool_snapshot(
                        prompt_pool,
                        self.runtime.exp_dir / "convergence" / f"{self.runtime._slot_label(slot_id)}_prompt_pool.json",
                    )
        return convergence_prompt_set

    def _run_slot_prompt_evolve(
        self,
        *,
        slot_id: int,
        prompt_text: str,
        grouped_train: Sequence[Dict[str, Any]],
        grouped_val: Sequence[Dict[str, Any]],
        eval_fn,
        log_dir: Optional[str],
    ) -> Tuple[int, str, Optional[PromptPool[PromptCandidate]]]:
        if not grouped_train or not grouped_val:
            return slot_id, prompt_text, None

        slot_label = self.runtime._slot_label(slot_id)
        empty_val_on_slot = self.runtime._evaluate_prompt_cached(
            prompt_text="",
            samples=list(grouped_val),
            eval_fn=eval_fn,
            split_name="val",
            label=f"convergence_{slot_label}_empty_val_floor",
            log_dir=log_dir,
            results_dir=self.runtime.exp_dir / "artifacts" / "convergence_empty_floor" / slot_label / "val",
        )

        seed_candidate = self.prompt_evolve.evaluate_candidate(
            slot_id=slot_id,
            prompt_text=prompt_text,
            grouped_train_samples=grouped_train,
            grouped_val_samples=grouped_val,
            all_train_samples=grouped_train,
            all_val_samples=grouped_val,
            eval_fn=eval_fn,
            log_dir=log_dir,
            label_prefix=f"convergence_{slot_label}_seed",
            full_eval=False,
            source="convergence_seed",
            parent_candidate_ids=[],
        )
        state = self.prompt_evolve.create_state(
            slot_id=slot_id,
            seed_candidate=seed_candidate,
            baseline_score=float(empty_val_on_slot.score),
            active=True,
            seed_offset=8000,
        )
        routed_train = {slot_id: list(grouped_train)}
        routed_val = {slot_id: list(grouped_val)}
        for round_idx in range(self.runtime.convergence_budget_per_slot):
            state, log_line = self.prompt_evolve.run_step(
                state=state,
                routed_train=routed_train,
                routed_val=routed_val,
                train_samples=grouped_train,
                val_samples=grouped_val,
                eval_fn=eval_fn,
                log_dir=log_dir,
                phase_tag=f"convergence_{slot_label}_{round_idx}",
                full_eval=False,
                step_idx=round_idx,
            )
            self.runtime._log_line(log_line)
        return slot_id, state.prompt_pool.best().prompt_text, state.prompt_pool
