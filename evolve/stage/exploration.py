from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evolve.unit import PromptEvolve, PromptEvolveState, RouterEvolve
from evolve.utils.runtime import group_samples_by_slot, slice_eval
from evolve.utils.structures import DatasetEval, EEVEECandidate, PromptCandidate


@dataclass
class PromptSetExplorationState:
    router_prompt: str
    routed_train: Dict[int, List[Dict[str, Any]]]
    routed_val: Dict[int, List[Dict[str, Any]]]
    states_by_slot: Dict[int, PromptEvolveState]
    active_slot_ids: List[int]
    history: List[float] = field(default_factory=list)
    steps_in_phase: int = 0
    group_steps_in_phase: int = 0
    checkpoint_best: float = 0.0
    next_slot_cursor: int = 0


class ExplorationStage:
    """Exploration orchestration: RouterEvolve -> grouping -> parallel single-slot PromptEvolve."""

    def __init__(
        self,
        runtime: Any,
        router_evolve: RouterEvolve,
        prompt_evolve: PromptEvolve,
    ):
        self.runtime = runtime
        self.router_evolve = router_evolve
        self.prompt_evolve = prompt_evolve

    def run(
        self,
        *,
        router_prompt: str,
        prompt_set: Sequence[str],
        train_samples: Sequence[Dict[str, Any]],
        val_samples: Sequence[Dict[str, Any]],
        empty_val_eval: DatasetEval,
        eval_fn,
        log_dir: Optional[str],
    ) -> Tuple[str, List[str], EEVEECandidate]:
        current_router_prompt = router_prompt
        current_prompt_set = list(prompt_set)
        remaining_budget = self.runtime.total_mini_step_budget
        phase = "router"
        router_state = None
        prompt_set_state: Optional[PromptSetExplorationState] = None

        while remaining_budget > 0:
            if phase == "router":
                router_score_weights = self.router_evolve.score_weights_for_remaining(remaining_budget)
                if router_state is None:
                    router_state = self.router_evolve.start(
                        router_prompt=current_router_prompt,
                        prompt_set=current_prompt_set,
                        train_samples=train_samples,
                        val_samples=val_samples,
                        eval_fn=eval_fn,
                        log_dir=log_dir,
                        phase_tag=f"router_evolve_{remaining_budget}",
                        score_weights=router_score_weights,
                    )
                    self.runtime.rlog.section("exploration: router evolution")
                    self.runtime._log_line(
                        f"router baseline score={router_state.baseline_router_score:.4f} "
                        f"weights=accuracy:{router_score_weights['accuracy']:.3f},"
                        f"consistency:{router_score_weights['consistency']:.3f},"
                        f"balance:{router_score_weights['balance']:.3f} remaining_budget={remaining_budget}"
                    )

                router_state = self.router_evolve.run_step(
                    state=router_state,
                    prompt_set=current_prompt_set,
                    train_samples=train_samples,
                    val_samples=val_samples,
                    log_dir=log_dir,
                    phase_tag=f"router_evolve_{remaining_budget}",
                    score_weights=router_score_weights,
                )
                remaining_budget -= 1
                self.runtime._log_line(
                    f"RouterEvolve steps={router_state.steps_in_phase} "
                    f"best_score={router_state.best_candidate.router_score:.4f} "
                    f"weights=accuracy:{router_score_weights['accuracy']:.3f},"
                    f"consistency:{router_score_weights['consistency']:.3f},"
                    f"balance:{router_score_weights['balance']:.3f} "
                    f"remaining_budget={remaining_budget}"
                )
                if (
                    router_state.steps_in_phase > 0
                    and router_state.steps_in_phase % self.runtime.router_window_size == 0
                ):
                    router_plateaued, router_state.checkpoint_best = self._plateaued(
                        history=router_state.history,
                        checkpoint_best=router_state.checkpoint_best,
                    )
                    if router_plateaued:
                        current_router_prompt = router_state.best_candidate.router_prompt
                        router_state = None
                        phase = "prompt"
                        self.runtime._log_line("switch phase: router -> prompt")
                continue

            if prompt_set_state is None:
                prompt_set_state = self._start_prompt_set_evolution(
                    router_prompt=current_router_prompt,
                    prompt_set=current_prompt_set,
                    train_samples=train_samples,
                    val_samples=val_samples,
                    empty_val_eval=empty_val_eval,
                    eval_fn=eval_fn,
                    log_dir=log_dir,
                    phase_tag=f"prompt_evolve_{remaining_budget}",
                )
                self.runtime.rlog.section("exploration: prompt evolution")
                self.runtime.rlog.log_prompt_part_table(
                    routed_train=prompt_set_state.routed_train,
                    routed_val=prompt_set_state.routed_val,
                    slot_count=self.runtime.slot_count,
                )
                self.runtime._log_line(
                    f"prompt set baseline score={prompt_set_state.checkpoint_best:.4f} "
                    f"active_slots={[self.runtime._slot_label(slot_id) for slot_id in prompt_set_state.active_slot_ids]}"
                )

            if not prompt_set_state.active_slot_ids:
                self.runtime._log_line("grouping produced no active prompt slots; returning to router evolution")
                phase = "router"
                prompt_set_state = None
                continue

            prompt_set_state, prompt_steps_taken, completed_prompt_group = self._run_parallel_prompt_evolve(
                state=prompt_set_state,
                train_samples=train_samples,
                val_samples=val_samples,
                eval_fn=eval_fn,
                log_dir=log_dir,
                phase_tag=f"prompt_evolve_{remaining_budget}",
                full_eval=True,
                max_steps=min(remaining_budget, len(prompt_set_state.active_slot_ids)),
            )
            if prompt_steps_taken <= 0:
                phase = "router"
                prompt_set_state = None
                continue
            remaining_budget -= prompt_steps_taken
            self.runtime._log_line(
                f"PromptEvolve parallel_steps={prompt_set_state.group_steps_in_phase} "
                f"slot_steps={prompt_set_state.steps_in_phase} "
                f"best_prompt_set_score={prompt_set_state.history[-1]:.4f} "
                f"remaining_budget={remaining_budget}"
            )
            if (
                completed_prompt_group
                and prompt_set_state.group_steps_in_phase > 0
                and prompt_set_state.group_steps_in_phase % self.runtime.prompt_window_size == 0
            ):
                prompt_plateaued, prompt_set_state.checkpoint_best = self._plateaued(
                    history=prompt_set_state.history,
                    checkpoint_best=prompt_set_state.checkpoint_best,
                )
                if prompt_plateaued:
                    current_prompt_set = self._best_prompt_set(prompt_set_state)
                    prompt_set_state = None
                    phase = "router"
                    self.runtime._log_line("switch phase: prompt -> router")

        if router_state is not None:
            current_router_prompt = router_state.best_candidate.router_prompt
        if prompt_set_state is not None:
            current_prompt_set = self._best_prompt_set(prompt_set_state)

        best_candidate = self.runtime._build_candidate_from_router_and_prompt_set(
            router_prompt=current_router_prompt,
            prompt_set=current_prompt_set,
            train_samples=train_samples,
            val_samples=val_samples,
            eval_fn=eval_fn,
            log_dir=log_dir,
            label_prefix="exploration_final",
        )
        self.runtime._persist_best_candidate(best_candidate)
        self.runtime.rlog.log_candidate("post exploration candidate", best_candidate)
        return current_router_prompt, current_prompt_set, best_candidate

    def _start_prompt_set_evolution(
        self,
        *,
        router_prompt: str,
        prompt_set: Sequence[str],
        train_samples: Sequence[Dict[str, Any]],
        val_samples: Sequence[Dict[str, Any]],
        empty_val_eval: DatasetEval,
        eval_fn,
        log_dir: Optional[str],
        phase_tag: str,
    ) -> PromptSetExplorationState:
        routed_train_by_qid = self.runtime._route_samples(
            router_prompt=router_prompt,
            prompt_set=prompt_set,
            samples=train_samples,
            call_prefix=f"{phase_tag}_train_route",
            log_dir=log_dir,
        )
        routed_val_by_qid = self.runtime._route_samples(
            router_prompt=router_prompt,
            prompt_set=prompt_set,
            samples=val_samples,
            call_prefix=f"{phase_tag}_val_route",
            log_dir=log_dir,
        )
        routed_train = group_samples_by_slot(routed_train_by_qid, train_samples, self.runtime.slot_count)
        routed_val = group_samples_by_slot(routed_val_by_qid, val_samples, self.runtime.slot_count)

        full_prompt_train_evals, full_prompt_val_evals = self.runtime._compute_full_prompt_evals(
            prompt_set=prompt_set,
            train_samples=train_samples,
            val_samples=val_samples,
            eval_fn=eval_fn,
            log_dir=log_dir,
            label_prefix=phase_tag,
            output_dir=self.runtime.exp_dir / "exploration" / phase_tag,
        )
        states_by_slot: Dict[int, PromptEvolveState] = {}
        active_slot_ids: List[int] = []
        for slot_id in range(self.runtime.slot_count):
            seed_candidate = PromptCandidate(
                slot_id=slot_id,
                prompt_text=prompt_set[slot_id],
                source="seed",
                train_eval=slice_eval(full_prompt_train_evals[slot_id], [sample["_qid"] for sample in routed_train.get(slot_id, [])]),
                val_eval=slice_eval(full_prompt_val_evals[slot_id], [sample["_qid"] for sample in routed_val.get(slot_id, [])]),
                parent_candidate_ids=[],
            )
            active = bool(routed_train.get(slot_id)) and bool(routed_val.get(slot_id))
            if active:
                active_slot_ids.append(slot_id)
            empty_val_on_slot = slice_eval(
                empty_val_eval,
                [sample["_qid"] for sample in routed_val.get(slot_id, [])],
            )
            states_by_slot[slot_id] = self.prompt_evolve.create_state(
                slot_id=slot_id,
                seed_candidate=seed_candidate,
                baseline_score=float(empty_val_on_slot.score),
                active=active,
                seed_offset=3100,
            )
        initial_score = self._prompt_set_score(states_by_slot)
        return PromptSetExplorationState(
            router_prompt=router_prompt,
            routed_train=routed_train,
            routed_val=routed_val,
            states_by_slot=states_by_slot,
            active_slot_ids=active_slot_ids,
            history=[initial_score],
            checkpoint_best=initial_score,
        )

    def _run_parallel_prompt_evolve(
        self,
        *,
        state: PromptSetExplorationState,
        train_samples: Sequence[Dict[str, Any]],
        val_samples: Sequence[Dict[str, Any]],
        eval_fn,
        log_dir: Optional[str],
        phase_tag: str,
        full_eval: bool,
        max_steps: Optional[int] = None,
    ) -> Tuple[PromptSetExplorationState, int, bool]:
        if not state.active_slot_ids:
            return state, 0, False
        ordered_slot_ids = [
            state.active_slot_ids[(state.next_slot_cursor + idx) % len(state.active_slot_ids)]
            for idx in range(len(state.active_slot_ids))
        ]
        steps_to_take = len(ordered_slot_ids) if max_steps is None else min(len(ordered_slot_ids), max_steps)
        slot_ids = ordered_slot_ids[:steps_to_take]
        if not slot_ids:
            return state, 0, False

        step_idx = state.group_steps_in_phase
        max_workers = max(1, min(self.runtime.max_parallel_slots, len(slot_ids)))
        updates: List[Tuple[int, PromptEvolveState, str]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.prompt_evolve.run_step,
                    state=state.states_by_slot[slot_id],
                    routed_train=state.routed_train,
                    routed_val=state.routed_val,
                    train_samples=train_samples,
                    val_samples=val_samples,
                    eval_fn=eval_fn,
                    log_dir=log_dir,
                    phase_tag=phase_tag,
                    full_eval=full_eval,
                    step_idx=step_idx,
                ): slot_id
                for slot_id in slot_ids
            }
            for future in as_completed(futures):
                slot_id = futures[future]
                evolved_state, log_line = future.result()
                updates.append((slot_id, evolved_state, log_line))

        for slot_id, evolved_state, log_line in sorted(updates, key=lambda item: item[0]):
            state.states_by_slot[slot_id] = evolved_state
            self.runtime._log_line(log_line)

        state.steps_in_phase += len(slot_ids)
        state.next_slot_cursor = (state.next_slot_cursor + len(slot_ids)) % len(state.active_slot_ids)
        completed_group = len(slot_ids) == len(state.active_slot_ids)
        if completed_group:
            state.group_steps_in_phase += 1
        state.history.append(self._prompt_set_score(state.states_by_slot))
        return state, len(slot_ids), completed_group

    @staticmethod
    def _prompt_set_score(states_by_slot: Dict[int, PromptEvolveState]) -> float:
        correct = sum(state.best_candidate.val_eval.correct for state in states_by_slot.values())
        total = sum(state.best_candidate.val_eval.total for state in states_by_slot.values())
        return (correct / total) if total else 0.0

    def _best_prompt_set(self, state: PromptSetExplorationState) -> List[str]:
        return [
            state.states_by_slot[slot_id].prompt_pool.best().prompt_text
            for slot_id in range(self.runtime.slot_count)
        ]

    def _plateaued(
        self,
        *,
        history: Sequence[float],
        checkpoint_best: float,
    ) -> Tuple[bool, float]:
        if not history:
            return False, checkpoint_best
        current_best = max(float(value) for value in history)
        return (current_best - checkpoint_best) < self.runtime.phase_switch_epsilon, current_best
