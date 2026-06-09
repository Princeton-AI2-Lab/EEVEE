from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from evolve.utils.runtime import dataset_eval_from_maps
from evolve.utils.structures import DatasetEval, RouterCandidate


class RouterPool:
    """Temporary router pool B_R and router scoring algorithms."""

    def __init__(self, runtime: Any, *, max_size: Optional[int] = None):
        self.runtime = runtime
        self.max_size = max_size
        self.items: List[RouterCandidate] = []

    def __len__(self) -> int:
        return len(self.items)

    @property
    def baseline_score(self) -> float:
        if not self.items:
            return 0.0
        return float(self.items[0].router_score)

    def add_seed(self, candidate: RouterCandidate) -> None:
        candidate.candidate_id = 0
        self.items = [candidate]

    def best(self) -> RouterCandidate:
        if not self.items:
            raise ValueError("Router pool is empty.")
        return max(
            self.items,
            key=lambda item: (item.router_score, item.val_score, -(item.candidate_id or 0)),
        )

    def sample(self) -> RouterCandidate:
        if not self.items:
            raise ValueError("Router pool is empty.")
        weights = self._normalize_probability_weights(
            [candidate.router_score for candidate in self.items],
            temperature=0.1,
        )
        return self.runtime._rng.choices(list(self.items), weights=weights, k=1)[0]

    def reference_candidates(self, *, parent_candidate: RouterCandidate, limit: int) -> List[RouterCandidate]:
        return [
            candidate
            for candidate in self.items
            if candidate is not parent_candidate
        ][:limit]

    def apply_score_weights(self, weights: Dict[str, float]) -> None:
        for candidate in self.items:
            candidate.router_score = self._compute_router_score(candidate, weights)

    def admit_if_improves_baseline(self, candidate: RouterCandidate) -> bool:
        if candidate.router_score <= self.baseline_score:
            return False
        candidate.candidate_id = max(
            (item.candidate_id for item in self.items if item.candidate_id is not None),
            default=-1,
        ) + 1
        self.items.append(candidate)
        self._trim_to_max_size()
        return any(item is candidate for item in self.items)

    def score_weights_for_remaining(self, remaining_budget: int) -> Dict[str, float]:
        if self.runtime.total_mini_step_budget <= 1:
            progress = 0.0
        else:
            progress = (float(remaining_budget) - 1.0) / (float(self.runtime.total_mini_step_budget) - 1.0)
        progress = max(0.0, min(1.0, progress))
        weights = {
            key: self.runtime.router_final_score_weights[key]
            + (self.runtime.router_score_weights[key] - self.runtime.router_final_score_weights[key]) * progress
            for key in ("accuracy", "consistency", "balance")
        }
        return self._normalize_score_weights(weights)

    @staticmethod
    def build_best_slot_targets(
        *,
        train_samples: Sequence[Dict[str, Any]],
        prompt_train_evals: Dict[int, DatasetEval],
        prompt_val_evals: Dict[int, DatasetEval],
    ) -> Dict[str, int]:
        targets: Dict[str, int] = {}
        for sample in train_samples:
            qid = str(sample["_qid"])
            correct_slots = [
                slot_id
                for slot_id, train_eval in prompt_train_evals.items()
                if train_eval.correctness_by_qid.get(qid, False)
            ]
            if not correct_slots:
                continue
            best_slot = max(correct_slots, key=lambda slot_id: (prompt_val_evals[slot_id].score, -slot_id))
            targets[qid] = best_slot
        return targets

    def evaluate_candidate(
        self,
        *,
        router_prompt: str,
        prompt_set: Sequence[str],
        prompt_train_evals: Dict[int, DatasetEval],
        prompt_val_evals: Dict[int, DatasetEval],
        train_samples: Sequence[Dict[str, Any]],
        val_samples: Sequence[Dict[str, Any]],
        log_dir: Optional[str],
        call_prefix: str,
        source: str,
        parent_candidate_ids: Sequence[int],
        score_weights: Optional[Dict[str, float]] = None,
    ) -> RouterCandidate:
        train_route_by_qid = self.runtime._route_samples(
            router_prompt=router_prompt,
            prompt_set=prompt_set,
            samples=train_samples,
            call_prefix=f"{call_prefix}_train_route",
            log_dir=log_dir,
        )
        val_route_by_qid = self.runtime._route_samples(
            router_prompt=router_prompt,
            prompt_set=prompt_set,
            samples=val_samples,
            call_prefix=f"{call_prefix}_val_route",
            log_dir=log_dir,
        )

        train_eval = dataset_eval_from_maps(
            split_name="train",
            correctness_by_qid={
                qid: prompt_train_evals[slot_id].correctness_by_qid.get(qid, False)
                if slot_id in prompt_train_evals
                else False
                for qid, slot_id in train_route_by_qid.items()
            },
            answers_by_qid={
                qid: prompt_train_evals[slot_id].answers_by_qid.get(qid, "")
                if slot_id in prompt_train_evals
                else ""
                for qid, slot_id in train_route_by_qid.items()
            },
        )
        val_eval = dataset_eval_from_maps(
            split_name="val",
            correctness_by_qid={
                qid: prompt_val_evals[slot_id].correctness_by_qid.get(qid, False)
                if slot_id in prompt_val_evals
                else False
                for qid, slot_id in val_route_by_qid.items()
            },
            answers_by_qid={
                qid: prompt_val_evals[slot_id].answers_by_qid.get(qid, "")
                if slot_id in prompt_val_evals
                else ""
                for qid, slot_id in val_route_by_qid.items()
            },
        )
        accuracy_score = max(0.0, min(1.0, float(val_eval.score)))
        consistency_score = self._compute_consistency_score(
            val_route_by_qid=val_route_by_qid,
            val_samples=val_samples,
        )
        balance_score = self._compute_balance_score(
            train_route_by_qid=train_route_by_qid,
            val_route_by_qid=val_route_by_qid,
        )
        active_weights = self._normalize_score_weights(score_weights or self.runtime.router_score_weights)
        router_score = (
            active_weights["accuracy"] * accuracy_score
            + active_weights["consistency"] * consistency_score
            + active_weights["balance"] * balance_score
        )
        return RouterCandidate(
            router_prompt=router_prompt,
            source=source,
            train_eval=train_eval,
            val_eval=val_eval,
            train_route_by_qid=train_route_by_qid,
            val_route_by_qid=val_route_by_qid,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            balance_score=balance_score,
            router_score=max(0.0, min(1.0, router_score)),
            parent_candidate_ids=[int(v) for v in parent_candidate_ids],
        )

    def _trim_to_max_size(self) -> None:
        if self.max_size is None:
            return
        max_size = max(1, int(self.max_size))
        seed_router = self.items[0]
        retained_candidates = sorted(
            self.items[1:],
            key=lambda item: (item.router_score, item.val_score, -(item.candidate_id or 0)),
            reverse=True,
        )[: max(0, max_size - 1)]
        self.items = [seed_router, *retained_candidates]

    @staticmethod
    def _normalize_probability_weights(values: Sequence[float], temperature: float = 0.1) -> List[float]:
        if not values:
            return []
        max_value = max(values)
        exps = [math.exp((float(value) - max_value) / max(temperature, 1e-6)) for value in values]
        total = sum(exps)
        if total <= 0:
            return [1.0 / len(values)] * len(values)
        return [value / total for value in exps]

    @staticmethod
    def _vector_distance(a: Sequence[float], b: Sequence[float]) -> float:
        if not a:
            return 0.0
        return sum(abs(float(x) - float(y)) for x, y in zip(a, b)) / len(a)

    @staticmethod
    def _centroid(vectors: Sequence[Sequence[float]]) -> List[float]:
        if not vectors:
            return []
        dim = len(vectors[0])
        return [
            sum(float(vector[idx]) for vector in vectors) / len(vectors)
            for idx in range(dim)
        ]

    def _compute_consistency_score(
        self,
        *,
        val_route_by_qid: Dict[str, int],
        val_samples: Sequence[Dict[str, Any]],
    ) -> float:
        prompt_hashes = self.runtime.eval_cache.prompt_hashes()
        if not prompt_hashes:
            return 0.0

        vectors_by_slot: Dict[int, List[List[float]]] = {
            slot_id: []
            for slot_id in range(self.runtime.slot_count)
        }
        for sample in val_samples:
            qid = str(sample["_qid"])
            slot_id = int(val_route_by_qid[qid])
            vector = [
                1.0
                if (
                    self.runtime.eval_cache.get(prompt_hash, "val", qid)
                    or type("MissingEval", (), {"is_correct": False})()
                ).is_correct
                else 0.0
                for prompt_hash in prompt_hashes
            ]
            vectors_by_slot[slot_id].append(vector)

        non_empty_slots = [slot_id for slot_id, vectors in vectors_by_slot.items() if vectors]
        if not non_empty_slots:
            return 0.0

        compact_numerator = 0.0
        compact_denominator = 0
        centroids: Dict[int, List[float]] = {}
        for slot_id in non_empty_slots:
            vectors = vectors_by_slot[slot_id]
            centroid = self._centroid(vectors)
            centroids[slot_id] = centroid
            slot_score = 1.0
            if vectors and centroid:
                slot_score = sum(1.0 - self._vector_distance(vector, centroid) for vector in vectors) / len(vectors)
            compact_numerator += slot_score * len(vectors)
            compact_denominator += len(vectors)
        compact = compact_numerator / compact_denominator if compact_denominator else 0.0

        separate_pairs = 0
        separate_sum = 0.0
        for idx, left_slot in enumerate(non_empty_slots):
            for right_slot in non_empty_slots[idx + 1 :]:
                separate_sum += self._vector_distance(centroids[left_slot], centroids[right_slot])
                separate_pairs += 1
        separate = separate_sum / separate_pairs if separate_pairs else 0.0

        compact_weight = self.runtime.consistency_component_weights["compact"]
        separate_weight = self.runtime.consistency_component_weights["separate"]
        denom = max(compact_weight + separate_weight, 1e-8)
        return max(0.0, min(1.0, (compact_weight * compact + separate_weight * separate) / denom))

    def _compute_balance_score(
        self,
        *,
        train_route_by_qid: Dict[str, int],
        val_route_by_qid: Dict[str, int],
    ) -> float:
        counts = [0] * self.runtime.slot_count
        for route_map in (train_route_by_qid, val_route_by_qid):
            for slot_id in route_map.values():
                if 0 <= int(slot_id) < self.runtime.slot_count:
                    counts[int(slot_id)] += 1
        total = sum(counts)
        if total <= 0 or self.runtime.slot_count <= 0:
            return 0.0
        used_slots = sum(1 for count in counts if count > 0)
        use = used_slots / self.runtime.slot_count
        if self.runtime.slot_count == 1:
            balance = 1.0
        else:
            uniform = 1.0 / self.runtime.slot_count
            proportions = [count / total for count in counts]
            total_variation = 0.5 * sum(abs(prop - uniform) for prop in proportions)
            max_total_variation = 1.0 - uniform
            balance = 1.0 - (total_variation / max(max_total_variation, 1e-8))
        use_weight = self.runtime.balance_component_weights["use"]
        distribution_weight = self.runtime.balance_component_weights["distribution"]
        denom = max(use_weight + distribution_weight, 1e-8)
        return max(0.0, min(1.0, (use_weight * use + distribution_weight * balance) / denom))

    @staticmethod
    def _normalize_score_weights(weights: Dict[str, float]) -> Dict[str, float]:
        normalized = {
            "accuracy": max(0.0, float(weights.get("accuracy", 0.0))),
            "consistency": max(0.0, float(weights.get("consistency", 0.0))),
            "balance": max(0.0, float(weights.get("balance", 0.0))),
        }
        total = sum(normalized.values())
        if total <= 0:
            return {"accuracy": 1.0, "consistency": 0.0, "balance": 0.0}
        return {key: value / total for key, value in normalized.items()}

    @staticmethod
    def _compute_router_score(candidate: RouterCandidate, weights: Dict[str, float]) -> float:
        score = (
            weights["accuracy"] * candidate.accuracy_score
            + weights["consistency"] * candidate.consistency_score
            + weights["balance"] * candidate.balance_score
        )
        return max(0.0, min(1.0, score))
