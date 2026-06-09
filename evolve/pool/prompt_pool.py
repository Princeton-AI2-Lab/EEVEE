from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Generic, List, Mapping, Optional, Sequence, Set, Tuple, TypeVar


T = TypeVar("T")


def _is_dominated(candidate_id: int, programs: Set[int], frontier_by_qid: Mapping[str, Set[int]]) -> bool:
    qid_fronts = [front for front in frontier_by_qid.values() if candidate_id in front]
    for front in qid_fronts:
        if not any(other_id in front for other_id in programs):
            return False
    return True


def _remove_dominated_candidates(
    frontier_by_qid: Mapping[str, Set[int]],
    scores_by_id: Mapping[int, float],
) -> Dict[str, Set[int]]:
    freq: Dict[int, int] = {}
    for front in frontier_by_qid.values():
        for candidate_id in front:
            freq[candidate_id] = freq.get(candidate_id, 0) + 1

    candidates = sorted(freq, key=lambda cid: (scores_by_id.get(cid, 0.0), cid))
    dominated: Set[int] = set()

    changed = True
    while changed:
        changed = False
        for candidate_id in candidates:
            if candidate_id in dominated:
                continue
            if _is_dominated(candidate_id, set(candidates).difference({candidate_id}).difference(dominated), frontier_by_qid):
                dominated.add(candidate_id)
                changed = True
                break

    return {
        qid: {candidate_id for candidate_id in front if candidate_id not in dominated}
        for qid, front in frontier_by_qid.items()
        if any(candidate_id not in dominated for candidate_id in front)
    }


def _select_candidate_from_pareto_front(
    frontier_by_qid: Mapping[str, Set[int]],
    scores_by_id: Mapping[int, float],
    rng: random.Random,
) -> int:
    reduced = _remove_dominated_candidates(frontier_by_qid, scores_by_id)
    sampling_pool: List[int] = []
    for front in reduced.values():
        for candidate_id in front:
            sampling_pool.append(candidate_id)
    if not sampling_pool:
        raise ValueError("Pareto-front pool is empty; cannot sample.")
    return rng.choice(sampling_pool)


class PromptPool(Generic[T]):
    def __init__(
        self,
        *,
        rng: Optional[random.Random] = None,
        max_size: Optional[int] = None,
    ):
        self.rng = rng or random.Random(0)
        self.max_size = max_size
        self.items: Dict[int, T] = {}
        self.scores_by_id: Dict[int, float] = {}
        self.frontier_by_qid: Dict[str, Set[int]] = {}
        self.next_id = 0

    def __len__(self) -> int:
        return len(self.items)

    def all_items(self) -> List[T]:
        return [self.items[candidate_id] for candidate_id in sorted(self.items)]

    def get(self, candidate_id: int) -> T:
        return self.items[candidate_id]

    def best(self) -> T:
        if not self.items:
            raise ValueError("Pool is empty.")
        best_id = max(self.scores_by_id, key=lambda candidate_id: (self.scores_by_id[candidate_id], -candidate_id))
        return self.items[best_id]

    def sample(self) -> T:
        if self.frontier_by_qid:
            candidate_id = _select_candidate_from_pareto_front(self.frontier_by_qid, self.scores_by_id, self.rng)
            return self.items[candidate_id]
        if not self.items:
            raise ValueError("Pool is empty.")
        fallback_id = max(self.scores_by_id, key=lambda candidate_id: (self.scores_by_id[candidate_id], -candidate_id))
        return self.items[fallback_id]

    def add_seed(self, item: T, *, correct_qids: Sequence[str], score: float) -> int:
        candidate_id = self.next_id
        self.next_id += 1
        if hasattr(item, "candidate_id"):
            setattr(item, "candidate_id", candidate_id)
        self.items[candidate_id] = item
        self.scores_by_id[candidate_id] = float(score)
        for qid in sorted(set(str(qid) for qid in correct_qids)):
            self.frontier_by_qid.setdefault(qid, set()).add(candidate_id)
        return candidate_id

    def try_add(self, item: T, *, correct_qids: Sequence[str], score: float) -> bool:
        normalized_qids = sorted(set(str(qid) for qid in correct_qids))
        if not normalized_qids:
            return False

        candidate_id = self.next_id
        self.next_id += 1
        if hasattr(item, "candidate_id"):
            setattr(item, "candidate_id", candidate_id)

        candidate_frontier = deepcopy(self.frontier_by_qid)
        for qid in normalized_qids:
            candidate_frontier.setdefault(qid, set()).add(candidate_id)
        candidate_scores = dict(self.scores_by_id)
        candidate_scores[candidate_id] = float(score)
        reduced = _remove_dominated_candidates(candidate_frontier, candidate_scores)
        survivors = {candidate for front in reduced.values() for candidate in front}
        if candidate_id not in survivors:
            return False

        self.items[candidate_id] = item
        self.scores_by_id = {cid: candidate_scores[cid] for cid in survivors}
        self.frontier_by_qid = {
            qid: {cid for cid in front if cid in survivors}
            for qid, front in reduced.items()
        }
        self.items = {cid: self.items[cid] for cid in sorted(self.items) if cid in survivors}

        if self.max_size is not None and len(self.items) > self.max_size:
            self._trim_to_max_size()
        return True

    def _trim_to_max_size(self) -> None:
        if self.max_size is None:
            return
        while len(self.items) > self.max_size:
            removable = [
                candidate_id
                for candidate_id in self.items
                if all(len(front) > 1 or candidate_id not in front for front in self.frontier_by_qid.values())
            ]
            if not removable:
                break
            worst_id = min(removable, key=lambda candidate_id: (self.scores_by_id[candidate_id], candidate_id))
            del self.items[worst_id]
            del self.scores_by_id[worst_id]
            self.frontier_by_qid = {
                qid: {candidate_id for candidate_id in front if candidate_id != worst_id}
                for qid, front in self.frontier_by_qid.items()
                if any(candidate_id != worst_id for candidate_id in front)
            }

    def reduced_frontier(self) -> Dict[str, Set[int]]:
        return _remove_dominated_candidates(self.frontier_by_qid, self.scores_by_id)

    def coverage_by_candidate(self) -> Dict[int, Set[str]]:
        reduced = self.reduced_frontier()
        return {
            candidate_id: {qid for qid, front in reduced.items() if candidate_id in front}
            for candidate_id in self.items
        }

    def select_prompt_set(self, prompt_set_size: int) -> List[Tuple[int, T, int]]:
        if not self.items:
            raise ValueError("Prompt pool is empty.")

        reduced_frontier = self.reduced_frontier()
        coverage_by_id = self.coverage_by_candidate()
        selected_ids: List[int] = []
        uncovered_qids = set(reduced_frontier)

        while len(selected_ids) < prompt_set_size:
            remaining_ids = [candidate_id for candidate_id in self.items if candidate_id not in selected_ids]
            if not remaining_ids or (selected_ids and not uncovered_qids):
                break
            best_id = max(
                remaining_ids,
                key=lambda candidate_id: (
                    len(coverage_by_id.get(candidate_id, set()) & uncovered_qids),
                    len(coverage_by_id.get(candidate_id, set())),
                    self.scores_by_id.get(candidate_id, 0.0),
                    -candidate_id,
                ),
            )
            selected_ids.append(best_id)
            uncovered_qids -= coverage_by_id.get(best_id, set())

        if not selected_ids:
            selected_ids = [
                max(
                    self.scores_by_id,
                    key=lambda candidate_id: (self.scores_by_id[candidate_id], -candidate_id),
                )
            ]

        while len(selected_ids) < prompt_set_size:
            selected_ids.append(selected_ids[-1])

        return [
            (candidate_id, self.items[candidate_id], len(coverage_by_id.get(candidate_id, set())))
            for candidate_id in selected_ids[:prompt_set_size]
        ]

    def save(self, path: Path, *, serializer) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "next_id": self.next_id,
            "scores_by_id": self.scores_by_id,
            "frontier_by_qid": {qid: sorted(front) for qid, front in self.frontier_by_qid.items()},
            "items": {str(candidate_id): serializer(item) for candidate_id, item in self.items.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
