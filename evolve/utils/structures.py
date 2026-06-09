from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def stable_hash_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def stable_hash_payload(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass
class DatasetEval:
    split_name: str
    score: float
    correct: int
    total: int
    correctness_by_qid: Dict[str, bool] = field(default_factory=dict)
    answers_by_qid: Dict[str, str] = field(default_factory=dict)

    @property
    def correct_qids(self) -> List[str]:
        return sorted(qid for qid, ok in self.correctness_by_qid.items() if ok)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetEval":
        return cls(
            split_name=str(data.get("split_name", "")),
            score=float(data.get("score", 0.0)),
            correct=int(data.get("correct", 0)),
            total=int(data.get("total", 0)),
            correctness_by_qid={str(k): bool(v) for k, v in dict(data.get("correctness_by_qid", {})).items()},
            answers_by_qid={str(k): str(v) for k, v in dict(data.get("answers_by_qid", {})).items()},
        )


@dataclass
class PromptCandidate:
    slot_id: int
    prompt_text: str
    source: str
    train_eval: DatasetEval
    val_eval: DatasetEval
    parent_candidate_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    candidate_id: Optional[int] = None
    created_at: str = field(default_factory=utc_now)

    @property
    def artifact_hash(self) -> str:
        return stable_hash_text(self.prompt_text)

    @property
    def val_correct_qids(self) -> List[str]:
        return self.val_eval.correct_qids

    @property
    def val_score(self) -> float:
        return self.val_eval.score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "prompt_text": self.prompt_text,
            "source": self.source,
            "train_eval": self.train_eval.to_dict(),
            "val_eval": self.val_eval.to_dict(),
            "parent_candidate_ids": list(self.parent_candidate_ids),
            "metadata": dict(self.metadata),
            "candidate_id": self.candidate_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptCandidate":
        return cls(
            slot_id=int(data["slot_id"]),
            prompt_text=str(data.get("prompt_text", "")),
            source=str(data.get("source", "")),
            train_eval=DatasetEval.from_dict(dict(data.get("train_eval", {}))),
            val_eval=DatasetEval.from_dict(dict(data.get("val_eval", {}))),
            parent_candidate_ids=[int(v) for v in data.get("parent_candidate_ids", [])],
            metadata=dict(data.get("metadata", {})),
            candidate_id=int(data["candidate_id"]) if data.get("candidate_id") is not None else None,
            created_at=str(data.get("created_at", utc_now())),
        )


@dataclass
class RouterCandidate:
    router_prompt: str
    source: str
    train_eval: DatasetEval
    val_eval: DatasetEval
    train_route_by_qid: Dict[str, int]
    val_route_by_qid: Dict[str, int]
    accuracy_score: float = 0.0
    consistency_score: float = 0.0
    balance_score: float = 0.0
    router_score: float = 0.0
    parent_candidate_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    candidate_id: Optional[int] = None
    created_at: str = field(default_factory=utc_now)

    @property
    def artifact_hash(self) -> str:
        return stable_hash_text(self.router_prompt)

    @property
    def val_correct_qids(self) -> List[str]:
        return self.val_eval.correct_qids

    @property
    def val_score(self) -> float:
        return self.val_eval.score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "router_prompt": self.router_prompt,
            "source": self.source,
            "train_eval": self.train_eval.to_dict(),
            "val_eval": self.val_eval.to_dict(),
            "train_route_by_qid": dict(self.train_route_by_qid),
            "val_route_by_qid": dict(self.val_route_by_qid),
            "accuracy_score": self.accuracy_score,
            "consistency_score": self.consistency_score,
            "balance_score": self.balance_score,
            "router_score": self.router_score,
            "parent_candidate_ids": list(self.parent_candidate_ids),
            "metadata": dict(self.metadata),
            "candidate_id": self.candidate_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouterCandidate":
        return cls(
            router_prompt=str(data.get("router_prompt", "")),
            source=str(data.get("source", "")),
            train_eval=DatasetEval.from_dict(dict(data.get("train_eval", {}))),
            val_eval=DatasetEval.from_dict(dict(data.get("val_eval", {}))),
            train_route_by_qid={str(k): int(v) for k, v in dict(data.get("train_route_by_qid", {})).items()},
            val_route_by_qid={str(k): int(v) for k, v in dict(data.get("val_route_by_qid", {})).items()},
            accuracy_score=float(data.get("accuracy_score", 0.0)),
            consistency_score=float(data.get("consistency_score", 0.0)),
            balance_score=float(data.get("balance_score", 0.0)),
            router_score=float(data.get("router_score", 0.0)),
            parent_candidate_ids=[int(v) for v in data.get("parent_candidate_ids", [])],
            metadata=dict(data.get("metadata", {})),
            candidate_id=int(data["candidate_id"]) if data.get("candidate_id") is not None else None,
            created_at=str(data.get("created_at", utc_now())),
        )


@dataclass
class EEVEECandidate:
    router_prompt: str
    prompt_set: List[str]
    prompt_set_ids: List[str]
    train_eval: DatasetEval
    val_eval: DatasetEval
    train_route_by_qid: Dict[str, int]
    val_route_by_qid: Dict[str, int]
    parent_candidate_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    candidate_id: Optional[int] = None
    created_at: str = field(default_factory=utc_now)

    @property
    def artifact_hash(self) -> str:
        return stable_hash_payload(
            {
                "router_prompt": self.router_prompt,
                "prompt_set": list(self.prompt_set),
            }
        )

    @property
    def val_correct_qids(self) -> List[str]:
        return self.val_eval.correct_qids

    @property
    def val_score(self) -> float:
        return self.val_eval.score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "router_prompt": self.router_prompt,
            "prompt_set": list(self.prompt_set),
            "prompt_set_ids": list(self.prompt_set_ids),
            "train_eval": self.train_eval.to_dict(),
            "val_eval": self.val_eval.to_dict(),
            "train_route_by_qid": dict(self.train_route_by_qid),
            "val_route_by_qid": dict(self.val_route_by_qid),
            "parent_candidate_ids": list(self.parent_candidate_ids),
            "candidate_id": self.candidate_id,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EEVEECandidate":
        parent_ids = data.get("parent_candidate_ids", [])
        candidate_id = data.get("candidate_id")
        return cls(
            router_prompt=str(data.get("router_prompt", "")),
            prompt_set=[str(v) for v in data.get("prompt_set", [])],
            prompt_set_ids=[str(v) for v in data.get("prompt_set_ids", [])],
            train_eval=DatasetEval.from_dict(dict(data.get("train_eval", {}))),
            val_eval=DatasetEval.from_dict(dict(data.get("val_eval", {}))),
            train_route_by_qid={str(k): int(v) for k, v in dict(data.get("train_route_by_qid", {})).items()},
            val_route_by_qid={str(k): int(v) for k, v in dict(data.get("val_route_by_qid", {})).items()},
            parent_candidate_ids=[int(v) for v in parent_ids],
            metadata=dict(data.get("metadata", {})),
            candidate_id=int(candidate_id) if candidate_id is not None else None,
            created_at=str(data.get("created_at", utc_now())),
        )
