from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CacheKey = Tuple[str, str, str]


@dataclass
class CachedEvaluation:
    answer: str
    is_correct: bool


class EvaluationCache:
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self._items: Dict[CacheKey, CachedEvaluation] = {}
        if self.storage_path is not None:
            self.load()

    def load(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        self._items = {
            (str(item["artifact_hash"]), str(item["split_name"]), str(item["qid"])): CachedEvaluation(
                answer=str(item.get("answer", "")),
                is_correct=bool(item.get("is_correct", False)),
            )
            for item in payload.get("items", [])
        }

    def save(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        items = []
        for (artifact_hash, split_name, qid), entry in sorted(self._items.items()):
            row = asdict(entry)
            row.update(
                {
                    "artifact_hash": artifact_hash,
                    "split_name": split_name,
                    "qid": qid,
                }
            )
            items.append(row)
        self.storage_path.write_text(
            json.dumps({"items": items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, artifact_hash: str, split_name: str, qid: str) -> Optional[CachedEvaluation]:
        return self._items.get((str(artifact_hash), str(split_name), str(qid)))

    def put(self, artifact_hash: str, split_name: str, qid: str, *, answer: str, is_correct: bool) -> None:
        self._items[(str(artifact_hash), str(split_name), str(qid))] = CachedEvaluation(
            answer=str(answer),
            is_correct=bool(is_correct),
        )

    def get_many(self, artifact_hash: str, split_name: str, qids: Iterable[str]) -> Dict[str, CachedEvaluation]:
        out: Dict[str, CachedEvaluation] = {}
        for qid in qids:
            entry = self.get(artifact_hash, split_name, qid)
            if entry is not None:
                out[str(qid)] = entry
        return out

    def prompt_hashes(self) -> List[str]:
        return sorted({artifact_hash for artifact_hash, _, _ in self._items})
