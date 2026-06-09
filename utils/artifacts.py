from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dev_manifest(
    *,
    exp_dir: Path,
    best_candidate_path: Optional[Path],
) -> None:
    manifest = {
        "events_jsonl": "events.jsonl",
        "eval_cache_json": "eval_cache.json",
        "best_candidate_json": best_candidate_path.name if best_candidate_path is not None else None,
        "initial_test_results_json": "initial_test_results.json",
        "final_test_results_json": "final_test_results.json",
        "final_test_routes_jsonl": "final_test_routes.jsonl",
        "notes": {
            "best_candidate_json": "Best validation candidate containing router R and prompt set P.",
            "initial_test_results_json": "Per-task empty-prompt baseline test results.",
            "final_test_routes_jsonl": "Per test question router decisions for the best candidate.",
            "eval_cache_json": "Reusable per-artifact, per-split cached answers and correctness.",
        },
    }
    (exp_dir / "dev_artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
