from __future__ import annotations

from typing import Tuple


def pick_mutation_step_winner(
    *,
    mutation_text: str,
    mutation_score: float,
    reflection_text: str,
    reflection_score: float,
) -> Tuple[str, str, float]:
    if reflection_score > mutation_score:
        return "reflection", reflection_text, reflection_score
    return "mutation", mutation_text, mutation_score
