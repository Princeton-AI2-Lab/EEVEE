import math as _math
from typing import Any, Dict, List, Optional


def sample_questions_proportional(
    samples: List[Dict[str, Any]],
    n: int,
    key: str = "_task_name",
    question_key: str = "question",
    seed: Optional[int] = None,
) -> List[str]:
    """Sample n questions from *samples*, keeping task proportions."""
    import random as _rnd

    rng = _rnd.Random(seed)
    if n <= 0 or not samples:
        return []

    groups: Dict[str, List[str]] = {}
    for s in samples:
        groups.setdefault(s.get(key, "?"), []).append(s.get(question_key, ""))
    total = len(samples)
    n = min(n, total)

    result: List[str] = []
    remaining = n
    group_items = sorted(groups.items())
    for i, (_, questions) in enumerate(group_items):
        if i == len(group_items) - 1:
            k = remaining
        else:
            k = max(1, _math.ceil(n * len(questions) / total))
            k = min(k, remaining, len(questions))
        result.extend(rng.sample(questions, k))
        remaining -= k
        if remaining <= 0:
            break

    rng.shuffle(result)
    return result
