from __future__ import annotations

import math


def wilson_interval(successes: int, trials: int, z_score: float = 1.96) -> tuple[float, float]:
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials")
    if trials == 0:
        return 0.0, 1.0
    proportion = successes / trials
    denominator = 1.0 + z_score**2 / trials
    center = (proportion + z_score**2 / (2 * trials)) / denominator
    margin = (
        z_score
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_score**2 / (4 * trials**2)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def elo_difference(score: float) -> float:
    if not 0 <= score <= 1:
        raise ValueError("Elo score must be between zero and one")
    clipped = min(max(score, 1e-6), 1.0 - 1e-6)
    return 400.0 * math.log10(clipped / (1.0 - clipped))
