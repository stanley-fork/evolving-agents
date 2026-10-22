"""Small statistics helpers: OLS slope, mean, seed-level bootstrap."""
from __future__ import annotations

import random
from typing import Callable


def mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def ols_slope(xs: list[float], ys: list[float]) -> float | None:
    """Least-squares slope of ys on xs (None if fewer than 2 points or no x variance)."""
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if len(pts) < 2:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pts) / sxx


def percentile_ci(values: list[float], level: float = 0.95) -> tuple[float, float] | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2

    def q(p: float) -> float:
        # linear interpolation between order statistics
        pos = p * (len(vals) - 1)
        i = int(pos)
        frac = pos - i
        return vals[i] if i + 1 >= len(vals) else vals[i] + frac * (vals[i + 1] - vals[i])

    return q(lo_q), q(hi_q)


def bootstrap(stat: Callable[[dict[str, list[int]]], float | None],
              seeds_by_family: dict[str, list[int]], n_boot: int = 2000,
              boot_seed: int = 0) -> list[float | None]:
    """Resample seeds with replacement inside each family (paired across configs:
    every config is evaluated on the same resampled seeds). Deterministic in boot_seed."""
    rng = random.Random(boot_seed)
    fams = sorted(seeds_by_family)
    out = []
    for _ in range(n_boot):
        sample = {f: [rng.choice(seeds_by_family[f]) for _ in seeds_by_family[f]] for f in fams}
        out.append(stat(sample))
    return out
