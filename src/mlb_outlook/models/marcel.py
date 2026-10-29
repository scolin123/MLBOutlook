"""Simplified Marcel baseline (see docs/PLAN.md, Phase 3 spec)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

DEFAULT_WEIGHTS: tuple[float, ...] = (5, 4, 3)
DEFAULT_REGRESSION_PA = 1200.0


def age_factor(age: float) -> float:
    """Multiplicative age adjustment; equals 1 at age 29."""
    slope = 0.006 if age < 29 else 0.003
    return 1 + (29 - age) * slope


def marcel_rate(
    seasons: Sequence[tuple[float, float]],
    target_age: float,
    league_rate: float,
    *,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    regression_pa: float = DEFAULT_REGRESSION_PA,
) -> float:
    """Marcel rate for one player.

    `seasons` is most-recent-first `(pa, rate)` pairs; seasons beyond `len(weights)` are ignored.
    """
    num = 0.0
    den = 0.0
    for w, (pa, rate) in zip(weights, seasons, strict=False):
        if pa > 0:
            num += w * pa * rate
            den += w * pa
    regressed = (num + regression_pa * league_rate) / (den + regression_pa)
    return regressed * age_factor(target_age)


def project_marcel(
    df: pd.DataFrame,
    target_season: int,
    league_rates: Mapping[int, float],
    *,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    regression_pa: float = DEFAULT_REGRESSION_PA,
) -> pd.DataFrame:
    """Project every player in a long table (`player_id, season, age, pa, rate`).

    Seasons >= `target_season` are dropped (leakage guard). Age in the target season is the age
    in the player's last observed prior season plus the years elapsed since.
    """
    lookback = [target_season - 1 - i for i in range(len(weights))]
    lg_pairs = [
        (w, league_rates[s]) for w, s in zip(weights, lookback, strict=True) if s in league_rates
    ]
    if not lg_pairs:
        raise ValueError(f"no league rates available for lookback seasons {lookback}")
    lg = sum(w * r for w, r in lg_pairs) / sum(w for w, _ in lg_pairs)

    past = df[df["season"] < target_season]
    rows = []
    for player_id, grp in past.groupby("player_id", sort=True):
        last = grp.loc[grp["season"].idxmax()]
        target_age = last["age"] + (target_season - last["season"])
        by_season = grp.set_index("season")
        seasons = []
        for s in lookback:
            if s in by_season.index:
                rec = by_season.loc[s]
                seasons.append((float(rec["pa"]), float(rec["rate"])))
            else:
                seasons.append((0.0, 0.0))
        rate = marcel_rate(seasons, target_age, lg, weights=weights, regression_pa=regression_pa)
        rows.append({"player_id": player_id, "target_season": target_season, "marcel_rate": rate})
    return pd.DataFrame(rows, columns=["player_id", "target_season", "marcel_rate"])
