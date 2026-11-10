"""Reporting-only diagnostics: who is in the population, and where the errors sit.

Nothing here changes a forecast. These are readings of a fold that the headline tables do
not carry, added before the design freeze so the holdout report can print them for fold 3
without anything being chosen afterwards:

- `population_accounting` -- of the players eligible by prior-season PA, how many played the
  target season at all. The scored rows are the ones who reached the target-season PA floor,
  so every probability and interval in this project is conditional on playing again, and this
  is the number that says how big that condition is.
- `coverage_by_group` and its two callers -- interval coverage inside a slice of the
  population (playing time, age), where an aggregate inside 75-85% can hide a slice outside
  it.

All of it reads the feature frames and the season table; none of it fits anything.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from mlb_outlook.config import Fold, FoldsConfig
from mlb_outlook.evaluation.harness import interval_metrics
from mlb_outlook.models.intervals import DEFAULT_PA_EDGES
from mlb_outlook.models.intervals import pa_buckets as _pa_buckets

POPULATION_COLUMNS = ["fold", "target_season", "eligible", "bucket", "n", "share"]
COVERAGE_COLUMNS = ["grouping", "bucket", "range", "n", "coverage", "width"]
# Age brackets for the coverage read: under 26, 26-30, 31 and up. The conventional pre-peak /
# peak / post-peak split, not fitted to anything -- the aging curve in `MarcelBaseline` turns
# at 29 and these are the brackets a scouting report would use.
DEFAULT_AGE_EDGES = (26, 31)


def population_accounting(
    df: pd.DataFrame, fold: Fold, cfg: FoldsConfig, *, target_season: int | None = None
) -> pd.DataFrame:
    """Attrition between eligibility and a scoreable outcome, for one target season.

    Eligible means what `build_features` means by it: a season at `target - 1` with at least
    `min_pa.feature_season` plate appearances. Those players are then split by what they did
    in the target season -- none, some but under `min_pa.target_season`, or enough to be
    scored. The last count is exactly the fold's test row count.

    This reads target-season plate appearances, so for a fold's own test target it is
    test-season data: playing time, not performance, but test-season data all the same. It is
    called for the selection folds from `run_folds.py` and for the holdout only from
    `run_holdout.py`.
    """
    target = fold.test_target if target_season is None else target_season
    feature_floor = cfg.min_pa.feature_season
    target_floor = cfg.min_pa.target_season
    prior = df.loc[(df["season"] == target - 1) & (df["pa"] >= feature_floor)]
    eligible = prior.drop_duplicates("player_id")["player_id"]
    if eligible.empty:
        raise ValueError(f"no player reached {feature_floor} PA in season {target - 1}")
    played = df.loc[df["season"] == target].drop_duplicates("player_id").set_index("player_id")
    pa = eligible.map(played["pa"]).fillna(0).to_numpy(dtype=float)

    counts = {
        "0 PA (did not play)": int((pa == 0).sum()),
        f"1-{target_floor - 1} PA (below the floor)": int(
            ((pa > 0) & (pa < target_floor)).sum()
        ),
        f"{target_floor}+ PA (scored)": int((pa >= target_floor).sum()),
    }
    total = len(eligible)
    if sum(counts.values()) != total:
        raise AssertionError("the three buckets must account for every eligible player")
    rows = [
        {
            "fold": fold.name,
            "target_season": int(target),
            "eligible": total,
            "bucket": bucket,
            "n": n,
            "share": n / total,
        }
        for bucket, n in counts.items()
    ]
    return pd.DataFrame(rows, columns=POPULATION_COLUMNS)


def bucket_labels(edges: Sequence[float], unit: str) -> list[str]:
    """Readable names for digitize indices: below the first edge, between, above the last."""
    cuts = [int(e) for e in edges]
    names = [f"{unit} < {cuts[0]}"]
    names += [f"{lo} <= {unit} < {hi}" for lo, hi in zip(cuts, cuts[1:], strict=False)]
    names.append(f"{unit} >= {cuts[-1]}")
    return names


def coverage_by_group(
    bounds: pd.DataFrame,
    y,
    groups: np.ndarray,
    names: Sequence[str],
    *,
    grouping: str,
) -> pd.DataFrame:
    """Coverage and mean width inside each group, skipping groups with no rows.

    An aggregate coverage inside the pre-registered 75-85% band can hide a slice outside it,
    and the slices that matter are the ones whose residual spread the bucketing does not
    model: part-time hitters, and the age extremes.
    """
    lo = bounds["lo"].to_numpy(dtype=float)
    hi = bounds["hi"].to_numpy(dtype=float)
    truth = np.asarray(y, dtype=float)
    if not lo.shape == hi.shape == truth.shape == groups.shape:
        raise ValueError("bounds, y and groups must be aligned 1-D arrays of the same length")
    rows = []
    for index in range(len(names)):
        inside = groups == index
        if not inside.any():
            continue
        coverage, width = interval_metrics(lo[inside], hi[inside], truth[inside])
        rows.append(
            {
                "grouping": grouping,
                "bucket": index,
                "range": names[index],
                "n": int(inside.sum()),
                "coverage": coverage,
                "width": width,
            }
        )
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def coverage_by_pa(
    bounds: pd.DataFrame, test: pd.DataFrame, *, edges: Sequence[float] = DEFAULT_PA_EDGES
) -> pd.DataFrame:
    """Coverage inside each lag-1 PA bucket -- the buckets the intervals are built on."""
    return coverage_by_group(
        bounds,
        test["y_woba"],
        _pa_buckets(test, edges),
        bucket_labels(edges, "pa_lag1"),
        grouping="pa_lag1",
    )


def coverage_by_age(
    bounds: pd.DataFrame, test: pd.DataFrame, *, edges: Sequence[float] = DEFAULT_AGE_EDGES
) -> pd.DataFrame:
    """Coverage inside each target-season age bracket.

    Age is in the feature set but not in the interval's bucketing, so a bracket outside the
    band would say the width ought to depend on it -- a v2 finding, not something to fix after
    the freeze.
    """
    groups = np.digitize(test["age_target"].to_numpy(dtype=float), np.asarray(edges, float))
    return coverage_by_group(
        bounds,
        test["y_woba"],
        groups,
        bucket_labels(edges, "age"),
        grouping="age_target",
    )
