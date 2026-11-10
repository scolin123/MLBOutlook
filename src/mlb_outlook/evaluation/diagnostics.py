"""Reporting-only diagnostics: who is in the population, and where the errors sit.

Nothing here changes a forecast. These are readings of a fold that the headline tables do
not carry, added before the design freeze so the holdout report can print them for fold 3
without anything being chosen afterwards:

- `population_accounting` -- of the players eligible by prior-season PA, how many played the
  target season at all. The scored rows are the ones who reached the target-season PA floor,
  so every probability and interval in this project is conditional on playing again, and this
  is the number that says how big that condition is.

All of it reads the feature frames and the season table; none of it fits anything.
"""

from __future__ import annotations

import pandas as pd

from mlb_outlook.config import Fold, FoldsConfig

POPULATION_COLUMNS = ["fold", "target_season", "eligible", "bucket", "n", "share"]


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
