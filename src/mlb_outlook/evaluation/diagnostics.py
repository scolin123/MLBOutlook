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
- `per_player_forecasts` -- one row per scored player with both forecasts, the signed gap
  between them, and which of the two came closer. The MAE comparison is a mean of this table,
  so this is where a reader checks whether an edge of 0.0023 is a small edge everywhere or a
  large edge on a handful of players.

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
PLAYER_ID_COLUMNS = ["player_id", "name", "target_season", "age_target", "pa_lag1", "y_pa"]
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


def per_player_forecasts(
    test: pd.DataFrame,
    model: pd.Series,
    baseline: pd.Series,
    *,
    names: pd.Series | None = None,
    bounds: pd.DataFrame | None = None,
    model_name: str = "ridge_rel",
    baseline_name: str = "marcel",
) -> pd.DataFrame:
    """One row per scored player: both forecasts, their difference, and who came closer.

    The headline MAE comparison is the mean of the `advantage` column, which is
    ``|baseline error| - |model error|`` so that positive means the model was closer. Reading
    it per player answers something the mean cannot: whether a 0.0023 edge is a small edge on
    most hitters or a large edge on a few. Rows are sorted by the signed gap between the two
    forecasts, descending, so the biggest disagreements sit at both ends of the file.

    `names` maps player_id to a display name (the feature frame does not carry one), and
    `bounds` adds the published interval and whether it covered the outcome.
    """
    if not model.index.equals(test.index) or not baseline.index.equals(test.index):
        raise ValueError("model and baseline forecasts must share the index of the test frame")
    diff_column = f"{model_name}_minus_{baseline_name}"
    truth = test["y_woba"].astype(float)
    out = pd.DataFrame(
        {
            "player_id": test["player_id"].to_numpy(),
            "name": (
                test["player_id"].map(names) if names is not None else pd.NA
            ),
            "target_season": test["target_season"].to_numpy(),
            "age_target": test["age_target"].to_numpy(),
            "pa_lag1": test["pa_lag1"].to_numpy(),
            "y_pa": test["y_pa"].to_numpy(),
            "y_woba": truth.to_numpy(),
            model_name: model.astype(float).to_numpy(),
            baseline_name: baseline.astype(float).to_numpy(),
            diff_column: (model.astype(float) - baseline.astype(float)).to_numpy(),
            f"abs_err_{model_name}": (model.astype(float) - truth).abs().to_numpy(),
            f"abs_err_{baseline_name}": (baseline.astype(float) - truth).abs().to_numpy(),
        },
        index=test.index,
    )
    out["advantage"] = (
        out[f"abs_err_{baseline_name}"] - out[f"abs_err_{model_name}"]
    )
    if bounds is not None:
        out["lo"] = bounds["lo"].to_numpy(dtype=float)
        out["hi"] = bounds["hi"].to_numpy(dtype=float)
        out["covered"] = (out["y_woba"] >= out["lo"]) & (out["y_woba"] <= out["hi"])
    return out.sort_values(diff_column, ascending=False, kind="stable").reset_index(drop=True)


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
