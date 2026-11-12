"""k nearest historical player-seasons, standardized, as of the query season.

A query is one player-season `(player_id, season)`: "who looked like this hitter did in
`season`, and what happened to them the year after?"

**The as-of rule.** Nothing after the query season is read, ever. The forecast date is the end
of `season`'s regular season, so `season` is complete and known; everything after it is not.
That is the line the rule draws:

- **Comp seasons** are strictly before `season`, so at most `season - 1`.
- **Comp outcomes** -- the following year we show -- run through `season` itself. A comp season
  of `season - 1` has its outcome in `season`, which is known at the forecast date.
- **League means and standardization statistics** come from the comp pool and from each
  season's own rows, none of them later than `season`.
- From `season` itself three things are read: the query's own row, that season's league means
  (to place the query league-relative), and the outcomes of `season - 1` comps.

An earlier version stopped comp seasons at `season - 2` so that outcomes too were strictly
before `season`. That was stricter than the forecast date requires and it threw away the most
recent comp year -- the one most likely to resemble a current hitter -- so the rule is now the
one above. The as-of guarantee is unchanged in substance: a comp and its outcome are both
knowable when the forecast is made.

**Features.** Age plus league-relative wOBA, xwOBA, K%, BB%, average exit velocity, barrel%
and hard-hit%. Each rate is relative to that season's PA-weighted league mean, so a 2016
line and a 2023 line are compared on the same footing. Each feature is z-scored with the
pool's mean and standard deviation and weighted equally; distance is Euclidean.

**Population.** Pool seasons need `min_pa` PA (100, the project's floor). 2020 is excluded as
a comp season by default because its 60-game rates are much noisier. A comp who did not play
the following year is still listed, with `next_pa = 0` and no outcome, because leaving them
out would make the comps' futures look better than they were.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

RATE_COLUMNS = ("woba", "xwoba", "k_pct", "bb_pct", "avg_ev", "barrel_pct", "hard_hit_pct")
COMP_FEATURES = ["age", *(f"{c}_rel" for c in RATE_COLUMNS)]
DEFAULT_K = 10
MIN_PA = 100
EXCLUDE_SEASONS = (2020,)


def league_relative(df: pd.DataFrame) -> pd.DataFrame:
    """Add `<rate>_rel` = rate minus that season's PA-weighted league mean, from `df` alone."""
    out = df.copy()
    for col in RATE_COLUMNS:
        ok = out[col].notna() & (out["pa"] > 0)
        weighted = (out[col] * out["pa"]).where(ok)
        means = weighted.groupby(out["season"]).sum() / out["pa"].where(ok).groupby(
            out["season"]
        ).sum()
        out[f"{col}_rel"] = out[col] - out["season"].map(means)
    return out


def comp_pool(
    df: pd.DataFrame,
    season: int,
    *,
    min_pa: int = MIN_PA,
    exclude_seasons: Sequence[int] = EXCLUDE_SEASONS,
) -> pd.DataFrame:
    """Candidate comp seasons for a query in `season`, each with its following season.

    Comp seasons are strictly before `season`; their outcomes run through `season` itself,
    which is complete at the forecast date. Reads nothing after `season`.
    """
    history = league_relative(df.loc[df["season"] < season])
    pool = history.loc[
        (history["season"] <= season - 1)
        & (history["pa"] >= min_pa)
        & ~history["season"].isin(list(exclude_seasons))
    ].dropna(subset=COMP_FEATURES)
    # Outcomes reach one season further than the comps do: a comp season of `season - 1` has
    # its following year in `season`, which is known when the forecast is made. Each season's
    # league mean is computed from its own rows, so widening the frame by one season does not
    # move the `_rel` columns of any earlier season.
    outcomes = league_relative(df.loc[df["season"] <= season])
    nxt = outcomes[["player_id", "season", "pa", "woba", "woba_rel"]].rename(
        columns={"pa": "next_pa", "woba": "next_woba", "woba_rel": "next_woba_rel"}
    )
    nxt = nxt.assign(season=nxt["season"] - 1)
    pool = pool.merge(nxt, on=["player_id", "season"], how="left", validate="one_to_one")
    pool["next_season"] = pool["season"] + 1
    pool["next_pa"] = pool["next_pa"].fillna(0).astype(int)
    pool["delta_woba_rel"] = pool["next_woba_rel"] - pool["woba_rel"]
    return pool.reset_index(drop=True)


def query_row(df: pd.DataFrame, player_id: int, season: int) -> pd.Series:
    """The query player-season with its league-relative features (league from `season` only)."""
    same = league_relative(df.loc[df["season"] == season])
    row = same.loc[same["player_id"] == player_id]
    if row.empty:
        raise KeyError(f"no row for player {player_id} in {season}")
    return row.iloc[0]


def find_comps(
    df: pd.DataFrame,
    player_id: int,
    season: int,
    *,
    k: int = DEFAULT_K,
    min_pa: int = MIN_PA,
    exclude_seasons: Sequence[int] = EXCLUDE_SEASONS,
) -> pd.DataFrame:
    """The `k` nearest earlier player-seasons of other players, nearest first.

    Raises `ValueError` if the query is missing a feature: imputing one would invent a comp.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    query = query_row(df, player_id, season)
    missing = [c for c in COMP_FEATURES if pd.isna(query[c])]
    if missing:
        raise ValueError(f"query {player_id}/{season} is missing {missing}")
    pool = comp_pool(df, season, min_pa=min_pa, exclude_seasons=exclude_seasons)
    pool = pool.loc[pool["player_id"] != player_id]
    if pool.empty:
        return pool.assign(distance=pd.Series(dtype=float))

    x = pool[COMP_FEATURES].to_numpy(dtype=float)
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    z = (x - mean) / sd
    q = (query[COMP_FEATURES].to_numpy(dtype=float) - mean) / sd
    pool = pool.assign(distance=np.sqrt(((z - q) ** 2).sum(axis=1)))
    nearest = pool.sort_values(["distance", "season", "player_id"], kind="stable").head(k)
    keep = [
        "player_id", *(["name"] if "name" in pool.columns else []), "season", "age", "pa",
        "woba", "xwoba", *COMP_FEATURES[1:], "distance",
        "next_season", "next_pa", "next_woba", "next_woba_rel", "delta_woba_rel",
    ]
    return nearest[keep].reset_index(drop=True)


def comps_summary(comps: pd.DataFrame, *, min_next_pa: int = MIN_PA) -> dict[str, float]:
    """What the comps did next, in one line: how many played, and their median change."""
    played = comps.loc[comps["next_pa"] >= min_next_pa]
    return {
        "n": len(comps),
        "n_played_next": len(played),
        "median_next_woba_rel": float(played["next_woba_rel"].median()) if len(played) else np.nan,
        "median_delta_woba_rel": (
            float(played["delta_woba_rel"].median()) if len(played) else np.nan
        ),
    }
