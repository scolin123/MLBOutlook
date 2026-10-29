"""As-of feature builder: only seasons strictly before the target season are used."""

import numpy as np
import pandas as pd

ASOF_COLUMNS = ["player_id", "target_season", "max_source_season", "n_source_seasons"]


def build_asof_rows(df: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """One row per player with at least one season < target_season."""
    prior = df.loc[df["season"] < target_season]
    grouped = prior.groupby("player_id")["season"]
    out = pd.DataFrame(
        {
            "max_source_season": grouped.max(),
            "n_source_seasons": grouped.nunique(),
        }
    ).reset_index()
    out["target_season"] = target_season
    out = out[ASOF_COLUMNS].astype(
        {"player_id": "int64", "target_season": "int64",
         "max_source_season": "int64", "n_source_seasons": "int64"}
    )
    return out.sort_values("player_id").reset_index(drop=True)


def _league_woba(df: pd.DataFrame, seasons: list[int]) -> dict[int, float]:
    """PA-weighted mean wOBA over all players, per season (NaN if the season has no rows)."""
    out: dict[int, float] = {}
    for s in seasons:
        rows = df.loc[df["season"] == s]
        total_pa = rows["pa"].sum()
        out[s] = float((rows["woba"] * rows["pa"]).sum() / total_pa) if total_pa > 0 else np.nan
    return out


def feature_columns(lags: int = 3) -> list[str]:
    """Output column order of `build_features` (without target columns)."""
    cols = ["player_id", "target_season", "max_source_season", "age_target"]
    for k in range(1, lags + 1):
        cols += [
            f"woba_lag{k}",
            f"pa_lag{k}",
            f"k_pct_lag{k}",
            f"bb_pct_lag{k}",
            f"lg_woba_lag{k}",
            f"miss_lag{k}",
        ]
    return cols


def build_features(
    df: pd.DataFrame,
    target_season: int,
    *,
    lags: int = 3,
    min_pa_feature: int = 100,
    min_pa_target: int = 100,
    include_target: bool = True,
) -> pd.DataFrame:
    """Lag features for `target_season`, built only from seasons < target_season."""
    prior = df.loc[df["season"] < target_season]
    last = prior.loc[(prior["season"] == target_season - 1) & (prior["pa"] >= min_pa_feature)]
    base = last[["player_id", "age"]].drop_duplicates("player_id").sort_values("player_id")
    out = pd.DataFrame(
        {
            "player_id": base["player_id"].astype("int64").to_numpy(),
            "target_season": np.int64(target_season),
            "age_target": (base["age"] + 1).astype("int64").to_numpy(),
        }
    )
    lag_seasons = [target_season - k for k in range(1, lags + 1)]
    league = _league_woba(prior, lag_seasons)
    used = np.full(len(out), -1, dtype="int64")
    for k, season in enumerate(lag_seasons, start=1):
        rows = prior.loc[prior["season"] == season].drop_duplicates("player_id")
        rows = rows.set_index("player_id")
        ids = out["player_id"]
        have = ids.isin(rows.index)
        out[f"woba_lag{k}"] = ids.map(rows["woba"]).astype("float64")
        out[f"pa_lag{k}"] = ids.map(rows["pa"]).fillna(0).astype("int64")
        out[f"k_pct_lag{k}"] = ids.map(rows["k_pct"]).astype("float64")
        out[f"bb_pct_lag{k}"] = ids.map(rows["bb_pct"]).astype("float64")
        out[f"lg_woba_lag{k}"] = league[season]
        out[f"miss_lag{k}"] = (~have).astype("int64")
        if not np.isnan(league[season]):
            used = np.maximum(used, season)
    out["max_source_season"] = used
    out = out[feature_columns(lags)]
    if include_target:
        tgt = df.loc[(df["season"] == target_season) & (df["pa"] >= min_pa_target)]
        tgt = tgt.drop_duplicates("player_id").set_index("player_id")
        out["y_woba"] = out["player_id"].map(tgt["woba"]).astype("float64")
        out["y_pa"] = out["player_id"].map(tgt["pa"])
        out = out.loc[out["y_pa"].notna()].copy()
        out["y_pa"] = out["y_pa"].astype("int64")
    return out.reset_index(drop=True)
