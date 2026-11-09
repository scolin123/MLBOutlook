"""As-of feature builder: only seasons strictly before the target season are used.

Columns prefixed ``y_`` are target-season outcomes used for fitting and scoring only; they
are absent when ``include_target=False`` (production), so no model may read them in
``predict``. League-relative work therefore uses ``lg_woba_hat`` (prior seasons only) to
put a relative forecast back on the wOBA scale.
"""

from collections.abc import Collection

import numpy as np
import pandas as pd

ASOF_COLUMNS = ["player_id", "target_season", "max_source_season", "n_source_seasons"]
TARGET_COLUMNS = ["y_woba", "y_pa", "y_lg_woba"]
# Statcast quality of contact, as ingested from Savant.
QOC_COLUMNS = ("xwoba", "avg_ev", "barrel_pct", "hard_hit_pct")
# Season rates that get both an absolute and a league-relative lag. Contact quality drifts
# with the environment too (league exit velocity and barrel rate have both risen), so the
# relative form is what a model should lean on.
RELATIVE_SOURCES = ("woba", *QOC_COLUMNS)


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


def _league_rate(df: pd.DataFrame, seasons: list[int], col: str = "woba") -> dict[int, float]:
    """PA-weighted league mean of `col` per season.

    NaN if the season has no usable rows, which also covers a source column the table does
    not carry at all. Players missing `col` (no tracked batted balls) are left out of both
    sides of the weighted mean rather than counted as zero.
    """
    out: dict[int, float] = {}
    if col not in df.columns:
        return dict.fromkeys(seasons, np.nan)
    for s in seasons:
        rows = df.loc[(df["season"] == s) & df[col].notna()]
        total_pa = rows["pa"].sum()
        out[s] = float((rows[col] * rows["pa"]).sum() / total_pa) if total_pa > 0 else np.nan
    return out


def feature_columns(lags: int = 3) -> list[str]:
    """Output column order of `build_features` (without target columns)."""
    cols = ["player_id", "target_season", "max_source_season", "age_target", "lg_woba_hat"]
    for k in range(1, lags + 1):
        for name in RELATIVE_SOURCES:
            cols += [f"{name}_lag{k}", f"{name}_rel_lag{k}"]
        cols += [
            f"pa_lag{k}",
            f"k_pct_lag{k}",
            f"bb_pct_lag{k}",
            f"lg_woba_lag{k}",
            f"miss_lag{k}",
            f"short_lag{k}",
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
    flag_seasons: Collection[int] = (),
) -> pd.DataFrame:
    """Lag features for `target_season`, built only from seasons < target_season.

    `flag_seasons` are seasons whose rates come from an abnormal schedule (2020). They are
    not dropped: each lag gets a `short_lag{k}` indicator so a model can fit them apart.

    Each rate in `RELATIVE_SOURCES` yields `<name>_lag{k}` and `<name>_rel_lag{k}` (the lag
    minus its own season's league mean). A source column the table does not carry gives NaN
    lags, never zero.
    """
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
    leagues = {name: _league_rate(prior, lag_seasons, name) for name in RELATIVE_SOURCES}
    league = leagues["woba"]
    # Forecast-date estimate of the target season's league level: last season's, a random walk.
    # The target season's own league wOBA is unknowable here, so it is a `y_` column instead.
    out["lg_woba_hat"] = league[target_season - 1]
    used = np.full(len(out), -1, dtype="int64")
    for k, season in enumerate(lag_seasons, start=1):
        rows = prior.loc[prior["season"] == season].drop_duplicates("player_id")
        rows = rows.set_index("player_id")
        ids = out["player_id"]
        have = ids.isin(rows.index)
        for name in RELATIVE_SOURCES:
            lagged = ids.map(rows[name]).astype("float64") if name in rows else np.nan
            out[f"{name}_lag{k}"] = lagged
            out[f"{name}_rel_lag{k}"] = out[f"{name}_lag{k}"] - leagues[name][season]
        out[f"pa_lag{k}"] = ids.map(rows["pa"]).fillna(0).astype("int64")
        out[f"k_pct_lag{k}"] = ids.map(rows["k_pct"]).astype("float64")
        out[f"bb_pct_lag{k}"] = ids.map(rows["bb_pct"]).astype("float64")
        out[f"lg_woba_lag{k}"] = league[season]
        out[f"miss_lag{k}"] = (~have).astype("int64")
        out[f"short_lag{k}"] = np.int64(season in flag_seasons)
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
        out["y_lg_woba"] = _league_rate(df, [target_season])[target_season]
    return out.reset_index(drop=True)
