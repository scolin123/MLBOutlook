"""As-of feature builder: only seasons strictly before the target season are used."""

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
