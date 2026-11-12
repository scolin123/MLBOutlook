"""Local-file data access for the app: the prod2027 forecasts, plus the two scored folds.

Everything player-level reads `artifacts/prod2027_*` and the parquet snapshot. The app shows
the **2027 forecasts**: made from data through 2026, with no outcome column anywhere, because
2027 has not been played. They are the project's prospective test -- committed before the
answer exists -- and `scripts/export_prod2027_forecasts.py` is what wrote them, from the
configuration `design-freeze-v1` pinned.

Seasons load through 2026, which is the last season the forecast reads. That is a deliberate
change from the backtest app, where 2026 was hidden: while fold 3 was the pending honest
test, browsing its season per player would have turned the single look into exactly the
fishing expedition the freeze ruled out. Fold 3 has now been run once and reported, FAIL and
all, so 2026 is no longer a held-out season -- it is the information set the 2027 forecast is
built on, and a forecast whose inputs the reader cannot see is not inspectable.

What is still out of reach is the holdout's *scoring*: `fold3_players.csv` is per-player 2026
forecast-against-outcome, and nothing in the app may read it. `holdout_artifact` is the only
route to any fold 3 file and `HOLDOUT_ARTIFACTS` is the whole allowlist, which holds the six
aggregate tables and no per-player one. The model-performance page is the only page that
reports a score at all: fold 2 (2025, the selection fold) and fold 3 (2026, the holdout), with
the pre-registered verdict read off the artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlb_outlook.comps import comps_summary, find_comps

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data" / "processed" / "hitter_seasons.parquet"

# The live forecast: the prospective test, and everything the player page and the board show.
FORECAST_RUN = "prod2027"
FORECAST_SEASON = 2027
FORECAST_FEATURES_THROUGH = 2026
# The last season with an outcome, and the last the forecast reads. Nothing later exists.
LAST_VISIBLE_SEASON = FORECAST_FEATURES_THROUGH

# The selection fold, scored. Reported on the model-performance page only.
FOLD = "fold2"
TARGET_SEASON = 2025
FEATURES_THROUGH = 2024

# The design-freeze holdout, scored. Aggregates only: see the module docstring.
HOLDOUT_FOLD = "fold3"
HOLDOUT_TARGET_SEASON = 2026
HOLDOUT_FEATURES_THROUGH = 2025
HOLDOUT_TAG = "design-freeze-v1"
HOLDOUT_ARTIFACTS = (
    "point", "verdict", "interval_coverage", "probabilities", "reliability", "population",
)

FORECAST_LABEL = (
    f"{FORECAST_SEASON} forecasts, made with data through {FORECAST_FEATURES_THROUGH}. "
    f"The {FORECAST_SEASON} season has not been played: there is no outcome on this page to "
    "check them against, and that is the point."
)
BACKTEST_LABEL = (
    f"Fold 2 backtest: {TARGET_SEASON} forecasts made from data through {FEATURES_THROUGH}, "
    f"shown beside what actually happened in {TARGET_SEASON}. Not a live projection."
)

PERCENTILE_METRICS = {
    "woba": ("wOBA", True),
    "xwoba": ("xwOBA", True),
    "k_pct": ("K%", False),
    "bb_pct": ("BB%", True),
    "avg_ev": ("Avg exit velo", True),
    "barrel_pct": ("Barrel%", True),
    "hard_hit_pct": ("Hard-hit%", True),
}


class MissingArtifacts(FileNotFoundError):
    """The forecasts or the fold 2 artifacts have not been generated yet."""


def required_files() -> list[Path]:
    fold2 = ["forecasts", "point", "interval_coverage", "probabilities", "reliability",
             "population", "verdict"]
    return [
        DATA,
        ARTIFACTS / f"{FORECAST_RUN}_forecasts.csv",
        ARTIFACTS / f"{FORECAST_RUN}_manifest.json",
        *(ARTIFACTS / f"{FOLD}_{n}.csv" for n in fold2),
    ]


def check_files() -> None:
    missing = [p for p in required_files() if not p.exists()]
    if missing:
        raise MissingArtifacts(
            "missing: " + ", ".join(str(p.relative_to(ROOT)) for p in missing)
            + ". Run scripts/run_holdout.py --rehearse fold2, then "
            "scripts/export_fold2_forecasts.py, then "
            "scripts/export_prod2027_forecasts.py."
        )


def seasons() -> pd.DataFrame:
    """The snapshot, cut at the last season the forecast reads."""
    df = pd.read_parquet(DATA)
    return df.loc[df["season"] <= LAST_VISIBLE_SEASON].reset_index(drop=True)


def forecasts() -> pd.DataFrame:
    """One row per forecast player: the 2027 rows the board and the player page share.

    No outcome column is added, and none exists in the file. `model_minus_marcel` is an alias
    the display code reads so the charts do not have to know the model's name.
    """
    f = pd.read_csv(ARTIFACTS / f"{FORECAST_RUN}_forecasts.csv")
    f["model_minus_marcel"] = f["ridge_rel_minus_marcel"]
    return f


def artifact(name: str) -> pd.DataFrame:
    """One fold 2 table. The selection fold's scores, for the model-performance page."""
    return pd.read_csv(ARTIFACTS / f"{FOLD}_{name}.csv")


def holdout_available() -> bool:
    """Has the holdout been run and its aggregate output committed?"""
    return all((ARTIFACTS / f"{HOLDOUT_FOLD}_{n}.csv").exists() for n in HOLDOUT_ARTIFACTS)


def holdout_artifact(name: str) -> pd.DataFrame:
    """One aggregate fold 3 table. Refuses anything not on `HOLDOUT_ARTIFACTS`.

    The guard is the point: `fold3_players.csv` exists beside these and is per-player 2026, so
    the allowlist is what stops a later page from reaching it through this function.
    """
    if name not in HOLDOUT_ARTIFACTS:
        raise ValueError(
            f"{name!r} is not an aggregate holdout table; allowed: {HOLDOUT_ARTIFACTS}"
        )
    return pd.read_csv(ARTIFACTS / f"{HOLDOUT_FOLD}_{name}.csv")


def verdict_label(verdict: pd.DataFrame) -> str:
    """PASS only if every pre-registered criterion passed, else FAIL.

    Read off the artifact rather than written down, so the app cannot state a verdict the
    numbers beside it do not support.
    """
    return "PASS" if bool(verdict["passed"].astype(bool).all()) else "FAIL"


def failed_criteria(verdict: pd.DataFrame) -> pd.DataFrame:
    return verdict.loc[~verdict["passed"].astype(bool)]


def forecast_manifest() -> dict:
    """What made the forecasts: model version, data hash, freeze tag, and when they were run."""
    path = ARTIFACTS / f"{FORECAST_RUN}_manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def history(df: pd.DataFrame, player_id: int) -> pd.DataFrame:
    return df.loc[df["player_id"] == player_id].sort_values("season").reset_index(drop=True)


def percentiles(df: pd.DataFrame, player_id: int, season: int = FORECAST_FEATURES_THROUGH,
                min_pa: int = 100) -> pd.DataFrame:
    """Percentile ranks (0-100, higher is better) among `season` hitters with `min_pa` PA."""
    pool = df.loc[(df["season"] == season) & (df["pa"] >= min_pa)]
    row = pool.loc[pool["player_id"] == player_id]
    out = []
    for col, (label, higher_better) in PERCENTILE_METRICS.items():
        if row.empty or pd.isna(row[col].iloc[0]):
            continue
        values = pool[col].dropna()
        value = float(row[col].iloc[0])
        share = float((values < value).mean() + 0.5 * (values == value).mean())
        pct = share if higher_better else 1.0 - share
        out.append({"metric": label, "value": value, "percentile": round(100 * pct)})
    return pd.DataFrame(out, columns=["metric", "value", "percentile"])


def comps(df: pd.DataFrame, player_id: int, k: int = 10) -> tuple[pd.DataFrame, dict] | None:
    """Comps as of the last season the forecast reads, which is its whole information set.

    Same rule and same `k` as the comps summary in the forecast file, so the table on the
    player page and the one-line summary in the committed CSV describe the same neighbours.
    """
    try:
        found = find_comps(df, player_id, FORECAST_FEATURES_THROUGH, k=k)
    except (KeyError, ValueError):
        return None
    return found, comps_summary(found)
