"""Local-file data access for the app. The fold 2 backtest, plus the fold 3 holdout scores.

Everything player-level reads `artifacts/fold2_*` and the parquet snapshot. The demo is the
fold 2 backtest: 2025 forecasts made from data through 2024, scored against what happened in
2025. Seasons after 2025 are dropped on load, so **no 2026 player row reaches any page** --
not a forecast, not an outcome, not a comp.

The one exception is deliberate and aggregate-only: the model-performance page reports the
fold 3 (2026) holdout result, because a page that scored the model while hiding the verdict
from its one honest test would be misleading. It reads the aggregate tables -- `fold3_point`,
`fold3_verdict`, `fold3_interval_coverage`, `fold3_probabilities`, `fold3_reliability`,
`fold3_population` -- and never `fold3_players.csv`, which is per-player 2026. `holdout_*`
below is the only route to any of it, and `HOLDOUT_ARTIFACTS` is the whole list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlb_outlook.comps import comps_summary, find_comps

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data" / "processed" / "hitter_seasons.parquet"

FOLD = "fold2"
TARGET_SEASON = 2025
FEATURES_THROUGH = 2024
LAST_VISIBLE_SEASON = TARGET_SEASON  # the backtest's outcome season; 2026 is never shown

# The design-freeze holdout. Scores only: see the module docstring.
HOLDOUT_FOLD = "fold3"
HOLDOUT_TARGET_SEASON = 2026
HOLDOUT_FEATURES_THROUGH = 2025
HOLDOUT_TAG = "design-freeze-v1"
HOLDOUT_ARTIFACTS = (
    "point", "verdict", "interval_coverage", "probabilities", "reliability", "population",
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
    """The fold 2 artifacts have not been generated yet."""


def required_files() -> list[Path]:
    names = ["forecasts", "point", "interval_coverage", "probabilities", "reliability",
             "population", "verdict"]
    return [DATA, *(ARTIFACTS / f"{FOLD}_{n}.csv" for n in names)]


def check_files() -> None:
    missing = [p for p in required_files() if not p.exists()]
    if missing:
        raise MissingArtifacts(
            "missing: " + ", ".join(str(p.relative_to(ROOT)) for p in missing)
            + ". Run scripts/run_holdout.py --rehearse fold2, then "
            "scripts/export_fold2_forecasts.py."
        )


def seasons() -> pd.DataFrame:
    """The snapshot, cut at the backtest's outcome season."""
    df = pd.read_parquet(DATA)
    return df.loc[df["season"] <= LAST_VISIBLE_SEASON].reset_index(drop=True)


def forecasts() -> pd.DataFrame:
    """One row per scored player, with display columns the board and player page share."""
    f = pd.read_csv(ARTIFACTS / f"{FOLD}_forecasts.csv")
    f["model_minus_marcel"] = f["ridge_rel_minus_marcel"]
    f["actual_minus_forecast"] = f["y_woba"] - f["ridge_rel"]
    return f


def artifact(name: str) -> pd.DataFrame:
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


def manifest() -> dict:
    path = ARTIFACTS / f"{FOLD}_manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def history(df: pd.DataFrame, player_id: int) -> pd.DataFrame:
    return df.loc[df["player_id"] == player_id].sort_values("season").reset_index(drop=True)


def percentiles(df: pd.DataFrame, player_id: int, season: int = FEATURES_THROUGH,
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
    """Comps for the player's last season before the target (the forecast's information set)."""
    try:
        found = find_comps(df, player_id, FEATURES_THROUGH, k=k)
    except (KeyError, ValueError):
        return None
    return found, comps_summary(found)
