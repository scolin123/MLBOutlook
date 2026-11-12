"""The production 2027 forecasts: the prospective test of the frozen model.

Fold 3 was the retrospective honest test and it returned FAIL on criterion 2 -- the model
was not shown to beat Marcel on 2026. This is the other kind of test, and the only one left
that fold 3's failure does not already contaminate: forecasts for a season that has not been
played, published before the outcome is knowable. Nothing here can be tuned to 2027, because
2027 does not exist yet.

**The configuration is the frozen one, unchanged.** Point forecast `ridge_rel`, 80% intervals
`resid_q_pa`, breakout/regression off the same `BucketedResidualDistribution` at the
registered 0.030 threshold against `weighted3`, extended window. Not one of those is an
argument to this script. What differs from a fold is only what season the features stop at:
the `prod2027` block in `configs/folds.yaml` trains on target seasons 2023-2026 (2016-2022
too, on the extended window, which is the window the holdout ran on), and the forecast frame
is built for 2027 with `include_target=False`, so no `y_` column exists for a model to read.

**The gate.** Refused unless the frozen code and config are still exactly what
`design-freeze-v1` tagged and the snapshot still hashes to what the freeze commit pinned. The
point of a prospective test is that the thing being tested is the thing that was frozen; if
either moved, this would be a forecast from some other model wearing the frozen model's name.
HEAD is *not* required to be at the tag -- the app and the comps have moved on since, and
neither is frozen code -- but the frozen paths themselves must not have.

**Write once.** Existing output is never overwritten without `--overwrite`. These rows are
the prospective test, so editing them after they are committed would destroy the only thing
that makes them evidence.

Writes:
    artifacts/prod2027_forecasts.csv  one row per forecast player
    artifacts/prod2027_manifest.json  tag, commit, data hash, model version, timestamp

Usage:
    uv run python scripts/export_prod2027_forecasts.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_outlook.comps import comps_summary, find_comps
from mlb_outlook.config import FoldsConfig, load_folds_config
from mlb_outlook.db.versions import file_sha256, frozen_code_commit, model_version
from mlb_outlook.features.asof import build_features
from mlb_outlook.models.baselines import MarcelBaseline
from mlb_outlook.models.intervals import DEFAULT_LEVEL, ResidualQuantileIntervals
from mlb_outlook.models.probabilities import (
    THRESHOLD,
    BucketedResidualDistribution,
    label_probabilities,
    weighted3_baseline,
)

DATA = Path("data/processed/hitter_seasons.parquet")
CONFIG = Path("configs/folds.yaml")
ARTIFACTS = Path("artifacts")
RUN = "prod2027"

FREEZE_TAG = "design-freeze-v1"
# The two things the freeze commit (ab710f79) pins, in its message. Hard-coded rather than
# read from a manifest so that this check cannot be satisfied by a file written beside it.
FROZEN_DATA_SHA256 = "b164c087e6186d989f52bebdcc1b1153f25f7de5d8f2f6b8ddb003c906039cd8"
FROZEN_DATA_ROWS = 9804

MODEL = "ridge_rel"
BASELINE = "marcel"
DIFF_COLUMN = f"{MODEL}_minus_{BASELINE}"
COMPS_K = 10

COLUMNS = [
    "player_id", "name", "target_season", "age_target", "pa_lag1",
    MODEL, "lo", "hi", "width",
    BASELINE, DIFF_COLUMN,
    "weighted3", "threshold", "p_breakout", "p_stable", "p_regression",
    "comps_n", "comps_played_next", "comps_median_next_woba_rel",
    "comps_median_delta_woba_rel",
]


class ProductionRefused(RuntimeError):
    """Raised when the frozen configuration this run claims to use is not the frozen one."""


# ---- the gate ----


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def tag_commit(tag: str = FREEZE_TAG) -> str | None:
    """The commit `tag` points at, or None if there is no such tag (or no git)."""
    return _git("rev-list", "-n", "1", tag)


def check_frozen(data: Path = DATA) -> tuple[str, str]:
    """Refuse unless the frozen paths are still at the tagged commit and the data still matches.

    Returns (freeze_commit, data_sha256) so the caller can record what it checked.
    """
    expected = tag_commit()
    if expected is None:
        raise ProductionRefused(
            f"there is no {FREEZE_TAG!r} tag in this repository (or git is unavailable), so "
            "there is nothing to check the frozen configuration against."
        )
    actual = frozen_code_commit()
    if actual != expected:
        raise ProductionRefused(
            f"the frozen paths were last touched by {actual}, but {FREEZE_TAG} is {expected}. "
            "The production forecast must come from the configuration the freeze pinned; a "
            "commit to src/mlb_outlook/models, features, evaluation, config.py or "
            "configs/folds.yaml means it would not. Revert that change, or this is a new "
            "model and needs its own freeze."
        )
    digest = file_sha256(data)
    if digest != FROZEN_DATA_SHA256:
        raise ProductionRefused(
            f"{data} hashes to {digest}, but the freeze commit pinned "
            f"{FROZEN_DATA_SHA256}. Savant revises historical lines, so a re-pull is a "
            "different experiment: these forecasts would not be comparable to fold 3's."
        )
    rows = len(pd.read_parquet(data, columns=["season"]))
    if rows != FROZEN_DATA_ROWS:
        raise ProductionRefused(
            f"{data} has {rows} rows; the freeze commit pinned {FROZEN_DATA_ROWS}."
        )
    return expected, digest


def check_writable(paths: Sequence[Path], *, overwrite: bool) -> None:
    """Refuse to overwrite committed prospective forecasts."""
    existing = [p for p in paths if p.exists()]
    if existing and not overwrite:
        listing = ", ".join(str(p) for p in existing)
        raise ProductionRefused(
            f"{listing} already exist(s). These are the prospective test and are not edited "
            "after they are committed: a forecast that can be revised once the season starts "
            "is not a forecast. Pass --overwrite only to fix a run that was never committed."
        )


# ---- the forecasts ----


def production_frames(
    df: pd.DataFrame, cfg: FoldsConfig
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """(train, forecast, train_targets) for the `prod2027` block on the extended window.

    The train side is every production target season with its outcome, plus the extended
    window's pre-2023 seasons -- the same `extended.extra_train_targets` the holdout used,
    dropping any at or after the forecast target, of which there are none. The forecast side
    is built with ``include_target=False``: there is no 2027 outcome to leak, and this makes
    that structural rather than a promise.
    """
    prod = cfg.production
    kwargs = {
        "min_pa_feature": cfg.min_pa.feature_season,
        "min_pa_target": cfg.min_pa.target_season,
        "flag_seasons": tuple(cfg.extended.flag_seasons),
    }
    extras = {s for s in cfg.extended.extra_train_targets if s < prod.forecast_target}
    targets = sorted(set(prod.train_targets) | extras)
    train = pd.concat(
        [build_features(df, t, include_target=True, **kwargs) for t in targets],
        ignore_index=True,
    )
    forecast = build_features(
        df, prod.forecast_target, include_target=False, **kwargs
    )
    leaked = [c for c in forecast.columns if c.startswith("y_")]
    if leaked:
        raise ProductionRefused(f"the forecast frame carries outcome columns: {leaked}")
    return train, forecast, targets


def comps_frame(df: pd.DataFrame, player_ids: Sequence[int], season: int) -> pd.DataFrame:
    """One `comps_summary` per player, as of `season`, or NaN where comps cannot be built.

    `season` is the last season the forecast reads, so the comps are drawn from exactly the
    information the forecast had: comp seasons strictly before it, their outcomes through it.
    A player whose line is missing a comp feature gets NaN rather than an imputed neighbour.
    """
    rows = []
    for pid in player_ids:
        try:
            summary = comps_summary(find_comps(df, int(pid), season, k=COMPS_K))
        except (KeyError, ValueError):
            summary = {
                "n": 0,
                "n_played_next": 0,
                "median_next_woba_rel": np.nan,
                "median_delta_woba_rel": np.nan,
            }
        rows.append({"player_id": int(pid), **summary})
    return pd.DataFrame(rows).rename(
        columns={
            "n": "comps_n",
            "n_played_next": "comps_played_next",
            "median_next_woba_rel": "comps_median_next_woba_rel",
            "median_delta_woba_rel": "comps_median_delta_woba_rel",
        }
    )


def build(df: pd.DataFrame, cfg: FoldsConfig, *, comps: bool = True) -> pd.DataFrame:
    """Fit the frozen configuration on the production window and forecast the target season."""
    train, forecast, _ = production_frames(df, cfg)
    y = train["y_woba"]

    intervals = ResidualQuantileIntervals().fit(train, y)
    point = intervals.predict(forecast)
    bounds = intervals.predict_interval(forecast)
    marcel = MarcelBaseline().fit(train, y).predict(forecast)

    dist = BucketedResidualDistribution().fit(train, y)
    probs = label_probabilities(
        dist, forecast, baseline=weighted3_baseline(forecast), threshold=THRESHOLD
    )
    # The interval model's point fit and the distribution's are the same `relative_ridge` fit
    # on the same rows. If they disagree, one of them saw different data and the probabilities
    # would not be the ones this interval implies.
    gap = float(np.nanmax(np.abs(probs["point"].to_numpy() - point.to_numpy())))
    if gap > 1e-9:
        raise ProductionRefused(f"the interval and distribution point fits differ by {gap:.2e}")

    # The real snapshot carries display names; the synthetic frames the tests use do not.
    names = (
        df.drop_duplicates("player_id").set_index("player_id")["name"]
        if "name" in df.columns
        else None
    )
    out = pd.DataFrame(
        {
            "player_id": forecast["player_id"].to_numpy(),
            "name": (
                forecast["player_id"].map(names).to_numpy() if names is not None else pd.NA
            ),
            "target_season": forecast["target_season"].to_numpy(),
            "age_target": forecast["age_target"].to_numpy(),
            "pa_lag1": forecast["pa_lag1"].to_numpy(),
            MODEL: point.to_numpy(dtype=float),
            "lo": bounds["lo"].to_numpy(dtype=float),
            "hi": bounds["hi"].to_numpy(dtype=float),
            BASELINE: marcel.to_numpy(dtype=float),
            "weighted3": probs["baseline"].to_numpy(dtype=float),
            "p_breakout": probs["p_breakout"].to_numpy(dtype=float),
            "p_stable": probs["p_stable"].to_numpy(dtype=float),
            "p_regression": probs["p_regression"].to_numpy(dtype=float),
        }
    )
    out["width"] = out["hi"] - out["lo"]
    out[DIFF_COLUMN] = out[MODEL] - out[BASELINE]
    out["threshold"] = THRESHOLD

    summaries = (
        comps_frame(df, out["player_id"].tolist(), int(cfg.production.forecast_target) - 1)
        if comps
        else pd.DataFrame({"player_id": out["player_id"]}).assign(
            comps_n=np.nan, comps_played_next=np.nan,
            comps_median_next_woba_rel=np.nan, comps_median_delta_woba_rel=np.nan,
        )
    )
    out = out.merge(summaries, on="player_id", how="left", validate="one_to_one")
    # Biggest disagreements at both ends, as `per_player_forecasts` orders the fold files.
    return (
        out[COLUMNS].sort_values(DIFF_COLUMN, ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def manifest(
    df: pd.DataFrame, cfg: FoldsConfig, out: pd.DataFrame, *, freeze_commit: str, data_sha: str
) -> dict:
    _, _, targets = production_frames(df, cfg)
    return {
        "run": RUN,
        "run_kind": "production",
        "target_season": int(cfg.production.forecast_target),
        "features_through": int(cfg.production.forecast_target) - 1,
        "config": "extended",
        "train_targets": targets,
        "n_forecast": len(out),
        "data_path": str(DATA).replace("\\", "/"),
        "data_sha256": data_sha,
        "data_rows": len(df),
        "model_version": model_version(),
        "freeze_tag": FREEZE_TAG,
        "freeze_commit": freeze_commit,
        "generated_from_commit": _git("rev-parse", "HEAD"),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "interval_level": DEFAULT_LEVEL,
        "threshold": THRESHOLD,
        "comps_k": COMPS_K,
    }


# ---- entry point ----


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output; only for a run that was never committed",
    )
    parser.add_argument(
        "--no-comps",
        action="store_true",
        help="leave the comps columns empty (they are the slow part); not for the real run",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    forecasts_path = ARTIFACTS / f"{RUN}_forecasts.csv"
    manifest_path = ARTIFACTS / f"{RUN}_manifest.json"
    try:
        freeze_commit, data_sha = check_frozen()
        check_writable([forecasts_path, manifest_path], overwrite=args.overwrite)
    except ProductionRefused as refused:
        print(f"refused: {refused}")
        return 1

    cfg = load_folds_config(CONFIG)
    df = pd.read_parquet(DATA)
    print(
        f"PRODUCTION RUN {RUN}: frozen configuration at {FREEZE_TAG} ({freeze_commit[:12]}), "
        f"snapshot sha256 {data_sha[:12]}..., {len(df)} rows"
    )
    out = build(df, cfg, comps=not args.no_comps)
    info = manifest(df, cfg, out, freeze_commit=freeze_commit, data_sha=data_sha)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(forecasts_path, index=False)
    manifest_path.write_text(json.dumps(info, indent=2) + "\n")
    print(
        f"\ntrained on target seasons {info['train_targets']} -> forecast "
        f"{info['target_season']}, {len(out)} players"
    )
    print(
        f"  {MODEL} mean {out[MODEL].mean():.4f}, {BASELINE} mean {out[BASELINE].mean():.4f}, "
        f"gap mean {out[DIFF_COLUMN].mean():+.4f} "
        f"(range {out[DIFF_COLUMN].min():+.4f} to {out[DIFF_COLUMN].max():+.4f})"
    )
    print(
        f"  {DEFAULT_LEVEL:.0%} interval mean width {out['width'].mean():.4f}; "
        f"mean P(breakout) {out['p_breakout'].mean():.3f}, "
        f"mean P(regression) {out['p_regression'].mean():.3f}"
    )
    print(f"\nwrote {forecasts_path} and {manifest_path}")
    print(json.dumps(info, indent=2))
    print(
        "\nThese are a prospective test: they are committed as they stand and never edited. "
        "2027 decides whether they were any good."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
