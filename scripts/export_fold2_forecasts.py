"""Per-player fold 2 forecasts with probabilities, for the database and the app's demo data.

`run_holdout.py --rehearse fold2` writes `fold2_players.csv` (point, Marcel, interval) but no
per-player breakout/regression probabilities: the scoring scripts only keep aggregates. This
adds them by calling the frozen code exactly as `run_probability_fold` does -- one
`BucketedResidualDistribution` fitted on fold 2's extended training frame, the registered
0.030 threshold, the `weighted3_baseline` -- without changing any of it.

It runs fold 2 only. Fold 2's target is 2025 and its features stop at 2024, so these are the
"2025 forecasts made from data through 2024" the app shows as a labelled backtest. Any fold
carrying a `frozen` key (the holdout) is refused.

Writes:
    artifacts/fold2_forecasts.csv  one row per scored player
    artifacts/fold2_manifest.json  data hash, model version, and when it was generated

Usage:
    uv run python scripts/run_holdout.py --rehearse fold2   # writes fold2_players.csv
    uv run python scripts/export_fold2_forecasts.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_outlook.config import load_folds_config
from mlb_outlook.db.versions import file_sha256, model_version
from mlb_outlook.evaluation.harness import fold_frames
from mlb_outlook.models.probabilities import (
    THRESHOLD,
    BucketedResidualDistribution,
    label_probabilities,
    weighted3_baseline,
)

DATA = Path("data/processed/hitter_seasons.parquet")
CONFIG = Path("configs/folds.yaml")
ARTIFACTS = Path("artifacts")
FOLD = "fold2"


def build(df: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Join fold 2's per-player file to the label probabilities for the same rows."""
    cfg = load_folds_config(CONFIG)
    fold = next(f for f in cfg.folds if f.name == FOLD)
    if fold.frozen is not None:
        raise SystemExit(f"refused: {fold.name} carries a frozen key; this is fold 2 only")
    train, test = fold_frames(df, fold, cfg, extended=True)
    dist = BucketedResidualDistribution().fit(train, train["y_woba"])
    probs = label_probabilities(dist, test, baseline=weighted3_baseline(test), threshold=THRESHOLD)
    probs.insert(0, "player_id", test["player_id"].to_numpy())
    probs = probs.rename(columns={"point": "point_dist", "baseline": "weighted3"})

    out = players.merge(probs, on="player_id", how="left", validate="one_to_one")
    # The distribution's point fit and the interval model's are the same ridge_rel fit; if
    # they disagree, the two files came from different code or data and must not be joined.
    gap = float(np.nanmax(np.abs(out["point_dist"] - out["ridge_rel"])))
    if gap > 1e-9:
        raise SystemExit(f"fold2_players.csv and the refit disagree by {gap:.2e}; regenerate it")
    out = out.drop(columns="point_dist")
    out["width"] = out["hi"] - out["lo"]
    out["threshold"] = THRESHOLD
    return out


def main() -> int:
    df = pd.read_parquet(DATA)
    players = pd.read_csv(ARTIFACTS / f"{FOLD}_players.csv")
    out = build(df, players)
    out.to_csv(ARTIFACTS / f"{FOLD}_forecasts.csv", index=False)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    manifest = {
        "fold": FOLD,
        "target_season": int(out["target_season"].iloc[0]),
        "features_through": int(out["target_season"].iloc[0]) - 1,
        "data_path": str(DATA).replace("\\", "/"),
        "data_sha256": file_sha256(DATA),
        "data_rows": len(df),
        "model_version": model_version(),
        "generated_from_commit": head,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "threshold": THRESHOLD,
    }
    (ARTIFACTS / f"{FOLD}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {ARTIFACTS / f'{FOLD}_forecasts.csv'} ({len(out)} rows) and the manifest")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
