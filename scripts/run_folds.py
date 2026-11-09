"""Run the unfrozen-safe folds (fold1, fold2) on real data and print an MAE table.

The final holdout (any fold with ``frozen: false``) is skipped, never run.

Usage:
    uv run python scripts/run_folds.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlb_outlook.config import load_folds_config
from mlb_outlook.evaluation.harness import fold_frames, paired_bootstrap_mae_diff, run_fold
from mlb_outlook.models.baselines import LastSeason, MarcelBaseline, Weighted3
from mlb_outlook.models.relative import relative_ridge
from mlb_outlook.models.ridge import RidgeForecaster

DATA = Path("data/processed/hitter_seasons.parquet")
OUT = Path("artifacts/fold_results.csv")
FACTORIES = {
    "last_season": LastSeason,
    "weighted3": Weighted3,
    "marcel": MarcelBaseline,
    "ridge": RidgeForecaster,
    "ridge_rel": relative_ridge,
}
BASELINE = "marcel"
CANDIDATES = ("ridge", "ridge_rel")


def main() -> None:
    df = pd.read_parquet(DATA)
    cfg = load_folds_config("configs/folds.yaml")
    results = []
    for fold in cfg.folds:
        if fold.frozen is False:
            print(f"skipping {fold.name}: final holdout, not frozen")
            continue
        results.append(run_fold(df, fold, cfg, FACTORIES))

        # Each candidate vs the Marcel baseline, paired bootstrap on absolute errors
        train, test = fold_frames(df, fold, cfg)
        errs = {}
        for name in (BASELINE, *CANDIDATES):
            model = FACTORIES[name]().fit(train, train["y_woba"])
            errs[name] = (model.predict(test) - test["y_woba"]).abs().to_numpy()
        for name in CANDIDATES:
            diff, lo, hi = paired_bootstrap_mae_diff(errs[name], errs[BASELINE])
            print(
                f"{fold.name}: {name} - {BASELINE} MAE = {diff:+.4f}  "
                f"(95% CI {lo:+.4f} to {hi:+.4f})"
            )

    table = pd.concat(results, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print()
    print(table.round(4).to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
