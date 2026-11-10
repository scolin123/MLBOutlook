"""Run the selection folds (fold1, fold2) on real data and print the results tables.

Each fold runs twice: the ``base`` training window from ``folds.yaml`` and the ``extended``
one that adds the pre-2023 target seasons. The final holdout is skipped, never run: any fold
carrying a ``frozen`` key at all is one, whichever way the flag is set. Reading the flag
rather than its value is deliberate -- after the freeze commit flips fold3 to
``frozen: true`` the `run_fold` guard stops refusing it, and the holdout still goes through
`scripts/run_holdout.py` and nothing else (DESIGN_FREEZE.md section 4).

Point models report MAE, RMSE and bias, each one both unweighted and weighted by the target
season's plate appearances -- one row one vote against where the plate appearances actually
went. Interval models report those plus the coverage and mean width of their 80% intervals,
checked against ``success_criteria`` in ``folds.yaml``.

Usage:
    uv run python scripts/run_folds.py
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd

from mlb_outlook.config import FoldsConfig, load_folds_config
from mlb_outlook.evaluation.harness import fold_frames, paired_bootstrap_mae_diff, run_fold
from mlb_outlook.models.baselines import LastSeason, MarcelBaseline, Weighted3
from mlb_outlook.models.intervals import (
    DEFAULT_LEVEL,
    DEFAULT_PA_EDGES,
    QuantileRegressionIntervals,
    ResidualQuantileIntervals,
    SplitConformalIntervals,
)
from mlb_outlook.models.relative import relative_ridge
from mlb_outlook.models.ridge import RidgeForecaster

DATA = Path("data/processed/hitter_seasons.parquet")
OUT = Path("artifacts/fold_results.csv")
# The interval models all carry the same point forecaster, ridge_rel, so their MAE stays
# comparable; the conformal pair refits it on a subset, which costs them a little.
FACTORIES = {
    "last_season": LastSeason,
    "weighted3": Weighted3,
    "marcel": MarcelBaseline,
    "ridge": RidgeForecaster,
    "ridge_rel": relative_ridge,
    "resid_q_pa": ResidualQuantileIntervals,
    "conformal": SplitConformalIntervals,
    "conformal_pa": partial(SplitConformalIntervals, pa_edges=DEFAULT_PA_EDGES),
    "quantile_reg": QuantileRegressionIntervals,
}
BASELINE = "marcel"
CANDIDATES = ("ridge", "ridge_rel")


def main() -> None:
    df = pd.read_parquet(DATA)
    cfg = load_folds_config("configs/folds.yaml")
    results = []
    for fold in cfg.folds:
        if fold.frozen is not None:
            print(
                f"skipping {fold.name}: final holdout, runs only through "
                "scripts/run_holdout.py"
            )
            continue
        for extended in (False, True):
            label = "extended" if extended else "base"
            results.append(run_fold(df, fold, cfg, FACTORIES, extended=extended))

            # Each candidate vs the Marcel baseline, paired bootstrap on absolute errors
            train, test = fold_frames(df, fold, cfg, extended=extended)
            errs = {}
            for name in (BASELINE, *CANDIDATES):
                model = FACTORIES[name]().fit(train, train["y_woba"])
                errs[name] = (model.predict(test) - test["y_woba"]).abs().to_numpy()
            for name in CANDIDATES:
                diff, lo, hi = paired_bootstrap_mae_diff(errs[name], errs[BASELINE])
                print(
                    f"{fold.name} {label:8s} {name:10s} - {BASELINE} MAE = {diff:+.4f}  "
                    f"(95% CI {lo:+.4f} to {hi:+.4f})"
                )

    table = pd.concat(results, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print()
    print(table.round(4).to_string(index=False))
    print_pa_weighted(table)
    print_interval_coverage(table, cfg)
    print(f"\nwrote {OUT}")


def print_pa_weighted(table: pd.DataFrame) -> None:
    """The weighted trio beside the unweighted one, per fold and window."""
    print()
    print("unweighted / PA-weighted by target-season PA:")
    print("  fold  config    model            MAE              RMSE             bias")
    for row in table.itertuples():
        print(
            f"  {row.fold} {row.config:9s} {row.model:12s} "
            f"{row.mae:.4f} / {row.mae_pa:.4f}  {row.rmse:.4f} / {row.rmse_pa:.4f}  "
            f"{row.bias:+.4f} / {row.bias_pa:+.4f}"
        )


def print_interval_coverage(table: pd.DataFrame, cfg: FoldsConfig) -> None:
    """Coverage and width of every interval model, against the pre-registered range."""
    intervals = table.loc[table["coverage"].notna()]
    if intervals.empty:
        return
    lo, hi = cfg.success_criteria.interval_80_coverage_range
    print()
    print(f"{DEFAULT_LEVEL:.0%} intervals, target coverage {lo:.0%}-{hi:.0%}:")
    for row in intervals.itertuples():
        verdict = "ok " if lo <= row.coverage <= hi else "OUT"
        print(
            f"  {verdict} {row.fold} {row.config:8s} {row.model:12s} "
            f"coverage {row.coverage:.3f}  mean width {row.width:.4f}"
        )


if __name__ == "__main__":
    main()
