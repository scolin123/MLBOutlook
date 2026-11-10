"""Breakout/regression probabilities on the selection fold only.

Reads the probabilities off the selected configuration's predictive distribution
(`ridge_rel`, extended window, residuals bucketed by lag-1 PA) and scores them against the
realized labels: Brier, the skill score against a base-rate baseline, a reliability table
over deciles of predicted probability, and the base rates themselves.

Definitions are pre-registered in `docs/PROJECT_PLAN.md` section 6 and were committed
before this script existed. 0.030 is the registered threshold; 0.020 and 0.040 are reported
as sensitivity and do not replace it.

Fold 2 is the only fold run. Fold 3 is the final holdout: it is looked up by name and never
touched, and `run_probability_fold` keeps the same `frozen: false` guard as `run_fold`.

Usage:
    uv run python scripts/run_probabilities.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlb_outlook.config import FoldsConfig, load_folds_config
from mlb_outlook.evaluation.calibration import run_probability_fold
from mlb_outlook.models.probabilities import LABELS, THRESHOLD

DATA = Path("data/processed/hitter_seasons.parquet")
SCORES_OUT = Path("artifacts/probability_results.csv")
RELIABILITY_OUT = Path("artifacts/probability_reliability.csv")
FOLD = "fold2"
EXTENDED = True


def selection_fold(cfg: FoldsConfig):
    """The fold probabilities are allowed to run on; anything unfrozen is refused."""
    by_name = {f.name: f for f in cfg.folds}
    fold = by_name.get(FOLD)
    if fold is None:
        raise SystemExit(f"{FOLD!r} is not in configs/folds.yaml: {sorted(by_name)}")
    if fold.frozen is False:
        raise SystemExit(f"{FOLD!r} is marked frozen: false; probabilities do not run on a holdout")
    return fold


def main() -> None:
    df = pd.read_parquet(DATA)
    cfg = load_folds_config("configs/folds.yaml")
    fold = selection_fold(cfg)
    scores, reliability = run_probability_fold(df, fold, cfg, extended=EXTENDED)

    SCORES_OUT.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORES_OUT, index=False)
    reliability.to_csv(RELIABILITY_OUT, index=False)
    print_report(scores, reliability)
    print(f"\nwrote {SCORES_OUT} and {RELIABILITY_OUT}")


def print_report(scores: pd.DataFrame, reliability: pd.DataFrame) -> None:
    """The registered threshold in full, then the sensitivity rows in one line each."""
    registered = scores.loc[scores["threshold"] == THRESHOLD]
    row = registered.iloc[0]
    print(f"{row.fold}, {row.config} window, threshold +/-{THRESHOLD:.3f}, n = {row.n}")

    for label in LABELS:
        r = registered.loc[registered["label"] == label].iloc[0]
        print(f"\n{label}:")
        print(
            f"  base rate   train {r.base_rate_train:.3f}   test {r.base_rate_test:.3f}   "
            f"mean predicted {r.mean_predicted:.3f} "
            f"(range {r.min_predicted:.3f}-{r.max_predicted:.3f})"
        )
        print(
            f"  Brier       {r.brier:.4f}   base rate {r.brier_base:.4f}   "
            f"skill {r.skill:+.4f}"
        )
        print(
            f"  model - base rate Brier {r.brier_diff:+.4f} "
            f"(95% CI {r.ci_low:+.4f} to {r.ci_high:+.4f})   "
            f"{'clear of zero' if r.ci_high < 0 or r.ci_low > 0 else 'spans zero'}"
        )
        print(f"  reliability gap (count-weighted mean |observed - predicted|) {r.ece:.4f}")
        part = reliability.loc[
            (reliability["label"] == label) & (reliability["threshold"] == THRESHOLD)
        ]
        print("    decile    n   predicted   observed       gap")
        for d in part.itertuples():
            print(
                f"    {d.decile:>6}  {d.n:>3}      {d.mean_predicted:.3f}      "
                f"{d.observed:.3f}   {d.gap:+.3f}"
            )

    print(f"\nthreshold sensitivity (registered threshold is {THRESHOLD:.3f}):")
    print("    label        thr   base(test)   mean p    Brier   base Brier    skill")
    for s in scores.itertuples():
        mark = " *" if s.threshold == THRESHOLD else "  "
        print(
            f"  {mark}{s.label:11s}{s.threshold:.3f}        {s.base_rate_test:.3f}    "
            f"{s.mean_predicted:.3f}   {s.brier:.4f}       {s.brier_base:.4f}  {s.skill:+.4f}"
        )


if __name__ == "__main__":
    main()
