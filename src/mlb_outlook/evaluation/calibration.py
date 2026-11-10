"""Scoring for the breakout/regression probabilities: Brier, skill, reliability.

Each probability is scored against the realized label on the same rows, and against a
base-rate baseline that predicts the fold's training-window base rate for every player --
the only base rate a forecaster has at the forecast date. The headline is the Brier skill
score, ``1 - BS_model / BS_base``: positive means the probabilities carry information the
base rate does not.

The skill comparison reuses `paired_bootstrap_mae_diff` from the harness. Brier is a mean
of per-row squared errors, so the same paired resampling over
(model squared error - base squared error) gives its CI, on the same rows.

The reliability table bins by equal-count deciles of predicted probability rather than
fixed 0.1-wide bins, because the predictions are skewed toward zero and fixed bins leave
most rows in one bucket. Pre-registered in `docs/PROJECT_PLAN.md` section 6.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from mlb_outlook.config import Fold, FoldsConfig
from mlb_outlook.evaluation.harness import (
    FoldNotFrozenError,
    fold_frames,
    paired_bootstrap_mae_diff,
)
from mlb_outlook.models.probabilities import (
    LABELS,
    SENSITIVITY_THRESHOLDS,
    BucketedResidualDistribution,
    label_probabilities,
    realized_labels,
    weighted3_baseline,
)

DEFAULT_BINS = 10
SCORE_COLUMNS = [
    "label",
    "config",
    "fold",
    "threshold",
    "n",
    "base_rate_train",
    "base_rate_test",
    "mean_predicted",
    "brier",
    "brier_base",
    "skill",
    "brier_diff",
    "ci_low",
    "ci_high",
    "ece",
    "min_predicted",
    "max_predicted",
]
RELIABILITY_COLUMNS = [
    "label",
    "config",
    "fold",
    "threshold",
    "decile",
    "n",
    "p_low",
    "p_high",
    "mean_predicted",
    "observed",
    "gap",
]


def _aligned(prob, outcome) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if p.shape != y.shape or p.ndim != 1 or p.size == 0:
        raise ValueError("prob and outcome must be aligned non-empty 1-D arrays")
    if np.isnan(p).any():
        raise ValueError("prob carries NaN; drop unscoreable rows before scoring")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("outcome must be boolean or 0/1")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("prob must lie in [0, 1]")
    return p, y


def squared_errors(prob, outcome) -> np.ndarray:
    """Per-row ``(p - y)^2``; the Brier score is the mean of these."""
    p, y = _aligned(prob, outcome)
    return (p - y) ** 2


def brier_score(prob, outcome) -> float:
    return float(np.mean(squared_errors(prob, outcome)))


def base_rate(outcome) -> float:
    """Share of rows with the label; the climatological forecast."""
    y = np.asarray(outcome, dtype=float)
    if y.size == 0:
        raise ValueError("base_rate needs at least one row")
    return float(np.mean(y))


def reliability_table(prob, outcome, *, n_bins: int = DEFAULT_BINS) -> pd.DataFrame:
    """Equal-count deciles of predicted probability against the observed rate.

    Equal predictions always land in the same bin, so ties collapse bins rather than being
    split across them: the table can come back with fewer than `n_bins` rows -- one, if
    every prediction is identical -- and the `n` column says how many rows each holds.
    """
    p, y = _aligned(prob, outcome)
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    edges = np.unique(np.quantile(p, np.linspace(0, 1, min(n_bins, p.size) + 1)))
    if edges.size < 2:  # one distinct prediction: a single bin holding everything
        bins = np.zeros(p.size, dtype=int)
    else:
        bins = np.clip(np.searchsorted(edges, p, side="left") - 1, 0, edges.size - 2)
    frame = pd.DataFrame({"p": p, "y": y, "bin": bins})
    rows = []
    for i, (_, grp) in enumerate(frame.groupby("bin", sort=True)):
        mean_p = float(grp["p"].mean())
        observed = float(grp["y"].mean())
        rows.append(
            {
                "decile": i + 1,
                "n": int(len(grp)),
                "p_low": float(grp["p"].min()),
                "p_high": float(grp["p"].max()),
                "mean_predicted": mean_p,
                "observed": observed,
                "gap": observed - mean_p,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Row-count-weighted mean absolute gap of a `reliability_table`."""
    if table.empty:
        raise ValueError("expected_calibration_error needs a non-empty reliability table")
    weights = table["n"].to_numpy(dtype=float)
    return float(np.average(np.abs(table["gap"].to_numpy(dtype=float)), weights=weights))


def score_probabilities(
    prob,
    outcome,
    *,
    base_rate_train: float,
    n_bins: int = DEFAULT_BINS,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Brier, the base-rate comparison, its bootstrap CI, and the reliability gap.

    `base_rate_train` is the baseline forecast: one number, the training-window rate, given
    to every player. Scoring it on the test rows is what the model has to beat.
    """
    p, y = _aligned(prob, outcome)
    if not 0 <= base_rate_train <= 1:
        raise ValueError(f"base_rate_train must lie in [0, 1], got {base_rate_train}")
    model_err = (p - y) ** 2
    base_err = (np.full_like(p, base_rate_train) - y) ** 2
    brier = float(np.mean(model_err))
    brier_base = float(np.mean(base_err))
    diff, lo, hi = paired_bootstrap_mae_diff(model_err, base_err, n_boot=n_boot, seed=seed)
    table = reliability_table(p, y, n_bins=n_bins)
    return {
        "n": int(p.size),
        "base_rate_train": float(base_rate_train),
        "base_rate_test": float(np.mean(y)),
        "mean_predicted": float(np.mean(p)),
        "brier": brier,
        "brier_base": brier_base,
        # Undefined only if the baseline is perfect, which needs a label nobody realized.
        "skill": float(1 - brier / brier_base) if brier_base > 0 else float("nan"),
        "brier_diff": diff,
        "ci_low": lo,
        "ci_high": hi,
        "ece": expected_calibration_error(table),
        "min_predicted": float(np.min(p)),
        "max_predicted": float(np.max(p)),
    }


def run_probability_fold(
    df: pd.DataFrame,
    fold: Fold,
    cfg: FoldsConfig,
    *,
    extended: bool = True,
    thresholds: Sequence[float] = SENSITIVITY_THRESHOLDS,
    allow_unfrozen: bool = False,
    n_bins: int = DEFAULT_BINS,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score both labels on one fold; returns (scores, reliability) frames.

    One distribution is fitted on the fold's training frame and every threshold reads the
    same fit -- sensitivity is about where the line sits, not about refitting. The base-rate
    baseline for a threshold is that threshold's base rate on the *training* rows.

    The holdout guard from `run_fold` applies: a fold marked ``frozen: false`` raises
    unless `allow_unfrozen` is set.
    """
    if fold.frozen is False and not allow_unfrozen:
        raise FoldNotFrozenError(
            f"fold {fold.name!r} is not frozen; pass allow_unfrozen=True after design freeze"
        )
    if not thresholds:
        raise ValueError("thresholds must be non-empty")
    train, test = fold_frames(df, fold, cfg, extended=extended)
    dist = BucketedResidualDistribution().fit(train, train["y_woba"])
    train_base = weighted3_baseline(train)
    test_base = weighted3_baseline(test)
    config = "extended" if extended else "base"
    scores: list[dict[str, object]] = []
    reliability: list[dict[str, object]] = []
    for threshold in thresholds:
        train_labels = realized_labels(train["y_woba"], train_base, threshold=threshold)
        test_labels = realized_labels(test["y_woba"], test_base, threshold=threshold)
        probs = label_probabilities(dist, test, baseline=test_base, threshold=threshold)
        # A row needs a baseline, an outcome and a probability to be scoreable at all.
        train_ok = train_base.notna() & train["y_woba"].notna()
        for label in LABELS:
            column = f"p_{label}"
            keep = probs[column].notna() & test_base.notna() & test["y_woba"].notna()
            p = probs.loc[keep, column]
            y = test_labels.loc[keep, label]
            tag = {"label": label, "config": config, "fold": fold.name, "threshold": threshold}
            scores.append(
                tag
                | score_probabilities(
                    p,
                    y,
                    base_rate_train=base_rate(train_labels.loc[train_ok, label]),
                    n_bins=n_bins,
                    n_boot=n_boot,
                    seed=seed,
                )
            )
            for row in reliability_table(p, y, n_bins=n_bins).to_dict("records"):
                reliability.append(tag | row)
    return (
        pd.DataFrame(scores, columns=SCORE_COLUMNS),
        pd.DataFrame(reliability, columns=RELIABILITY_COLUMNS),
    )
