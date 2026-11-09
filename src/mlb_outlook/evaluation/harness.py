"""Per-fold evaluation harness: fresh models, as-of features, guarded holdout."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from mlb_outlook.config import Fold, FoldsConfig
from mlb_outlook.features.asof import build_features
from mlb_outlook.models.baselines import Forecaster


class FoldNotFrozenError(RuntimeError):
    """Raised when a fold marked ``frozen: False`` is run before design freeze."""


def fold_frames(
    df: pd.DataFrame, fold: Fold, cfg: FoldsConfig, *, extended: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) feature frames for one fold.

    With ``extended=True`` the train frame also covers ``cfg.extended.extra_train_targets``
    (the pre-2023 seasons); the test frame is identical either way.
    """
    kwargs = {
        "include_target": True,
        "min_pa_feature": cfg.min_pa.feature_season,
        "min_pa_target": cfg.min_pa.target_season,
        "flag_seasons": tuple(cfg.extended.flag_seasons),
    }
    targets = cfg.extended_train_targets(fold) if extended else fold.train_targets
    train = pd.concat([build_features(df, t, **kwargs) for t in targets], ignore_index=True)
    test = build_features(df, fold.test_target, **kwargs)
    return train, test


RESULT_COLUMNS = ["model", "config", "fold", "n_train", "n_test", "mae", "bias"]


def run_fold(
    df: pd.DataFrame,
    fold: Fold,
    cfg: FoldsConfig,
    factories: dict[str, Callable[[], Forecaster]],
    *,
    allow_unfrozen: bool = False,
    extended: bool = False,
) -> pd.DataFrame:
    """Fit a fresh model per factory on the fold's train frame and score the test frame."""
    if fold.frozen is False and not allow_unfrozen:
        raise FoldNotFrozenError(
            f"fold {fold.name!r} is not frozen; pass allow_unfrozen=True after design freeze"
        )
    train, test = fold_frames(df, fold, cfg, extended=extended)
    rows = []
    for name, factory in factories.items():
        model = factory()
        model.fit(train, train["y_woba"])
        pred = model.predict(test)
        err = pred.to_numpy(dtype=float) - test["y_woba"].to_numpy(dtype=float)
        rows.append(
            {
                "model": name,
                "config": "extended" if extended else "base",
                "fold": fold.name,
                "n_train": len(train),
                "n_test": len(test),
                "mae": float(np.mean(np.abs(err))),
                "bias": float(np.mean(err)),
            }
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def run_all(
    df: pd.DataFrame,
    cfg: FoldsConfig,
    factories: dict[str, Callable[[], Forecaster]],
    *,
    allow_unfrozen: bool = False,
    extended: bool = False,
) -> pd.DataFrame:
    """Run every fold in ``cfg.folds`` (not production) and concatenate the results."""
    parts = [
        run_fold(df, fold, cfg, factories, allow_unfrozen=allow_unfrozen, extended=extended)
        for fold in cfg.folds
    ]
    return pd.concat(parts, ignore_index=True)


def paired_bootstrap_mae_diff(
    errors_a, errors_b, *, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap ``mean(a) - mean(b)`` over resampled pairs; returns (diff, ci_low, ci_high)."""
    a = np.asarray(errors_a, dtype=float)
    b = np.asarray(errors_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("errors_a and errors_b must be aligned non-empty 1-D arrays")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)
