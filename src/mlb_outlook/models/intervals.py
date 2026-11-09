"""Prediction intervals around a point forecaster.

Every approach here wraps a point-forecast factory and describes the spread of that
model's residuals, so the interval rides on whatever forecaster is selected. All of it is
fitted inside the training window: ``fit`` sees only training rows and ``predict_interval``
reads the same feature columns as the point model.

Three approaches, meant to be compared:

- `ResidualQuantileIntervals` -- empirical residual quantiles inside a playing-time bucket.
- `SplitConformalIntervals` -- quantile of absolute residuals on held-out calibration rows.
- `QuantileRegressionIntervals` -- linear quantile regression of the residuals on the
  features the point model uses.

The first and third use in-sample residuals and can come out narrow; only the conformal
one holds its calibration rows out of the point fit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlb_outlook.models.baselines import Forecaster
from mlb_outlook.models.relative import relative_ridge
from mlb_outlook.models.ridge import RELATIVE_FEATURES

DEFAULT_LEVEL = 0.80
# Lag-1 PA cut points: a part-time season says less about next year than a full one does.
DEFAULT_PA_EDGES = (300, 500)
MIN_BUCKET = 50
INTERVAL_COLUMNS = ["lo", "hi"]

Summarize = Callable[[np.ndarray], tuple[float, float]]


@runtime_checkable
class IntervalForecaster(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> IntervalForecaster: ...

    def predict(self, X: pd.DataFrame) -> pd.Series: ...

    def predict_interval(self, X: pd.DataFrame) -> pd.DataFrame: ...


def pa_buckets(X: pd.DataFrame, edges: Sequence[float] = DEFAULT_PA_EDGES) -> np.ndarray:
    """Bucket index per row from `pa_lag1`: 0 below the first edge, then one per edge."""
    return np.digitize(X["pa_lag1"].to_numpy(dtype=float), np.asarray(edges, dtype=float))


def conformal_quantile(values: np.ndarray, level: float) -> float:
    """The ceil((n + 1) * level)-th smallest of `values`, or the largest if that is past n.

    This is the split-conformal quantile, which is why it is a rank rather than an
    interpolated percentile: the finite-sample coverage guarantee is about ranks.
    """
    n = values.size
    if n == 0:
        raise ValueError("conformal_quantile needs at least one calibration residual")
    rank = int(np.ceil((n + 1) * level))
    if rank >= n:
        return float(np.max(values))
    return float(np.sort(values)[rank - 1])


class _ResidualIntervals:
    """Shared plumbing: fit a point model, then offset its forecast by a residual spread.

    Subclasses fill in `fit`. Offsets are looked up per playing-time bucket when
    `pa_edges` is set, falling back to the pooled pair for a bucket with too few rows.
    """

    def __init__(
        self,
        factory: Callable[[], Forecaster] = relative_ridge,
        level: float = DEFAULT_LEVEL,
        *,
        pa_edges: Sequence[float] = (),
        min_bucket: int = MIN_BUCKET,
    ) -> None:
        if not 0 < level < 1:
            raise ValueError(f"level must be in (0, 1), got {level}")
        self.factory = factory
        self.level = level
        self.pa_edges = tuple(pa_edges)
        self.min_bucket = min_bucket
        self.model_: Forecaster | None = None
        self.pooled_: tuple[float, float] | None = None
        self.offsets_: dict[int, tuple[float, float]] = {}

    @property
    def quantiles(self) -> tuple[float, float]:
        """The two residual quantiles that bracket `level` symmetrically."""
        tail = (1 - self.level) / 2
        return tail, 1 - tail

    def _fit_point(self, X: pd.DataFrame, y: pd.Series | None) -> np.ndarray:
        """Fit the point model on the whole frame and return its residuals."""
        if y is None:
            raise ValueError(f"{type(self).__name__}.fit requires y")
        self.model_ = self.factory().fit(X, y)
        return np.asarray(y, dtype=float) - self.model_.predict(X).to_numpy(dtype=float)

    def _store_offsets(self, resid: np.ndarray, X: pd.DataFrame, summarize: Summarize) -> None:
        self.pooled_ = summarize(resid)
        self.offsets_ = {}
        if not self.pa_edges:
            return
        buckets = pa_buckets(X, self.pa_edges)
        for b in np.unique(buckets):
            inside = resid[buckets == b]
            if inside.size >= self.min_bucket:
                self.offsets_[int(b)] = summarize(inside)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")
        return self.model_.predict(X)

    def _offsets(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.pooled_ is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")
        if not self.pa_edges:
            pairs = [self.pooled_] * len(X)
        else:
            pairs = [self.offsets_.get(int(b), self.pooled_) for b in pa_buckets(X, self.pa_edges)]
        return (
            np.array([p[0] for p in pairs], dtype=float),
            np.array([p[1] for p in pairs], dtype=float),
        )

    def predict_interval(self, X: pd.DataFrame) -> pd.DataFrame:
        """Lower and upper bounds, one row per row of `X`, always straddling the forecast."""
        point = self.predict(X).to_numpy(dtype=float)
        lo_off, hi_off = self._offsets(X)
        return pd.DataFrame(
            {
                "lo": np.minimum(point + lo_off, point),
                "hi": np.maximum(point + hi_off, point),
            },
            index=X.index,
            columns=INTERVAL_COLUMNS,
        )


class ResidualQuantileIntervals(_ResidualIntervals):
    """Empirical residual quantiles, taken inside a lag-1 PA bucket.

    The residuals are in-sample: the point model has already fitted these rows, so the
    spread is the optimistic one. What it buys is width that varies with playing time.
    """

    def __init__(
        self,
        factory: Callable[[], Forecaster] = relative_ridge,
        level: float = DEFAULT_LEVEL,
        *,
        pa_edges: Sequence[float] = DEFAULT_PA_EDGES,
        min_bucket: int = MIN_BUCKET,
    ) -> None:
        super().__init__(factory, level, pa_edges=pa_edges, min_bucket=min_bucket)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ResidualQuantileIntervals:
        resid = self._fit_point(X, y)
        lo_q, hi_q = self.quantiles

        def summarize(r: np.ndarray) -> tuple[float, float]:
            return float(np.quantile(r, lo_q)), float(np.quantile(r, hi_q))

        self._store_offsets(resid, X, summarize)
        return self


class SplitConformalIntervals(_ResidualIntervals):
    """Split conformal: calibrate the absolute residual on rows held out of the point fit.

    The calibration rows are unseen by the point model, so the quantile is not optimistic.
    The cost is a point model fitted on fewer rows, which shows up in its MAE. With
    `pa_edges` the quantile is taken per playing-time bucket (Mondrian conformal) so the
    width can vary; without them every row gets the same width.

    The split is a seeded row sample rather than the last training season, because the
    base window of fold 1 holds a single target season and would leave nothing to fit on.
    """

    def __init__(
        self,
        factory: Callable[[], Forecaster] = relative_ridge,
        level: float = DEFAULT_LEVEL,
        *,
        cal_fraction: float = 0.3,
        seed: int = 0,
        pa_edges: Sequence[float] = (),
        min_bucket: int = MIN_BUCKET,
    ) -> None:
        super().__init__(factory, level, pa_edges=pa_edges, min_bucket=min_bucket)
        if not 0 < cal_fraction < 1:
            raise ValueError(f"cal_fraction must be in (0, 1), got {cal_fraction}")
        self.cal_fraction = cal_fraction
        self.seed = seed
        self.n_calibration_: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> SplitConformalIntervals:
        if y is None:
            raise ValueError("SplitConformalIntervals.fit requires y")
        values = np.asarray(y, dtype=float)
        n_cal = int(round(len(X) * self.cal_fraction))
        if n_cal < 1 or n_cal >= len(X):
            raise ValueError(
                f"a {self.cal_fraction:.0%} calibration split of {len(X)} rows leaves "
                "nothing to fit or nothing to calibrate on"
            )
        order = np.random.default_rng(self.seed).permutation(len(X))
        cal, fit = order[:n_cal], order[n_cal:]
        self.n_calibration_ = n_cal
        fit_rows = X.iloc[fit]
        self.model_ = self.factory().fit(fit_rows, pd.Series(values[fit], index=fit_rows.index))
        cal_rows = X.iloc[cal]
        resid = values[cal] - self.model_.predict(cal_rows).to_numpy(dtype=float)

        def summarize(r: np.ndarray) -> tuple[float, float]:
            q = conformal_quantile(np.abs(r), self.level)
            return -q, q

        self._store_offsets(resid, cal_rows, summarize)
        return self


class QuantileRegressionIntervals(_ResidualIntervals):
    """Linear quantile regression of the point model's residuals on its own features.

    This generalises the PA buckets: the width can respond to every feature instead of one
    cut of playing time. The residuals are in-sample, as in `ResidualQuantileIntervals`.
    """

    def __init__(
        self,
        factory: Callable[[], Forecaster] = relative_ridge,
        level: float = DEFAULT_LEVEL,
        *,
        feature_cols: Sequence[str] = RELATIVE_FEATURES,
        alpha: float = 1e-4,
    ) -> None:
        super().__init__(factory, level)
        self.feature_cols = tuple(feature_cols)
        self.alpha = alpha
        self.bounds_: tuple[Pipeline, Pipeline] | None = None

    def _matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[list(self.feature_cols)].astype(float)

    def _fit_quantile(self, X: pd.DataFrame, resid: np.ndarray, q: float) -> Pipeline:
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("quantile", QuantileRegressor(quantile=q, alpha=self.alpha, solver="highs")),
            ]
        )
        return pipe.fit(self._matrix(X), resid)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> QuantileRegressionIntervals:
        resid = self._fit_point(X, y)
        lo_q, hi_q = self.quantiles
        self.bounds_ = (
            self._fit_quantile(X, resid, lo_q),
            self._fit_quantile(X, resid, hi_q),
        )
        # Kept as the pooled fallback so an unfitted-state check has something to read.
        self.pooled_ = (float(np.quantile(resid, lo_q)), float(np.quantile(resid, hi_q)))
        return self

    def _offsets(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.bounds_ is None:
            raise RuntimeError("QuantileRegressionIntervals is not fitted; call fit() first")
        lo_model, hi_model = self.bounds_
        matrix = self._matrix(X)
        return lo_model.predict(matrix), hi_model.predict(matrix)
