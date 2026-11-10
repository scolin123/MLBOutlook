"""Breakout and regression probabilities, read off the forecast distribution.

The definitions are pre-registered in `docs/PROJECT_PLAN.md` section 6 and were committed
before anything here was computed. For next-season wOBA `y` and a weighted 3-year baseline
`b`: breakout is ``y >= b + 0.030``, regression is ``y <= b - 0.030``, stable is the rest.
The three are mutually exclusive, so the probabilities sum to 1.

There is no classifier. `BucketedResidualDistribution` is the same construction as the
selected `ResidualQuantileIntervals` -- a point fit plus its training residuals inside the
row's lag-1 PA bucket -- keeping the whole residual sample instead of two quantiles, so any
threshold can be read off it. The sample's 10th and 90th percentiles are the published 80%
interval by construction, which is what stops a probability from contradicting the forecast.

The distribution is empirical: a probability is a share of a bucket's residual sample, so
its granularity is 1/n for that bucket.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from mlb_outlook.models.baselines import Forecaster, Weighted3
from mlb_outlook.models.intervals import DEFAULT_PA_EDGES, MIN_BUCKET, pa_buckets
from mlb_outlook.models.relative import relative_ridge

# Pre-registered threshold, in wOBA points, either side of the baseline.
THRESHOLD = 0.030
# Reported alongside it as sensitivity; the registered number does not move because a
# neighbour scores better.
SENSITIVITY_THRESHOLDS = (0.020, THRESHOLD, 0.040)
LABELS = ("breakout", "regression")
PROBABILITY_COLUMNS = ["point", "baseline", "p_breakout", "p_stable", "p_regression"]


class BucketedResidualDistribution:
    """Predictive distribution: a point forecast plus training residuals in a PA bucket.

    `fit` fits the point model on the whole training frame and keeps its residuals, pooled
    and split by lag-1 PA bucket. A bucket with fewer than `min_bucket` rows falls back to
    the pooled sample, exactly as the interval version does. With no `pa_edges` every row
    reads the pooled sample.
    """

    def __init__(
        self,
        factory: Callable[[], Forecaster] = relative_ridge,
        *,
        pa_edges: Sequence[float] = DEFAULT_PA_EDGES,
        min_bucket: int = MIN_BUCKET,
    ) -> None:
        self.factory = factory
        self.pa_edges = tuple(pa_edges)
        self.min_bucket = min_bucket
        self.model_: Forecaster | None = None
        self.pooled_: np.ndarray | None = None
        self.samples_: dict[int, np.ndarray] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> BucketedResidualDistribution:
        if y is None:
            raise ValueError("BucketedResidualDistribution.fit requires y")
        self.model_ = self.factory().fit(X, y)
        resid = np.asarray(y, dtype=float) - self.model_.predict(X).to_numpy(dtype=float)
        self.pooled_ = resid
        self.samples_ = {}
        if not self.pa_edges:
            return self
        buckets = pa_buckets(X, self.pa_edges)
        for b in np.unique(buckets):
            inside = resid[buckets == b]
            if inside.size >= self.min_bucket:
                self.samples_[int(b)] = inside
        return self

    def residual_sample(self, bucket: int = 0) -> np.ndarray:
        """The residual sample for `bucket`, or the pooled one if that bucket is thin."""
        if self.pooled_ is None:
            raise RuntimeError("BucketedResidualDistribution is not fitted; call fit() first")
        return self.samples_.get(int(bucket), self.pooled_)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("BucketedResidualDistribution is not fitted; call fit() first")
        return self.model_.predict(X)

    def _buckets(self, X: pd.DataFrame) -> np.ndarray:
        if not self.pa_edges:
            return np.zeros(len(X), dtype=int)
        return pa_buckets(X, self.pa_edges)

    def _tail(self, X: pd.DataFrame, thresholds, *, upper: bool) -> pd.Series:
        """Share of the bucket's residual sample that puts `y` past each row's threshold.

        `upper=True` gives ``P(y >= t)``, otherwise ``P(y <= t)``. A row with a missing
        threshold or a missing point forecast gets NaN rather than a silent zero.
        """
        point = self.predict(X).to_numpy(dtype=float)
        t = np.asarray(thresholds, dtype=float)
        if t.shape != point.shape:
            raise ValueError(f"expected {point.shape[0]} thresholds, got {t.shape}")
        needed = t - point  # the residual the outcome would have to come in at
        out = np.full(len(X), np.nan)
        known = ~np.isnan(needed)
        buckets = self._buckets(X)
        for b in np.unique(buckets[known]):
            rows = known & (buckets == b)
            sample = self.residual_sample(int(b))[None, :]
            cut = needed[rows][:, None]
            hits = sample >= cut if upper else sample <= cut
            out[rows] = hits.mean(axis=1)
        return pd.Series(out, index=X.index, dtype=float)

    def prob_at_least(self, X: pd.DataFrame, thresholds) -> pd.Series:
        """``P(y >= t)`` per row under the empirical distribution."""
        return self._tail(X, thresholds, upper=True)

    def prob_at_most(self, X: pd.DataFrame, thresholds) -> pd.Series:
        """``P(y <= t)`` per row under the empirical distribution."""
        return self._tail(X, thresholds, upper=False)

    def quantile(self, X: pd.DataFrame, q: float) -> pd.Series:
        """The `q` quantile of each row's distribution; at 0.10/0.90 this is the interval."""
        if not 0 <= q <= 1:
            raise ValueError(f"q must be in [0, 1], got {q}")
        point = self.predict(X).to_numpy(dtype=float)
        offsets = np.array(
            [float(np.quantile(self.residual_sample(int(b)), q)) for b in self._buckets(X)]
        )
        return pd.Series(point + offsets, index=X.index, dtype=float)


def weighted3_baseline(X: pd.DataFrame) -> pd.Series:
    """The pre-registered baseline: the PA- and recency-weighted 3-year wOBA average.

    `Weighted3` does not train, so this needs no training frame and reads the same at the
    forecast date as it does in hindsight.
    """
    return Weighted3().fit(X).predict(X)


def label_probabilities(
    dist: BucketedResidualDistribution,
    X: pd.DataFrame,
    *,
    baseline: pd.Series | None = None,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """Per-row point forecast, baseline, and the three label probabilities.

    `baseline` defaults to `weighted3_baseline(X)`. The two tails are disjoint for any
    positive `threshold`, so `p_stable` is their complement and the row sums to 1.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    b = weighted3_baseline(X) if baseline is None else baseline.astype(float)
    if not b.index.equals(X.index):
        raise ValueError("baseline must share the index of X")
    up = dist.prob_at_least(X, b.to_numpy(dtype=float) + threshold)
    down = dist.prob_at_most(X, b.to_numpy(dtype=float) - threshold)
    return pd.DataFrame(
        {
            "point": dist.predict(X),
            "baseline": b,
            "p_breakout": up,
            "p_stable": 1.0 - up - down,
            "p_regression": down,
        },
        index=X.index,
        columns=PROBABILITY_COLUMNS,
    )


def realized_labels(
    y: pd.Series, baseline: pd.Series, *, threshold: float = THRESHOLD
) -> pd.DataFrame:
    """The outcome side: boolean `breakout` and `regression` columns per row.

    A row with a missing outcome or baseline is False on both, which is the same as saying
    it is not evidence either way; callers drop those rows before scoring.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    actual = np.asarray(y, dtype=float)
    b = np.asarray(baseline, dtype=float)
    return pd.DataFrame(
        {
            "breakout": actual >= b + threshold,
            "regression": actual <= b - threshold,
        },
        index=y.index,
    )
