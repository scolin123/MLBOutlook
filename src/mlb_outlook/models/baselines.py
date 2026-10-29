"""Baseline forecasters sharing one interface; inputs are the `build_features` frame."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import pandas as pd

from mlb_outlook.models.marcel import DEFAULT_REGRESSION_PA, DEFAULT_WEIGHTS, marcel_rate


@runtime_checkable
class Forecaster(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Forecaster: ...

    def predict(self, X: pd.DataFrame) -> pd.Series: ...


class LastSeason:
    """Predicts the previous season's wOBA."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> LastSeason:
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X["woba_lag1"].astype(float).rename(None)


def _weighted_woba(row: pd.Series, weights: Sequence[float]) -> float:
    num = 0.0
    den = 0.0
    for k, w in enumerate(weights, start=1):
        pa = row[f"pa_lag{k}"]
        woba = row[f"woba_lag{k}"]
        if pa > 0 and not math.isnan(woba):
            num += w * pa * woba
            den += w * pa
    return num / den if den > 0 else math.nan


class Weighted3:
    """PA- and recency-weighted mean of the last seasons; no regression, no age adjustment."""

    def __init__(self, weights: Sequence[float] = DEFAULT_WEIGHTS) -> None:
        self.weights = tuple(weights)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Weighted3:
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        values = [_weighted_woba(row, self.weights) for _, row in X.iterrows()]
        return pd.Series(values, index=X.index, dtype=float)


class MarcelBaseline:
    """Marcel projection per feature row, delegating the formula to `marcel_rate`."""

    def __init__(
        self,
        weights: Sequence[float] = DEFAULT_WEIGHTS,
        regression_pa: float = DEFAULT_REGRESSION_PA,
    ) -> None:
        self.weights = tuple(weights)
        self.regression_pa = regression_pa

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> MarcelBaseline:
        return self

    def _league_rate(self, row: pd.Series) -> float:
        num = 0.0
        den = 0.0
        for k, w in enumerate(self.weights, start=1):
            lg = row[f"lg_woba_lag{k}"]
            if not math.isnan(lg):
                num += w * lg
                den += w
        return num / den if den > 0 else math.nan

    def predict(self, X: pd.DataFrame) -> pd.Series:
        values = []
        for _, row in X.iterrows():
            seasons = [
                (float(row[f"pa_lag{k}"]), float(row[f"woba_lag{k}"]))
                for k in range(1, len(self.weights) + 1)
            ]
            seasons = [(pa, 0.0 if math.isnan(r) else r) for pa, r in seasons]
            lg = self._league_rate(row)
            if math.isnan(lg):
                values.append(math.nan)
                continue
            values.append(
                marcel_rate(
                    seasons,
                    row["age_target"],
                    lg,
                    weights=self.weights,
                    regression_pa=self.regression_pa,
                )
            )
        return pd.Series(values, index=X.index, dtype=float)
