"""Ridge regression forecaster over the `build_features` frame."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = tuple(
    f"{name}_lag{k}"
    for name in ("woba", "pa", "k_pct", "bb_pct", "lg_woba", "miss")
    for k in (1, 2, 3)
) + ("age_target",)


class RidgeForecaster:
    """Median-impute, standardise, then ridge; all state is learned inside `fit`."""

    def __init__(self, alpha: float = 10.0, feature_cols: Sequence[str] | None = None) -> None:
        self.alpha = alpha
        self.feature_cols = tuple(feature_cols) if feature_cols is not None else DEFAULT_FEATURES
        self.pipeline_: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> RidgeForecaster:
        if y is None:
            raise ValueError("RidgeForecaster.fit requires y")
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )
        pipe.fit(X[list(self.feature_cols)].astype(float), pd.Series(y).to_numpy(dtype=float))
        self.pipeline_ = pipe
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.pipeline_ is None:
            raise RuntimeError("RidgeForecaster is not fitted; call fit() before predict()")
        values = self.pipeline_.predict(X[list(self.feature_cols)].astype(float))
        return pd.Series(values, index=X.index, dtype=float)
