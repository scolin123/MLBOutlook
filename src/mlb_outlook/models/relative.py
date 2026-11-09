"""League-relative forecasting: model the gap to the league, not the raw rate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlb_outlook.models.baselines import Forecaster
from mlb_outlook.models.ridge import RELATIVE_FEATURES, RidgeForecaster

TARGET_LEAGUE_COL = "y_lg_woba"
ESTIMATE_LEAGUE_COL = "lg_woba_hat"


class LeagueRelative:
    """Fit an inner forecaster on league-relative wOBA, then put the league level back.

    This takes the shared season environment out of the training signal. The inner model is
    fitted on ``y - y_lg_woba``, so every row is centred on its own target season's league
    wOBA -- known at the forecast date, because each training target season precedes the test
    target. Predictions are re-levelled with ``lg_woba_hat``, which is built from prior
    seasons only, so ``predict`` never needs target-season data.
    """

    def __init__(self, inner: Forecaster) -> None:
        self.inner = inner

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> LeagueRelative:
        if y is None:
            raise ValueError("LeagueRelative.fit requires y")
        if TARGET_LEAGUE_COL not in X.columns:
            raise ValueError(f"LeagueRelative.fit requires the {TARGET_LEAGUE_COL!r} column")
        centred = np.asarray(y, dtype=float) - X[TARGET_LEAGUE_COL].to_numpy(dtype=float)
        self.inner.fit(X, pd.Series(centred, index=X.index))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if ESTIMATE_LEAGUE_COL not in X.columns:
            raise ValueError(f"LeagueRelative.predict requires the {ESTIMATE_LEAGUE_COL!r} column")
        relative = self.inner.predict(X)
        return (relative + X[ESTIMATE_LEAGUE_COL].astype(float)).rename(None)


def relative_ridge(alpha: float = 10.0) -> LeagueRelative:
    """Ridge on league-relative lag features, re-levelled with `lg_woba_hat`."""
    return LeagueRelative(RidgeForecaster(alpha=alpha, feature_cols=RELATIVE_FEATURES))
