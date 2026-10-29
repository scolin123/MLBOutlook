"""Tests for the ridge forecaster: fit-only state, determinism, NaN handling, integration."""

import numpy as np
import pandas as pd
import pytest

from mlb_outlook.config import load_folds_config
from mlb_outlook.evaluation.harness import fold_frames, run_all
from mlb_outlook.features.asof import build_features
from mlb_outlook.models.baselines import Forecaster, LastSeason, MarcelBaseline, Weighted3
from mlb_outlook.models.ridge import DEFAULT_FEATURES, RidgeForecaster

CFG = load_folds_config("configs/folds.yaml")


@pytest.fixture
def frames(seasons_df):
    return fold_frames(seasons_df, CFG.folds[0], CFG)


def _params(model):
    pipe = model.pipeline_
    return (
        pipe["imputer"].statistics_.copy(),
        pipe["scaler"].mean_.copy(),
        pipe["scaler"].scale_.copy(),
        pipe["ridge"].coef_.copy(),
        float(pipe["ridge"].intercept_),
    )


def test_satisfies_protocol():
    assert isinstance(RidgeForecaster(), Forecaster)


def test_scaler_mean_equals_training_means(frames):
    train, _ = frames
    model = RidgeForecaster().fit(train, train["y_woba"])
    cols = list(DEFAULT_FEATURES)
    # Impute with the median first, then the scaler mean is the mean of the imputed columns.
    imputed = train[cols].astype(float).fillna(train[cols].astype(float).median())
    np.testing.assert_allclose(model.pipeline_["scaler"].mean_, imputed.mean().to_numpy())
    complete = [c for c in cols if not train[c].isna().any()]
    for c in complete:
        i = cols.index(c)
        assert model.pipeline_["scaler"].mean_[i] == pytest.approx(train[c].mean())


def test_predict_does_not_change_fitted_state(frames):
    train, test = frames
    model = RidgeForecaster().fit(train, train["y_woba"])
    before = _params(model)
    extreme = test.copy()
    for c in DEFAULT_FEATURES:
        extreme[c] = 1e9
    model.predict(extreme)
    model.predict(test)
    after = _params(model)
    for b, a in zip(before, after, strict=True):
        np.testing.assert_array_equal(b, a)


def test_repeat_predict_and_independent_fits_identical(frames):
    train, test = frames
    m1 = RidgeForecaster().fit(train, train["y_woba"])
    m2 = RidgeForecaster().fit(train, train["y_woba"])
    p1 = m1.predict(test)
    pd.testing.assert_series_equal(p1, m1.predict(test))
    pd.testing.assert_series_equal(p1, m2.predict(test))
    assert p1.index.equals(test.index)


def test_nan_lags_handled(frames):
    train, test = frames
    assert train["woba_lag3"].isna().any()
    model = RidgeForecaster().fit(train, train["y_woba"])
    pred = model.predict(test)
    assert len(pred) == len(test)
    assert not pred.isna().any()


def test_unfitted_predict_raises(frames):
    _, test = frames
    with pytest.raises(RuntimeError, match="not fitted"):
        RidgeForecaster().predict(test)


def test_fit_recovers_exact_line():
    x = np.arange(20, dtype=float)
    X = pd.DataFrame({"a": x}, index=range(100, 120))
    y = pd.Series(3.0 * x + 2.0, index=X.index)
    model = RidgeForecaster(alpha=1e-9, feature_cols=["a"]).fit(X, y)
    pred = model.predict(pd.DataFrame({"a": [25.0, -3.0]}, index=["p", "q"]))
    assert pred.index.tolist() == ["p", "q"]
    assert pred.to_numpy() == pytest.approx([77.0, -7.0], abs=1e-6)


FACTORIES = {
    "last_season": LastSeason,
    "weighted3": Weighted3,
    "marcel": MarcelBaseline,
    "ridge": RidgeForecaster,
}


def test_integration_beats_last_season(seasons_df):
    res = run_all(seasons_df, CFG, FACTORIES, allow_unfrozen=True)
    mae = res.groupby("model")["mae"].mean()
    assert res["fold"].nunique() == len(CFG.folds)
    assert mae["marcel"] < mae["last_season"]
    assert mae["ridge"] < mae["last_season"]


@pytest.mark.parametrize("fold", CFG.folds, ids=lambda f: f.name)
def test_truncating_future_does_not_change_ridge_predictions(seasons_df, fold):
    def preds(data):
        train, _ = fold_frames(data, fold, CFG)
        model = RidgeForecaster().fit(train, train["y_woba"])
        test_x = build_features(
            data,
            fold.test_target,
            include_target=False,
            min_pa_feature=CFG.min_pa.feature_season,
        )
        return model.predict(test_x)

    full = preds(seasons_df)
    truncated = preds(seasons_df[seasons_df["season"] < fold.test_target])
    assert len(full) > 0
    pd.testing.assert_series_equal(full, truncated)
