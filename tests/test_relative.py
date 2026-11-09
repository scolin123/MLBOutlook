"""Tests for the league-relative wrapper: centring, re-levelling, and no target-season reads."""

import numpy as np
import pandas as pd
import pytest

from mlb_outlook.config import load_folds_config
from mlb_outlook.evaluation.harness import fold_frames
from mlb_outlook.features.asof import TARGET_COLUMNS
from mlb_outlook.models.baselines import Forecaster
from mlb_outlook.models.relative import LeagueRelative, relative_ridge
from mlb_outlook.models.ridge import RidgeForecaster

CFG = load_folds_config("configs/folds.yaml")
FOLD2 = next(f for f in CFG.folds if f.name == "fold2")


@pytest.fixture
def frames(seasons_df):
    return fold_frames(seasons_df, FOLD2, CFG)


class Spy:
    """Inner forecaster that records the centred target and predicts a constant."""

    def __init__(self, value=0.0):
        self.value = value
        self.seen_y = None
        self.seen_cols = None

    def fit(self, X, y=None):
        self.seen_y = pd.Series(y).copy()
        self.seen_cols = list(X.columns)
        return self

    def predict(self, X):
        return pd.Series(self.value, index=X.index, dtype=float)


def test_satisfies_protocol():
    assert isinstance(relative_ridge(), Forecaster)


def test_fit_centres_target_on_its_own_season_league(frames):
    train, _ = frames
    spy = Spy()
    LeagueRelative(spy).fit(train, train["y_woba"])
    expected = train["y_woba"] - train["y_lg_woba"]
    pd.testing.assert_series_equal(spy.seen_y, expected, check_names=False)
    assert train["y_lg_woba"].nunique() == len(FOLD2.train_targets)  # one level per season


def test_predict_adds_the_prior_season_league_estimate(frames):
    train, test = frames
    model = LeagueRelative(Spy(value=0.01)).fit(train, train["y_woba"])
    pred = model.predict(test)
    pd.testing.assert_series_equal(pred, test["lg_woba_hat"] + 0.01, check_names=False)


def test_predict_ignores_target_season_columns(frames):
    train, test = frames
    model = relative_ridge().fit(train, train["y_woba"])
    clean = model.predict(test)
    poisoned = test.copy()
    for col in TARGET_COLUMNS:
        poisoned[col] = 9.99
    pd.testing.assert_series_equal(clean, model.predict(poisoned))
    # Production frames (include_target=False) carry no y_ columns at all.
    pd.testing.assert_series_equal(clean, model.predict(test.drop(columns=TARGET_COLUMNS)))


def test_fit_requires_y_and_the_league_column(frames):
    train, _ = frames
    with pytest.raises(ValueError, match="requires y"):
        relative_ridge().fit(train, None)
    with pytest.raises(ValueError, match="y_lg_woba"):
        relative_ridge().fit(train.drop(columns=["y_lg_woba"]), train["y_woba"])


def test_predict_requires_the_league_estimate(frames):
    train, test = frames
    model = relative_ridge().fit(train, train["y_woba"])
    with pytest.raises(ValueError, match="lg_woba_hat"):
        model.predict(test.drop(columns=["lg_woba_hat"]))


def test_shifting_every_season_shifts_predictions_by_the_same_amount(seasons_df):
    shift = 0.01
    shifted = seasons_df.copy()
    shifted["woba"] += shift
    base_train, base_test = fold_frames(seasons_df, FOLD2, CFG)
    up_train, up_test = fold_frames(shifted, FOLD2, CFG)
    base = relative_ridge().fit(base_train, base_train["y_woba"]).predict(base_test)
    up = relative_ridge().fit(up_train, up_train["y_woba"]).predict(up_test)
    assert len(base) == len(up)
    np.testing.assert_allclose(up.to_numpy(), base.to_numpy() + shift, atol=1e-12)


def _fold2_bias(data, factory):
    train, test = fold_frames(data, FOLD2, CFG)
    model = factory().fit(train, train["y_woba"])
    return float((model.predict(test) - test["y_woba"]).mean())


def test_league_level_step_leaves_relative_bias_unchanged(seasons_df):
    """A step down in the league from the last training season on. The relative model
    re-levels from the prior season, so its bias is unmoved; plain ridge's is not."""
    step = 0.015
    stepped = seasons_df.copy()
    stepped.loc[stepped["season"] >= FOLD2.train_targets[-1], "woba"] -= step
    assert _fold2_bias(stepped, relative_ridge) == pytest.approx(
        _fold2_bias(seasons_df, relative_ridge), abs=1e-12
    )
    plain_shift = _fold2_bias(stepped, RidgeForecaster) - _fold2_bias(seasons_df, RidgeForecaster)
    assert abs(plain_shift) > 0.002
