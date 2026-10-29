"""Baselines against hand-computed values (league rate 0.320 in every lag season)."""

import math

import pandas as pd
import pytest

from mlb_outlook.models.baselines import Forecaster, LastSeason, MarcelBaseline, Weighted3

NAN = math.nan


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "age_target": [29, 25, 33, 24, 30],
            "woba_lag1": [0.400, 0.400, 0.400, 0.350, NAN],
            "woba_lag2": [0.360, 0.360, 0.360, NAN, NAN],
            "woba_lag3": [0.320, 0.320, 0.320, NAN, NAN],
            "pa_lag1": [600, 600, 600, 300, 0],
            "pa_lag2": [500, 500, 500, 0, 0],
            "pa_lag3": [400, 400, 400, 0, 0],
            "lg_woba_lag1": [0.320] * 5,
            "lg_woba_lag2": [0.320] * 5,
            "lg_woba_lag3": [0.320] * 5,
            "miss_lag1": [0, 0, 0, 0, 1],
            "miss_lag2": [0, 0, 0, 1, 1],
            "miss_lag3": [0, 0, 0, 1, 1],
        },
        index=[10, 11, 12, 13, 14],
    )


def test_last_season():
    pred = LastSeason().predict(_frame())
    assert pred.loc[10] == pytest.approx(0.400, abs=1e-6)
    assert pred.loc[13] == pytest.approx(0.350, abs=1e-6)
    assert math.isnan(pred.loc[14])


def test_weighted3():
    pred = Weighted3().predict(_frame())
    # (5*600*.400 + 4*500*.360 + 3*400*.320) / (3000 + 2000 + 1200) = 2304 / 6200
    assert pred.loc[10] == pytest.approx(0.371613, abs=1e-6)
    assert pred.loc[13] == pytest.approx(0.350, abs=1e-6)
    assert math.isnan(pred.loc[14])


def test_marcel_ages():
    pred = MarcelBaseline().predict(_frame())
    # (2304 + 1200*.320) / (6200 + 1200) = 2688 / 7400 = 0.363243, age factor 1 at 29
    assert pred.loc[10] == pytest.approx(0.363243, abs=1e-6)
    # age 25: * (1 + 4 * .006) = 1.024
    assert pred.loc[11] == pytest.approx(0.371961, abs=1e-6)
    # age 33: * (1 - 4 * .003) = 0.988
    assert pred.loc[12] == pytest.approx(0.358884, abs=1e-6)


def test_marcel_missing_seasons():
    pred = MarcelBaseline().predict(_frame())
    # (5*300*.350 + 1200*.320) / (1500 + 1200) = 909 / 2700 = .336667; age 24: * 1.03
    assert pred.loc[13] == pytest.approx(0.346767, abs=1e-6)


def test_marcel_no_history_regresses_to_league():
    pred = MarcelBaseline().predict(_frame())
    # no PA at all: league rate 0.320 * (1 + (29 - 30) * .003) = 0.319040
    assert pred.loc[14] == pytest.approx(0.319040, abs=1e-6)


def test_marcel_renormalised_league_rate():
    X = _frame().iloc[[0]].copy()
    X["lg_woba_lag3"] = NAN
    X["lg_woba_lag1"] = 0.300
    X["lg_woba_lag2"] = 0.330
    # league = (5*.300 + 4*.330) / 9 = 0.313333; (2304 + 1200*.313333) / 7400 = 2680 / 7400
    assert MarcelBaseline().predict(X).iloc[0] == pytest.approx(0.362162, abs=1e-6)


@pytest.mark.parametrize("model", [LastSeason(), Weighted3(), MarcelBaseline()])
def test_interface(model):
    X = _frame()
    assert isinstance(model, Forecaster)
    assert model.fit(X) is model
    assert model.fit(X, pd.Series(0.3, index=X.index)) is model
    pred = model.predict(X)
    assert isinstance(pred, pd.Series)
    assert list(pred.index) == list(X.index)
