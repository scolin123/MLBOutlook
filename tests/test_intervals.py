"""Tests for prediction intervals: calibration, shape, no test-season reads, coverage."""

import numpy as np
import pandas as pd
import pytest

from mlb_outlook.config import load_folds_config
from mlb_outlook.evaluation.harness import fold_frames, interval_metrics, run_fold
from mlb_outlook.features.asof import TARGET_COLUMNS
from mlb_outlook.models.intervals import (
    DEFAULT_LEVEL,
    INTERVAL_COLUMNS,
    IntervalForecaster,
    QuantileRegressionIntervals,
    ResidualQuantileIntervals,
    SplitConformalIntervals,
    conformal_quantile,
    pa_buckets,
)
from mlb_outlook.models.relative import relative_ridge

CFG = load_folds_config("configs/folds.yaml")
FOLD2 = next(f for f in CFG.folds if f.name == "fold2")
# Quantile regression fits a linear program per bound, so it is kept out of the parametrised
# sweeps and exercised on its own.
FAST_MODELS = [
    ResidualQuantileIntervals,
    SplitConformalIntervals,
    lambda: SplitConformalIntervals(pa_edges=(300, 500)),
]
FAST_IDS = ["resid_q_pa", "conformal", "conformal_pa"]


@pytest.fixture
def frames(seasons_df):
    return fold_frames(seasons_df, FOLD2, CFG, extended=True)


def _fit(factory, frames):
    train, test = frames
    return factory().fit(train, train["y_woba"]), test


def _is_constant(bounds: pd.DataFrame) -> bool:
    """One width for every row, up to the rounding of adding and subtracting a forecast."""
    width = (bounds["hi"] - bounds["lo"]).to_numpy()
    return bool(np.ptp(width) < 1e-9)


# ---- the helpers ----


def test_pa_buckets_edges():
    X = pd.DataFrame({"pa_lag1": [100, 299, 300, 499, 500, 700]})
    assert pa_buckets(X, (300, 500)).tolist() == [0, 0, 1, 1, 2, 2]
    assert pa_buckets(X, ()).tolist() == [0] * 6  # no edges: one pooled bucket


def test_conformal_quantile_rank_and_cap():
    values = np.arange(1.0, 11.0)  # 1..10
    # ceil((10 + 1) * 0.8) = 9, so the 9th smallest value
    assert conformal_quantile(values, 0.80) == pytest.approx(9.0)
    # A level the sample cannot resolve falls back to the maximum rather than extrapolating.
    assert conformal_quantile(values, 0.99) == pytest.approx(10.0)
    assert conformal_quantile(np.array([2.0]), 0.80) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="at least one"):
        conformal_quantile(np.array([]), 0.80)


def test_interval_metrics_by_hand():
    y = np.array([0.30, 0.32, 0.34, 0.36])
    lo = np.array([0.29, 0.30, 0.30, 0.30])
    hi = np.array([0.31, 0.31, 0.40, 0.34])  # rows 0 and 2 cover, 1 and 3 do not
    coverage, width = interval_metrics(lo, hi, y)
    assert coverage == pytest.approx(0.5)
    assert width == pytest.approx((0.02 + 0.01 + 0.10 + 0.04) / 4)


def test_interval_metrics_counts_the_bounds_as_covered():
    coverage, _ = interval_metrics([0.30, 0.32], [0.31, 0.33], [0.30, 0.33])
    assert coverage == pytest.approx(1.0)


def test_interval_metrics_validates():
    with pytest.raises(ValueError, match="aligned non-empty"):
        interval_metrics([0.1], [0.2, 0.3], [0.15])
    with pytest.raises(ValueError, match="at least its lower bound"):
        interval_metrics([0.3], [0.2], [0.25])


# ---- shared behaviour ----


@pytest.mark.parametrize("factory", FAST_MODELS, ids=FAST_IDS)
def test_satisfies_protocol(factory):
    assert isinstance(factory(), IntervalForecaster)


@pytest.mark.parametrize("factory", FAST_MODELS, ids=FAST_IDS)
def test_interval_frame_shape_and_order(factory, frames):
    model, test = _fit(factory, frames)
    bounds = model.predict_interval(test)
    assert list(bounds.columns) == INTERVAL_COLUMNS
    assert bounds.index.equals(test.index)
    assert bounds.notna().all().all()
    point = model.predict(test)
    assert (bounds["lo"] <= point).all() and (point <= bounds["hi"]).all()


@pytest.mark.parametrize("factory", FAST_MODELS, ids=FAST_IDS)
def test_predict_interval_ignores_target_season_columns(factory, frames):
    model, test = _fit(factory, frames)
    clean = model.predict_interval(test)
    poisoned = test.copy()
    for col in TARGET_COLUMNS:
        poisoned[col] = 9.99
    pd.testing.assert_frame_equal(clean, model.predict_interval(poisoned))
    pd.testing.assert_frame_equal(clean, model.predict_interval(test.drop(columns=TARGET_COLUMNS)))


@pytest.mark.parametrize("factory", FAST_MODELS, ids=FAST_IDS)
def test_unfitted_raises(factory, frames):
    _, test = frames
    with pytest.raises(RuntimeError, match="not fitted"):
        factory().predict_interval(test)
    with pytest.raises(ValueError, match="requires y"):
        factory().fit(test, None)


@pytest.mark.parametrize("factory", FAST_MODELS, ids=FAST_IDS)
def test_repeat_fits_are_identical(factory, frames):
    train, test = frames
    a = factory().fit(train, train["y_woba"]).predict_interval(test)
    b = factory().fit(train, train["y_woba"]).predict_interval(test)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("factory", FAST_MODELS, ids=FAST_IDS)
def test_wider_level_gives_wider_intervals(factory, frames):
    train, test = frames

    def width(level):
        model = factory()
        model.level = level
        bounds = model.fit(train, train["y_woba"]).predict_interval(test)
        return float((bounds["hi"] - bounds["lo"]).mean())

    assert width(0.50) < width(DEFAULT_LEVEL) < width(0.95)


@pytest.mark.parametrize("factory", FAST_MODELS, ids=FAST_IDS)
def test_coverage_on_training_rows_is_near_the_level(factory, frames):
    """Coverage on the rows the spread was calibrated on should land near the level."""
    train, _ = frames
    model = factory().fit(train, train["y_woba"])
    bounds = model.predict_interval(train)
    coverage, _ = interval_metrics(bounds["lo"], bounds["hi"], train["y_woba"])
    assert abs(coverage - DEFAULT_LEVEL) < 0.06


def test_bad_level_rejected():
    for level in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="level must be"):
            ResidualQuantileIntervals(level=level)


# ---- per-approach behaviour ----


def test_residual_quantile_point_forecast_matches_the_bare_model(frames):
    train, test = frames
    wrapped = ResidualQuantileIntervals().fit(train, train["y_woba"])
    bare = relative_ridge().fit(train, train["y_woba"])
    # The whole frame is used for the point fit, so the forecast is the unwrapped one.
    pd.testing.assert_series_equal(wrapped.predict(test), bare.predict(test))


def test_residual_quantile_width_varies_by_playing_time(frames):
    train, test = frames
    model = ResidualQuantileIntervals().fit(train, train["y_woba"])
    assert len(model.offsets_) > 1
    bounds = model.predict_interval(test)
    width = (bounds["hi"] - bounds["lo"]).groupby(pa_buckets(test)).mean()
    # Part-time seasons carry more uncertainty into the next one than full seasons do.
    assert width.loc[0] > width.loc[width.index.max()]


def test_residual_quantile_falls_back_to_pooled_for_a_thin_bucket(frames):
    train, test = frames
    model = ResidualQuantileIntervals(min_bucket=10**6).fit(train, train["y_woba"])
    assert model.offsets_ == {}
    assert _is_constant(model.predict_interval(test))  # one pooled width for everyone


def test_conformal_holds_calibration_rows_out_of_the_point_fit(frames):
    train, test = frames
    model = SplitConformalIntervals(cal_fraction=0.3).fit(train, train["y_woba"])
    assert model.n_calibration_ == round(0.3 * len(train))
    bare = relative_ridge().fit(train, train["y_woba"])
    # Fitted on 70% of the rows, so the point forecast is not the full-data one.
    assert not np.allclose(model.predict(test), bare.predict(test))


def test_conformal_width_is_symmetric_and_constant_without_buckets(frames):
    train, test = frames
    model = SplitConformalIntervals().fit(train, train["y_woba"])
    lo_off, hi_off = model.pooled_
    assert lo_off == pytest.approx(-hi_off)
    assert _is_constant(model.predict_interval(test))


def test_conformal_split_must_leave_both_sides_usable(frames):
    train, _ = frames
    with pytest.raises(ValueError, match="cal_fraction must be"):
        SplitConformalIntervals(cal_fraction=0.0)
    with pytest.raises(ValueError, match="nothing to fit"):
        SplitConformalIntervals(cal_fraction=0.99).fit(train.head(2), train.head(2)["y_woba"])


def test_conformal_seed_changes_the_split_but_not_much_else(frames):
    train, test = frames
    widths = []
    for seed in (0, 1, 2):
        model = SplitConformalIntervals(seed=seed).fit(train, train["y_woba"])
        bounds = model.predict_interval(test)
        widths.append(float((bounds["hi"] - bounds["lo"]).mean()))
    assert len(set(widths)) == 3  # a different split gives a different quantile
    assert max(widths) - min(widths) < 0.01  # but not a wildly different one


def test_quantile_regression_width_responds_to_features(frames):
    train, test = frames
    model = QuantileRegressionIntervals().fit(train, train["y_woba"])
    bounds = model.predict_interval(test)
    assert list(bounds.columns) == INTERVAL_COLUMNS
    width = bounds["hi"] - bounds["lo"]
    assert width.nunique() == len(test)  # a width per row, not per bucket
    assert (width > 0).all()
    coverage, _ = interval_metrics(bounds["lo"], bounds["hi"], test["y_woba"])
    assert 0.5 < coverage < 1.0


def test_quantile_regression_unfitted_raises(frames):
    _, test = frames
    with pytest.raises(RuntimeError, match="not fitted"):
        QuantileRegressionIntervals().predict_interval(test)


# ---- through the harness ----


@pytest.mark.parametrize("factory", FAST_MODELS, ids=FAST_IDS)
def test_run_fold_reports_coverage_and_width(seasons_df, factory):
    out = run_fold(
        seasons_df, FOLD2, CFG, {"iv": factory}, allow_unfrozen=True, extended=True
    ).iloc[0]
    assert 0.0 <= out["coverage"] <= 1.0
    assert out["width"] > 0
    assert out["mae"] > 0


def test_run_fold_leaves_coverage_blank_for_point_models(seasons_df):
    out = run_fold(seasons_df, FOLD2, CFG, {"ridge_rel": relative_ridge}, allow_unfrozen=True)
    assert out.iloc[0]["coverage"] != out.iloc[0]["coverage"]  # NaN
    assert out.iloc[0]["width"] != out.iloc[0]["width"]
