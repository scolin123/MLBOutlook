"""Tests for the breakout/regression probabilities: definitions, coherence, plumbing."""

import numpy as np
import pandas as pd
import pytest

from mlb_outlook.config import load_folds_config
from mlb_outlook.evaluation.harness import fold_frames
from mlb_outlook.features.asof import TARGET_COLUMNS
from mlb_outlook.models.intervals import DEFAULT_PA_EDGES, ResidualQuantileIntervals, pa_buckets
from mlb_outlook.models.probabilities import (
    PROBABILITY_COLUMNS,
    THRESHOLD,
    BucketedResidualDistribution,
    label_probabilities,
    realized_labels,
    weighted3_baseline,
)

CFG = load_folds_config("configs/folds.yaml")
SELECTION_FOLD = next(f for f in CFG.folds if f.name == "fold2")


@pytest.fixture
def frames(seasons_df):
    return fold_frames(seasons_df, SELECTION_FOLD, CFG, extended=True)


@pytest.fixture
def dist(frames):
    train, _ = frames
    return BucketedResidualDistribution().fit(train, train["y_woba"])


class ConstantForecaster:
    """Predicts one number, so a probability becomes a countable share of a known sample."""

    def __init__(self, value: float) -> None:
        self.value = value

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return pd.Series(self.value, index=X.index, dtype=float)


# ---- the definitions ----


def test_threshold_is_the_registered_one():
    """0.030 is pre-registered in docs/PROJECT_PLAN.md; a change needs a doc note."""
    assert THRESHOLD == 0.030


def test_realized_labels_are_closed_at_the_threshold():
    baseline = pd.Series([0.300, 0.300, 0.300, 0.300, 0.300])
    y = pd.Series([0.330, 0.3299, 0.270, 0.2701, 0.300])
    labels = realized_labels(y, baseline, threshold=0.030)
    # >= and <= , so landing exactly on the line counts.
    assert labels["breakout"].tolist() == [True, False, False, False, False]
    assert labels["regression"].tolist() == [False, False, True, False, False]


def test_realized_labels_are_mutually_exclusive(frames):
    _, test = frames
    labels = realized_labels(test["y_woba"], weighted3_baseline(test))
    assert not (labels["breakout"] & labels["regression"]).any()


def test_realized_labels_leave_a_missing_baseline_unlabelled():
    labels = realized_labels(pd.Series([0.400]), pd.Series([np.nan]))
    assert not labels["breakout"].iloc[0]
    assert not labels["regression"].iloc[0]


def test_bad_threshold_rejected(dist, frames):
    _, test = frames
    for threshold in (0.0, -0.030):
        with pytest.raises(ValueError, match="threshold must be positive"):
            label_probabilities(dist, test, threshold=threshold)
        with pytest.raises(ValueError, match="threshold must be positive"):
            realized_labels(test["y_woba"], weighted3_baseline(test), threshold=threshold)


def test_baseline_is_the_weighted_three_year_average(frames):
    """The pre-registered baseline is Weighted3, PA- and recency-weighted over three lags."""
    _, test = frames
    weights = (5, 4, 3)
    full = test.loc[
        np.all([test[f"pa_lag{k}"] > 0 for k in (1, 2, 3)], axis=0)
        & np.all([test[f"woba_lag{k}"].notna() for k in (1, 2, 3)], axis=0)
    ]
    assert not full.empty
    row = full.iloc[0]
    lags = list(zip((1, 2, 3), weights, strict=True))
    num = sum(w * row[f"pa_lag{k}"] * row[f"woba_lag{k}"] for k, w in lags)
    den = sum(w * row[f"pa_lag{k}"] for k, w in lags)
    assert weighted3_baseline(test).loc[row.name] == pytest.approx(num / den)


# ---- the distribution ----


def test_probabilities_are_counts_over_the_bucket_sample():
    """P(y >= t) is the share of the bucket's residuals that reach the threshold."""
    train = pd.DataFrame({"pa_lag1": [600] * 4, "y": [0.1, 0.2, 0.3, 0.4]})
    dist = BucketedResidualDistribution(
        factory=lambda: ConstantForecaster(0.25), pa_edges=()
    ).fit(train, train["y"])
    # Residuals are -0.15, -0.05, +0.05, +0.15 around a forecast of 0.25.
    assert sorted(dist.residual_sample()) == pytest.approx([-0.15, -0.05, 0.05, 0.15])
    X = pd.DataFrame({"pa_lag1": [600]})
    assert dist.prob_at_least(X, [0.31]).iloc[0] == pytest.approx(0.25)  # one of four
    assert dist.prob_at_least(X, [0.20]).iloc[0] == pytest.approx(0.75)  # three of four
    assert dist.prob_at_most(X, [0.19]).iloc[0] == pytest.approx(0.25)
    # Closed at the boundary, the same way the labels are: 0.30 needs r >= +0.05, and
    # +0.05 is in the sample, so it counts.
    assert dist.prob_at_least(X, [0.30]).iloc[0] == pytest.approx(0.50)
    assert dist.prob_at_most(X, [0.20]).iloc[0] == pytest.approx(0.50)
    assert dist.prob_at_least(X, [0.40]).iloc[0] == pytest.approx(0.25)
    assert dist.prob_at_least(X, [0.4001]).iloc[0] == pytest.approx(0.0)


def test_tails_bracket_the_whole_sample(dist, frames):
    _, test = frames
    assert (dist.prob_at_least(test, np.full(len(test), -9.0)) == 1.0).all()
    assert (dist.prob_at_least(test, np.full(len(test), 9.0)) == 0.0).all()
    assert (dist.prob_at_most(test, np.full(len(test), 9.0)) == 1.0).all()


def test_quantiles_reproduce_the_selected_intervals(frames):
    """The 10th and 90th percentiles of the distribution are the published 80% interval.

    This is the coherence the plan registers: the probabilities and the intervals are two
    readings of one object, so they cannot disagree.
    """
    train, test = frames
    dist = BucketedResidualDistribution().fit(train, train["y_woba"])
    bounds = ResidualQuantileIntervals().fit(train, train["y_woba"]).predict_interval(test)
    assert dist.quantile(test, 0.10).to_numpy() == pytest.approx(bounds["lo"].to_numpy())
    assert dist.quantile(test, 0.90).to_numpy() == pytest.approx(bounds["hi"].to_numpy())


def test_point_forecast_matches_the_bare_interval_model(frames):
    """Same factory, same full-data fit, so the forecast is the selected one."""
    train, test = frames
    dist = BucketedResidualDistribution().fit(train, train["y_woba"])
    wrapped = ResidualQuantileIntervals().fit(train, train["y_woba"])
    pd.testing.assert_series_equal(dist.predict(test), wrapped.predict(test))


def test_buckets_hold_their_own_residuals(dist):
    assert set(dist.samples_) == {0, 1, 2}
    assert sum(s.size for s in dist.samples_.values()) == dist.pooled_.size
    # Part-time seasons scatter more, which is the point of bucketing at all.
    spread = {b: float(np.std(s)) for b, s in dist.samples_.items()}
    assert spread[0] > spread[max(spread)]


def test_thin_bucket_falls_back_to_pooled(frames):
    train, test = frames
    dist = BucketedResidualDistribution(min_bucket=10**6).fit(train, train["y_woba"])
    assert dist.samples_ == {}
    assert dist.residual_sample(2) is dist.pooled_
    probs = label_probabilities(dist, test)
    # One pooled sample for everyone, so rows with the same gap to the baseline agree.
    assert probs["p_breakout"].notna().all()


def test_no_edges_means_one_pooled_bucket(frames):
    train, _ = frames
    dist = BucketedResidualDistribution(pa_edges=()).fit(train, train["y_woba"])
    assert dist.samples_ == {}
    assert dist.residual_sample(7) is dist.pooled_


def test_quantile_rejects_an_out_of_range_q(dist, frames):
    _, test = frames
    with pytest.raises(ValueError, match="q must be in"):
        dist.quantile(test, 1.5)


def test_unfitted_raises(frames):
    _, test = frames
    with pytest.raises(RuntimeError, match="not fitted"):
        BucketedResidualDistribution().predict(test)
    with pytest.raises(RuntimeError, match="not fitted"):
        BucketedResidualDistribution().residual_sample(0)
    with pytest.raises(ValueError, match="requires y"):
        BucketedResidualDistribution().fit(test, None)


def test_threshold_count_must_match_the_rows(dist, frames):
    _, test = frames
    with pytest.raises(ValueError, match="thresholds"):
        dist.prob_at_least(test, [0.300, 0.310])


def test_repeat_fits_are_identical(frames):
    train, test = frames
    a = BucketedResidualDistribution().fit(train, train["y_woba"])
    b = BucketedResidualDistribution().fit(train, train["y_woba"])
    pd.testing.assert_frame_equal(label_probabilities(a, test), label_probabilities(b, test))


# ---- the label table ----


def test_label_probabilities_shape_and_sum(dist, frames):
    _, test = frames
    probs = label_probabilities(dist, test)
    assert list(probs.columns) == PROBABILITY_COLUMNS
    assert probs.index.equals(test.index)
    assert probs.notna().all().all()
    total = probs[["p_breakout", "p_stable", "p_regression"]].sum(axis=1)
    assert total.to_numpy() == pytest.approx(np.ones(len(test)))
    for col in ("p_breakout", "p_stable", "p_regression"):
        assert ((probs[col] >= 0) & (probs[col] <= 1)).all()


def test_a_wider_threshold_makes_both_tails_smaller(dist, frames):
    _, test = frames
    narrow = label_probabilities(dist, test, threshold=0.020)
    wide = label_probabilities(dist, test, threshold=0.040)
    assert (wide["p_breakout"] <= narrow["p_breakout"]).all()
    assert (wide["p_regression"] <= narrow["p_regression"]).all()
    assert wide["p_stable"].mean() > narrow["p_stable"].mean()


def test_breakout_probability_rises_with_the_forecast_over_the_baseline(frames):
    """A forecast further above the recent record is a likelier breakout, by construction."""
    train, test = frames
    dist = BucketedResidualDistribution(pa_edges=()).fit(train, train["y_woba"])
    probs = label_probabilities(dist, test)
    gap = (probs["point"] - probs["baseline"]).to_numpy()
    order = np.argsort(gap)
    ranked = probs["p_breakout"].to_numpy()[order]
    assert np.all(np.diff(ranked) >= -1e-12)
    assert np.corrcoef(gap, probs["p_breakout"])[0, 1] > 0.9


def test_a_missing_baseline_gives_nan_not_zero(dist, frames):
    _, test = frames
    poisoned = weighted3_baseline(test)
    poisoned.iloc[0] = np.nan
    probs = label_probabilities(dist, test, baseline=poisoned)
    assert np.isnan(probs["p_breakout"].iloc[0])
    assert np.isnan(probs["p_regression"].iloc[0])
    assert probs["p_breakout"].iloc[1:].notna().all()


def test_baseline_index_must_match(dist, frames):
    _, test = frames
    with pytest.raises(ValueError, match="index of X"):
        shifted = weighted3_baseline(test).set_axis(np.arange(len(test)) + 100)
        label_probabilities(dist, test, baseline=shifted)


def test_probabilities_ignore_target_season_columns(dist, frames):
    """Nothing on the probability path may read the season it is forecasting."""
    _, test = frames
    clean = label_probabilities(dist, test)
    poisoned = test.copy()
    for col in TARGET_COLUMNS:
        poisoned[col] = 9.99
    pd.testing.assert_frame_equal(clean, label_probabilities(dist, poisoned))
    pd.testing.assert_frame_equal(
        clean, label_probabilities(dist, test.drop(columns=TARGET_COLUMNS))
    )


def test_width_of_the_stable_band_tracks_playing_time(dist, frames):
    """Part-time hitters get more probability in the tails, as their intervals are wider."""
    _, test = frames
    probs = label_probabilities(dist, test)
    tails = probs["p_breakout"] + probs["p_regression"]
    by_bucket = tails.groupby(pa_buckets(test, DEFAULT_PA_EDGES)).mean()
    assert by_bucket.loc[0] > by_bucket.loc[by_bucket.index.max()]
