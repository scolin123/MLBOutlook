"""Where the interval and probability residuals come from: the training window, only.

Both published uncertainty objects -- the selected `resid_q_pa` intervals and the
`BucketedResidualDistribution` the breakout/regression probabilities are read off -- are a
point fit plus a sample of that fit's residuals. The claim this module exists to prove is
that the sample is drawn from the training window and nothing else, so no part of a
published interval or probability is informed by an outcome it is scored against.

Three separate things are checked, because each could fail without the others:

1. **Identity.** The stored residual sample *is* the training rows' residuals -- same
   length, same values, bucket by bucket -- recomputed here from the training frame.
2. **Independence of the test outcomes.** Truncating the data at the test season, so the
   target season's rows do not exist at all, leaves the fitted object and every bound and
   probability it produces bit-identical.
3. **Independence of the test frame at predict time.** Covered for the published pair by
   `test_predict_interval_ignores_target_season_columns` and
   `test_probabilities_ignore_target_season_columns`; the counts here add that the fit never
   saw as many rows as train and test together.

`docs/DESIGN_FREEZE.md` section 2 states the provenance in prose and points here.
"""

import numpy as np
import pandas as pd
import pytest

from mlb_outlook.config import load_folds_config
from mlb_outlook.evaluation.harness import fold_frames
from mlb_outlook.models.intervals import (
    DEFAULT_LEVEL,
    DEFAULT_PA_EDGES,
    MIN_BUCKET,
    QuantileRegressionIntervals,
    ResidualQuantileIntervals,
    SplitConformalIntervals,
    pa_buckets,
)
from mlb_outlook.models.probabilities import (
    THRESHOLD,
    BucketedResidualDistribution,
    label_probabilities,
)
from mlb_outlook.models.relative import relative_ridge

CFG = load_folds_config("configs/folds.yaml")
FOLD2 = next(f for f in CFG.folds if f.name == "fold2")


@pytest.fixture
def frames(seasons_df):
    return fold_frames(seasons_df, FOLD2, CFG, extended=True)


def training_residuals(train: pd.DataFrame) -> np.ndarray:
    """The residuals of the selected forecaster, fitted on the training frame, recomputed."""
    model = relative_ridge().fit(train, train["y_woba"])
    return train["y_woba"].to_numpy(dtype=float) - model.predict(train).to_numpy(dtype=float)


# ---- 1. the sample is the training rows' residuals ----


def test_the_distribution_holds_exactly_the_training_residuals(frames):
    train, _ = frames
    dist = BucketedResidualDistribution().fit(train, train["y_woba"])
    expected = training_residuals(train)
    assert dist.pooled_.shape == (len(train),)
    np.testing.assert_allclose(np.sort(dist.pooled_), np.sort(expected), atol=1e-12)


def test_every_bucket_sample_is_that_buckets_training_residuals(frames):
    train, _ = frames
    dist = BucketedResidualDistribution().fit(train, train["y_woba"])
    expected = training_residuals(train)
    buckets = pa_buckets(train, DEFAULT_PA_EDGES)
    assert dist.samples_, "the fold's buckets are all above min_bucket, so some are stored"
    for bucket, sample in dist.samples_.items():
        inside = expected[buckets == bucket]
        assert sample.size == inside.size >= MIN_BUCKET
        np.testing.assert_allclose(np.sort(sample), np.sort(inside), atol=1e-12)
    # And the samples partition the training rows: no row counted twice, none invented.
    assert sum(s.size for s in dist.samples_.values()) <= len(train)


def test_the_interval_offsets_are_quantiles_of_those_same_residuals(frames):
    train, _ = frames
    model = ResidualQuantileIntervals().fit(train, train["y_woba"])
    expected = training_residuals(train)
    buckets = pa_buckets(train, DEFAULT_PA_EDGES)
    tail = (1 - DEFAULT_LEVEL) / 2
    for bucket, (lo, hi) in model.offsets_.items():
        inside = expected[buckets == bucket]
        assert lo == pytest.approx(float(np.quantile(inside, tail)), abs=1e-12)
        assert hi == pytest.approx(float(np.quantile(inside, 1 - tail)), abs=1e-12)


def test_the_published_interval_and_probability_read_one_sample(frames):
    """The 10th/90th percentiles of the probability distribution are the published bounds."""
    train, test = frames
    dist = BucketedResidualDistribution().fit(train, train["y_woba"])
    intervals = ResidualQuantileIntervals().fit(train, train["y_woba"])
    bounds = intervals.predict_interval(test)
    np.testing.assert_allclose(dist.quantile(test, 0.10), bounds["lo"], atol=1e-12)
    np.testing.assert_allclose(dist.quantile(test, 0.90), bounds["hi"], atol=1e-12)


def test_the_conformal_sample_is_the_held_out_calibration_rows(frames):
    """The fallback's residuals are training rows too -- held out of the fit, not of the fold."""
    train, _ = frames
    model = SplitConformalIntervals().fit(train, train["y_woba"])
    assert model.n_calibration_ == round(len(train) * model.cal_fraction)
    assert model.n_calibration_ < len(train)


# ---- 2. the test season's outcomes are not an input ----


def truncated(seasons_df: pd.DataFrame) -> pd.DataFrame:
    """The snapshot with the test season and everything after it removed."""
    return seasons_df.loc[seasons_df["season"] < FOLD2.test_target].copy()


def test_the_training_frame_does_not_depend_on_the_test_season(seasons_df):
    train, _ = fold_frames(seasons_df, FOLD2, CFG, extended=True)
    train_cut, _ = fold_frames(truncated(seasons_df), FOLD2, CFG, extended=True)
    pd.testing.assert_frame_equal(train, train_cut)


@pytest.mark.parametrize(
    "factory",
    [ResidualQuantileIntervals, SplitConformalIntervals, QuantileRegressionIntervals],
    ids=["resid_q_pa", "conformal", "quantile_reg"],
)
def test_intervals_are_unchanged_when_the_test_season_is_deleted(seasons_df, factory):
    _, test = fold_frames(seasons_df, FOLD2, CFG, extended=True)
    train_cut, _ = fold_frames(truncated(seasons_df), FOLD2, CFG, extended=True)
    train, _ = fold_frames(seasons_df, FOLD2, CFG, extended=True)
    full = factory().fit(train, train["y_woba"]).predict_interval(test)
    cut = factory().fit(train_cut, train_cut["y_woba"]).predict_interval(test)
    pd.testing.assert_frame_equal(full, cut)


def test_probabilities_are_unchanged_when_the_test_season_is_deleted(seasons_df):
    train, test = fold_frames(seasons_df, FOLD2, CFG, extended=True)
    train_cut, _ = fold_frames(truncated(seasons_df), FOLD2, CFG, extended=True)
    full = label_probabilities(
        BucketedResidualDistribution().fit(train, train["y_woba"]), test, threshold=THRESHOLD
    )
    cut = label_probabilities(
        BucketedResidualDistribution().fit(train_cut, train_cut["y_woba"]),
        test,
        threshold=THRESHOLD,
    )
    pd.testing.assert_frame_equal(full, cut)


def test_poisoned_test_outcomes_do_not_move_the_fit(frames):
    """The strongest version: wreck every outcome in the test frame, change nothing."""
    train, test = frames
    dist = BucketedResidualDistribution().fit(train, train["y_woba"])
    poisoned = test.copy()
    poisoned["y_woba"] = 9.99
    poisoned["y_pa"] = 10**6
    poisoned["y_lg_woba"] = 9.99
    pd.testing.assert_frame_equal(
        label_probabilities(dist, test), label_probabilities(dist, poisoned)
    )


# ---- 3. the fit saw the training rows and no more ----


def test_the_fit_never_saw_the_test_rows(frames):
    train, test = frames
    dist = BucketedResidualDistribution().fit(train, train["y_woba"])
    assert dist.pooled_.size == len(train)
    assert dist.pooled_.size != len(train) + len(test)
