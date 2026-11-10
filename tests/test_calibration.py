"""Tests for probability scoring: Brier, base rates, reliability deciles, the fold runner."""

import numpy as np
import pandas as pd
import pytest

from mlb_outlook.config import Fold, load_folds_config
from mlb_outlook.evaluation.calibration import (
    PRECISION_KS,
    RELIABILITY_COLUMNS,
    SCORE_COLUMNS,
    base_rate,
    brier_score,
    expected_calibration_error,
    precision_at_k,
    reliability_table,
    run_probability_fold,
    score_probabilities,
    squared_errors,
)
from mlb_outlook.evaluation.harness import FoldNotFrozenError
from mlb_outlook.models.probabilities import LABELS, SENSITIVITY_THRESHOLDS, THRESHOLD

CFG = load_folds_config("configs/folds.yaml")
SELECTION_FOLD = next(f for f in CFG.folds if f.name == "fold2")
# An unfrozen holdout, built here rather than read out of folds.yaml: the freeze commit
# flips fold3 to `frozen: true`, and the guard still has to be tested after that.
HOLDOUT = Fold(name="holdout", train_targets=[2023, 2024, 2025], test_target=2026, frozen=False)


# ---- Brier and base rates ----


def test_brier_score_by_hand():
    prob = [0.9, 0.2, 0.5]
    outcome = [1, 0, 1]
    expected = (0.01 + 0.04 + 0.25) / 3
    assert brier_score(prob, outcome) == pytest.approx(expected)
    assert squared_errors(prob, outcome) == pytest.approx([0.01, 0.04, 0.25])


def test_brier_bounds():
    assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert brier_score([0.0, 1.0], [1, 0]) == pytest.approx(1.0)


def test_brier_accepts_booleans():
    assert brier_score([1.0, 0.0], [True, False]) == pytest.approx(0.0)


def test_base_rate_is_the_share_of_labels():
    assert base_rate([True, False, False, False]) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="at least one row"):
        base_rate([])


def test_scoring_validates_its_inputs():
    with pytest.raises(ValueError, match="aligned non-empty"):
        brier_score([0.5], [1, 0])
    with pytest.raises(ValueError, match="carries NaN"):
        brier_score([0.5, np.nan], [1, 0])
    with pytest.raises(ValueError, match="boolean or 0/1"):
        brier_score([0.5, 0.5], [1, 2])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([1.5, 0.5], [1, 0])


# ---- reliability ----


def test_reliability_table_bins_into_equal_count_deciles():
    rng = np.random.default_rng(0)
    prob = rng.random(500)
    outcome = rng.random(500) < prob  # perfectly calibrated by construction
    table = reliability_table(prob, outcome)
    assert len(table) == 10
    assert table["n"].sum() == 500
    assert table["n"].tolist() == [50] * 10
    assert table["decile"].tolist() == list(range(1, 11))
    # Bins are ordered and do not overlap.
    assert table["mean_predicted"].is_monotonic_increasing
    assert (table["p_low"].to_numpy()[1:] >= table["p_high"].to_numpy()[:-1]).all()
    assert table["gap"].to_numpy() == pytest.approx(
        (table["observed"] - table["mean_predicted"]).to_numpy()
    )
    assert expected_calibration_error(table) < 0.08


def test_reliability_table_collapses_when_every_prediction_is_the_same():
    """Ties drop duplicate edges rather than splitting equal predictions across deciles."""
    table = reliability_table([0.3] * 40, [True] * 12 + [False] * 28)
    assert len(table) == 1
    assert table["n"].iloc[0] == 40
    assert table["observed"].iloc[0] == pytest.approx(0.3)
    assert table["gap"].iloc[0] == pytest.approx(0.0)


def test_reliability_table_on_a_biased_forecast():
    """A forecast that is too low everywhere shows a positive gap in every decile."""
    rng = np.random.default_rng(1)
    truth = 0.2 + rng.random(2000) * 0.6
    outcome = rng.random(2000) < truth
    table = reliability_table(truth - 0.1, outcome)  # understated by a tenth everywhere
    # The bias shows up in the aggregate; a single decile of 200 rows still has +/-0.035
    # of sampling noise in it, which is the honest reading of a reliability row.
    assert np.average(table["gap"], weights=table["n"]) == pytest.approx(0.1, abs=0.03)
    assert (table["gap"] > 0).mean() >= 0.8
    assert expected_calibration_error(table) > 0.05


def test_reliability_table_rejects_too_few_bins():
    with pytest.raises(ValueError, match="at least 2"):
        reliability_table([0.2, 0.8], [0, 1], n_bins=1)


def test_expected_calibration_error_is_count_weighted():
    table = pd.DataFrame({"n": [90, 10], "gap": [0.0, 0.5]})
    assert expected_calibration_error(table) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="non-empty"):
        expected_calibration_error(table.head(0))


# ---- precision at k ----


def test_precision_at_k_takes_the_top_k_by_probability():
    prob = [0.1, 0.9, 0.4, 0.8, 0.2]
    outcome = [0, 1, 0, 1, 1]
    assert precision_at_k(prob, outcome, 2) == pytest.approx(1.0)  # 0.9 and 0.8, both hits
    assert precision_at_k(prob, outcome, 3) == pytest.approx(2 / 3)  # plus 0.4, a miss
    assert precision_at_k(prob, outcome, 5) == pytest.approx(0.6)  # every row: the base rate


def test_precision_at_k_above_the_row_count_scores_every_row():
    assert precision_at_k([0.9, 0.1], [1, 0], 25) == pytest.approx(0.5)


def test_precision_at_k_breaks_ties_by_row_order():
    """Equal probabilities are taken in input order, so the number is reproducible."""
    prob = [0.5, 0.5, 0.5, 0.5]
    assert precision_at_k(prob, [1, 1, 0, 0], 2) == pytest.approx(1.0)
    assert precision_at_k(prob, [0, 0, 1, 1], 2) == pytest.approx(0.0)


def test_precision_at_k_rejects_a_useless_k():
    with pytest.raises(ValueError, match="k must be at least 1"):
        precision_at_k([0.5, 0.5], [1, 0], 0)


def test_precision_at_k_of_a_perfect_ranking_is_one():
    outcome = np.array([1] * 20 + [0] * 80)
    prob = np.where(outcome == 1, 0.9, 0.1)
    assert precision_at_k(prob, outcome, 10) == pytest.approx(1.0)
    assert precision_at_k(prob, outcome, 25) == pytest.approx(0.8)  # 20 hits in 25 rows


# ---- the combined score ----


def test_score_against_the_base_rate_it_was_handed():
    """Predicting the training base rate for everyone must score exactly zero skill."""
    outcome = np.array([1] * 30 + [0] * 70)
    rate = 0.3
    out = score_probabilities(np.full(100, rate), outcome, base_rate_train=rate)
    assert out["brier"] == pytest.approx(out["brier_base"])
    assert out["skill"] == pytest.approx(0.0)
    assert out["brier_diff"] == pytest.approx(0.0)
    assert out["base_rate_test"] == pytest.approx(0.3)
    assert out["n"] == 100


def test_a_sharper_forecast_scores_positive_skill():
    outcome = np.array([1] * 50 + [0] * 50)
    prob = np.where(outcome == 1, 0.8, 0.2)
    out = score_probabilities(prob, outcome, base_rate_train=0.5)
    assert out["skill"] > 0.5
    assert out["brier_diff"] < 0  # model minus base, so negative is the model winning
    assert out["ci_high"] < 0  # and clear of zero
    assert out["mean_predicted"] == pytest.approx(0.5)
    assert (out["min_predicted"], out["max_predicted"]) == (0.2, 0.8)
    # The ranking is perfect here, so every top-k slice is a hit.
    assert all(out[f"precision_at_{k}"] == pytest.approx(1.0) for k in PRECISION_KS)


def test_score_reports_precision_for_every_registered_k():
    outcome = np.array([1] * 30 + [0] * 70)
    out = score_probabilities(np.full(100, 0.3), outcome, base_rate_train=0.3)
    for k in PRECISION_KS:
        # A flat forecast has no ranking, so precision@k is whatever the first k rows hold.
        assert out[f"precision_at_{k}"] == pytest.approx(1.0)
    assert set(f"precision_at_{k}" for k in PRECISION_KS) <= set(SCORE_COLUMNS)


def test_score_rejects_an_impossible_base_rate():
    with pytest.raises(ValueError, match=r"base_rate_train must lie in \[0, 1\]"):
        score_probabilities([0.3], [1], base_rate_train=1.4)


# ---- the fold runner ----


def test_run_probability_fold_shape(seasons_df):
    scores, reliability = run_probability_fold(seasons_df, SELECTION_FOLD, CFG, extended=True)
    assert list(scores.columns) == SCORE_COLUMNS
    assert list(reliability.columns) == RELIABILITY_COLUMNS
    assert set(scores["label"]) == set(LABELS)
    assert sorted(set(scores["threshold"])) == sorted(SENSITIVITY_THRESHOLDS)
    assert len(scores) == len(LABELS) * len(SENSITIVITY_THRESHOLDS)
    assert (scores["fold"] == SELECTION_FOLD.name).all()
    assert (scores["config"] == "extended").all()
    assert (scores["n"] > 0).all()
    assert scores["brier"].between(0, 1).all()
    assert scores["base_rate_train"].between(0, 1).all()
    assert scores["base_rate_test"].between(0, 1).all()
    assert scores.notna().all().all()


def test_run_probability_fold_reliability_rows_cover_every_score(seasons_df):
    scores, reliability = run_probability_fold(seasons_df, SELECTION_FOLD, CFG, extended=True)
    for row in scores.itertuples():
        part = reliability[
            (reliability["label"] == row.label) & (reliability["threshold"] == row.threshold)
        ]
        assert not part.empty
        assert part["n"].sum() == row.n


def test_run_probability_fold_narrower_threshold_labels_more_players(seasons_df):
    scores, _ = run_probability_fold(seasons_df, SELECTION_FOLD, CFG, extended=True)
    breakout = scores[scores["label"] == "breakout"].set_index("threshold")
    assert breakout.loc[0.020, "base_rate_test"] >= breakout.loc[0.040, "base_rate_test"]
    assert breakout.loc[0.020, "mean_predicted"] >= breakout.loc[0.040, "mean_predicted"]


def test_run_probability_fold_refuses_the_holdout(seasons_df):
    with pytest.raises(FoldNotFrozenError, match="not frozen"):
        run_probability_fold(seasons_df, HOLDOUT, CFG)


def test_run_probability_fold_single_threshold(seasons_df):
    scores, _ = run_probability_fold(
        seasons_df, SELECTION_FOLD, CFG, thresholds=(THRESHOLD,), n_boot=50
    )
    assert len(scores) == len(LABELS)
    with pytest.raises(ValueError, match="non-empty"):
        run_probability_fold(seasons_df, SELECTION_FOLD, CFG, thresholds=())
