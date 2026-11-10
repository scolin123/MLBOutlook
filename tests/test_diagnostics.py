"""Tests for the reporting-only diagnostics: coverage slices and population accounting."""

import numpy as np
import pandas as pd
import pytest

from mlb_outlook.config import load_folds_config
from mlb_outlook.evaluation.diagnostics import (
    COVERAGE_COLUMNS,
    DEFAULT_AGE_EDGES,
    POPULATION_COLUMNS,
    bucket_labels,
    coverage_by_age,
    coverage_by_group,
    coverage_by_pa,
    population_accounting,
)
from mlb_outlook.evaluation.harness import fold_frames
from mlb_outlook.models.intervals import DEFAULT_PA_EDGES, ResidualQuantileIntervals

CFG = load_folds_config("configs/folds.yaml")
FOLD2 = next(f for f in CFG.folds if f.name == "fold2")


# ---- coverage by slice ----


def test_bucket_labels_cover_every_digitize_index():
    assert bucket_labels((300, 500), "pa_lag1") == [
        "pa_lag1 < 300",
        "300 <= pa_lag1 < 500",
        "pa_lag1 >= 500",
    ]
    assert bucket_labels(DEFAULT_AGE_EDGES, "age") == [
        "age < 26",
        "26 <= age < 31",
        "age >= 31",
    ]
    assert len(bucket_labels((400,), "pa_lag1")) == 2


def test_coverage_by_group_counts_each_group():
    bounds = pd.DataFrame({"lo": [0.2, 0.2, 0.4, 0.2], "hi": [0.4, 0.4, 0.5, 0.4]})
    y = [0.3, 0.3, 0.3, 0.3]  # the third row sits below its own lower bound
    groups = np.array([0, 0, 1, 2])
    table = coverage_by_group(bounds, y, groups, ["a", "b", "c"], grouping="g")
    assert list(table.columns) == COVERAGE_COLUMNS
    assert list(table["n"]) == [2, 1, 1]
    assert list(table["coverage"]) == [1.0, 0.0, 1.0]
    assert (table["grouping"] == "g").all()
    assert list(table["width"]) == [0.2, pytest.approx(0.1), 0.2]


def test_coverage_by_group_skips_an_empty_group():
    bounds = pd.DataFrame({"lo": [0.2], "hi": [0.4]})
    table = coverage_by_group(bounds, [0.3], np.array([2]), ["a", "b", "c"], grouping="g")
    assert list(table["bucket"]) == [2]


def test_coverage_by_group_validates_alignment():
    bounds = pd.DataFrame({"lo": [0.2, 0.2], "hi": [0.4, 0.4]})
    with pytest.raises(ValueError, match="aligned"):
        coverage_by_group(bounds, [0.3], np.array([0, 0]), ["a"], grouping="g")


def test_coverage_by_pa_uses_the_interval_buckets(seasons_df):
    _, test = fold_frames(seasons_df, FOLD2, CFG, extended=True)
    train, _ = fold_frames(seasons_df, FOLD2, CFG, extended=True)
    bounds = ResidualQuantileIntervals().fit(train, train["y_woba"]).predict_interval(test)
    table = coverage_by_pa(bounds, test)
    assert list(table["range"]) == bucket_labels(DEFAULT_PA_EDGES, "pa_lag1")
    assert table["n"].sum() == len(test)
    assert (table["grouping"] == "pa_lag1").all()


def test_coverage_by_age_splits_under_26_26_to_30_and_31_plus(seasons_df):
    _, test = fold_frames(seasons_df, FOLD2, CFG, extended=True)
    train, _ = fold_frames(seasons_df, FOLD2, CFG, extended=True)
    bounds = ResidualQuantileIntervals().fit(train, train["y_woba"]).predict_interval(test)
    table = coverage_by_age(bounds, test)
    assert list(table["range"]) == ["age < 26", "26 <= age < 31", "age >= 31"]
    assert table["n"].sum() == len(test)
    # The brackets have to agree with the frame's own age column.
    ages = test["age_target"]
    assert int(table["n"].iloc[0]) == int((ages < 26).sum())
    assert int(table["n"].iloc[-1]) == int((ages >= 31).sum())


def test_both_groupings_cover_the_same_rows(seasons_df):
    """Two slices of one set of bounds, so the totals must match."""
    _, test = fold_frames(seasons_df, FOLD2, CFG, extended=True)
    train, _ = fold_frames(seasons_df, FOLD2, CFG, extended=True)
    bounds = ResidualQuantileIntervals().fit(train, train["y_woba"]).predict_interval(test)
    by_pa = coverage_by_pa(bounds, test)
    by_age = coverage_by_age(bounds, test)
    assert by_pa["n"].sum() == by_age["n"].sum() == len(test)
    weighted = lambda t: (t["coverage"] * t["n"]).sum() / t["n"].sum()  # noqa: E731
    assert weighted(by_pa) == pytest.approx(weighted(by_age))


# ---- population accounting ----


def test_population_accounting_shape(seasons_df):
    table = population_accounting(seasons_df, FOLD2, CFG)
    assert list(table.columns) == POPULATION_COLUMNS
    assert len(table) == 3
    assert (table["fold"] == FOLD2.name).all()
    assert (table["target_season"] == FOLD2.test_target).all()


def test_the_three_buckets_account_for_every_eligible_player(seasons_df):
    table = population_accounting(seasons_df, FOLD2, CFG)
    eligible = int(table["eligible"].iloc[0])
    assert table["n"].sum() == eligible
    assert table["share"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("fold", CFG.folds, ids=lambda f: f.name)
def test_the_scored_bucket_is_the_folds_test_frame(seasons_df, fold):
    """The 100+ PA count has to be the rows the fold actually scores, or one of them is wrong."""
    table = population_accounting(seasons_df, fold, CFG)
    _, test = fold_frames(seasons_df, fold, CFG, extended=True)
    scored = int(table["n"].iloc[-1])
    assert table["bucket"].iloc[-1].startswith(f"{CFG.min_pa.target_season}+")
    assert scored == len(test)


def test_eligibility_reads_the_prior_season_pa_floor():
    """Only the season before the target counts, and only at or above the feature floor."""
    df = pd.DataFrame(
        {
            # 1: eligible and plays. 2: eligible, never plays. 3: eligible, 40 PA.
            # 4: 99 PA in the prior season, so never eligible. 5: eligible two years early.
            "player_id": [1, 1, 2, 3, 3, 4, 5, 5],
            "season": [2024, 2025, 2024, 2024, 2025, 2024, 2023, 2025],
            "age": 28,
            "pa": [500, 500, 400, 300, 40, 99, 600, 600],
            "woba": 0.320,
            "k_pct": 0.2,
            "bb_pct": 0.08,
        }
    )
    table = population_accounting(df, FOLD2, CFG)  # target 2025, so eligibility is 2024
    assert int(table["eligible"].iloc[0]) == 3  # players 1, 2, 3
    assert list(table["n"]) == [1, 1, 1]  # player 2 none, player 3 below floor, player 1 scored


def test_a_target_season_with_nobody_eligible_raises():
    df = pd.DataFrame(
        {
            "player_id": [1],
            "season": [2024],
            "age": [28],
            "pa": [10],  # below min_pa.feature_season
            "woba": [0.320],
            "k_pct": [0.2],
            "bb_pct": [0.08],
        }
    )
    with pytest.raises(ValueError, match="no player reached"):
        population_accounting(df, FOLD2, CFG)


def test_the_target_season_can_be_overridden(seasons_df):
    """Used to account for a training target rather than the fold's test target."""
    table = population_accounting(seasons_df, FOLD2, CFG, target_season=2024)
    assert (table["target_season"] == 2024).all()
    _, test = fold_frames(seasons_df, FOLD2, CFG)
    assert int(table["n"].iloc[-1]) != len(test)  # a different season, so a different count
