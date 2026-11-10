"""Tests for the reporting-only diagnostics: population accounting."""

import pandas as pd
import pytest

from mlb_outlook.config import load_folds_config
from mlb_outlook.evaluation.diagnostics import POPULATION_COLUMNS, population_accounting
from mlb_outlook.evaluation.harness import fold_frames

CFG = load_folds_config("configs/folds.yaml")
FOLD2 = next(f for f in CFG.folds if f.name == "fold2")


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
