"""Leakage guard.

The contract: every feature row for target season T must be derived only from
data dated before the forecast date for T. The builder exposes, per feature
row, `max_source_season` (the latest season of any input) and the tests assert
max_source_season < target_season for every row.
"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlb_outlook.features.asof import ASOF_COLUMNS, build_asof_rows, build_features
from synthetic import make_seasons

TARGETS = range(2020, 2027)


def assert_no_leakage(rows: pd.DataFrame) -> None:
    bad = rows[rows["max_source_season"] >= rows["target_season"]]
    if len(bad):
        raise AssertionError(f"{len(bad)} rows use data from the target season or later")


@pytest.mark.parametrize("target", TARGETS)
def test_features_use_only_prior_seasons(seasons_df, target):
    rows = build_asof_rows(seasons_df, target)
    assert len(rows) > 0
    assert (rows.max_source_season < rows.target_season).all()
    assert_no_leakage(rows)


@pytest.mark.parametrize("target", TARGETS)
def test_poison_future_rows_do_not_change_output(seasons_df, target):
    clean = build_asof_rows(seasons_df, target)
    ids = seasons_df["player_id"].unique()[:20]
    poison = pd.DataFrame(
        {
            "player_id": list(ids) + [10**6],
            "season": [target + i % 3 for i in range(len(ids) + 1)],
            "age": 30,
            "pa": 10**6,
            "woba": 9.99,
            "k_pct": 9.99,
            "bb_pct": 9.99,
        }
    )
    poisoned = pd.concat([seasons_df, poison], ignore_index=True)
    assert_frame_equal(build_asof_rows(poisoned, target), clean)


def test_negative_control_leaky_frame_raises():
    leaky = pd.DataFrame(
        {
            "player_id": [1, 2],
            "target_season": [2022, 2022],
            "max_source_season": [2021, 2022],
            "n_source_seasons": [1, 2],
        }
    )
    with pytest.raises(AssertionError):
        assert_no_leakage(leaky)


def test_player_with_only_future_rows_is_excluded(seasons_df):
    extra = pd.DataFrame(
        {
            "player_id": [999999],
            "season": [2024],
            "age": [25],
            "pa": [500],
            "woba": [0.330],
            "k_pct": [0.2],
            "bb_pct": [0.08],
        }
    )
    df = pd.concat([seasons_df, extra], ignore_index=True)
    assert 999999 not in build_asof_rows(df, 2024)["player_id"].values
    assert 999999 in build_asof_rows(df, 2025)["player_id"].values


def test_target_before_all_data_is_empty_with_columns(seasons_df):
    rows = build_asof_rows(seasons_df, 2018)
    assert len(rows) == 0
    assert list(rows.columns) == ASOF_COLUMNS


def test_counts_and_max_source_season():
    df = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2],
            "season": [2020, 2021, 2023, 2022],
            "age": [25, 26, 28, 30],
            "pa": [500] * 4,
            "woba": [0.3] * 4,
            "k_pct": [0.2] * 4,
            "bb_pct": [0.08] * 4,
        }
    )
    rows = build_asof_rows(df, 2023)
    assert rows["player_id"].tolist() == [1, 2]
    assert rows["max_source_season"].tolist() == [2021, 2022]
    assert rows["n_source_seasons"].tolist() == [2, 1]


@pytest.mark.parametrize("include_target", [True, False])
@pytest.mark.parametrize("target", TARGETS)
def test_build_features_use_only_prior_seasons(seasons_df, target, include_target):
    rows = build_features(seasons_df, target, include_target=include_target)
    assert len(rows) > 0
    assert (rows.max_source_season < rows.target_season).all()
    assert_no_leakage(rows)


@pytest.mark.parametrize("target", TARGETS)
def test_build_features_poison_future_rows_do_not_change_output(seasons_df, target):
    ids = seasons_df["player_id"].unique()[:20]
    poison = pd.DataFrame(
        {
            "player_id": list(ids) + [10**6],
            "season": [target + 1 + i % 3 for i in range(len(ids) + 1)],
            "age": 30,
            "pa": 10**6,
            "woba": 9.99,
            "k_pct": 9.99,
            "bb_pct": 9.99,
        }
    )
    poisoned = pd.concat([seasons_df, poison], ignore_index=True)
    # Poison starts after the target season, so even the target columns must be unchanged.
    for include_target in (True, False):
        assert_frame_equal(
            build_features(poisoned, target, include_target=include_target),
            build_features(seasons_df, target, include_target=include_target),
        )


@pytest.mark.parametrize("target", TARGETS)
def test_build_features_target_season_rows_do_not_change_features(seasons_df, target):
    ids = seasons_df["player_id"].unique()[:20]
    poison = pd.DataFrame(
        {
            "player_id": ids,
            "season": target,
            "age": 30,
            "pa": 10**6,
            "woba": 9.99,
            "k_pct": 9.99,
            "bb_pct": 9.99,
        }
    )
    poisoned = pd.concat([seasons_df, poison], ignore_index=True)
    assert_frame_equal(
        build_features(poisoned, target, include_target=False),
        build_features(seasons_df, target, include_target=False),
    )


def test_make_seasons_is_deterministic():
    assert_frame_equal(make_seasons(seed=7), make_seasons(seed=7))
    assert not make_seasons(seed=7).equals(make_seasons(seed=8))
