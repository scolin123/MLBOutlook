"""Tests for the feature-group ablations: group hygiene, column plumbing, guard, metrics."""

import numpy as np
import pandas as pd
import pytest

from mlb_outlook.config import Fold, load_folds_config
from mlb_outlook.evaluation.ablation import (
    ABLATION_COLUMNS,
    FULL,
    GROUPS,
    LEAGUE_LEVEL_COLUMNS,
    Group,
    ablated_features,
    ablated_ridge,
    run_ablations,
    verdict,
)
from mlb_outlook.evaluation.harness import FoldNotFrozenError, fold_frames
from mlb_outlook.models.relative import relative_ridge
from mlb_outlook.models.ridge import RELATIVE_FEATURES

CFG = load_folds_config("configs/folds.yaml")
SELECTION_FOLD = CFG.folds[1]
# An unfrozen holdout, built here rather than read out of folds.yaml: the freeze commit
# flips fold3 to `frozen: true`, and the guard still has to be tested after that.
HOLDOUT = Fold(name="holdout", train_targets=[2023, 2024, 2025], test_target=2026, frozen=False)


@pytest.fixture
def frames(seasons_df):
    return fold_frames(seasons_df, SELECTION_FOLD, CFG, extended=True)


def test_group_names_are_unique():
    names = [g.name for g in GROUPS]
    assert len(set(names)) == len(names)
    assert FULL not in names


def test_every_dropped_column_exists_in_the_base_set():
    for group in GROUPS:
        assert group.drop
        assert set(group.drop) <= set(RELATIVE_FEATURES), group.name


CORE = tuple(f"woba_rel_lag{k}" for k in (1, 2, 3))


def test_groups_partition_everything_but_the_core_woba_lags():
    """The lagged wOBA skill is the model; no group removes it.

    `league_rel` is left out of the partition: it is a swap, and it overlaps the relative
    contact columns that `contact_quality` drops.
    """
    seen: set[str] = set()
    for group in (g for g in GROUPS if g.name != "league_rel"):
        assert not seen & set(group.drop), group.name
        seen |= set(group.drop)
    assert seen == set(RELATIVE_FEATURES) - set(CORE)


def test_relative_model_carries_no_absolute_league_level():
    # The documented no-op: there is no lg_woba_lag column in ridge_rel to ablate.
    assert not set(LEAGUE_LEVEL_COLUMNS) & set(RELATIVE_FEATURES)


def test_full_group_keeps_the_base_set_unchanged():
    assert ablated_features(Group(FULL)) == tuple(RELATIVE_FEATURES)


@pytest.mark.parametrize("group", GROUPS, ids=lambda g: g.name)
def test_ablated_features_drop_adds_and_order(group):
    cols = ablated_features(group)
    assert not set(group.drop) & set(cols)
    assert set(group.add) <= set(cols)
    survivors = [c for c in RELATIVE_FEATURES if c not in group.drop]
    assert list(cols[: len(survivors)]) == survivors
    assert len(set(cols)) == len(cols)
    assert len(cols) == len(RELATIVE_FEATURES) - len(group.drop) + len(group.add)


def test_league_swap_is_the_same_size_and_has_no_relative_rates():
    cols = ablated_features(next(g for g in GROUPS if g.name == "league_rel"))
    assert len(cols) == len(RELATIVE_FEATURES)
    assert not [c for c in cols if "_rel_lag" in c]
    assert "woba_lag1" in cols and "barrel_pct_lag3" in cols


def test_unknown_dropped_column_raises():
    with pytest.raises(ValueError, match="absent from the base set"):
        ablated_features(Group("typo", ("woba_rel_lag9",)))


def test_dropping_everything_raises():
    with pytest.raises(ValueError, match="no features"):
        ablated_features(Group("all", tuple(RELATIVE_FEATURES)))


@pytest.mark.parametrize("ci", [(0.001, 0.002), (-0.002, -0.001), (-0.001, 0.002), (0.0, 0.001)])
def test_verdict_reads_the_ci(ci):
    lo, hi = ci
    expected = {True: "helps", False: "hurts"}
    if lo > 0:
        assert verdict(lo, hi) == expected[True]
    elif hi < 0:
        assert verdict(lo, hi) == expected[False]
    else:
        assert verdict(lo, hi) == "unclear"


@pytest.mark.parametrize("group", GROUPS, ids=lambda g: g.name)
def test_ablated_model_uses_only_its_own_columns(frames, group):
    """Deleting the dropped columns from the test frame must not move the forecast."""
    train, test = frames
    model = ablated_ridge(group)().fit(train, train["y_woba"])
    full = model.predict(test)
    stripped = model.predict(test.drop(columns=list(group.drop)))
    pd.testing.assert_series_equal(full, stripped)
    assert model.inner.feature_cols == ablated_features(group)


def test_full_row_matches_the_bare_selected_forecaster(seasons_df, frames):
    train, test = frames
    bare = relative_ridge().fit(train, train["y_woba"])
    expected = float(
        np.mean(np.abs(bare.predict(test).to_numpy(float) - test["y_woba"].to_numpy(float)))
    )
    table = run_ablations(seasons_df, SELECTION_FOLD, CFG, extended=True, n_boot=50)
    row = table.loc[table["group"] == FULL].iloc[0]
    assert row.mae == pytest.approx(expected)
    assert row.n_features == len(RELATIVE_FEATURES)
    assert (row.mae_vs_full, row.ci_low, row.ci_high) == (0.0, 0.0, 0.0)
    assert row.verdict == FULL


def test_run_ablations_shape_and_columns(seasons_df):
    table = run_ablations(seasons_df, SELECTION_FOLD, CFG, extended=True, n_boot=50)
    assert list(table.columns) == ABLATION_COLUMNS
    assert table["group"].tolist() == [FULL, *(g.name for g in GROUPS)]
    assert (table["fold"] == SELECTION_FOLD.name).all()
    assert (table["config"] == "extended").all()
    assert table[["mae", "coverage", "width"]].notna().all().all()
    assert (table["coverage"].between(0, 1)).all()
    assert (table["width"] > 0).all()
    assert set(table["verdict"]) <= {FULL, "helps", "hurts", "unclear"}
    assert (table["ci_low"] <= table["mae_vs_full"] + 1e-12).all()
    assert (table["mae_vs_full"] <= table["ci_high"] + 1e-12).all()


def test_mae_vs_full_is_the_difference_of_the_mae_column(seasons_df):
    table = run_ablations(seasons_df, SELECTION_FOLD, CFG, extended=True, n_boot=50)
    full_mae = float(table.loc[table["group"] == FULL, "mae"].iloc[0])
    diffs = table["mae"] - full_mae
    np.testing.assert_allclose(table["mae_vs_full"].to_numpy(), diffs.to_numpy(), atol=1e-12)


def test_run_ablations_is_deterministic(seasons_df):
    kwargs = {"extended": True, "n_boot": 50}
    first = run_ablations(seasons_df, SELECTION_FOLD, CFG, **kwargs)
    second = run_ablations(seasons_df, SELECTION_FOLD, CFG, **kwargs)
    pd.testing.assert_frame_equal(first, second)


def test_holdout_fold_is_refused(seasons_df):
    with pytest.raises(FoldNotFrozenError, match="not frozen"):
        run_ablations(seasons_df, HOLDOUT, CFG, extended=True, n_boot=50)


def test_holdout_season_never_reaches_an_ablation(seasons_df):
    """Truncating the selection fold's test season and later leaves the fit unchanged."""
    train, _ = frames_for(seasons_df)
    truncated, _ = frames_for(seasons_df[seasons_df["season"] < SELECTION_FOLD.test_target])
    pd.testing.assert_frame_equal(train, truncated)


def frames_for(df):
    return fold_frames(df, SELECTION_FOLD, CFG, extended=True)
