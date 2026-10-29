"""Tests for the evaluation harness: membership, leakage, fresh models, guard, bootstrap."""

import numpy as np
import pandas as pd
import pytest

from mlb_outlook.config import Fold, load_folds_config
from mlb_outlook.evaluation.harness import (
    FoldNotFrozenError,
    fold_frames,
    paired_bootstrap_mae_diff,
    run_all,
    run_fold,
)
from mlb_outlook.models.baselines import LastSeason

CFG = load_folds_config("configs/folds.yaml")
COLS = ["player_id", "season", "age", "pa", "woba", "k_pct", "bb_pct"]


@pytest.mark.parametrize("fold", CFG.folds, ids=lambda f: f.name)
def test_fold_frames_target_membership(seasons_df, fold):
    train, test = fold_frames(seasons_df, fold, CFG)
    assert len(train) > 0 and len(test) > 0
    assert set(train["target_season"]) <= set(fold.train_targets)
    assert set(test["target_season"]) == {fold.test_target}


@pytest.mark.parametrize("fold", CFG.folds, ids=lambda f: f.name)
def test_test_season_never_influences_training(seasons_df, fold):
    full_train, _ = fold_frames(seasons_df, fold, CFG)
    truncated = seasons_df[seasons_df["season"] < fold.test_target]
    trunc_train, _ = fold_frames(truncated, fold, CFG)
    pd.testing.assert_frame_equal(full_train, trunc_train)


class Spy:
    instances = 0

    def __init__(self):
        Spy.instances += 1
        self.seen = None

    def fit(self, X, y=None):
        self.seen = set(X["target_season"])
        Spy.last_seen = self.seen
        assert y is not None and len(y) == len(X)
        return self

    def predict(self, X):
        return X["woba_lag1"]


@pytest.mark.parametrize("fold", CFG.folds, ids=lambda f: f.name)
def test_spy_sees_only_train_targets_and_fresh_model(seasons_df, fold):
    Spy.instances = 0
    Spy.last_seen = None
    factories = {"spy_a": Spy, "spy_b": Spy}
    run_fold(seasons_df, fold, CFG, factories, allow_unfrozen=True)
    assert Spy.instances == 2  # one factory call per model per fold
    assert Spy.last_seen == set(fold.train_targets)
    assert fold.test_target not in Spy.last_seen


def test_run_all_calls_factory_once_per_fold(seasons_df):
    Spy.instances = 0
    out = run_all(seasons_df, CFG, {"spy": Spy}, allow_unfrozen=True)
    assert Spy.instances == len(CFG.folds)
    assert out["fold"].tolist() == [f.name for f in CFG.folds]


def test_last_season_hand_computed_mae_and_bias():
    rows = [
        (1, 2022, 25, 300, 0.300, 0.2, 0.08),
        (1, 2023, 26, 300, 0.340, 0.2, 0.08),
        (1, 2024, 27, 300, 0.320, 0.2, 0.08),
        (2, 2022, 25, 300, 0.280, 0.2, 0.08),
        (2, 2023, 26, 300, 0.300, 0.2, 0.08),
        (2, 2024, 27, 300, 0.340, 0.2, 0.08),
        (3, 2023, 26, 300, 0.360, 0.2, 0.08),
        (3, 2024, 27, 300, 0.330, 0.2, 0.08),
    ]
    df = pd.DataFrame(rows, columns=COLS)
    fold = Fold(name="mini", train_targets=[2023], test_target=2024)
    out = run_fold(df, fold, CFG, {"last": LastSeason})
    # Test preds: .340, .300, .360 vs truth .320, .340, .330 -> errors +.02, -.04, +.03
    # mae = (.02 + .04 + .03) / 3 = .03; bias = (.02 - .04 + .03) / 3 = .01 / 3
    # Train: only players 1 and 2 have 2022 rows, so n_train = 2.
    row = out.iloc[0]
    assert row["model"] == "last" and row["fold"] == "mini"
    assert row["n_train"] == 2 and row["n_test"] == 3
    assert row["mae"] == pytest.approx(0.03, abs=1e-9)
    assert row["bias"] == pytest.approx(0.01 / 3, abs=1e-9)


def test_unfrozen_fold_guard(seasons_df):
    fold = Fold(name="holdout", train_targets=[2023, 2024, 2025], test_target=2026, frozen=False)
    with pytest.raises(FoldNotFrozenError):
        run_fold(seasons_df, fold, CFG, {"last": LastSeason})
    out = run_fold(seasons_df, fold, CFG, {"last": LastSeason}, allow_unfrozen=True)
    assert len(out) == 1


def test_frozen_none_or_true_is_not_guarded(seasons_df):
    for frozen in (None, True):
        fold = Fold(name="f", train_targets=[2023], test_target=2024, frozen=frozen)
        assert len(run_fold(seasons_df, fold, CFG, {"last": LastSeason})) == 1


def test_run_all_blocks_unfrozen_holdout(seasons_df):
    with pytest.raises(FoldNotFrozenError):
        run_all(seasons_df, CFG, {"last": LastSeason})


def test_bootstrap_clear_winner():
    rng = np.random.default_rng(1)
    b = rng.uniform(0.02, 0.06, size=200)
    a = b - rng.uniform(0.005, 0.01, size=200)
    diff, lo, hi = paired_bootstrap_mae_diff(a, b)
    assert diff < 0
    assert lo < diff < hi
    assert hi < 0


def test_bootstrap_identical_arrays():
    a = np.random.default_rng(2).uniform(0.01, 0.05, size=100)
    diff, lo, hi = paired_bootstrap_mae_diff(a, a.copy())
    assert diff == 0.0
    assert lo <= 0.0 <= hi


def test_bootstrap_seed_reproducible_and_validates():
    rng = np.random.default_rng(3)
    a, b = rng.uniform(size=50), rng.uniform(size=50)
    assert paired_bootstrap_mae_diff(a, b, seed=5) == paired_bootstrap_mae_diff(a, b, seed=5)
    with pytest.raises(ValueError):
        paired_bootstrap_mae_diff(a, b[:10])
