"""Tests for the evaluation harness: membership, leakage, fresh models, guard, bootstrap."""

import numpy as np
import pandas as pd
import pytest

from mlb_outlook.config import Fold, load_folds_config
from mlb_outlook.evaluation.harness import (
    RESULT_COLUMNS,
    WEIGHT_COLUMN,
    FoldNotFrozenError,
    error_metrics,
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
    # The holdout is forced unfrozen here rather than read as `folds.yaml` leaves it: the
    # freeze commit flips fold3 to `frozen: true`, and this guard still has to be tested
    # after that. `run_all` runs every fold it is given, so the guard is the only thing
    # standing between it and a holdout.
    folds = [f.model_copy(update={"frozen": False}) if f.frozen is not None else f
             for f in CFG.folds]
    with pytest.raises(FoldNotFrozenError):
        run_all(seasons_df, CFG.model_copy(update={"folds": folds}), {"last": LastSeason})


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


# ---- error metrics, unweighted and PA-weighted ----


def test_error_metrics_by_hand():
    err = np.array([0.02, -0.04, 0.06])
    out = error_metrics(err)
    assert out["mae"] == pytest.approx(0.04)
    assert out["bias"] == pytest.approx(0.04 / 3)
    assert out["rmse"] == pytest.approx(np.sqrt((0.0004 + 0.0016 + 0.0036) / 3))
    assert "mae_pa" not in out


def test_pa_weighting_follows_the_plate_appearances():
    err = np.array([0.02, -0.04])
    out = error_metrics(err, [100.0, 300.0])
    assert out["mae"] == pytest.approx(0.03)  # one row one vote
    assert out["mae_pa"] == pytest.approx((100 * 0.02 + 300 * 0.04) / 400)
    assert out["bias_pa"] == pytest.approx((100 * 0.02 - 300 * 0.04) / 400)
    assert out["rmse_pa"] == pytest.approx(
        np.sqrt((100 * 0.0004 + 300 * 0.0016) / 400)
    )


def test_equal_weights_reproduce_the_unweighted_metrics():
    err = np.array([0.01, -0.03, 0.05, -0.02])
    out = error_metrics(err, np.full(4, 500.0))
    for metric in ("mae", "rmse", "bias"):
        assert out[f"{metric}_pa"] == pytest.approx(out[metric])


def test_rmse_is_at_least_mae_and_bias_keeps_its_sign():
    err = np.array([-0.05, -0.01, 0.02])
    out = error_metrics(err, [600.0, 200.0, 150.0])
    assert out["rmse"] >= out["mae"]
    assert out["rmse_pa"] >= out["mae_pa"]
    assert out["bias_pa"] < out["bias"] < 0  # the big miss is the big-PA row


def test_error_metrics_validates_its_inputs():
    with pytest.raises(ValueError, match="non-empty 1-D"):
        error_metrics([])
    with pytest.raises(ValueError, match="align"):
        error_metrics([0.01, 0.02], [100.0])
    with pytest.raises(ValueError, match="non-negative"):
        error_metrics([0.01, 0.02], [100.0, -1.0])
    with pytest.raises(ValueError, match="non-negative"):
        error_metrics([0.01, 0.02], [100.0, np.nan])
    with pytest.raises(ValueError, match="sum to zero"):
        error_metrics([0.01, 0.02], [0.0, 0.0])


def test_run_fold_reports_both_readings_for_every_model(seasons_df):
    fold = next(f for f in CFG.folds if f.name == "fold2")
    out = run_fold(seasons_df, fold, CFG, {"last": LastSeason}, extended=True)
    row = out.iloc[0]
    assert set(RESULT_COLUMNS) >= {"mae", "rmse", "bias", "mae_pa", "rmse_pa", "bias_pa"}
    for column in ("mae", "rmse", "mae_pa", "rmse_pa"):
        assert row[column] > 0
    # The weight column is a target-season outcome, so it stays on the scoring side.
    assert WEIGHT_COLUMN.startswith("y_")


def test_pa_weighted_metrics_match_a_hand_weighting(seasons_df):
    """The harness must be weighting by the test frame's y_pa and nothing else."""
    fold = next(f for f in CFG.folds if f.name == "fold2")
    train, test = fold_frames(seasons_df, fold, CFG, extended=True)
    model = LastSeason().fit(train, train["y_woba"])
    err = model.predict(test).to_numpy(dtype=float) - test["y_woba"].to_numpy(dtype=float)
    weights = test[WEIGHT_COLUMN].to_numpy(dtype=float)
    out = run_fold(seasons_df, fold, CFG, {"last": LastSeason}, extended=True).iloc[0]
    assert out["mae_pa"] == pytest.approx(np.average(np.abs(err), weights=weights))
    assert out["bias_pa"] == pytest.approx(np.average(err, weights=weights))


# ---- extended training window (configs/folds.yaml: extended.extra_train_targets) ----

# The synthetic fixture starts in 2018, so extended targets before 2019 have no prior
# season to build features from and contribute no rows.
FIRST_USABLE_TARGET = 2019


@pytest.mark.parametrize("fold", CFG.folds, ids=lambda f: f.name)
def test_extended_train_frame_adds_the_extra_targets(seasons_df, fold):
    base, _ = fold_frames(seasons_df, fold, CFG)
    ext, _ = fold_frames(seasons_df, fold, CFG, extended=True)
    expected = {t for t in CFG.extended_train_targets(fold) if t >= FIRST_USABLE_TARGET}
    assert set(ext["target_season"]) == expected
    assert set(base["target_season"]) < set(ext["target_season"])
    assert len(ext) > len(base)
    assert 2020 not in set(ext["target_season"])  # short season is flagged, never a target


@pytest.mark.parametrize("fold", CFG.folds, ids=lambda f: f.name)
def test_extended_leaves_the_test_frame_alone(seasons_df, fold):
    _, base_test = fold_frames(seasons_df, fold, CFG)
    _, ext_test = fold_frames(seasons_df, fold, CFG, extended=True)
    pd.testing.assert_frame_equal(base_test, ext_test)


@pytest.mark.parametrize("fold", CFG.folds, ids=lambda f: f.name)
def test_extended_training_stays_before_the_test_target(seasons_df, fold):
    ext, _ = fold_frames(seasons_df, fold, CFG, extended=True)
    assert (ext["target_season"] < fold.test_target).all()
    assert (ext["max_source_season"] < ext["target_season"]).all()
    truncated = seasons_df[seasons_df["season"] < fold.test_target]
    trunc_ext, _ = fold_frames(truncated, fold, CFG, extended=True)
    pd.testing.assert_frame_equal(ext, trunc_ext)


@pytest.mark.parametrize("fold", CFG.folds, ids=lambda f: f.name)
def test_extended_scores_only_differ_for_models_that_train(seasons_df, fold):
    kwargs = {"allow_unfrozen": True}
    base = run_fold(seasons_df, fold, CFG, {"last": LastSeason}, **kwargs).iloc[0]
    ext = run_fold(seasons_df, fold, CFG, {"last": LastSeason}, extended=True, **kwargs).iloc[0]
    assert base["config"] == "base" and ext["config"] == "extended"
    assert ext["n_train"] > base["n_train"]
    assert ext["n_test"] == base["n_test"]
    # LastSeason ignores the training frame, so a wider window cannot move its score.
    assert ext["mae"] == base["mae"] and ext["bias"] == base["bias"]
    assert ext["mae_pa"] == base["mae_pa"] and ext["rmse_pa"] == base["rmse_pa"]


def test_result_columns(seasons_df):
    out = run_all(seasons_df, CFG, {"last": LastSeason}, allow_unfrozen=True, extended=True)
    assert list(out.columns) == RESULT_COLUMNS
    assert (out["config"] == "extended").all()

