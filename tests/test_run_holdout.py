"""Tests for `scripts/run_holdout.py`: the freeze gate, the report, and the verdict.

The holdout script has to be right the first time -- it runs once, on a fold nothing has
been fitted to -- so it is written and tested before the freeze, as `DESIGN_FREEZE.md`
section 4 requires. Everything here runs on synthetic seasons or on hand-built frames; the
real fold 3 is never touched, and the gate is tested by proving it refuses.

The script is not an importable module, so it is loaded from its path.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from mlb_outlook.config import Fold, load_folds_config
from mlb_outlook.evaluation.calibration import RELIABILITY_COLUMNS, SCORE_COLUMNS
from mlb_outlook.evaluation.diagnostics import COVERAGE_COLUMNS, POPULATION_COLUMNS
from mlb_outlook.evaluation.harness import RESULT_COLUMNS
from mlb_outlook.models.probabilities import LABELS, THRESHOLD

SCRIPT = Path("scripts/run_holdout.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("run_holdout", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


holdout = _load_script()

CFG = load_folds_config("configs/folds.yaml")
SELECTION_FOLD = next(f for f in CFG.folds if f.name == "fold2")
HOLDOUT_FOLD = next(f for f in CFG.folds if f.name == holdout.HOLDOUT_FOLD)


def frozen_holdout() -> Fold:
    """The holdout fold as the freeze commit will leave it: ``frozen: true``."""
    return HOLDOUT_FOLD.model_copy(update={"frozen": True})


def unfrozen_holdout() -> Fold:
    """The holdout fold as it stands before the freeze commit: ``frozen: false``.

    Built by hand rather than read off `CFG`, so that every test here keeps testing what it
    says it does once the freeze commit flips the flag in `folds.yaml`.
    """
    return HOLDOUT_FOLD.model_copy(update={"frozen": False})


# ---- the gate ----


def test_the_holdout_is_refused_while_the_config_says_unfrozen():
    with pytest.raises(holdout.HoldoutRefused, match="frozen: False"):
        holdout.check_gate(unfrozen_holdout())


def test_the_holdout_is_refused_when_head_is_untagged(monkeypatch):
    monkeypatch.setattr(holdout, "exact_tag", lambda: None)
    with pytest.raises(holdout.HoldoutRefused, match="untagged"):
        holdout.check_gate(frozen_holdout())


def test_the_holdout_is_refused_under_the_wrong_tag(monkeypatch):
    monkeypatch.setattr(holdout, "exact_tag", lambda: "v0.1.0")
    with pytest.raises(holdout.HoldoutRefused, match="v0.1.0"):
        holdout.check_gate(frozen_holdout())


def test_the_gate_opens_only_on_the_freeze_tag(monkeypatch):
    monkeypatch.setattr(holdout, "exact_tag", lambda: holdout.FREEZE_TAG)
    monkeypatch.setattr(holdout, "dirty_paths", list)
    holdout.check_gate(frozen_holdout())  # all three conditions met: no refusal


def test_the_holdout_is_refused_on_a_dirty_tree(monkeypatch):
    """An edit on top of the tagged commit is not what the tag describes."""
    monkeypatch.setattr(holdout, "exact_tag", lambda: holdout.FREEZE_TAG)
    monkeypatch.setattr(
        holdout, "dirty_paths", lambda: [" M src/mlb_outlook/models/ridge.py"]
    )
    with pytest.raises(holdout.HoldoutRefused, match="uncommitted or untracked"):
        holdout.check_gate(frozen_holdout())


def test_the_dirty_tree_refusal_names_the_paths(monkeypatch):
    monkeypatch.setattr(holdout, "exact_tag", lambda: holdout.FREEZE_TAG)
    monkeypatch.setattr(holdout, "dirty_paths", lambda: [f"?? scratch{i}.py" for i in range(25)])
    with pytest.raises(holdout.HoldoutRefused) as refused:
        holdout.check_gate(frozen_holdout())
    message = str(refused.value)
    assert "scratch0.py" in message and "and 5 more" in message


def test_the_flag_and_the_tag_are_checked_before_the_tree(monkeypatch):
    """A dirty tree is the last complaint, so the first message names the real blocker."""
    monkeypatch.setattr(holdout, "exact_tag", lambda: None)
    monkeypatch.setattr(holdout, "dirty_paths", lambda: [" M x.py"])
    with pytest.raises(holdout.HoldoutRefused, match="untagged"):
        holdout.check_gate(frozen_holdout())


def test_the_tag_alone_does_not_open_the_gate(monkeypatch):
    """A tag on a commit whose config still says frozen: false is not a freeze."""
    monkeypatch.setattr(holdout, "exact_tag", lambda: holdout.FREEZE_TAG)
    with pytest.raises(holdout.HoldoutRefused, match="frozen: False"):
        holdout.check_gate(unfrozen_holdout())


def test_the_rehearsal_path_refuses_every_holdout():
    with pytest.raises(holdout.HoldoutRefused, match="frozen flag"):
        holdout.check_rehearsal(HOLDOUT_FOLD)
    with pytest.raises(holdout.HoldoutRefused, match="frozen flag"):
        holdout.check_rehearsal(frozen_holdout())
    # A fold with no frozen key at all is a selection fold, which is what it is for.
    assert SELECTION_FOLD.frozen is None
    holdout.check_rehearsal(SELECTION_FOLD)


def test_resolve_fold_rejects_an_unknown_name():
    with pytest.raises(holdout.HoldoutRefused, match="not in"):
        holdout.resolve_fold(CFG, "fold9")


def test_main_refuses_the_holdout_without_the_freeze_tag(monkeypatch, capsys):
    """The default invocation refuses, and the suite can never reach the single look.

    `exact_tag` is patched to None rather than left alone on purpose. Once the tag exists
    this is the only thing standing between `pytest` -- which section 5 runs immediately
    before the holdout -- and a `main([])` that would spend the one look from inside a test.
    """
    monkeypatch.setattr(holdout, "exact_tag", lambda: None)
    assert holdout.main([]) == 1
    out = capsys.readouterr().out
    assert "refused" in out


def test_main_refuses_a_rehearsal_on_the_holdout(capsys):
    assert holdout.main(["--rehearse", holdout.HOLDOUT_FOLD]) == 1
    assert "refused" in capsys.readouterr().out


def test_main_refuses_an_unknown_rehearsal_fold(capsys):
    assert holdout.main(["--rehearse", "fold9"]) == 1
    assert "refused" in capsys.readouterr().out





# ---- the verdict ----


def _point_table(mae: float, mae_marcel: float, coverage: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": [holdout.MODEL, holdout.BASELINE, holdout.INTERVALS],
            "mae": [mae, mae_marcel, mae],
            "coverage": [None, None, coverage],
        }
    )


def test_verdict_passes_when_all_three_criteria_are_met():
    table = _point_table(0.0263, 0.0285, 0.815)
    verdict = holdout.verdict_table(table, (-0.0023, -0.0036, -0.0008), CFG)
    assert list(verdict["criterion"]) == [1, 2, 3]
    assert verdict["passed"].all()


def test_verdict_fails_criterion_2_when_the_ci_crosses_zero():
    """Beats Marcel but the CI spans zero: the named failure mode from section 6."""
    table = _point_table(0.0280, 0.0285, 0.800)
    verdict = holdout.verdict_table(table, (-0.0005, -0.0021, +0.0009), CFG)
    assert list(verdict["passed"]) == [True, False, True]


def test_verdict_fails_criterion_1_when_marcel_wins():
    table = _point_table(0.0290, 0.0285, 0.800)
    verdict = holdout.verdict_table(table, (+0.0005, -0.0009, +0.0021), CFG)
    assert list(verdict["passed"]) == [False, False, True]


@pytest.mark.parametrize("coverage", [0.74, 0.86])
def test_verdict_fails_criterion_3_outside_the_registered_range(coverage):
    table = _point_table(0.0263, 0.0285, coverage)
    verdict = holdout.verdict_table(table, (-0.0023, -0.0036, -0.0008), CFG)
    assert list(verdict["passed"]) == [True, True, False]


@pytest.mark.parametrize("coverage", [0.75, 0.85])
def test_criterion_3_is_inclusive_at_both_edges(coverage):
    table = _point_table(0.0263, 0.0285, coverage)
    verdict = holdout.verdict_table(table, (-0.0023, -0.0036, -0.0008), CFG)
    assert verdict["passed"].all()


def test_a_failed_verdict_prints_the_fail_and_the_rules(capsys):
    table = _point_table(0.0290, 0.0285, 0.700)
    holdout.print_verdict(holdout.verdict_table(table, (+0.0005, -0.0009, +0.0021), CFG))
    out = capsys.readouterr().out
    assert "VERDICT: FAIL" in out
    assert "not retried" in out


# ---- the whole report, on a selection fold ----


def test_run_holdout_reports_and_writes_every_frame(seasons_df, tmp_path, capsys):
    frames = holdout.run_holdout(seasons_df, SELECTION_FOLD, CFG, out_dir=tmp_path)
    assert set(frames) == {
        "point",
        "interval_coverage",
        "population",
        "players",
        "probabilities",
        "reliability",
        "verdict",
    }
    for name in frames:
        written = tmp_path / f"{SELECTION_FOLD.name}_{name}.csv"
        assert written.exists(), name
        assert len(pd.read_csv(written)) == len(frames[name])

    point = frames["point"]
    assert list(point.columns) == RESULT_COLUMNS
    # The three baselines, the forecaster, and the selected interval model: nothing else.
    assert list(point["model"]) == [*holdout.BASELINES, holdout.MODEL, holdout.INTERVALS]
    assert (point["config"] == "extended").all()
    assert (point["fold"] == SELECTION_FOLD.name).all()
    assert point.set_index("model").loc[holdout.INTERVALS, "coverage"] > 0

    out = capsys.readouterr().out
    assert "PASS / FAIL" in out
    assert "precision@10" in out
    assert "conditional on" in out  # the population caveat is printed, not just written


def test_run_holdout_slices_the_coverage_two_ways(seasons_df, tmp_path):
    """Both groupings, on the same bounds, each accounting for every test row."""
    frames = holdout.run_holdout(seasons_df, SELECTION_FOLD, CFG, out_dir=tmp_path)
    coverage = frames["interval_coverage"]
    assert list(coverage.columns) == COVERAGE_COLUMNS
    assert set(coverage["grouping"]) == {"pa_lag1", "age_target"}
    n_test = int(frames["point"].set_index("model").loc[holdout.MODEL, "n_test"])
    for _, part in coverage.groupby("grouping"):
        assert part["n"].sum() == n_test


def test_run_holdout_writes_a_per_player_file(seasons_df, tmp_path):
    """The per-player rows have to reconcile with the headline MAE comparison."""
    frames = holdout.run_holdout(seasons_df, SELECTION_FOLD, CFG, out_dir=tmp_path)
    players = frames["players"]
    point = frames["point"].set_index("model")
    assert f"{holdout.MODEL}_minus_{holdout.BASELINE}" in players.columns
    assert len(players) == int(point.loc[holdout.MODEL, "n_test"])
    expected = float(point.loc[holdout.BASELINE, "mae"]) - float(point.loc[holdout.MODEL, "mae"])
    assert players["advantage"].mean() == pytest.approx(expected)
    assert players["covered"].mean() == pytest.approx(
        float(point.loc[holdout.INTERVALS, "coverage"])
    )


def test_run_holdout_accounts_for_the_population(seasons_df, tmp_path):
    frames = holdout.run_holdout(seasons_df, SELECTION_FOLD, CFG, out_dir=tmp_path)
    population = frames["population"]
    assert list(population.columns) == POPULATION_COLUMNS
    scored = int(population["n"].iloc[-1])
    assert scored == int(frames["point"].set_index("model").loc[holdout.MODEL, "n_test"])


def test_run_holdout_scores_the_registered_threshold_only(seasons_df, tmp_path):
    """No threshold sweep on the holdout: 0.030 is registered and the others are not run."""
    frames = holdout.run_holdout(seasons_df, SELECTION_FOLD, CFG, out_dir=tmp_path)
    scores = frames["probabilities"]
    assert list(scores.columns) == SCORE_COLUMNS
    assert list(frames["reliability"].columns) == RELIABILITY_COLUMNS
    assert set(scores["threshold"]) == {THRESHOLD}
    assert set(scores["label"]) == set(LABELS)
    assert len(scores) == len(LABELS)


def test_run_holdout_point_mae_matches_the_interval_models_point_fit(seasons_df, tmp_path):
    """`resid_q_pa` wraps `ridge_rel`, so the two MAE rows have to be the same number."""
    point = holdout.run_holdout(
        seasons_df, SELECTION_FOLD, CFG, out_dir=tmp_path
    )["point"].set_index("model")
    assert point.loc[holdout.INTERVALS, "mae"] == pytest.approx(
        point.loc[holdout.MODEL, "mae"]
    )
