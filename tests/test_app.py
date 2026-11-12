"""The Streamlit app: pages render headless, the labelling is right, the guards hold.

The app shows the prod2027 forecasts. Three things these tests are for:

- every page says which season it is about, and the forecast pages say there is no outcome;
- nothing player-level carries a 2027 outcome column, because there is no such thing;
- the fold 3 holdout is still reachable only as aggregate tables. 2026 season lines do now
  appear -- they are what the 2027 forecast is built from -- but the per-player holdout
  scoring rows in `fold3_players.csv` are not reachable from any page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from components import data  # noqa: E402
from components.style import percentile_color, signed, woba  # noqa: E402

HOME = APP / "Home.py"
PLAYER = APP / "pages" / "1_Player.py"
BOARD = APP / "pages" / "2_Projection_disagreements.py"
PERFORMANCE = APP / "pages" / "3_Model_performance.py"
PAGES = [HOME, PLAYER, BOARD, PERFORMANCE]
# Which banner each page must carry: the forecast pages make a claim about an unplayed
# season, the performance page one about two scored folds. Mixing them up is the mistake
# worth a test.
FORECAST_PAGES = [HOME, PLAYER, BOARD]
SCORED_PAGES = [PERFORMANCE]


def _have_artifacts() -> bool:
    return all(p.exists() for p in data.required_files())


needs_artifacts = pytest.mark.skipif(
    not _have_artifacts(), reason="fold 2 artifacts not generated; see components/data.py"
)


# ---- wording and scope, no artifacts needed ----


def test_no_page_calls_a_player_undervalued():
    for path in APP.rglob("*.py"):
        assert "undervalued" not in path.read_text(encoding="utf-8").lower(), path


def test_every_page_declares_which_season_it_is_about():
    """`setup` takes the banner with no default, so a page cannot inherit the wrong claim."""
    import inspect

    from components.style import setup

    assert "banner" in inspect.signature(setup).parameters
    assert inspect.signature(setup).parameters["banner"].default is inspect.Parameter.empty
    for page in FORECAST_PAGES:
        assert "setup(" in page.read_text(encoding="utf-8"), page
        assert "data.FORECAST_LABEL" in page.read_text(encoding="utf-8"), page
    for page in SCORED_PAGES:
        assert "data.BACKTEST_LABEL" in page.read_text(encoding="utf-8"), page


def test_the_forecast_label_says_the_season_has_not_been_played():
    assert "2027 forecasts" in data.FORECAST_LABEL
    assert "data through 2026" in data.FORECAST_LABEL
    assert "has not been played" in data.FORECAST_LABEL
    assert "Fold 2 backtest" in data.BACKTEST_LABEL
    assert "Not a live projection" in data.BACKTEST_LABEL


def test_the_forecast_is_prod2027_from_data_through_2026():
    assert data.FORECAST_RUN == "prod2027"
    assert data.FORECAST_SEASON == 2027
    assert data.FORECAST_FEATURES_THROUGH == 2026
    # The last season that exists, and the last the forecast reads. They are the same season.
    assert data.LAST_VISIBLE_SEASON == 2026


def test_the_scored_folds_are_still_fold2_and_fold3():
    assert data.FOLD == "fold2"
    assert data.TARGET_SEASON == 2025
    assert data.FEATURES_THROUGH == 2024


def test_the_holdout_is_reachable_only_as_aggregate_tables():
    """The one way 2026 reaches a page is `holdout_artifact`, and it refuses per-player rows."""
    assert data.HOLDOUT_FOLD == "fold3"
    assert data.HOLDOUT_TARGET_SEASON == 2026
    assert "players" not in data.HOLDOUT_ARTIFACTS
    assert "forecasts" not in data.HOLDOUT_ARTIFACTS
    for name in ("players", "forecasts", "../fold2_players"):
        with pytest.raises(ValueError, match="aggregate holdout table"):
            data.holdout_artifact(name)


def test_only_the_data_module_names_the_holdout_fold():
    """`components/data.py` is the single gateway. A page that built its own `fold3_*` path
    would bypass the allowlist and could read the per-player 2026 rows."""
    gateway = APP / "components" / "data.py"
    assert "fold3" in gateway.read_text(encoding="utf-8")
    for path in APP.rglob("*.py"):
        if path == gateway:
            continue
        assert "fold3" not in path.read_text(encoding="utf-8"), path


def test_the_verdict_label_is_read_off_the_artifact():
    frame = pd.DataFrame({"criterion": [1, 2, 3], "passed": [True, True, True]})
    assert data.verdict_label(frame) == "PASS"
    assert data.failed_criteria(frame).empty
    frame.loc[1, "passed"] = False
    assert data.verdict_label(frame) == "FAIL"
    assert list(data.failed_criteria(frame)["criterion"]) == [2]


def test_formatters():
    assert woba(0.345) == ".345"
    assert woba(float("nan")) == "-"
    assert signed(0.012) == "+.012"
    assert signed(-0.012) == "-.012"


def test_percentile_colors_run_blue_to_red():
    assert percentile_color(0) == "#2F6DB5"
    assert percentile_color(100) == "#D22D49"
    assert percentile_color(50) == "#BEC4CE"
    assert percentile_color(-5) == percentile_color(0)


# ---- real artifacts ----


@needs_artifacts
def test_loaded_seasons_stop_at_the_last_feature_season():
    assert data.seasons()["season"].max() == data.FORECAST_FEATURES_THROUGH


@needs_artifacts
def test_forecasts_are_the_prod2027_rows():
    f = data.forecasts()
    assert len(f) == 465
    assert set(f["target_season"]) == {2027}
    assert ((f["p_breakout"] + f["p_stable"] + f["p_regression"]) - 1).abs().max() < 1e-9
    assert (f["lo"] <= f["hi"]).all()
    assert (f["model_minus_marcel"] == f["ridge_rel_minus_marcel"]).all()


@needs_artifacts
def test_nothing_player_level_claims_a_2027_outcome():
    """The season has not happened. A column saying otherwise would be fabricated."""
    f = data.forecasts()
    assert not [c for c in f.columns if c.startswith("y_")]
    assert not {"covered", "advantage", "actual_minus_forecast"} & set(f.columns)


@needs_artifacts
def test_the_forecast_manifest_names_the_frozen_model():
    info = data.forecast_manifest()
    assert info["run"] == "prod2027"
    assert info["target_season"] == 2027
    assert info["features_through"] == 2026
    assert info["freeze_tag"] == data.HOLDOUT_TAG


@needs_artifacts
def test_percentiles_are_bounded():
    df = data.seasons()
    pid = int(data.forecasts()["player_id"].iloc[0])
    pcts = data.percentiles(df, pid)
    assert len(pcts) > 0
    assert pcts["percentile"].between(0, 100).all()


@needs_artifacts
@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders_without_error(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    expected = data.BACKTEST_LABEL if page in SCORED_PAGES else data.FORECAST_LABEL
    assert any(expected in m.value for m in at.markdown), page


@needs_artifacts
@pytest.mark.skipif(not data.holdout_available(), reason="fold 3 holdout not run")
def test_model_performance_shows_the_holdout_verdict_and_both_folds():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PERFORMANCE), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]

    verdict = data.holdout_artifact("verdict")
    label = data.verdict_label(verdict)
    assert label == "FAIL", "the committed fold 3 artifact says FAIL; the page must say so"
    assert any(f"Fold 3 holdout verdict: {label}" in h.value for h in at.subheader)
    # A fail is announced as an error, names the criterion that missed, and is not dressed up.
    assert len(at.error) == 1
    shown = at.error[0].value
    assert f"Verdict {label}" in shown
    assert "not shown to beat" in shown
    assert "partial success" in shown
    assert not at.success

    # Both folds' criteria, side by side, and the fold 3 column marks the failure.
    criteria = at.dataframe[0].value
    assert list(criteria["fold 3 result"]) == ["pass", "FAIL", "pass"]
    assert "fold 2 (selection)" in criteria.columns

    # Aggregate only: nothing on this page is per player, so no 2026 hitter can reach it.
    for shown_frame in at.dataframe:
        columns = set(shown_frame.value.columns)
        assert not columns & {"player_id", "name", "y_woba"}, columns


@needs_artifacts
def test_board_sorts_change_the_order():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(BOARD), default_timeout=120).run()
    first = at.dataframe[0].value.iloc[0]["name"]
    at.selectbox[0].set_value("Model below Marcel").run()
    assert not at.exception
    assert at.dataframe[0].value.iloc[0]["name"] != first


@needs_artifacts
def test_player_page_follows_the_selection():
    from streamlit.testing.v1 import AppTest

    f = data.forecasts().sort_values("name")
    target = int(f["player_id"].iloc[5])
    at = AppTest.from_file(str(PLAYER), default_timeout=120).run()
    at.selectbox[0].set_value(target).run()
    assert not at.exception
    assert at.title[0].value == f["name"].iloc[5]
