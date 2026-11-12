"""The Streamlit app: pages render headless, data stops at 2025, wording rules hold."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from components import data  # noqa: E402
from components.style import percentile_color, signed, woba  # noqa: E402

PAGES = [
    APP / "Home.py",
    APP / "pages" / "1_Player.py",
    APP / "pages" / "2_Projection_disagreements.py",
    APP / "pages" / "3_Model_performance.py",
]


def _have_artifacts() -> bool:
    return all(p.exists() for p in data.required_files())


needs_artifacts = pytest.mark.skipif(
    not _have_artifacts(), reason="fold 2 artifacts not generated; see components/data.py"
)


# ---- wording and scope, no artifacts needed ----


def test_no_page_calls_a_player_undervalued():
    for path in APP.rglob("*.py"):
        assert "undervalued" not in path.read_text(encoding="utf-8").lower(), path


def test_every_page_carries_the_backtest_banner():
    for page in PAGES:
        assert "setup(" in page.read_text(encoding="utf-8"), page
    assert "Fold 2 backtest" in data.BACKTEST_LABEL
    assert "Not a live projection" in data.BACKTEST_LABEL


def test_the_backtest_is_fold2_and_never_the_holdout_season():
    assert data.FOLD == "fold2"
    assert data.TARGET_SEASON == 2025
    assert data.FEATURES_THROUGH == 2024
    assert data.LAST_VISIBLE_SEASON < 2026


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
def test_loaded_seasons_stop_at_2025():
    assert data.seasons()["season"].max() == 2025


@needs_artifacts
def test_forecasts_are_the_fold2_rows():
    f = data.forecasts()
    assert len(f) == 362
    assert set(f["target_season"]) == {2025}
    assert ((f["p_breakout"] + f["p_stable"] + f["p_regression"]) - 1).abs().max() < 1e-9
    assert (f["lo"] <= f["hi"]).all()


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
    assert any("Fold 2 backtest" in m.value for m in at.markdown)


@needs_artifacts
def test_board_sorts_change_the_order():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGES[2]), default_timeout=120).run()
    first = at.dataframe[0].value.iloc[0]["name"]
    at.selectbox[0].set_value("Model below Marcel").run()
    assert not at.exception
    assert at.dataframe[0].value.iloc[0]["name"] != first


@needs_artifacts
def test_player_page_follows_the_selection():
    from streamlit.testing.v1 import AppTest

    f = data.forecasts().sort_values("name")
    target = int(f["player_id"].iloc[5])
    at = AppTest.from_file(str(PAGES[1]), default_timeout=120).run()
    at.selectbox[0].set_value(target).run()
    assert not at.exception
    assert at.title[0].value == f["name"].iloc[5]
