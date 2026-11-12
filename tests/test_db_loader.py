"""Unit tests for the loader's row builders and version helpers; no database needed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlb_outlook.db import loader
from mlb_outlook.db.versions import MODEL_NAME, file_sha256, model_version

SCHEMA = Path(__file__).resolve().parents[1] / "sql" / "001_schema.sql"


def test_clean_turns_missing_into_none_and_numpy_into_python():
    assert loader.clean(float("nan")) is None
    assert loader.clean(pd.NA) is None
    assert loader.clean(None) is None
    assert loader.clean(np.int64(3)) == 3 and type(loader.clean(np.int64(3))) is int
    assert type(loader.clean(np.float64(0.5))) is float
    assert loader.clean(np.bool_(True)) is True
    assert loader.clean("x") == "x"


def test_read_env_file_skips_comments_and_strips_quotes(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\n\nDATABASE_URL='mysql+pymysql://u:p@h/db'\nOTHER = 1\n")
    assert loader.read_env_file(env) == {"DATABASE_URL": "mysql+pymysql://u:p@h/db", "OTHER": "1"}
    assert loader.read_env_file(tmp_path / "missing") == {}


def test_database_url_prefers_the_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DATABASE_URL=from-file\n")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert loader.database_url(env) == "from-file"
    monkeypatch.setenv("DATABASE_URL", "from-env")
    assert loader.database_url(env) == "from-env"


def test_schema_splits_into_the_six_tables():
    statements = loader.schema_statements(SCHEMA.read_text())
    tables = [s.split()[5] for s in statements]
    assert tables == [
        "data_versions", "players", "player_seasons", "model_runs",
        "predictions", "evaluation_results",
    ]
    assert all(s.startswith("CREATE TABLE IF NOT EXISTS") for s in statements)
    assert not any("--" in s for s in statements)


def test_every_prediction_row_carries_model_and_data_version():
    forecasts = pd.DataFrame(
        {c: [0.3, np.nan] for c in loader.PREDICTION_COLUMNS}
        | {"player_id": [1, 2], "target_season": [2025, 2025]}
    )
    rows = loader.prediction_rows(
        forecasts, model_run_id=7, model_version="m@abc", data_version_id=3
    )
    assert len(rows) == 2
    for row in rows:
        assert row["model_version"] == "m@abc"
        assert row["data_version_id"] == 3
        assert row["model_run_id"] == 7
        assert set(loader.PREDICTION_COLUMNS.values()) <= set(row)
    assert rows[1]["point_woba"] is None  # NaN never becomes zero


def test_production_prediction_rows_leave_the_outcome_columns_out():
    """2027 has no outcome, so a production row must not carry one -- not even as NULL input."""
    columns = loader.PRODUCTION_PREDICTION_COLUMNS
    assert set(columns) == set(loader.PREDICTION_COLUMNS) - {"y_woba", "y_pa"}
    assert "actual_woba" not in columns.values()
    assert "actual_pa" not in columns.values()

    forecasts = pd.DataFrame(
        {c: [0.3] for c in columns} | {"player_id": [1], "target_season": [2027]}
    )
    rows = loader.prediction_rows(
        forecasts, model_run_id=9, model_version="m@abc", data_version_id=4, columns=columns
    )
    assert len(rows) == 1
    assert set(rows[0]) == {"model_run_id", "model_version", "data_version_id"} | set(
        columns.values()
    )
    assert rows[0]["target_season"] == 2027


def test_prediction_rows_refuse_a_frame_missing_columns():
    with pytest.raises(ValueError, match="missing"):
        loader.prediction_rows(
            pd.DataFrame({"player_id": [1]}), model_run_id=1, model_version="m", data_version_id=1
        )


def test_player_rows_keep_the_latest_name():
    df = pd.DataFrame(
        {"player_id": [1, 1, 2], "season": [2020, 2021, 2020], "name": ["Old", "New", "B"]}
    )
    names = {r["player_id"]: r["name"] for r in loader.player_rows(df)}
    assert names == {1: "New", 2: "B"}


def test_evaluation_rows_melt_metrics_and_skip_labels():
    point = pd.DataFrame(
        {"model": ["marcel"], "config": ["extended"], "fold": ["fold2"], "mae": [0.0285],
         "coverage": [np.nan]}
    )
    rows = loader.evaluation_rows("point", point, model_run_id=1)
    assert {(r["model"], r["metric"]) for r in rows} == {("marcel", "mae"), ("marcel", "coverage")}
    assert next(r for r in rows if r["metric"] == "coverage")["value"] is None

    verdict = pd.DataFrame(
        {"criterion": [1], "description": ["x"], "value": ["0.0263 vs 0.0285"], "passed": [True]}
    )
    rows = loader.evaluation_rows("verdict", verdict, model_run_id=1)
    assert rows == [
        {"model_run_id": 1, "source": "verdict", "model": "", "slice": "1",
         "metric": "passed", "value": 1.0}
    ]


def test_evaluation_slices_join_grouping_and_range():
    coverage = pd.DataFrame(
        {"grouping": ["age"], "bucket": [0], "range": ["age < 26"], "n": [77], "coverage": [0.87]}
    )
    rows = loader.evaluation_rows("interval_coverage", coverage, model_run_id=1)
    assert {r["slice"] for r in rows} == {"age/age < 26"}
    assert {r["metric"] for r in rows} == {"n", "coverage"}  # bucket is a slice id, not a metric


def test_snapshot_summary():
    df = pd.DataFrame({"player_id": [1, 1, 2], "season": [2015, 2016, 2026]})
    assert loader.snapshot_summary(df) == {
        "n_rows": 3, "n_players": 2, "first_season": 2015, "last_season": 2026
    }


def test_file_sha256(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    assert file_sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_model_version_names_the_frozen_code():
    version = model_version()
    assert version.startswith(f"{MODEL_NAME}@")
    suffix = version.split("@", 1)[1]
    assert suffix == "unversioned" or (len(suffix) == 12 and int(suffix, 16) >= 0)
