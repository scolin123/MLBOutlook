"""Integration tests against a running MySQL. Skipped cleanly when there is none.

They load the real snapshot and the fold 2 artifacts into the database `DATABASE_URL` points
at. The load is idempotent (it replaces only this snapshot and this run), so running them
against the dev database leaves it as `python -m mlb_outlook.db.loader` would.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

sa = pytest.importorskip("sqlalchemy")

from mlb_outlook.db import loader  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/hitter_seasons.parquet"
ARTIFACTS = ROOT / "artifacts"


def _engine():
    url = loader.database_url(ROOT / ".env")
    if not url:
        pytest.skip("DATABASE_URL not set; MySQL integration tests skipped")
    engine = sa.create_engine(url, connect_args={"connect_timeout": 3})
    try:
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 -- any connection failure means "no server"
        pytest.skip(f"MySQL not reachable ({type(exc).__name__}); integration tests skipped")
    return engine


@pytest.fixture(scope="module")
def loaded():
    engine = _engine()
    needed = [DATA, ARTIFACTS / "fold2_forecasts.csv", ARTIFACTS / "fold2_manifest.json"]
    if not all(p.exists() for p in needed):
        pytest.skip("snapshot or fold 2 artifacts missing; run the export first")
    ids = loader.load_all(
        engine,
        data=DATA,
        artifacts=ARTIFACTS,
        config=ROOT / "configs/folds.yaml",
        schema=ROOT / "sql/001_schema.sql",
    )
    return engine, ids


def _scalar(engine, sql, **params):
    with engine.connect() as conn:
        return conn.execute(sa.text(sql), params).scalar_one()


def test_the_snapshot_loads_with_its_hash(loaded):
    engine, ids = loaded
    manifest = json.loads((ARTIFACTS / "fold2_manifest.json").read_text())
    d = ids["data_version_id"]
    assert _scalar(engine, "SELECT sha256 FROM data_versions WHERE data_version_id=:d", d=d) == (
        manifest["data_sha256"]
    )
    n = _scalar(engine, "SELECT COUNT(*) FROM player_seasons WHERE data_version_id=:d", d=d)
    assert n == len(pd.read_parquet(DATA))


def test_every_prediction_names_its_model_and_data_version(loaded):
    engine, ids = loaded
    run = ids["model_run_id"]
    forecasts = pd.read_csv(ARTIFACTS / "fold2_forecasts.csv")
    assert _scalar(engine, "SELECT COUNT(*) FROM predictions WHERE model_run_id=:r", r=run) == (
        len(forecasts)
    )
    bad = _scalar(
        engine,
        "SELECT COUNT(*) FROM predictions p JOIN model_runs m USING (model_run_id) "
        "WHERE p.model_run_id=:r AND (p.model_version <> m.model_version "
        "OR p.data_version_id <> m.data_version_id OR p.model_version = '')",
        r=run,
    )
    assert bad == 0


def test_the_loaded_forecasts_reproduce_fold2_mae(loaded):
    engine, ids = loaded
    mae = _scalar(
        engine,
        "SELECT AVG(ABS(point_woba - actual_woba)) FROM predictions WHERE model_run_id=:r",
        r=ids["model_run_id"],
    )
    assert float(mae) == pytest.approx(0.0263, abs=5e-5)


def test_the_production_run_is_loaded_as_production_with_no_outcome(loaded):
    """prod2027 sits beside fold 2 under its own run_kind, with 2027 still unanswered."""
    engine, ids = loaded
    run = ids["production_run_id"]
    if run is None:
        pytest.skip("prod2027 artifacts not generated")
    manifest = json.loads((ARTIFACTS / "prod2027_manifest.json").read_text())
    forecasts = pd.read_csv(ARTIFACTS / "prod2027_forecasts.csv")

    with engine.connect() as conn:
        row = conn.execute(
            sa.text(
                "SELECT run_kind, fold, config, target_season, features_through, "
                "train_targets, model_version FROM model_runs WHERE model_run_id=:r"
            ),
            {"r": run},
        ).one()
    assert row.run_kind == "production"
    assert row.fold == "prod2027"
    assert row.config == "extended"
    assert row.target_season == 2027
    assert row.features_through == 2026
    assert row.train_targets == ",".join(str(t) for t in manifest["train_targets"])
    assert row.model_version == manifest["model_version"]

    assert _scalar(engine, "SELECT COUNT(*) FROM predictions WHERE model_run_id=:r", r=run) == (
        len(forecasts)
    )
    # Every row says what made it, and none of them claims to know what happened in 2027.
    assert _scalar(
        engine,
        "SELECT COUNT(*) FROM predictions p JOIN model_runs m USING (model_run_id) "
        "WHERE p.model_run_id=:r AND (p.model_version <> m.model_version "
        "OR p.data_version_id <> m.data_version_id)",
        r=run,
    ) == 0
    assert _scalar(
        engine,
        "SELECT COUNT(*) FROM predictions WHERE model_run_id=:r "
        "AND (actual_woba IS NOT NULL OR actual_pa IS NOT NULL)",
        r=run,
    ) == 0
    # A production run has no evaluation tables: there is nothing to score yet.
    assert _scalar(
        engine, "SELECT COUNT(*) FROM evaluation_results WHERE model_run_id=:r", r=run
    ) == 0


def test_the_production_and_backtest_runs_share_the_data_version(loaded):
    engine, ids = loaded
    if ids["production_run_id"] is None:
        pytest.skip("prod2027 artifacts not generated")
    versions = {
        _scalar(engine, "SELECT data_version_id FROM model_runs WHERE model_run_id=:r", r=r)
        for r in (ids["model_run_id"], ids["production_run_id"])
    }
    assert versions == {ids["data_version_id"]}


def test_the_load_is_idempotent(loaded):
    engine, ids = loaded
    before = {
        t: _scalar(engine, f"SELECT COUNT(*) FROM {t}")
        for t in ("players", "player_seasons", "model_runs", "predictions", "evaluation_results")
    }
    again = loader.load_all(
        engine,
        data=DATA,
        artifacts=ARTIFACTS,
        config=ROOT / "configs/folds.yaml",
        schema=ROOT / "sql/001_schema.sql",
    )
    assert again["data_version_id"] == ids["data_version_id"]
    after = {t: _scalar(engine, f"SELECT COUNT(*) FROM {t}") for t in before}
    assert after == before
