"""Load the data snapshot and the fold 2 backtest artifacts into MySQL.

Idempotent: re-running replaces the snapshot's player-seasons and the fold 2 run (its
predictions and evaluation rows cascade), and leaves other data versions and runs alone.

    docker compose up -d
    uv run python scripts/run_holdout.py --rehearse fold2
    uv run python scripts/export_fold2_forecasts.py
    uv run python -m mlb_outlook.db.loader

The connection string is `DATABASE_URL`, from the environment or from `.env`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from mlb_outlook.config import load_folds_config
from mlb_outlook.db.versions import file_sha256

DATA = Path("data/processed/hitter_seasons.parquet")
ARTIFACTS = Path("artifacts")
CONFIG = Path("configs/folds.yaml")
SCHEMA = Path("sql/001_schema.sql")
FOLD = "fold2"
CONFIG_NAME = "extended"

SEASON_COLUMNS = [
    "player_id", "season", "age", "pa", "woba", "xwoba",
    "k_pct", "bb_pct", "avg_ev", "barrel_pct", "hard_hit_pct",
]
# fold2_forecasts.csv column -> predictions column
PREDICTION_COLUMNS = {
    "player_id": "player_id",
    "target_season": "target_season",
    "age_target": "age_target",
    "pa_lag1": "pa_lag1",
    "ridge_rel": "point_woba",
    "lo": "lo80",
    "hi": "hi80",
    "marcel": "marcel_woba",
    "weighted3": "weighted3_woba",
    "threshold": "threshold_woba",
    "p_breakout": "p_breakout",
    "p_stable": "p_stable",
    "p_regression": "p_regression",
    "y_woba": "actual_woba",
    "y_pa": "actual_pa",
}
# Evaluation artifacts: file suffix -> (columns naming the model, columns naming the slice).
EVALUATION_SOURCES = {
    "point": (("model",), ()),
    "interval_coverage": ((), ("grouping", "range")),
    "probabilities": (("label",), ()),
    "reliability": (("label",), ("decile",)),
    "population": ((), ("bucket",)),
    "verdict": ((), ("criterion",)),
}
# Identifying or text columns that are not metrics.
NON_METRICS = {
    "model", "config", "fold", "label", "grouping", "range", "bucket", "decile",
    "criterion", "description", "value", "target_season",
}


# ---- connection ----


def read_env_file(path: str | Path = ".env") -> dict[str, str]:
    """`KEY=value` lines from a dotenv file; comments and blank lines skipped."""
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip("'\"")
    return out


def database_url(env_file: str | Path = ".env") -> str | None:
    return os.environ.get("DATABASE_URL") or read_env_file(env_file).get("DATABASE_URL")


def make_engine(url: str | None = None) -> Engine:
    url = url or database_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not set (environment or .env); see .env.example")
    return sa.create_engine(url, pool_pre_ping=True, future=True)


# ---- pure row builders (unit-tested without a database) ----


def clean(value):
    """NaN / NA -> None, numpy scalars -> Python scalars, so the driver gets plain values."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def records(frame: pd.DataFrame) -> list[dict]:
    return [{k: clean(v) for k, v in row.items()} for row in frame.to_dict("records")]


def schema_statements(sql: str) -> list[str]:
    """Split a schema file into statements, dropping `--` comments."""
    stripped = re.sub(r"--[^\n]*", "", sql)
    return [s.strip() for s in stripped.split(";") if s.strip()]


def snapshot_summary(df: pd.DataFrame) -> dict[str, int]:
    return {
        "n_rows": len(df),
        "n_players": int(df["player_id"].nunique()),
        "first_season": int(df["season"].min()),
        "last_season": int(df["season"].max()),
    }


def player_rows(df: pd.DataFrame) -> list[dict]:
    """One row per player, the most recent name the snapshot carries."""
    latest = df.sort_values("season").drop_duplicates("player_id", keep="last")
    return records(latest[["player_id", "name"]].fillna({"name": ""}))


def season_rows(df: pd.DataFrame, data_version_id: int) -> list[dict]:
    frame = df[SEASON_COLUMNS].copy()
    frame.insert(0, "data_version_id", data_version_id)
    return records(frame)


def prediction_rows(
    forecasts: pd.DataFrame, *, model_run_id: int, model_version: str, data_version_id: int
) -> list[dict]:
    missing = set(PREDICTION_COLUMNS) - set(forecasts.columns)
    if missing:
        raise ValueError(f"forecasts are missing columns: {sorted(missing)}")
    frame = forecasts[list(PREDICTION_COLUMNS)].rename(columns=PREDICTION_COLUMNS)
    frame.insert(0, "model_run_id", model_run_id)
    frame.insert(1, "model_version", model_version)
    frame.insert(2, "data_version_id", data_version_id)
    return records(frame)


def evaluation_rows(source: str, frame: pd.DataFrame, *, model_run_id: int) -> list[dict]:
    """Melt one evaluation artifact into (model, slice, metric, value) rows."""
    model_cols, slice_cols = EVALUATION_SOURCES[source]
    out: list[dict] = []
    for row in frame.to_dict("records"):
        model = "/".join(str(row[c]) for c in model_cols)
        slice_ = "/".join(str(row[c]) for c in slice_cols)
        for metric, value in row.items():
            if metric in NON_METRICS:
                continue
            value = clean(value)
            if isinstance(value, bool):
                value = float(value)
            if value is not None and not isinstance(value, int | float):
                continue  # text columns (e.g. a verdict label) are not metrics
            out.append(
                {
                    "model_run_id": model_run_id,
                    "source": source,
                    "model": model,
                    "slice": slice_,
                    "metric": metric,
                    "value": None if value is None else float(value),
                }
            )
    return out


# ---- database operations ----


def apply_schema(conn: Connection, path: Path = SCHEMA) -> None:
    for statement in schema_statements(path.read_text(encoding="utf-8")):
        conn.exec_driver_sql(statement)


def _insert(conn: Connection, table: str, rows: Sequence[dict], chunk: int = 2000) -> None:
    if not rows:
        return
    columns = list(rows[0])
    sql = sa.text(
        f"INSERT INTO {table} ({', '.join(columns)}) "
        f"VALUES ({', '.join(':' + c for c in columns)})"
    )
    for start in range(0, len(rows), chunk):
        conn.execute(sql, list(rows[start : start + chunk]))


def upsert_data_version(conn: Connection, df: pd.DataFrame, sha256: str, source: str) -> int:
    summary = snapshot_summary(df)
    conn.execute(
        sa.text(
            "INSERT INTO data_versions "
            "(sha256, source_path, n_rows, n_players, first_season, last_season) "
            "VALUES (:sha, :src, :n_rows, :n_players, :first_season, :last_season) "
            "ON DUPLICATE KEY UPDATE source_path = VALUES(source_path)"
        ),
        {"sha": sha256, "src": source, **summary},
    )
    return int(
        conn.execute(
            sa.text("SELECT data_version_id FROM data_versions WHERE sha256 = :sha"),
            {"sha": sha256},
        ).scalar_one()
    )


def load_snapshot(conn: Connection, df: pd.DataFrame, sha256: str, source: str) -> int:
    """Players (upserted) and this data version's player-seasons (replaced)."""
    data_version_id = upsert_data_version(conn, df, sha256, source)
    players = player_rows(df)
    conn.execute(
        sa.text(
            "INSERT INTO players (player_id, name) VALUES (:player_id, :name) "
            "ON DUPLICATE KEY UPDATE name = VALUES(name)"
        ),
        players,
    )
    conn.execute(
        sa.text("DELETE FROM player_seasons WHERE data_version_id = :d"), {"d": data_version_id}
    )
    _insert(conn, "player_seasons", season_rows(df, data_version_id))
    return data_version_id


def load_fold_run(
    conn: Connection,
    *,
    data_version_id: int,
    manifest: dict,
    train_targets: Iterable[int],
    forecasts: pd.DataFrame,
    evaluations: dict[str, pd.DataFrame],
) -> int:
    """Replace the (model_version, data version, fold, config) run and load its rows."""
    key = {
        "mv": manifest["model_version"],
        "d": data_version_id,
        "fold": manifest["fold"],
        "config": CONFIG_NAME,
    }
    conn.execute(
        sa.text(
            "DELETE FROM model_runs WHERE model_version = :mv AND data_version_id = :d "
            "AND fold = :fold AND config = :config"
        ),
        key,
    )
    generated = manifest.get("generated_at")
    conn.execute(
        sa.text(
            "INSERT INTO model_runs (model_version, data_version_id, run_kind, fold, config, "
            "train_targets, target_season, features_through, git_commit, generated_at) "
            "VALUES (:mv, :d, 'backtest', :fold, :config, :train, :target, :through, "
            ":commit, :generated)"
        ),
        key
        | {
            "train": ",".join(str(t) for t in train_targets),
            "target": manifest["target_season"],
            "through": manifest["features_through"],
            "commit": manifest.get("generated_from_commit") or None,
            "generated": generated[:19].replace("T", " ") if generated else None,
        },
    )
    run_id = int(
        conn.execute(
            sa.text(
                "SELECT model_run_id FROM model_runs WHERE model_version = :mv "
                "AND data_version_id = :d AND fold = :fold AND config = :config"
            ),
            key,
        ).scalar_one()
    )
    _insert(
        conn,
        "predictions",
        prediction_rows(
            forecasts,
            model_run_id=run_id,
            model_version=manifest["model_version"],
            data_version_id=data_version_id,
        ),
    )
    for source, frame in evaluations.items():
        _insert(conn, "evaluation_results", evaluation_rows(source, frame, model_run_id=run_id))
    return run_id


def load_all(
    engine: Engine,
    *,
    data: Path = DATA,
    artifacts: Path = ARTIFACTS,
    config: Path = CONFIG,
    schema: Path = SCHEMA,
) -> dict[str, int]:
    manifest = json.loads((artifacts / f"{FOLD}_manifest.json").read_text())
    sha = file_sha256(data)
    if sha != manifest["data_sha256"]:
        raise RuntimeError(
            f"{data} hashes to {sha}, but the fold 2 artifacts were built from "
            f"{manifest['data_sha256']}; regenerate them before loading"
        )
    df = pd.read_parquet(data)
    cfg = load_folds_config(config)
    fold = next(f for f in cfg.folds if f.name == FOLD)
    forecasts = pd.read_csv(artifacts / f"{FOLD}_forecasts.csv")
    evaluations = {
        source: pd.read_csv(artifacts / f"{FOLD}_{source}.csv") for source in EVALUATION_SOURCES
    }
    with engine.begin() as conn:
        apply_schema(conn, schema)
    with engine.begin() as conn:
        data_version_id = load_snapshot(conn, df, sha, str(data).replace("\\", "/"))
        run_id = load_fold_run(
            conn,
            data_version_id=data_version_id,
            manifest=manifest,
            train_targets=cfg.extended_train_targets(fold),
            forecasts=forecasts,
            evaluations=evaluations,
        )
    return {"data_version_id": data_version_id, "model_run_id": run_id}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", help="override DATABASE_URL")
    args = parser.parse_args(argv)
    engine = make_engine(args.url)
    ids = load_all(engine)
    with engine.connect() as conn:
        for table in ("data_versions", "players", "player_seasons", "model_runs",
                      "predictions", "evaluation_results"):
            n = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            print(f"  {table:20s} {n:>7}")
    print(f"loaded: {ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
