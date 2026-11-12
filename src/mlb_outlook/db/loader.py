"""Load the data snapshot, the fold 2 backtest and the prod2027 forecasts into MySQL.

Two runs, distinguished by `model_runs.run_kind`:

- **backtest** -- fold 2, with its evaluation tables and the 2025 outcomes, so the
  `predictions` rows carry `actual_woba` and `actual_pa`.
- **production** -- prod2027, loaded when `artifacts/prod2027_*` are there. No evaluation
  tables and no outcome columns, because 2027 has not happened: the whole point of these rows
  is that they are in the database before the answer is.

Both carry their model version and data version on every prediction row, so a row read on
its own still says what made it.

Idempotent: re-running replaces the snapshot's player-seasons and each run it loads (whose
predictions and evaluation rows cascade), and leaves other data versions and runs alone.

    docker compose up -d
    uv run python scripts/run_holdout.py --rehearse fold2
    uv run python scripts/export_fold2_forecasts.py
    uv run python scripts/export_prod2027_forecasts.py
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
PRODUCTION_RUN = "prod2027"

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
# prod2027_forecasts.csv has no outcome columns to map: the season has not been played, so
# actual_woba and actual_pa stay NULL until it has.
PRODUCTION_PREDICTION_COLUMNS = {
    source: target
    for source, target in PREDICTION_COLUMNS.items()
    if not source.startswith("y_")
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
    forecasts: pd.DataFrame,
    *,
    model_run_id: int,
    model_version: str,
    data_version_id: int,
    columns: dict[str, str] = PREDICTION_COLUMNS,
) -> list[dict]:
    """Prediction rows for one run. `columns` maps forecast columns to database columns.

    A production run passes `PRODUCTION_PREDICTION_COLUMNS`, which leaves out the outcome
    pair; a backtest passes the default, which includes it.
    """
    missing = set(columns) - set(forecasts.columns)
    if missing:
        raise ValueError(f"forecasts are missing columns: {sorted(missing)}")
    frame = forecasts[list(columns)].rename(columns=columns)
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


def replace_run(
    conn: Connection,
    *,
    data_version_id: int,
    manifest: dict,
    run_kind: str,
    fold: str,
    config: str,
    train_targets: Iterable[int],
) -> int:
    """Delete and re-insert one (model_version, data version, fold, config) run; return its id.

    That four-column key is the table's unique key, so this is what makes a reload idempotent:
    the old run's predictions and evaluation rows cascade away with it.
    """
    key = {
        "mv": manifest["model_version"],
        "d": data_version_id,
        "fold": fold,
        "config": config,
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
            "VALUES (:mv, :d, :kind, :fold, :config, :train, :target, :through, "
            ":commit, :generated)"
        ),
        key
        | {
            "kind": run_kind,
            "train": ",".join(str(t) for t in train_targets),
            "target": manifest["target_season"],
            "through": manifest["features_through"],
            "commit": manifest.get("generated_from_commit") or None,
            "generated": generated[:19].replace("T", " ") if generated else None,
        },
    )
    return int(
        conn.execute(
            sa.text(
                "SELECT model_run_id FROM model_runs WHERE model_version = :mv "
                "AND data_version_id = :d AND fold = :fold AND config = :config"
            ),
            key,
        ).scalar_one()
    )


def load_fold_run(
    conn: Connection,
    *,
    data_version_id: int,
    manifest: dict,
    train_targets: Iterable[int],
    forecasts: pd.DataFrame,
    evaluations: dict[str, pd.DataFrame],
) -> int:
    """Replace the fold's backtest run and load its predictions and evaluation rows."""
    run_id = replace_run(
        conn,
        data_version_id=data_version_id,
        manifest=manifest,
        run_kind="backtest",
        fold=manifest["fold"],
        config=CONFIG_NAME,
        train_targets=train_targets,
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


def load_production_run(
    conn: Connection, *, data_version_id: int, manifest: dict, forecasts: pd.DataFrame
) -> int:
    """Replace the production run and load its predictions. No evaluation rows exist yet.

    `run_kind` is 'production' and the outcome columns stay NULL, which is the difference that
    matters: a query can tell a forecast whose season has been played from one whose has not
    without knowing which fold names mean what.
    """
    run_id = replace_run(
        conn,
        data_version_id=data_version_id,
        manifest=manifest,
        run_kind="production",
        fold=manifest["run"],
        config=manifest["config"],
        train_targets=manifest["train_targets"],
    )
    _insert(
        conn,
        "predictions",
        prediction_rows(
            forecasts,
            model_run_id=run_id,
            model_version=manifest["model_version"],
            data_version_id=data_version_id,
            columns=PRODUCTION_PREDICTION_COLUMNS,
        ),
    )
    return run_id


def _check_hash(data: Path, sha: str, manifest: dict, what: str) -> None:
    if sha != manifest["data_sha256"]:
        raise RuntimeError(
            f"{data} hashes to {sha}, but the {what} artifacts were built from "
            f"{manifest['data_sha256']}; regenerate them before loading"
        )


def load_all(
    engine: Engine,
    *,
    data: Path = DATA,
    artifacts: Path = ARTIFACTS,
    config: Path = CONFIG,
    schema: Path = SCHEMA,
) -> dict[str, int | None]:
    """Load the snapshot, the fold 2 backtest, and prod2027 if its artifacts are present."""
    manifest = json.loads((artifacts / f"{FOLD}_manifest.json").read_text())
    sha = file_sha256(data)
    _check_hash(data, sha, manifest, "fold 2")
    df = pd.read_parquet(data)
    cfg = load_folds_config(config)
    fold = next(f for f in cfg.folds if f.name == FOLD)
    forecasts = pd.read_csv(artifacts / f"{FOLD}_forecasts.csv")
    evaluations = {
        source: pd.read_csv(artifacts / f"{FOLD}_{source}.csv") for source in EVALUATION_SOURCES
    }

    prod_manifest_path = artifacts / f"{PRODUCTION_RUN}_manifest.json"
    prod_forecasts_path = artifacts / f"{PRODUCTION_RUN}_forecasts.csv"
    prod: tuple[dict, pd.DataFrame] | None = None
    if prod_manifest_path.exists() and prod_forecasts_path.exists():
        prod_manifest = json.loads(prod_manifest_path.read_text())
        _check_hash(data, sha, prod_manifest, PRODUCTION_RUN)
        prod = (prod_manifest, pd.read_csv(prod_forecasts_path))

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
        production_run_id = (
            load_production_run(
                conn,
                data_version_id=data_version_id,
                manifest=prod[0],
                forecasts=prod[1],
            )
            if prod is not None
            else None
        )
    return {
        "data_version_id": data_version_id,
        "model_run_id": run_id,
        "production_run_id": production_run_id,
    }


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
