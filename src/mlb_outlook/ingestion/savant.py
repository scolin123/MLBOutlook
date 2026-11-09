"""Pull MLB hitter season lines from Baseball Savant's custom leaderboard.

One CSV request per season. Savant keys players by MLBAM id and aggregates
players traded mid-season into a single line, so no ID mapping is needed here.

Raw responses are cached to ``data/raw/savant_batting_<year>.csv`` and never
re-downloaded unless ``--refresh`` is passed. The cleaned table is written to
``data/processed/hitter_seasons.parquet`` with the columns the feature builder
expects: player_id, season, age, pa, woba, xwoba, k_pct, bb_pct and the
quality-of-contact fields avg_ev, barrel_pct, hard_hit_pct (plus name).

A cached CSV that predates a change to ``SELECTIONS`` is re-downloaded; the
quality-of-contact fields are blank for hitters with no tracked batted balls and
stay missing (never zero).

Usage:
    uv run python -m mlb_outlook.ingestion.savant --start 2015 --end 2026
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://baseballsavant.mlb.com/leaderboard/custom"
SELECTIONS = (
    "pa,k_percent,bb_percent,woba,xwoba,player_age,"
    "exit_velocity_avg,barrel_batted_rate,hard_hit_percent"
)
RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/hitter_seasons.parquet")
COLUMNS = [
    "player_id",
    "name",
    "season",
    "age",
    "pa",
    "woba",
    "xwoba",
    "k_pct",
    "bb_pct",
    "avg_ev",
    "barrel_pct",
    "hard_hit_pct",
]
HEADERS = {"User-Agent": "mlb-outlook research project (personal, non-commercial)"}

# Savant column -> our column
RENAME = {
    "player_id": "player_id",
    "year": "season",
    "player_age": "age",
    "pa": "pa",
    "woba": "woba",
    "xwoba": "xwoba",
    "k_percent": "k_pct",
    "bb_percent": "bb_pct",
    "exit_velocity_avg": "avg_ev",
    "barrel_batted_rate": "barrel_pct",
    "hard_hit_percent": "hard_hit_pct",
}
# Savant reports these as percentages; we store fractions.
PERCENT_COLUMNS = ("k_pct", "bb_pct", "barrel_pct", "hard_hit_pct")


def season_params(year: int) -> dict[str, str]:
    return {
        "year": str(year),
        "type": "batter",
        "filter": "",
        "min": "1",
        "selections": SELECTIONS,
        "sort": "pa",
        "sortDir": "desc",
        "csv": "true",
    }


def fetch_season_csv(year: int, session: requests.Session | None = None) -> str:
    """Download one season's leaderboard as CSV text. Raises on HTTP or format errors."""
    s = session or requests.Session()
    r = s.get(URL, params=season_params(year), headers=HEADERS, timeout=60)
    r.raise_for_status()
    text = r.content.decode("utf-8-sig")
    first_line = text.splitlines()[0] if text else ""
    if "player_id" not in first_line:
        raise ValueError(f"{year}: response is not the expected CSV (first line: {text[:120]!r})")
    return text


def header_columns(text: str) -> list[str]:
    """Column names from a CSV header line (good enough to spot a stale cache)."""
    first = text.splitlines()[0] if text else ""
    return [c.strip().strip('"') for c in first.split(",")]


def parse_season_csv(text: str, year: int) -> pd.DataFrame:
    """Turn Savant CSV text into rows of COLUMNS. Rates become fractions (22.5 -> 0.225)."""
    raw = pd.read_csv(io.StringIO(text))
    raw.columns = [c.strip().strip('"') for c in raw.columns]
    missing = [c for c in RENAME if c not in raw.columns]
    if missing:
        raise ValueError(f"{year}: Savant CSV missing columns {missing}; got {list(raw.columns)}")
    name_col = next((c for c in raw.columns if "name" in c.lower()), None)
    df = raw[list(RENAME)].rename(columns=RENAME)
    df["name"] = raw[name_col].astype(str).str.strip() if name_col else ""
    if (df["season"] != year).any():
        raise ValueError(f"{year}: CSV contains rows for other seasons")
    df = df.dropna(subset=["player_id", "pa", "woba"])
    df = df[df["pa"] > 0].copy()
    for col in PERCENT_COLUMNS:
        df[col] = df[col].astype(float) / 100.0
    df = df.astype(
        {"player_id": "int64", "season": "int64", "age": "int64", "pa": "int64",
         "woba": "float64", "xwoba": "float64", "avg_ev": "float64"}
    )
    if df["player_id"].duplicated().any():
        raise ValueError(f"{year}: duplicate player_id rows (expected one line per player)")
    return df[COLUMNS].sort_values(["season", "player_id"]).reset_index(drop=True)


def validate(df: pd.DataFrame) -> list[str]:
    """Sanity checks on the combined table. Returns a list of problems (empty = fine)."""
    problems = []
    if df.duplicated(["player_id", "season"]).any():
        problems.append("duplicate (player_id, season) rows")
    for c in ("woba", "xwoba"):
        if not df[c].dropna().between(0, 2.1).all():
            problems.append(f"{c} outside [0, 2.1]")
    for c in ("k_pct", "bb_pct", "barrel_pct", "hard_hit_pct"):
        if not df[c].dropna().between(0, 1).all():
            problems.append(f"{c} outside [0, 1]")
    # A hitter with one tracked batted ball can average anything, so the loose bound is wide.
    if not df["avg_ev"].dropna().between(20, 115).all():
        problems.append("avg_ev outside [20, 115] mph")
    if not df["age"].between(16, 50).all():
        problems.append("age outside [16, 50]")
    # Quality of contact needs tracked batted balls; it should be there for regulars.
    regulars = df[df["pa"] >= 100]
    for c in ("xwoba", "avg_ev", "barrel_pct", "hard_hit_pct"):
        missing = regulars[c].isna().mean()
        if missing > 0.02:
            problems.append(f"{c}: {missing:.1%} of 100+ PA hitters missing")
    if not regulars["avg_ev"].dropna().between(75, 100).all():
        problems.append("avg_ev outside [75, 100] mph for a 100+ PA hitter")
    for season, g in df.groupby("season"):
        lg = (g["woba"] * g["pa"]).sum() / g["pa"].sum()
        if not 0.28 <= lg <= 0.35:
            problems.append(f"{season}: league wOBA {lg:.3f} looks wrong")
        min_rows = 250 if season == 2020 else 400
        if (g["pa"] >= 100).sum() < min_rows:
            problems.append(f"{season}: only {(g['pa'] >= 100).sum()} hitters with 100+ PA")
    return problems


def pull(start: int, end: int, *, refresh: bool = False, pause: float = 2.0) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    frames = []
    for year in range(start, end + 1):
        cache = RAW_DIR / f"savant_batting_{year}.csv"
        cached = cache.read_text(encoding="utf-8") if cache.exists() else None
        stale = cached is not None and any(c not in header_columns(cached) for c in RENAME)
        if cached is not None and not refresh and not stale:
            text = cached
            print(f"{year}: cached")
        else:
            if stale:
                print(f"{year}: cached CSV predates the current fields, refetching")
            text = fetch_season_csv(year, session)
            cache.write_text(text, encoding="utf-8")
            print(f"{year}: downloaded")
            time.sleep(pause)
        frames.append(parse_season_csv(text, year))
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", type=int, default=2015)
    p.add_argument("--end", type=int, default=2026)
    p.add_argument("--refresh", action="store_true", help="re-download cached seasons")
    args = p.parse_args(argv)

    df = pull(args.start, args.end, refresh=args.refresh)
    problems = validate(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"\nwrote {OUT_PATH}: {len(df)} rows, {df['player_id'].nunique()} players")
    summary = df.groupby("season").apply(
        lambda g: pd.Series(
            {
                "hitters": len(g),
                "100+PA": int((g["pa"] >= 100).sum()),
                "lg_woba": round((g["woba"] * g["pa"]).sum() / g["pa"].sum(), 3),
                "lg_xwoba": round((g["xwoba"] * g["pa"]).sum() / g["pa"].sum(), 3),
                "no_qoc": int(g.loc[g["pa"] >= 100, "avg_ev"].isna().sum()),
            }
        ),
        include_groups=False,
    )
    print(summary.to_string())
    if problems:
        print("\nVALIDATION PROBLEMS:")
        for prob in problems:
            print(" -", prob)
        raise SystemExit(1)
    print("\nvalidation: ok")


if __name__ == "__main__":
    main()
