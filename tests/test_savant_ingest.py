"""Offline tests for the Savant parser and validator (no network)."""

import pandas as pd
import pytest

from mlb_outlook.ingestion.savant import COLUMNS, parse_season_csv, validate

CSV = (
    '﻿"last_name, first_name","player_id","year","player_age","pa",'
    '"k_percent","bb_percent","woba"\n'
    '"Judge, Aaron",592450,2024,32,704,24.9,18.9,0.476\n'
    '"Ohtani, Shohei",660271,2024,29,731,22.0,11.4,0.431\n'
    '"Pitcher, Some",111111,2024,30,0,,,\n'
)


def test_parse_columns_and_units():
    df = parse_season_csv(CSV.lstrip("﻿"), 2024)
    assert list(df.columns) == COLUMNS
    assert df["player_id"].tolist() == [592450, 660271]  # 0-PA row dropped, sorted by id
    judge = df.iloc[0]
    assert judge["name"] == "Judge, Aaron"
    assert (judge["season"], judge["age"], judge["pa"]) == (2024, 32, 704)
    assert judge["k_pct"] == pytest.approx(0.249)
    assert judge["bb_pct"] == pytest.approx(0.189)
    assert judge["woba"] == pytest.approx(0.476)


def test_parse_rejects_missing_column():
    bad = CSV.replace('"woba"', '"xwoba"')
    with pytest.raises(ValueError, match="missing columns"):
        parse_season_csv(bad, 2024)


def test_parse_rejects_wrong_season():
    with pytest.raises(ValueError, match="other seasons"):
        parse_season_csv(CSV, 2023)


def test_parse_rejects_duplicate_players():
    dup = CSV + '"Judge, Aaron",592450,2024,32,10,20.0,10.0,0.300\n'
    with pytest.raises(ValueError, match="duplicate"):
        parse_season_csv(dup, 2024)


def _season(season, n=450, woba=0.315):
    return pd.DataFrame(
        {
            "player_id": range(n),
            "name": "x",
            "season": season,
            "age": 27,
            "pa": 300,
            "woba": woba,
            "k_pct": 0.22,
            "bb_pct": 0.08,
        }
    )


def test_validate_clean_table():
    assert validate(pd.concat([_season(2023), _season(2024)])) == []


def test_validate_flags_problems():
    bad = _season(2024, woba=0.5)
    bad.loc[0, "k_pct"] = 24.9  # forgot to divide by 100
    probs = validate(pd.concat([bad, _season(2023, n=100)]))
    text = " | ".join(probs)
    assert "league wOBA" in text
    assert "k_pct" in text
    assert "2023: only 100" in text


def test_validate_short_2020_threshold():
    assert validate(_season(2020, n=300)) == []
