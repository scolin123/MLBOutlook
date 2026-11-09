"""Offline tests for the Savant parser and validator (no network)."""

import numpy as np
import pandas as pd
import pytest

from mlb_outlook.ingestion.savant import (
    COLUMNS,
    RENAME,
    header_columns,
    parse_season_csv,
    validate,
)

HEADER = (
    '"last_name, first_name","player_id","year","player_age","pa",'
    '"k_percent","bb_percent","woba","xwoba",'
    '"exit_velocity_avg","barrel_batted_rate","hard_hit_percent"'
)
CSV = (
    "\ufeff" + HEADER + "\n"
    '"Judge, Aaron",592450,2024,32,704,24.9,18.9,0.476,0.467,95.6,26.9,61.0\n'
    '"Ohtani, Shohei",660271,2024,29,731,22.0,11.4,0.431,0.411,94.0,20.4,55.4\n'
    '"Nocontact, Danny",222222,2024,24,3,100.0,0.0,0.000,0.000,,,\n'
    '"Pitcher, Some",111111,2024,30,0,,,,,,,\n'
)
OLD_HEADER = (
    '"last_name, first_name","player_id","year","pa","k_percent","bb_percent","woba","player_age"'
)


def test_parse_columns_and_units():
    df = parse_season_csv(CSV.lstrip("\ufeff"), 2024)
    assert list(df.columns) == COLUMNS
    # 0-PA row dropped, sorted by id
    assert df["player_id"].tolist() == [222222, 592450, 660271]
    judge = df.loc[df["player_id"] == 592450].iloc[0]
    assert judge["name"] == "Judge, Aaron"
    assert (judge["season"], judge["age"], judge["pa"]) == (2024, 32, 704)
    assert judge["k_pct"] == pytest.approx(0.249)
    assert judge["bb_pct"] == pytest.approx(0.189)
    assert judge["woba"] == pytest.approx(0.476)


def test_parse_contact_quality_units():
    judge = parse_season_csv(CSV, 2024).set_index("player_id").loc[592450]
    assert judge["xwoba"] == pytest.approx(0.467)  # rate, left alone
    assert judge["avg_ev"] == pytest.approx(95.6)  # mph, left alone
    assert judge["barrel_pct"] == pytest.approx(0.269)  # percent -> fraction
    assert judge["hard_hit_pct"] == pytest.approx(0.610)


def test_parse_keeps_absent_contact_quality_missing():
    row = parse_season_csv(CSV, 2024).set_index("player_id").loc[222222]
    for col in ("avg_ev", "barrel_pct", "hard_hit_pct"):
        assert np.isnan(row[col]), f"{col} must stay missing, never 0"
    assert row["pa"] == 3  # the hitter is kept; only the untracked fields are missing


def test_parse_rejects_missing_column():
    bad = CSV.replace('"exit_velocity_avg"', '"spam"')
    with pytest.raises(ValueError, match="missing columns"):
        parse_season_csv(bad, 2024)


def test_parse_rejects_wrong_season():
    with pytest.raises(ValueError, match="other seasons"):
        parse_season_csv(CSV, 2023)


def test_parse_rejects_duplicate_players():
    dup = CSV + '"Judge, Aaron",592450,2024,32,10,20.0,10.0,0.300,0.300,90.0,10.0,40.0\n'
    with pytest.raises(ValueError, match="duplicate"):
        parse_season_csv(dup, 2024)


def test_header_columns_spots_a_stale_cache():
    assert all(c in header_columns(HEADER) for c in RENAME)
    stale = [c for c in RENAME if c not in header_columns(OLD_HEADER)]
    assert stale == ["xwoba", "exit_velocity_avg", "barrel_batted_rate", "hard_hit_percent"]


def _season(season, n=450, woba=0.315, **overrides):
    data = {
        "player_id": range(n),
        "name": "x",
        "season": season,
        "age": 27,
        "pa": 300,
        "woba": woba,
        "xwoba": woba,
        "k_pct": 0.22,
        "bb_pct": 0.08,
        "avg_ev": 88.0,
        "barrel_pct": 0.07,
        "hard_hit_pct": 0.40,
    }
    return pd.DataFrame({**data, **overrides})


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


def test_validate_flags_contact_quality_problems():
    bad = _season(2024, barrel_pct=7.0, hard_hit_pct=40.0)  # percentages, not fractions
    text = " | ".join(validate(bad))
    assert "barrel_pct outside [0, 1]" in text
    assert "hard_hit_pct outside [0, 1]" in text


def test_validate_flags_implausible_exit_velocity():
    assert "avg_ev outside [20, 115]" in " | ".join(validate(_season(2024, avg_ev=130.0)))
    # In range overall but impossible for a regular.
    assert "100+ PA hitter" in " | ".join(validate(_season(2024, avg_ev=110.0)))


def test_validate_allows_untracked_contact_quality_for_part_timers():
    df = pd.concat([_season(2024), _season(2024, n=20).assign(player_id=range(500, 520), pa=5)])
    df.loc[df["pa"] == 5, ["avg_ev", "barrel_pct", "hard_hit_pct", "xwoba"]] = np.nan
    assert validate(df) == []


def test_validate_flags_missing_contact_quality_for_regulars():
    df = _season(2024)
    df.loc[: len(df) // 2, "xwoba"] = np.nan
    assert "xwoba" in " | ".join(validate(df))


def test_validate_short_2020_threshold():
    assert validate(_season(2020, n=300)) == []
