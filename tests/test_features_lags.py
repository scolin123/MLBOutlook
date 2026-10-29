"""Hand-checked tests for `build_features` on a tiny fixture.

Fixture (player, season: age, pa, woba, k_pct, bb_pct):
  P1 2021: 25, 500, .300, .20, .08   2022: 26, 400, .320, .22, .09
     2023: 27, 600, .340, .18, .10   2024: 28, 550, .350, .19, .11
  P2 2022: 29, 200, .300, .24, .06   2023: 30, 300, .280, .25, .07
     2024: 31,  50, .250, .26, .05   (target pa below 100)
  P3 2023: 22,  80, .400, .15, .12   2024: 23, 400, .310, .16, .13
     (feature pa below 100)
"""

import numpy as np
import pandas as pd
import pytest

from mlb_outlook.features.asof import build_features, feature_columns

COLS = ["player_id", "season", "age", "pa", "woba", "k_pct", "bb_pct"]
ROWS = [
    (1, 2021, 25, 500, 0.300, 0.20, 0.08),
    (1, 2022, 26, 400, 0.320, 0.22, 0.09),
    (1, 2023, 27, 600, 0.340, 0.18, 0.10),
    (1, 2024, 28, 550, 0.350, 0.19, 0.11),
    (2, 2022, 29, 200, 0.300, 0.24, 0.06),
    (2, 2023, 30, 300, 0.280, 0.25, 0.07),
    (2, 2024, 31, 50, 0.250, 0.26, 0.05),
    (3, 2023, 22, 80, 0.400, 0.15, 0.12),
    (3, 2024, 23, 400, 0.310, 0.16, 0.13),
]


@pytest.fixture
def tiny():
    return pd.DataFrame(ROWS, columns=COLS)


def test_column_names_and_order(tiny):
    out = build_features(tiny, 2024)
    expected = ["player_id", "target_season", "max_source_season", "age_target"]
    for k in (1, 2, 3):
        expected += [
            f"woba_lag{k}",
            f"pa_lag{k}",
            f"k_pct_lag{k}",
            f"bb_pct_lag{k}",
            f"lg_woba_lag{k}",
            f"miss_lag{k}",
        ]
    assert list(out.columns) == expected + ["y_woba", "y_pa"]
    assert feature_columns(3) == expected
    assert list(build_features(tiny, 2024, include_target=False).columns) == expected


def test_lag_alignment_and_filters(tiny):
    out = build_features(tiny, 2024)
    # P2 dropped: target pa 50 < 100. P3 dropped: lag1 pa 80 < 100. Only P1 remains.
    assert out["player_id"].tolist() == [1]
    row = out.iloc[0]
    assert row["target_season"] == 2024
    assert row["age_target"] == 28  # age 27 in 2023, plus 1
    assert (row["woba_lag1"], row["woba_lag2"], row["woba_lag3"]) == (0.340, 0.320, 0.300)
    assert (row["pa_lag1"], row["pa_lag2"], row["pa_lag3"]) == (600, 400, 500)
    assert (row["k_pct_lag1"], row["k_pct_lag2"], row["k_pct_lag3"]) == (0.18, 0.22, 0.20)
    assert (row["bb_pct_lag1"], row["bb_pct_lag2"], row["bb_pct_lag3"]) == (0.10, 0.09, 0.08)
    assert (row["miss_lag1"], row["miss_lag2"], row["miss_lag3"]) == (0, 0, 0)
    assert row["max_source_season"] == 2023
    assert (row["y_woba"], row["y_pa"]) == (0.350, 550)


def test_min_pa_feature_and_target_filters(tiny):
    both_off = build_features(tiny, 2024, min_pa_feature=0, min_pa_target=0)
    assert both_off["player_id"].tolist() == [1, 2, 3]
    only_feature_off = build_features(tiny, 2024, min_pa_feature=0)
    assert only_feature_off["player_id"].tolist() == [1, 3]  # P2 target pa 50
    only_target_off = build_features(tiny, 2024, min_pa_target=0)
    assert only_target_off["player_id"].tolist() == [1, 2]  # P3 lag1 pa 80


def test_league_rate_by_hand(tiny):
    out = build_features(tiny, 2024)
    row = out.iloc[0]
    # 2023: (600*.340 + 300*.280 + 80*.400) / (600 + 300 + 80) = 320 / 980
    assert row["lg_woba_lag1"] == pytest.approx(320 / 980)
    # 2022: (400*.320 + 200*.300) / 600 = 188 / 600
    assert row["lg_woba_lag2"] == pytest.approx(188 / 600)
    # 2021: only P1: 500*.300 / 500
    assert row["lg_woba_lag3"] == pytest.approx(0.300)


def test_league_rate_includes_players_not_in_output(tiny):
    # P3 (80 PA, .400) is filtered out of the rows but still counts in the league rate.
    without_p3 = tiny[tiny["player_id"] != 3]
    with_p3 = build_features(tiny, 2024).iloc[0]["lg_woba_lag1"]
    without = build_features(without_p3, 2024).iloc[0]["lg_woba_lag1"]
    assert without == pytest.approx(288 / 900)  # (204 + 84) / 900
    assert with_p3 != pytest.approx(without)


def test_missing_season_fill(tiny):
    out = build_features(tiny, 2024, include_target=False)
    assert out["player_id"].tolist() == [1, 2]  # P3 excluded by feature pa
    p2 = out.loc[out["player_id"] == 2].iloc[0]
    assert p2["age_target"] == 31  # age 30 in 2023, plus 1
    assert (p2["woba_lag1"], p2["woba_lag2"]) == (0.280, 0.300)
    assert np.isnan(p2["woba_lag3"])
    assert np.isnan(p2["k_pct_lag3"]) and np.isnan(p2["bb_pct_lag3"])
    assert p2["pa_lag3"] == 0
    assert (p2["miss_lag1"], p2["miss_lag2"], p2["miss_lag3"]) == (0, 0, 1)
    # league rate for 2021 is still present even though P2 has no row
    assert p2["lg_woba_lag3"] == pytest.approx(0.300)


def test_league_nan_when_season_has_no_rows(tiny):
    out = build_features(tiny[tiny["season"] >= 2022], 2024, include_target=False)
    assert out["lg_woba_lag3"].isna().all()
    # max_source_season is the latest season used, still < target
    assert (out["max_source_season"] == 2023).all()


def test_include_target_false_keeps_player_without_target_row(tiny):
    # Target 2025 has no rows at all. P1 (550 PA) and P3 (400 PA) qualify; P2 has 50 PA.
    with_target = build_features(tiny, 2025)
    assert len(with_target) == 0
    assert "y_woba" in with_target.columns
    prod = build_features(tiny, 2025, include_target=False)
    assert prod["player_id"].tolist() == [1, 3]
    assert "y_woba" not in prod.columns and "y_pa" not in prod.columns
    assert (prod["target_season"] == 2025).all()
    assert prod.iloc[0]["woba_lag1"] == 0.350
    assert prod.iloc[0]["age_target"] == 29
    assert prod["max_source_season"].tolist() == [2024, 2024]


def test_lags_parameter(tiny):
    out = build_features(tiny, 2024, lags=2)
    assert "woba_lag3" not in out.columns
    assert "woba_lag2" in out.columns
    assert list(out.columns)[: len(feature_columns(2))] == feature_columns(2)


def test_empty_when_target_before_data(tiny):
    out = build_features(tiny, 2019, include_target=False)
    assert len(out) == 0
    assert list(out.columns) == feature_columns(3)
