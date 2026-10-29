import pandas as pd
import pytest

from mlb_outlook.models.marcel import age_factor, marcel_rate, project_marcel

LG = 0.320
COLS = ["player_id", "season", "age", "pa", "rate"]
THREE = [(600, 0.400), (500, 0.360), (400, 0.320)]
# 5*600*.400 + 4*500*.360 + 3*400*.320 = 1200 + 720 + 384 = 2304
# weighted PA = 3000 + 2000 + 1200 = 6200
# regressed = (2304 + 1200*.320) / (6200 + 1200) = 2688 / 7400 = 0.363243
REGRESSED = 2688 / 7400


def approx(x):
    return pytest.approx(x, abs=1e-6)


def test_three_seasons_age_29():
    assert marcel_rate(THREE, 29, LG) == approx(0.363243)


def test_three_seasons_age_25():
    # 0.363243 * (1 + 4*.006) = 0.363243 * 1.024
    assert marcel_rate(THREE, 25, LG) == approx(0.371961)


def test_three_seasons_age_33():
    # 0.363243 * (1 - 4*.003) = 0.363243 * 0.988
    assert marcel_rate(THREE, 33, LG) == approx(0.358884)


def test_rookie_one_season():
    # (5*300*.350 + 1200*.320) / (1500 + 1200) = (525 + 384) / 2700 = 0.336667; * 1.03 (age 24)
    assert marcel_rate([(300, 0.350)], 24, LG) == approx(0.346767)


def test_no_history():
    # lg * (1 + 2*.006) = .320 * 1.012
    assert marcel_rate([], 27, LG) == approx(0.32384)


def test_custom_parameters():
    # weights (1,), K=100: (1*100*.400 + 100*.320) / (100 + 100) = 72/200 = 0.36
    assert marcel_rate([(100, 0.400)], 29, LG, weights=(1,), regression_pa=100) == approx(0.36)


@pytest.mark.parametrize(
    ("age", "expected"),
    [(20, 1.054), (28, 1.006), (29, 1.0), (30, 0.997), (35, 0.982)],
)
def test_age_factor_boundaries(age, expected):
    assert age_factor(age) == approx(expected)


def _frame():
    # Player 1: seasons 2022-2024 as THREE (oldest last), age 28 in 2024 -> 29 in 2025.
    # Player 2: rookie 2024 only, age 23 in 2024 -> 24 in 2025.
    # Player 3: missing oldest season (2022): 2024 (600, .400), 2023 (500, .360);
    # age 29 in 2024 -> 30.
    return pd.DataFrame(
        [
            (1, 2024, 28, 600, 0.400),
            (1, 2023, 27, 500, 0.360),
            (1, 2022, 26, 400, 0.320),
            (2, 2024, 23, 300, 0.350),
            (3, 2024, 29, 600, 0.400),
            (3, 2023, 28, 500, 0.360),
        ],
        columns=COLS,
    )


LEAGUE = {2022: LG, 2023: LG, 2024: LG}


def test_wrapper_two_players_and_missing_oldest_season():
    out = project_marcel(_frame(), 2025, LEAGUE).set_index("player_id")
    assert list(out["target_season"]) == [2025, 2025, 2025]
    assert out.loc[1, "marcel_rate"] == approx(0.363243)  # age 29 -> factor 1
    assert out.loc[2, "marcel_rate"] == approx(0.346767)
    # P3: (5*600*.4 + 4*500*.36 + 384) / (3000 + 2000 + 1200)
    # = (1200 + 720 + 384) / 6200 = 2304/6200 = 0.371613; age 30 -> * 0.997 = 0.370498
    assert out.loc[3, "marcel_rate"] == approx(2304 / 6200 * 0.997)


def test_wrapper_ignores_future_seasons():
    base = project_marcel(_frame(), 2025, LEAGUE)
    future = pd.DataFrame(
        [(1, 2025, 29, 700, 9.0), (1, 2026, 30, 700, 9.0), (2, 2025, 24, 700, 9.0)],
        columns=COLS,
    )
    both = pd.concat([_frame(), future], ignore_index=True)
    out = project_marcel(both, 2025, {**LEAGUE, 2025: 9.0, 2026: 9.0})
    pd.testing.assert_frame_equal(out, base)


def test_wrapper_league_rate_weighted_and_renormalised():
    # Only 2024 (w=5) and 2022 (w=3) have league rates: lg = (5*.330 + 3*.310) / 8 = 0.3225
    out = project_marcel(
        pd.DataFrame([(1, 2024, 28, 0, 0.0)], columns=COLS),
        2025,
        {2024: 0.330, 2022: 0.310},
    )
    assert out.loc[0, "marcel_rate"] == approx(0.3225)  # no PA -> lg; age 29 in 2025


def test_wrapper_age_projected_from_last_observed_season():
    # Last observed 2021 (age 24) for target 2025 -> age 28 (factor 1.006).
    # 2021 is outside the 2022-2024 lookback, so no PA is used: result is lg * 1.006.
    df = pd.DataFrame([(1, 2021, 24, 500, 0.400)], columns=COLS)
    out = project_marcel(df, 2025, {2022: LG, 2023: LG, 2024: LG})
    assert out.loc[0, "marcel_rate"] == approx(LG * (1 + 1 * 0.006))
