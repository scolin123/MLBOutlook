"""Seeded synthetic player-season generator (offline test data)."""

import numpy as np
import pandas as pd

COLUMNS = ["player_id", "season", "age", "pa", "woba", "k_pct", "bb_pct"]


def aging(age: int) -> float:
    if age <= 27:
        return 0.001 * (27 - age)
    return -0.0015 * (age - 27)


def make_seasons(seasons=range(2018, 2027), n_players=300, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seasons = list(seasons)
    first, last = seasons[0], seasons[-1]
    rows = []
    for pid in range(1, n_players + 1):
        talent = rng.normal(0.320, 0.025)
        debut = int(rng.choice(seasons))
        length = int(rng.integers(3, 10))
        debut_age = int(rng.integers(21, 29))
        for i in range(length):
            season = debut + i
            if season > last or season < first:
                break
            age = debut_age + i
            pa = int(rng.integers(80, 651))
            woba = talent + aging(age) + rng.normal(0, 0.045 * np.sqrt(300 / pa))
            k_pct = 0.22 + 0.5 * (0.320 - talent) + rng.normal(0, 0.02)
            bb_pct = 0.08 + 0.4 * (talent - 0.320) + rng.normal(0, 0.01)
            rows.append((pid, season, age, pa, woba, k_pct, bb_pct))
    df = pd.DataFrame(rows, columns=COLUMNS)
    for col in ("k_pct", "bb_pct"):
        df.loc[rng.random(len(df)) < 0.03, col] = np.nan
    return df
