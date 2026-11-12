"""Historical comps: the as-of rule first, then the mechanics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_outlook.comps import COMP_FEATURES, comp_pool, comps_summary, find_comps
from mlb_outlook.comps.neighbors import league_relative

QUERY_SEASON = 2024


@pytest.fixture
def query(seasons_df):
    """A player with a complete 2024 row and some history."""
    rows = seasons_df.loc[seasons_df["season"] == QUERY_SEASON].dropna()
    return int(rows["player_id"].iloc[0])


# ---- the as-of rule ----


def test_no_comp_or_outcome_comes_from_after_the_query_season(seasons_df, query):
    """The as-of rule: comps are strictly earlier, outcomes reach the query season at most."""
    comps = find_comps(seasons_df, query, QUERY_SEASON, k=25)
    assert len(comps) == 25
    assert (comps["season"] <= QUERY_SEASON - 1).all()
    assert (comps["next_season"] <= QUERY_SEASON).all()


def test_the_most_recent_comp_year_is_available(seasons_df, query):
    """A comp from `season - 1` is allowed: its outcome lands in the query season, which is
    complete at the forecast date. The pool reaches that far, and the outcome comes with it."""
    pool = comp_pool(seasons_df, QUERY_SEASON)
    recent = pool.loc[pool["season"] == QUERY_SEASON - 1]
    assert len(recent) > 0
    assert (recent["next_season"] == QUERY_SEASON).all()
    # Their outcomes are populated, not left as the did-not-play fallback.
    assert (recent["next_pa"] > 0).any()
    assert recent["next_woba"].notna().any()


def test_later_seasons_cannot_change_the_answer(seasons_df, query):
    before = find_comps(seasons_df, query, QUERY_SEASON, k=15)
    poisoned = seasons_df.copy()
    later = poisoned["season"] > QUERY_SEASON
    for col in ("pa", "woba", "xwoba", "k_pct", "bb_pct", "avg_ev", "barrel_pct", "hard_hit_pct"):
        poisoned.loc[later, col] = poisoned.loc[later, col] * 7 + 3
    pd.testing.assert_frame_equal(before, find_comps(poisoned, query, QUERY_SEASON, k=15))
    dropped = seasons_df.loc[~later]
    pd.testing.assert_frame_equal(before, find_comps(dropped, query, QUERY_SEASON, k=15))


def test_the_query_season_is_read_only_through_relative_position(seasons_df, query):
    """Shift every query-season rate by a constant: the league means move with it, so nothing
    league-relative changes. Which comps are chosen, in what order, at what distance, and what
    their outcomes were *relative to the league* must all be identical. Only the raw wOBA of an
    outcome that falls in the query season moves, because the whole season moved."""
    before = find_comps(seasons_df, query, QUERY_SEASON, k=15)
    shifted = seasons_df.copy()
    same = shifted["season"] == QUERY_SEASON
    for col in ("woba", "xwoba", "k_pct", "bb_pct", "avg_ev", "barrel_pct", "hard_hit_pct"):
        shifted.loc[same, col] += 0.05
    after = find_comps(shifted, query, QUERY_SEASON, k=15)
    invariant = [c for c in before.columns if c != "next_woba"]
    pd.testing.assert_frame_equal(
        before[invariant], after[invariant], check_exact=False, rtol=1e-9, atol=1e-12
    )
    # The one column that may move, and only for outcomes inside the query season.
    moved = before["next_woba"] != after["next_woba"]
    assert (before.loc[moved, "next_season"] == QUERY_SEASON).all()
    # And a comp pool built for the query season reads nothing after it.
    pool = comp_pool(seasons_df, QUERY_SEASON)
    assert pool["season"].max() <= QUERY_SEASON - 1
    assert pool["next_season"].max() <= QUERY_SEASON


def test_the_query_player_is_never_his_own_comp(seasons_df, query):
    comps = find_comps(seasons_df, query, QUERY_SEASON, k=200)
    assert query not in set(comps["player_id"])


# ---- mechanics ----


def test_an_identical_earlier_season_is_the_nearest_comp(seasons_df, query):
    df = seasons_df.copy()
    q = df.loc[(df["player_id"] == query) & (df["season"] == QUERY_SEASON)].iloc[0]
    twin_id = int(df["player_id"].max()) + 1
    twin = q.copy()
    twin["player_id"] = twin_id
    twin["season"] = 2020 + 1  # 2021: an eligible comp season, next season 2022
    # Make the twin's league-relative line match the query's exactly.
    rel = league_relative(df.loc[df["season"].isin([QUERY_SEASON, 2021])])
    shift = {
        c: float(
            rel.loc[(rel["player_id"] == query) & (rel["season"] == QUERY_SEASON), f"{c}_rel"]
            .iloc[0]
        )
        for c in ("woba", "xwoba", "k_pct", "bb_pct", "avg_ev", "barrel_pct", "hard_hit_pct")
    }
    league_2021 = {
        c: float(np.average(df.loc[df["season"] == 2021, c].dropna(),
                            weights=df.loc[(df["season"] == 2021) & df[c].notna(), "pa"]))
        for c in shift
    }
    for c, value in shift.items():
        twin[c] = league_2021[c] + value
    # The pool floor; one 100-PA row moves the 2021 league mean by a negligible amount.
    twin["pa"] = 100
    nxt = twin.copy()
    nxt["season"] = 2022
    df = pd.concat([df, pd.DataFrame([twin, nxt])], ignore_index=True)
    comps = find_comps(df, query, QUERY_SEASON, k=3)
    assert comps["player_id"].iloc[0] == twin_id
    assert comps["season"].iloc[0] == 2021
    assert comps["next_season"].iloc[0] == 2022
    assert comps["distance"].iloc[0] < comps["distance"].iloc[1]


def test_comps_are_sorted_and_k_is_respected(seasons_df, query):
    comps = find_comps(seasons_df, query, QUERY_SEASON, k=7)
    assert len(comps) == 7
    assert comps["distance"].is_monotonic_increasing


def test_a_comp_who_did_not_play_next_year_is_kept_with_zero_pa(seasons_df, query):
    pool = comp_pool(seasons_df, QUERY_SEASON)
    gone = pool.loc[pool["next_pa"] == 0]
    assert len(gone) > 0
    assert gone["next_woba"].isna().all()


def test_short_season_and_low_pa_seasons_are_not_comps(seasons_df):
    pool = comp_pool(seasons_df, QUERY_SEASON)
    assert 2020 not in set(pool["season"])
    assert (pool["pa"] >= 100).all()
    assert pool[COMP_FEATURES].notna().all().all()


def test_a_query_missing_a_feature_is_refused(seasons_df, query):
    df = seasons_df.copy()
    df.loc[(df["player_id"] == query) & (df["season"] == QUERY_SEASON), "avg_ev"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        find_comps(df, query, QUERY_SEASON)


def test_an_unknown_query_raises(seasons_df):
    with pytest.raises(KeyError):
        find_comps(seasons_df, -1, QUERY_SEASON)


def test_league_relative_centres_each_season(seasons_df):
    rel = league_relative(seasons_df)
    for _, rows in rel.groupby("season"):
        ok = rows["woba_rel"].notna()
        assert np.average(rows.loc[ok, "woba_rel"], weights=rows.loc[ok, "pa"]) == pytest.approx(
            0, abs=1e-12
        )


def test_summary_counts_only_comps_who_played(seasons_df, query):
    comps = find_comps(seasons_df, query, QUERY_SEASON, k=20)
    summary = comps_summary(comps)
    assert summary["n"] == 20
    assert summary["n_played_next"] == int((comps["next_pa"] >= 100).sum())
