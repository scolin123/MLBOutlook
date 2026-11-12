"""Player page: history, the 2027 forecast, probabilities, the Marcel gap, and comps."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from components import cached, charts, data  # noqa: E402
from components.style import card, pct, setup, signed, woba  # noqa: E402

setup("Player", data.FORECAST_LABEL)
cached.require_files()

forecasts = cached.forecasts().sort_values("name")
df = cached.seasons()

ids = forecasts["player_id"].tolist()
names = dict(zip(forecasts["player_id"], forecasts["name"], strict=True))
default = st.query_params.get("player")
index = ids.index(int(default)) if default and default.isdigit() and int(default) in ids else 0
player_id = st.selectbox("Player", ids, index=index, format_func=lambda i: names[i])
st.query_params["player"] = str(player_id)

row = forecasts.loc[forecasts["player_id"] == player_id].iloc[0]
hist = data.history(df, player_id)

st.title(row["name"])
st.caption(
    f"Age {int(row['age_target'])} in {data.FORECAST_SEASON} | {int(row['pa_lag1'])} PA in "
    f"{data.FORECAST_FEATURES_THROUGH}"
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    card(f"{data.FORECAST_SEASON} forecast", woba(row["ridge_rel"]),
         f"80% interval {woba(row['lo'])} - {woba(row['hi'])}")
with c2:
    card("vs Marcel", signed(row["model_minus_marcel"]), f"Marcel {woba(row['marcel'])}")
with c3:
    card("Breakout / regression", f"{pct(row['p_breakout'])} / {pct(row['p_regression'])}",
         f"vs 3-yr weighted baseline {woba(row['weighted3'])} +/- .030")
with c4:
    # Where the outcome card used to be. There is no 2027 outcome, and the nearest honest
    # thing is what hitters who looked like this one went on to do.
    card("Comps' next year",
         f"{int(row['comps_played_next'])} of {int(row['comps_n'])} played",
         f"median change {signed(row['comps_median_delta_woba_rel'])} (league-relative)")

left, right = st.columns([3, 2])
with left:
    st.plotly_chart(charts.history_chart(hist, row), use_container_width=True)
    st.markdown("**Probability of each outcome**")
    st.plotly_chart(charts.probability_bar(row), use_container_width=True,
                    config={"displayModeBar": False})
    st.caption(
        f"Breakout = {data.FORECAST_SEASON} wOBA at least .030 above the weighted three-year "
        "baseline; regression = at least .030 below. Read off the same distribution as the "
        "interval, so they cannot contradict it."
    )
with right:
    pcts = data.percentiles(df, player_id)
    if pcts.empty:
        st.info(f"No {data.FORECAST_FEATURES_THROUGH} percentile data (fewer than 100 PA).")
    else:
        st.plotly_chart(charts.percentile_chart(pcts), use_container_width=True,
                        config={"displayModeBar": False})

st.subheader("Season lines")
table = hist.loc[hist["season"] <= data.FORECAST_FEATURES_THROUGH,
                 ["season", "age", "pa", "woba", "xwoba", "k_pct", "bb_pct", "avg_ev",
                  "barrel_pct", "hard_hit_pct"]]
st.dataframe(
    table.sort_values("season", ascending=False),
    hide_index=True,
    use_container_width=True,
    column_config={
        "season": st.column_config.NumberColumn(format="%d"),
        "woba": st.column_config.NumberColumn("wOBA", format="%.3f"),
        "xwoba": st.column_config.NumberColumn("xwOBA", format="%.3f"),
        "k_pct": st.column_config.NumberColumn("K%", format="%.3f"),
        "bb_pct": st.column_config.NumberColumn("BB%", format="%.3f"),
        "avg_ev": st.column_config.NumberColumn("Avg EV", format="%.1f"),
        "barrel_pct": st.column_config.NumberColumn("Barrel%", format="%.3f"),
        "hard_hit_pct": st.column_config.NumberColumn("Hard-hit%", format="%.3f"),
    },
)
st.caption(
    f"Every season the {data.FORECAST_SEASON} forecast was allowed to read, and no later one."
)

st.subheader("Historical comps")
st.caption(
    f"The 10 nearest player-seasons to this hitter's {data.FORECAST_FEATURES_THROUGH}, on age "
    "and league-relative wOBA, xwOBA, K%, BB% and contact quality. Comp seasons are before "
    f"{data.FORECAST_FEATURES_THROUGH} and their following years run through it, so all of it "
    "was knowable when the forecast was made."
)
found = cached.comps(player_id)
if found is None:
    st.info(
        f"No comps: the {data.FORECAST_FEATURES_THROUGH} line is missing a comp feature."
    )
else:
    comps, summary = found
    st.markdown(
        f"{summary['n_played_next']} of {summary['n']} comps had 100+ PA the next year; their "
        f"median league-relative wOBA the next year was "
        f"**{signed(summary['median_next_woba_rel'])}**, a median change of "
        f"**{signed(summary['median_delta_woba_rel'])}**."
    )
    st.dataframe(
        comps[["name", "season", "age", "pa", "woba", "xwoba", "distance", "next_pa",
               "next_woba", "delta_woba_rel"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "season": st.column_config.NumberColumn(format="%d"),
            "woba": st.column_config.NumberColumn("wOBA", format="%.3f"),
            "xwoba": st.column_config.NumberColumn("xwOBA", format="%.3f"),
            "distance": st.column_config.NumberColumn(format="%.2f"),
            "next_pa": st.column_config.NumberColumn("next PA"),
            "next_woba": st.column_config.NumberColumn("next wOBA", format="%.3f"),
            "delta_woba_rel": st.column_config.NumberColumn("change (rel.)", format="%+.3f"),
        },
    )
