"""Projection disagreements: where the model and Marcel part ways on 2027, and the tails."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from components import cached, charts, data  # noqa: E402
from components.style import setup  # noqa: E402

SORTS = {
    "Model above Marcel": ("model_minus_marcel", False),
    "Model below Marcel": ("model_minus_marcel", True),
    "Breakout probability": ("p_breakout", False),
    "Regression probability": ("p_regression", False),
    "Widest interval": ("width", False),
    "Narrowest interval": ("width", True),
    "Comps improved most": ("comps_median_delta_woba_rel", False),
    "Comps declined most": ("comps_median_delta_woba_rel", True),
}

setup("Projection disagreements", data.FORECAST_LABEL)
cached.require_files()
st.title("Projection disagreements")
st.caption(
    f"Where this model's {data.FORECAST_SEASON} forecast and Marcel's differ, and who sits in "
    "the probability tails. A disagreement is between two forecasts; it is not a valuation, "
    "and neither side has been checked against an outcome, because there is not one yet."
)

board = cached.forecasts()

c1, c2 = st.columns([2, 2])
with c1:
    sort = st.selectbox("Sort by", list(SORTS))
with c2:
    min_pa = st.slider(f"Min {data.FORECAST_FEATURES_THROUGH} PA", 100, 600, 100, step=50)

column, ascending = SORTS[sort]
view = board.loc[board["pa_lag1"] >= min_pa].sort_values(column, ascending=ascending)

columns = ["name", "age_target", "pa_lag1", "ridge_rel", "marcel", "model_minus_marcel",
           "lo", "hi", "width", "p_breakout", "p_stable", "p_regression",
           "comps_played_next", "comps_median_delta_woba_rel"]
st.dataframe(
    view[columns],
    hide_index=True,
    use_container_width=True,
    height=520,
    column_config={
        "name": "Player",
        "age_target": st.column_config.NumberColumn(f"Age {data.FORECAST_SEASON}"),
        "pa_lag1": st.column_config.NumberColumn(f"PA {data.FORECAST_FEATURES_THROUGH}"),
        "ridge_rel": st.column_config.NumberColumn("Model", format="%.3f"),
        "marcel": st.column_config.NumberColumn("Marcel", format="%.3f"),
        "model_minus_marcel": st.column_config.NumberColumn("Model - Marcel", format="%+.3f"),
        "lo": st.column_config.NumberColumn("80% low", format="%.3f"),
        "hi": st.column_config.NumberColumn("80% high", format="%.3f"),
        "width": st.column_config.NumberColumn("Width", format="%.3f"),
        "p_breakout": st.column_config.ProgressColumn("Breakout", format="%.2f",
                                                      min_value=0, max_value=1),
        "p_stable": st.column_config.NumberColumn("Stable", format="%.2f"),
        "p_regression": st.column_config.ProgressColumn("Regression", format="%.2f",
                                                        min_value=0, max_value=1),
        "comps_played_next": st.column_config.NumberColumn("Comps played /10"),
        "comps_median_delta_woba_rel": st.column_config.NumberColumn(
            "Comps' median change", format="%+.3f"
        ),
    },
)
st.caption(
    f"{len(view)} players, all forecast for {data.FORECAST_SEASON} from data through "
    f"{data.FORECAST_FEATURES_THROUGH}. The last two columns are what the 10 nearest "
    "historical comps did the year after the season they resemble. Open one on the Player "
    "page for the full profile."
)
st.plotly_chart(charts.disagreement_scatter(view), use_container_width=True)
