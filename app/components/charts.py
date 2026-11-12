"""Plotly figure builders. Pure functions of data frames, so they can be tested headless."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from components.data import FEATURES_THROUGH, TARGET_SEASON
from components.style import BLUE, FORECAST, GREY, NAVY, RED, percentile_color

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Helvetica, Arial, sans-serif", size=12, color="#333"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    hoverlabel=dict(bgcolor="white"),
)


def history_chart(history: pd.DataFrame, forecast: pd.Series) -> go.Figure:
    """wOBA and xwOBA through the last feature season, the forecast with its 80% interval,
    and the target season's actual wOBA marked as hindsight."""
    seen = history.loc[history["season"] <= FEATURES_THROUGH]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seen["season"], y=seen["woba"], name="wOBA",
                             mode="lines+markers", line=dict(color=NAVY, width=2.5)))
    fig.add_trace(go.Scatter(x=seen["season"], y=seen["xwoba"], name="xwOBA",
                             mode="lines+markers", line=dict(color=GREY, dash="dot")))
    fig.add_trace(go.Scatter(
        x=[TARGET_SEASON], y=[forecast["ridge_rel"]], name=f"{TARGET_SEASON} forecast (80%)",
        mode="markers", marker=dict(color=FORECAST, size=13, symbol="diamond"),
        error_y=dict(type="data", symmetric=False,
                     array=[forecast["hi"] - forecast["ridge_rel"]],
                     arrayminus=[forecast["ridge_rel"] - forecast["lo"]],
                     color=FORECAST, thickness=2, width=8),
    ))
    fig.add_trace(go.Scatter(
        x=[TARGET_SEASON], y=[forecast["marcel"]], name="Marcel",
        mode="markers", marker=dict(color=BLUE, size=10, symbol="line-ew-open",
                                    line=dict(width=3)),
    ))
    fig.add_trace(go.Scatter(
        x=[TARGET_SEASON], y=[forecast["y_woba"]], name=f"{TARGET_SEASON} actual (hindsight)",
        mode="markers", marker=dict(color=RED, size=10, symbol="x"),
    ))
    fig.add_vrect(x0=FEATURES_THROUGH + 0.5, x1=TARGET_SEASON + 0.5, fillcolor="#FFF4E8",
                  line_width=0, layer="below")
    fig.update_layout(**LAYOUT, title="wOBA history and the backtest forecast", height=380)
    fig.update_xaxes(dtick=1, title=None)
    fig.update_yaxes(title="wOBA", tickformat=".3f")
    return fig


def probability_bar(row: pd.Series) -> go.Figure:
    parts = [("Regression", row["p_regression"], BLUE), ("Stable", row["p_stable"], GREY),
             ("Breakout", row["p_breakout"], RED)]
    fig = go.Figure()
    for label, value, color in parts:
        fig.add_trace(go.Bar(x=[value], y=[""], orientation="h", name=label,
                             marker_color=color, text=f"{label} {value:.0%}",
                             textposition="inside", insidetextanchor="middle",
                             hovertemplate=f"{label}: %{{x:.1%}}<extra></extra>"))
    fig.update_layout(**LAYOUT | dict(margin=dict(l=10, r=10, t=10, b=10)), barmode="stack",
                      height=90, showlegend=False)
    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=False)
    return fig


def percentile_chart(pcts: pd.DataFrame) -> go.Figure:
    pcts = pcts.iloc[::-1]
    colors = [percentile_color(p) for p in pcts["percentile"]]
    fig = go.Figure(go.Bar(
        x=pcts["percentile"], y=pcts["metric"], orientation="h", marker_color=colors,
        text=pcts["percentile"], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x}th percentile<extra></extra>",
    ))
    fig.update_layout(**LAYOUT, height=40 + 34 * len(pcts),
                      title=f"{FEATURES_THROUGH} percentile ranks (100+ PA hitters)")
    fig.update_xaxes(range=[0, 108], showgrid=False, showticklabels=False)
    return fig


def reliability_chart(reliability: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                             line=dict(color=GREY, dash="dash")))
    for label, color in (("breakout", RED), ("regression", BLUE)):
        part = reliability.loc[reliability["label"] == label]
        fig.add_trace(go.Scatter(
            x=part["mean_predicted"], y=part["observed"], mode="lines+markers",
            name=label, line=dict(color=color), customdata=part[["decile", "n"]],
            hovertemplate="decile %{customdata[0]} (n=%{customdata[1]})<br>"
                          "predicted %{x:.3f}<br>observed %{y:.3f}<extra></extra>",
        ))
    top = float(max(reliability["mean_predicted"].max(), reliability["observed"].max())) + 0.05
    fig.update_layout(**LAYOUT, title="Reliability by decile (registered 0.030 threshold)",
                      height=420)
    fig.update_xaxes(title="mean predicted probability", range=[0, top])
    fig.update_yaxes(title="observed rate", range=[0, top])
    return fig


def coverage_chart(coverage: pd.DataFrame, band: tuple[float, float] = (0.75, 0.85)) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=coverage["range"], y=coverage["coverage"],
        marker_color=[NAVY if g == "pa_lag1" else GREY for g in coverage["grouping"]],
        text=[f"{c:.1%}<br>n={n}" for c, n in zip(coverage["coverage"], coverage["n"],
                                                   strict=True)],
        textposition="outside",
    ))
    fig.add_hrect(y0=band[0], y1=band[1], fillcolor="#E8F3E8", line_width=0, layer="below")
    fig.add_hline(y=0.80, line=dict(color=GREY, dash="dot"))
    fig.update_layout(**LAYOUT, title="80% interval coverage by slice (target band 75-85%)",
                      height=380)
    fig.update_yaxes(range=[0.5, 1.0], tickformat=".0%")
    return fig


def disagreement_scatter(board: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=board["marcel"], y=board["ridge_rel"], mode="markers",
        marker=dict(color=board["model_minus_marcel"], colorscale=[[0, BLUE], [0.5, "#D6D9DF"],
                                                                    [1, RED]],
                    cmid=0, size=7, line=dict(width=0.5, color="white"),
                    colorbar=dict(title="model - Marcel")),
        text=board["name"],
        hovertemplate="%{text}<br>Marcel %{x:.3f}<br>model %{y:.3f}<extra></extra>",
    ))
    lo = float(min(board["marcel"].min(), board["ridge_rel"].min())) - 0.005
    hi = float(max(board["marcel"].max(), board["ridge_rel"].max())) + 0.005
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                             line=dict(color=GREY, dash="dash"), hoverinfo="skip"))
    fig.update_layout(**LAYOUT, title="Model forecast vs Marcel", height=420, showlegend=False)
    fig.update_xaxes(title="Marcel wOBA", tickformat=".3f")
    fig.update_yaxes(title="model wOBA", tickformat=".3f")
    return fig
