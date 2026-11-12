"""DiamondIQ app entry point: `uv run streamlit run app/Home.py` from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from components import data  # noqa: E402
from components.style import card, setup, signed, woba  # noqa: E402

setup("Home", data.FORECAST_LABEL)
st.title("DiamondIQ")
st.caption(
    f"MLB hitter forecasting: next-season wOBA with calibrated uncertainty. "
    f"Showing the {data.FORECAST_SEASON} forecasts, made with data through "
    f"{data.FORECAST_FEATURES_THROUGH}."
)

try:
    data.check_files()
except data.MissingArtifacts as exc:
    st.error(str(exc))
    st.stop()

forecasts = data.forecasts()
verdict = data.artifact("verdict")

c1, c2, c3, c4 = st.columns(4)
with c1:
    card("Players forecast", f"{len(forecasts)}",
         f"{data.FORECAST_SEASON}, 100+ PA in {data.FORECAST_FEATURES_THROUGH}")
with c2:
    card("Mean gap to Marcel", signed(forecasts["model_minus_marcel"].mean()),
         f"model minus Marcel, across all {len(forecasts)}")
if data.holdout_available():
    # The holdout verdict is the headline number in this project, pass or fail, and it belongs
    # next to the forecasts: it is the only honest measurement of how good they are likely be.
    holdout = data.holdout_artifact("verdict")
    point3 = data.holdout_artifact("point").set_index("model")
    met = int(holdout["passed"].astype(bool).sum())
    with c3:
        card("Fold 3 holdout verdict", data.verdict_label(holdout),
             f"{met} of {len(holdout)} pre-registered criteria met")
    with c4:
        card("Holdout MAE vs Marcel",
             f"{woba(point3.loc['ridge_rel', 'mae'])} / {woba(point3.loc['marcel', 'mae'])}",
             f"fold 3 ({data.HOLDOUT_TARGET_SEASON}), the one untouched season")
else:
    with c3:
        passed = int(verdict["passed"].astype(bool).sum())
        card("Fold 2 criteria met", f"{passed} / {len(verdict)}", "pre-registered")
    with c4:
        card("Fold 3 holdout", "not run", "no honest test of these forecasts yet")

st.markdown(
    f"""
### What's here

- **Player** -- one hitter: wOBA/xwOBA history through {data.FORECAST_FEATURES_THROUGH}, the
  {data.FORECAST_SEASON} forecast with its 80% interval, breakout / stable / regression
  probabilities, the gap to Marcel, and historical comps with what they did next.
- **Projection disagreements** -- every forecast player, sortable by where the model and
  Marcel disagree most, by breakout or regression probability, or by interval width.
- **Model performance** -- point error against the baselines, interval coverage, the
  probability scores and their reliability, for the fold 2 backtest ({data.TARGET_SEASON}) and
  for the fold 3 ({data.HOLDOUT_TARGET_SEASON}) holdout, with the pre-registered verdict.

### How to read it

The forecasts are for **{data.FORECAST_SEASON}**, built from seasons through
**{data.FORECAST_FEATURES_THROUGH}** by the configuration the `{data.HOLDOUT_TAG}` tag froze.
{data.FORECAST_SEASON} has not been played, so there is no outcome column anywhere on the
Player page or the board, and nothing here can have been fitted to the season it forecasts.
They were committed to the repository and the database before the answer existed, which is
what makes them a test rather than a demonstration.

A projection disagreement is a statement about two forecasts, not a claim that a player is
mispriced. Breakout and regression are defined against a weighted three-year baseline, plus or
minus .030 wOBA.

**The model failed its holdout.** On fold 3 -- {data.HOLDOUT_TARGET_SEASON}, the one season
nothing was fitted to or chosen on -- it beat Marcel's MAE, but the 95% interval on that
difference contains zero, so one of the three pre-registered criteria was not met and the
verdict is FAIL. The model is **not shown to beat Marcel**. Read every forecast here with
that in mind: Model performance has the numbers, and nothing was re-fitted after they were
seen.
"""
)
info = data.forecast_manifest()
if info:
    st.caption(
        f"Model {info.get('model_version')} (frozen at `{info.get('freeze_tag')}`) | "
        f"data sha256 {info.get('data_sha256', '')[:12]}... "
        f"({info.get('data_rows')} rows) | forecasts generated {info.get('generated_at')}"
    )
