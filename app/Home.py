"""DiamondIQ app entry point: `uv run streamlit run app/Home.py` from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from components import data  # noqa: E402
from components.style import card, setup, woba  # noqa: E402

setup("Home")
st.title("DiamondIQ")
st.caption("MLB hitter forecasting: next-season wOBA with calibrated uncertainty.")

try:
    data.check_files()
except data.MissingArtifacts as exc:
    st.error(str(exc))
    st.stop()

forecasts = data.forecasts()
point = data.artifact("point").set_index("model")
verdict = data.artifact("verdict")

c1, c2, c3, c4 = st.columns(4)
with c1:
    card("Players forecast", f"{len(forecasts)}", f"{data.TARGET_SEASON}, 100+ PA in both years")
with c2:
    card("Model MAE", woba(point.loc["ridge_rel", "mae"]),
         f"Marcel {woba(point.loc['marcel', 'mae'])}")
with c3:
    card("80% interval coverage", f"{point.loc['resid_q_pa', 'coverage']:.1%}", "target 75-85%")
with c4:
    passed = int(verdict["passed"].astype(bool).sum())
    card("Fold 2 criteria met", f"{passed} / {len(verdict)}", "pre-registered")

st.markdown(
    f"""
### What's here

- **Player** -- one hitter: wOBA/xwOBA history, the {data.TARGET_SEASON} forecast with its 80%
  interval, breakout / stable / regression probabilities, the gap to Marcel, and historical
  comps with what they did next.
- **Projection disagreements** -- every forecast player, sortable by where the model and
  Marcel disagree most, by breakout or regression probability, or by interval width.
- **Model performance** -- the fold 2 tables: point error against the baselines, interval
  coverage, the probability scores and their reliability.

### How to read it

Everything is the **fold 2 backtest**: the model was fitted on seasons before
{data.TARGET_SEASON}, given data through {data.FEATURES_THROUGH}, and its forecasts are shown
next to what happened in {data.TARGET_SEASON}. A projection disagreement is a statement about
two forecasts, not a claim that a player is mispriced. Breakout and regression are defined
against a weighted three-year baseline, plus or minus .030 wOBA.

The final holdout season (2026) is not shown anywhere in this app.
"""
)
info = data.manifest()
if info:
    st.caption(
        f"Model {info.get('model_version')} | data sha256 {info.get('data_sha256', '')[:12]}... "
        f"({info.get('data_rows')} rows) | generated {info.get('generated_at')}"
    )
