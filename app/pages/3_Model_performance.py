"""Model performance on fold 2: point error, interval coverage, probability scores."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from components import cached, charts, data  # noqa: E402
from components.style import setup  # noqa: E402

setup("Model performance")
cached.require_files()
st.title("Model performance")
st.caption(
    f"Fold 2: trained on target seasons through {data.FEATURES_THROUGH}, tested on "
    f"{data.TARGET_SEASON}. This is the fold the configuration was selected on, so these "
    "numbers are optimistic; the untouched holdout is fold 3."
)

st.subheader("Pre-registered criteria")
verdict = cached.artifact("verdict")
st.dataframe(verdict, hide_index=True, use_container_width=True)

st.subheader("Point forecasts")
point = cached.artifact("point")
point = point.loc[point["mae"].notna()]
st.dataframe(
    point[["model", "n_test", "mae", "rmse", "bias", "mae_pa", "rmse_pa", "bias_pa"]],
    hide_index=True,
    use_container_width=True,
    column_config={
        c: st.column_config.NumberColumn(c.replace("_pa", " (PA-wtd)").upper(), format="%.4f")
        for c in ("mae", "rmse", "bias", "mae_pa", "rmse_pa", "bias_pa")
    },
)
st.caption("Unweighted MAE is the criterion; PA-weighted columns are reporting only.")

st.subheader("80% intervals")
coverage = cached.artifact("interval_coverage")
st.plotly_chart(charts.coverage_chart(coverage), use_container_width=True)

st.subheader("Breakout and regression probabilities")
probs = cached.artifact("probabilities")
st.dataframe(
    probs[["label", "n", "base_rate_train", "base_rate_test", "mean_predicted", "brier",
           "brier_base", "skill", "ci_low", "ci_high", "ece", "precision_at_10",
           "precision_at_25"]],
    hide_index=True,
    use_container_width=True,
    column_config={c: st.column_config.NumberColumn(format="%.3f")
                   for c in probs.columns if c not in ("label", "n")},
)
st.plotly_chart(charts.reliability_chart(cached.artifact("reliability")),
                use_container_width=True)

st.subheader("Who is scored")
st.dataframe(cached.artifact("population"), hide_index=True, use_container_width=True,
             column_config={"share": st.column_config.NumberColumn(format="%.3f")})
st.caption(
    "Every metric above is conditional on the player reaching 100 PA in the target season."
)
