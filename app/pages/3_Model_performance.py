"""Model performance: the fold 3 holdout verdict first, then fold 2 and fold 3 in detail."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import cached, charts, data  # noqa: E402
from components.style import setup  # noqa: E402

POINT_COLUMNS = ["model", "n_test", "mae", "rmse", "bias", "mae_pa", "rmse_pa", "bias_pa"]
POINT_FORMATS = ("mae", "rmse", "bias", "mae_pa", "rmse_pa", "bias_pa")
PROB_COLUMNS = [
    "label", "n", "base_rate_train", "base_rate_test", "mean_predicted", "brier",
    "brier_base", "skill", "ci_low", "ci_high", "ece", "precision_at_10", "precision_at_25",
]


def point_table(point: pd.DataFrame) -> None:
    point = point.loc[point["mae"].notna()]
    st.dataframe(
        point[POINT_COLUMNS],
        hide_index=True,
        use_container_width=True,
        column_config={
            c: st.column_config.NumberColumn(c.replace("_pa", " (PA-wtd)").upper(),
                                             format="%.4f")
            for c in POINT_FORMATS
        },
    )
    st.caption("Unweighted MAE is the criterion; PA-weighted columns are reporting only.")


def probability_table(probs: pd.DataFrame) -> None:
    st.dataframe(
        probs[PROB_COLUMNS],
        hide_index=True,
        use_container_width=True,
        column_config={c: st.column_config.NumberColumn(format="%.3f")
                       for c in probs.columns if c not in ("label", "n")},
    )


def fold_detail(point: pd.DataFrame, coverage: pd.DataFrame, probs: pd.DataFrame,
                reliability: pd.DataFrame, population: pd.DataFrame) -> None:
    st.markdown("**Point forecasts**")
    point_table(point)
    st.markdown("**80% intervals**")
    st.plotly_chart(charts.coverage_chart(coverage), use_container_width=True)
    st.markdown("**Breakout and regression probabilities**")
    probability_table(probs)
    st.plotly_chart(charts.reliability_chart(reliability), use_container_width=True)
    st.markdown("**Who is scored**")
    st.dataframe(population, hide_index=True, use_container_width=True,
                 column_config={"share": st.column_config.NumberColumn(format="%.3f")})
    st.caption(
        "Every metric above is conditional on the player reaching 100 PA in the target season."
    )


setup("Model performance", data.BACKTEST_LABEL)
cached.require_files()
st.title("Model performance")

verdict2 = cached.artifact("verdict")
have_holdout = data.holdout_available()

if not have_holdout:
    st.caption(
        f"Fold 2: trained on target seasons through {data.FEATURES_THROUGH}, tested on "
        f"{data.TARGET_SEASON}. This is the fold the configuration was selected on, so these "
        "numbers are optimistic; the untouched holdout is fold 3, which has not been run."
    )
    st.subheader("Pre-registered criteria, fold 2")
    st.dataframe(verdict2, hide_index=True, use_container_width=True)
    fold_detail(
        cached.artifact("point"), cached.artifact("interval_coverage"),
        cached.artifact("probabilities"), cached.artifact("reliability"),
        cached.artifact("population"),
    )
    st.stop()

verdict3 = cached.holdout_artifact("verdict")
label = data.verdict_label(verdict3)
failed = data.failed_criteria(verdict3)

st.caption(
    f"Two folds. **Fold 2** ({data.TARGET_SEASON}) is the fold this configuration was selected "
    f"on, so its numbers are optimistic by construction. **Fold 3** "
    f"({data.HOLDOUT_TARGET_SEASON}) is the design-freeze holdout: run once, from the commit "
    f"tagged `{data.HOLDOUT_TAG}`, on a season nothing was fitted to or chosen on. The verdict "
    "is read off fold 3."
)

st.subheader(f"Fold 3 holdout verdict: {label}")
if label == "PASS":
    st.success("Every pre-registered criterion held on the holdout season.")
else:
    missed = ", ".join(str(c) for c in failed["criterion"])
    st.error(
        f"**Verdict {label}.** Criterion {missed} was not met on the holdout, and the "
        "pre-registered rule is that a fail is any one criterion missing -- not two out of "
        "three, and not a partial success. On fold 3 the model is **not shown to beat "
        "Marcel**: the MAE difference is -0.0011 with a 95% interval of -0.0024 to +0.0003, "
        "which contains zero. Nothing was re-fitted or re-run after that number was seen."
    )

criteria = verdict3[["criterion", "description"]].copy()
criteria["fold 2 (selection)"] = verdict2["value"].to_numpy()
criteria["fold 3 (holdout)"] = verdict3["value"].to_numpy()
criteria["fold 3 result"] = [
    "pass" if p else "FAIL" for p in verdict3["passed"].astype(bool)
]
st.dataframe(criteria, hide_index=True, use_container_width=True)
st.caption(
    "The three criteria are the ones in `configs/folds.yaml`, written before fold 2 was run "
    "and unchanged since. The probability scores below were deliberately not registered as a "
    "criterion -- they read off a distribution these three already test -- so they do not "
    "offset the one that failed."
)

st.subheader("Fold 2 against fold 3")
point2 = cached.artifact("point").set_index("model")
point3 = cached.holdout_artifact("point").set_index("model")
probs2 = cached.artifact("probabilities").set_index("label")
probs3 = cached.holdout_artifact("probabilities").set_index("label")
pop2 = cached.artifact("population")
pop3 = cached.holdout_artifact("population")


def scored_share(population: pd.DataFrame) -> float:
    return float(population["share"].iloc[-1])


def edge(point: pd.DataFrame, column: str) -> float:
    """Marcel minus the model, so positive means the model was more accurate."""
    return float(point.loc["marcel", column] - point.loc["ridge_rel", column])


comparison = pd.DataFrame(
    [
        ("ridge_rel MAE", point2.loc["ridge_rel", "mae"], point3.loc["ridge_rel", "mae"]),
        ("Marcel MAE", point2.loc["marcel", "mae"], point3.loc["marcel", "mae"]),
        ("edge over Marcel", edge(point2, "mae"), edge(point3, "mae")),
        ("edge over Marcel, PA-weighted", edge(point2, "mae_pa"), edge(point3, "mae_pa")),
        ("ridge_rel RMSE", point2.loc["ridge_rel", "rmse"], point3.loc["ridge_rel", "rmse"]),
        ("Marcel RMSE", point2.loc["marcel", "rmse"], point3.loc["marcel", "rmse"]),
        ("ridge_rel bias", point2.loc["ridge_rel", "bias"], point3.loc["ridge_rel", "bias"]),
        ("Marcel bias", point2.loc["marcel", "bias"], point3.loc["marcel", "bias"]),
        ("80% interval coverage", point2.loc["resid_q_pa", "coverage"],
         point3.loc["resid_q_pa", "coverage"]),
        ("mean interval width", point2.loc["resid_q_pa", "width"],
         point3.loc["resid_q_pa", "width"]),
        ("breakout Brier skill", probs2.loc["breakout", "skill"],
         probs3.loc["breakout", "skill"]),
        ("regression Brier skill", probs2.loc["regression", "skill"],
         probs3.loc["regression", "skill"]),
        ("share of eligible players scored", scored_share(pop2), scored_share(pop3)),
    ],
    columns=[
        "reading",
        f"fold 2 ({data.TARGET_SEASON}, {int(point2.loc['ridge_rel', 'n_test'])} rows)",
        f"fold 3 ({data.HOLDOUT_TARGET_SEASON}, "
        f"{int(point3.loc['ridge_rel', 'n_test'])} rows)",
    ],
)
st.dataframe(
    comparison, hide_index=True, use_container_width=True,
    column_config={
        c: st.column_config.NumberColumn(format="%.4f") for c in comparison.columns[1:]
    },
)
st.markdown(
    """
**Why the edge over Marcel halved.** Fold 2's margin is a selected number and fold 3's is not.
Three model decisions, four interval approaches, six feature-group ablations and six
probability-threshold readings were all compared on the same 362 fold-2 rows, keeping whichever
scored best; a maximum over many noisy comparisons on one sample overstates how the winner does
on a fresh one. Fold 3 is the first measurement on a sample that had no say in the
configuration. What was never a selection target barely moved: Marcel's own MAE and bias, the
mean interval width, and the PA-weighted edge. 2026 was not a strange season -- the true edge is
smaller than fold 2 suggested.
"""
)

f3, f2 = st.tabs(
    [f"Fold 3 holdout ({data.HOLDOUT_TARGET_SEASON})",
     f"Fold 2 selection ({data.TARGET_SEASON})"]
)
with f3:
    st.caption(
        f"Run once from the tagged commit, on {data.HOLDOUT_TARGET_SEASON} outcomes with "
        f"features through {data.HOLDOUT_FEATURES_THROUGH}. Aggregate scores only: the "
        f"per-player {data.HOLDOUT_TARGET_SEASON} scoring rows are not reachable from this "
        f"app, so no hitter's holdout result can be looked up one at a time."
    )
    fold_detail(
        cached.holdout_artifact("point"), cached.holdout_artifact("interval_coverage"),
        cached.holdout_artifact("probabilities"), cached.holdout_artifact("reliability"),
        cached.holdout_artifact("population"),
    )
    st.caption(
        f"{1 - scored_share(pop3):.1%} of eligible players are not scored here at all, the "
        "largest share of the three folds. The top breakout decile observed 0.556 against "
        "0.523 predicted, against fold 2's 0.649 against 0.496 -- the under-confidence fold 2 "
        "showed did not reproduce, which is why no calibration map was fitted."
    )
with f2:
    st.caption(
        f"The selection fold: {data.TARGET_SEASON} outcomes, features through "
        f"{data.FEATURES_THROUGH}. Every modelling choice was made by looking at these rows, "
        "so read them as optimistic."
    )
    fold_detail(
        cached.artifact("point"), cached.artifact("interval_coverage"),
        cached.artifact("probabilities"), cached.artifact("reliability"), pop2,
    )

st.caption(
    "Full write-up, including the secondary evidence and the limitations that follow from the "
    "verdict: `docs/MODEL_CARD.md`. What the freeze pinned and the pass/fail rules written "
    "before the run: `docs/DESIGN_FREEZE.md`."
)
