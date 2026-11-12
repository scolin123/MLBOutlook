"""Do the numbers in the docs match the artifacts? (Pre-freeze checklist item 4, and after.)

`docs/DESIGN_FREEZE.md` section 4 requires that the artifacts be regenerated from the commit
being tagged and that the numbers in the README and `PROJECT_PLAN.md` match them. Doing that
by eye across five tables, six ablation rows, twenty reliability deciles and three sliced
coverage tables is not a check anybody repeats honestly, so it is a script.

Every expected value below was read out of the docs and is written here to the precision the
docs quote. That duplication is the point: if a number moves in one place and not the other,
this fails and names both values.

Run it after regenerating everything:

    uv run python scripts/run_folds.py
    uv run python scripts/run_ablations.py
    uv run python scripts/run_probabilities.py
    uv run python scripts/run_holdout.py --rehearse fold2
    uv run python scripts/check_docs.py

Exit code is 0 when everything matches and 1 otherwise. Numbers the docs quote that no
artifact carries are recomputed here from the snapshot rather than skipped.

The `fold3_*` section covers the holdout write-up (README, `PROJECT_PLAN.md`, `MODEL_CARD.md`).
It reads the committed `artifacts/fold3_*.csv` from the single look and never regenerates them:
fold 3 is run once, so there is no command above that reproduces them, and `run_holdout.py`
refuses to try unless HEAD is the tagged freeze commit with a clean tree. It also asserts the
shape of the claim the write-up leads with -- criterion 2 failed, criteria 1 and 3 passed, so the
verdict is FAIL -- because that sentence is the one thing in the docs a silently changed artifact
could turn into a lie.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlb_outlook.config import load_folds_config
from mlb_outlook.evaluation.harness import fold_frames
from mlb_outlook.models.intervals import DEFAULT_PA_EDGES, pa_buckets
from mlb_outlook.models.marcel import DEFAULT_REGRESSION_PA, age_factor, marcel_rate

ARTIFACTS = Path("artifacts")
DATA = Path("data/processed/hitter_seasons.parquet")
CONFIG = Path("configs/folds.yaml")
# The holdout writes these per fold; the docs quote the fold 2 rehearsal and the fold 3 run.
REHEARSAL_FOLD = "fold2"
HOLDOUT_FOLD = "fold3"

fails: list[str] = []
checks = 0


def eq(label: str, got, want, tol: float = 5e-5) -> None:
    """Compare one artifact value against the doc's, at the doc's own precision."""
    global checks
    checks += 1
    ok = abs(float(got) - float(want)) <= tol
    if not ok:
        fails.append(f"{label}: artifact {float(got):.6f} vs doc {want}")
    print(f"  {'ok ' if ok else 'BAD'} {label:58s} {float(got):.4f} vs doc {want}")


def note(message: str) -> None:
    print(f"  --  {message}")


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def read(name: str) -> pd.DataFrame | None:
    path = ARTIFACTS / name
    if not path.exists():
        fails.append(f"{path} is missing; regenerate the artifacts first")
        print(f"  BAD {path} is missing")
        return None
    return pd.read_csv(path)


# ---- fold results: MAE, RMSE, PA-weighted, intervals, biases ----

section("fold_results.csv  (README + PROJECT_PLAN fold, PA-weighted and interval tables)")
folds = read("fold_results.csv")
if folds is not None:
    f = folds.set_index(["fold", "config", "model"])

    print("\n-- MAE, the headline table --")
    mae_doc = {
        ("fold1", "base"): {
            "last_season": 0.0359, "weighted3": 0.0333, "marcel": 0.0310,
            "ridge": 0.0322, "ridge_rel": 0.0310,
        },
        ("fold1", "extended"): {"ridge": 0.0312, "ridge_rel": 0.0282},
        ("fold2", "base"): {
            "last_season": 0.0325, "weighted3": 0.0306, "marcel": 0.0285,
            "ridge": 0.0265, "ridge_rel": 0.0266,
        },
        ("fold2", "extended"): {"ridge": 0.0289, "ridge_rel": 0.0263},
    }
    for (fold, config), models in mae_doc.items():
        for model, want in models.items():
            eq(f"MAE {fold} {config} {model}", f.loc[(fold, config, model), "mae"], want)

    print("\n-- the PA-weighted table, fold 2 extended (unweighted / PA-weighted) --")
    weighted_doc = {
        # model: (mae, mae_pa, rmse, rmse_pa, bias, bias_pa)
        "last_season": (0.0325, 0.0298, 0.0427, 0.0385, 0.0015, -0.0019),
        "weighted3": (0.0306, 0.0273, 0.0401, 0.0358, 0.0053, 0.0004),
        "marcel": (0.0285, 0.0248, 0.0368, 0.0320, 0.0093, 0.0034),
        "ridge": (0.0289, 0.0283, 0.0364, 0.0355, -0.0141, -0.0190),
        "ridge_rel": (0.0263, 0.0240, 0.0335, 0.0306, -0.0013, -0.0062),
    }
    for model, values in weighted_doc.items():
        row = f.loc[("fold2", "extended", model)]
        for column, want in zip(
            ("mae", "mae_pa", "rmse", "rmse_pa", "bias", "bias_pa"), values, strict=True
        ):
            eq(f"{column:8s} fold2 extended {model}", row[column], want)

    print("\n-- fold 1 extended, the PA-weighted edge the docs say survives there --")
    eq("mae_pa fold1 extended ridge_rel", f.loc[("fold1", "extended", "ridge_rel"), "mae_pa"],
       0.0248)
    eq("mae_pa fold1 extended marcel", f.loc[("fold1", "extended", "marcel"), "mae_pa"], 0.0275)

    print("\n-- 80% intervals, fold 2 extended (coverage / mean width) --")
    for model, (cov, width) in {
        "resid_q_pa": (0.815, 0.0874),
        "conformal": (0.840, 0.0917),
        "conformal_pa": (0.837, 0.0916),
        "quantile_reg": (0.801, 0.0857),
    }.items():
        row = f.loc[("fold2", "extended", model)]
        eq(f"coverage fold2 extended {model}", row["coverage"], cov, tol=5e-4)
        eq(f"width    fold2 extended {model}", row["width"], width)

    print("\n-- fold 1 base window, the narrow-window weakness --")
    eq("coverage fold1 base resid_q_pa",
       f.loc[("fold1", "base", "resid_q_pa"), "coverage"], 0.628, tol=5e-4)
    eq("coverage fold1 base quantile_reg",
       f.loc[("fold1", "base", "quantile_reg"), "coverage"], 0.480, tol=5e-4)

    print("\n-- biases and row counts quoted in the docs --")
    eq("bias fold1 extended ridge_rel", f.loc[("fold1", "extended", "ridge_rel"), "bias"], 0.0111)
    eq("bias fold1 extended marcel", f.loc[("fold1", "extended", "marcel"), "bias"], 0.0151)
    eq("n_test fold1", f.loc[("fold1", "base", "marcel"), "n_test"], 358, tol=0)
    eq("n_test fold2", f.loc[("fold2", "base", "marcel"), "n_test"], 362, tol=0)
    eq("n_train fold1 extended", f.loc[("fold1", "extended", "marcel"), "n_train"], 2351, tol=0)
    eq("n_train fold2 extended", f.loc[("fold2", "extended", "marcel"), "n_train"], 2709, tol=0)
    note(
        "PROJECT_PLAN's decision-sequence numbers (+0.0058 -> -0.0017, 0.0333 -> 0.0303, "
        "0.0271) describe configurations that no longer exist and are annotated as historical"
    )

# ---- population accounting ----

section("population_accounting.csv  (PROJECT_PLAN population table)")
population = read("population_accounting.csv")
if population is not None:
    p = population.set_index(["fold", "bucket"])
    for fold, (eligible, zero, partial, scored) in {
        "fold1": (461, 50, 53, 358),
        "fold2": (455, 38, 55, 362),
    }.items():
        part = population.loc[population["fold"] == fold]
        eq(f"{fold} eligible", part["eligible"].iloc[0], eligible, tol=0)
        eq(f"{fold} 0 PA", part["n"].iloc[0], zero, tol=0)
        eq(f"{fold} 1-99 PA", part["n"].iloc[1], partial, tol=0)
        eq(f"{fold} 100+ PA (scored)", part["n"].iloc[2], scored, tol=0)
        eq(f"{fold} scored share", part["share"].iloc[2], scored / eligible, tol=5e-4)
    del p

# ---- ablations ----

section("ablation_results.csv  (README + PROJECT_PLAN ablation table)")
ablations = read("ablation_results.csv")
if ablations is not None:
    a = ablations.set_index("group")
    abl_doc = {
        # group: (n_features, n_dropped, mae, mae_vs_full, ci_low, ci_high, coverage, width)
        "full": (31, 0, 0.0263, 0.0, 0.0, 0.0, 0.815, 0.0874),
        "contact_quality": (19, 12, 0.0271, 0.0008, -0.0001, 0.0017, 0.831, 0.0899),
        "k_bb": (25, 6, 0.0268, 0.0006, 0.0002, 0.0009, 0.815, 0.0883),
        "age": (30, 1, 0.0266, 0.0003, -0.0003, 0.0009, 0.807, 0.0876),
        "pa_flags": (22, 9, 0.0263, 0.0000, -0.0005, 0.0005, 0.809, 0.0865),
        "league_rel": (31, 15, 0.0262, -0.0000, -0.0002, 0.0002, 0.818, 0.0875),
    }
    for group, (nf, nd, mae, vs, lo, hi, cov, width) in abl_doc.items():
        row = a.loc[group]
        eq(f"{group} n_features", row["n_features"], nf, tol=0)
        eq(f"{group} n_dropped", row["n_dropped"], nd, tol=0)
        eq(f"{group} mae", row["mae"], mae)
        eq(f"{group} mae_vs_full", row["mae_vs_full"], vs)
        eq(f"{group} ci_low", row["ci_low"], lo)
        eq(f"{group} ci_high", row["ci_high"], hi)
        eq(f"{group} coverage", row["coverage"], cov, tol=5e-4)
        eq(f"{group} width", row["width"], width)
    checks += 1
    verdicts = list(a["verdict"])
    if verdicts != ["full", "unclear", "helps", "unclear", "unclear", "unclear"]:
        fails.append(f"ablation verdicts changed: {verdicts}")
    print(f"  ok  verdicts {verdicts}")

# ---- probabilities ----

section("probability_results.csv  (README + PROJECT_PLAN probability tables)")
probabilities = read("probability_results.csv")
if probabilities is not None:
    registered = probabilities.loc[probabilities["threshold"] == 0.03].set_index("label")
    prob_doc = {
        "breakout": {
            "n": 362, "base_rate_train": 0.154, "base_rate_test": 0.171,
            "mean_predicted": 0.171, "brier": 0.1061, "brier_base": 0.1422, "skill": 0.254,
            "brier_diff": -0.0361, "ci_low": -0.0501, "ci_high": -0.0226, "ece": 0.057,
            "precision_at_10": 0.900, "precision_at_25": 0.680,
        },
        "regression": {
            "n": 362, "base_rate_train": 0.279, "base_rate_test": 0.257,
            "mean_predicted": 0.273, "brier": 0.1679, "brier_base": 0.1914, "skill": 0.123,
            "brier_diff": -0.0235, "ci_low": -0.0373, "ci_high": -0.0091, "ece": 0.039,
            "precision_at_10": 0.800, "precision_at_25": 0.600,
        },
    }
    for label, wanted in prob_doc.items():
        for column, want in wanted.items():
            eq(f"{label} {column}", registered.loc[label, column], want,
               tol=0 if column == "n" else 5e-4)

    print("\n-- threshold sensitivity and the base rates beside it --")
    sens = probabilities.set_index(["label", "threshold"])
    for label, skills in {
        "breakout": {0.02: 0.225, 0.03: 0.254, 0.04: 0.200},
        "regression": {0.02: 0.147, 0.03: 0.123, 0.04: 0.130},
    }.items():
        for threshold, want in skills.items():
            eq(f"{label} skill at {threshold}", sens.loc[(label, threshold), "skill"], want,
               tol=5e-4)
    for label, rates in {
        "breakout": {0.02: 0.232, 0.04: 0.113},
        "regression": {0.02: 0.343, 0.04: 0.157},
    }.items():
        for threshold, want in rates.items():
            eq(f"{label} base rate test at {threshold}",
               sens.loc[(label, threshold), "base_rate_test"], want, tol=5e-4)

section("probability_reliability.csv  (PROJECT_PLAN decile table)")
reliability = read("probability_reliability.csv")
if reliability is not None:
    r = reliability.loc[reliability["threshold"] == 0.03].set_index(["label", "decile"])
    deciles = {
        "breakout": [
            (0.019, 0.000), (0.051, 0.000), (0.074, 0.056), (0.097, 0.111), (0.116, 0.079),
            (0.150, 0.029), (0.186, 0.139), (0.228, 0.333), (0.289, 0.306), (0.496, 0.649),
        ],
        "regression": [
            (0.056, 0.054), (0.107, 0.111), (0.142, 0.250), (0.179, 0.108), (0.224, 0.171),
            (0.262, 0.243), (0.312, 0.257), (0.375, 0.361), (0.442, 0.444), (0.626, 0.568),
        ],
    }
    for label, rows in deciles.items():
        for decile, (predicted, observed) in enumerate(rows, start=1):
            row = r.loc[(label, decile)]
            eq(f"{label} decile {decile} predicted", row["mean_predicted"], predicted, tol=5e-4)
            eq(f"{label} decile {decile} observed", row["observed"], observed, tol=5e-4)
    checks += 1
    counts = sorted(set(r["n"]))
    if min(counts) < 33 or max(counts) > 40:
        fails.append(f"decile counts outside the documented spread: {counts}")
    print(f"  ok  decile n values {counts}")

# ---- the holdout's own tables, from the fold 2 rehearsal ----

section(f"{REHEARSAL_FOLD}_*.csv  (PROJECT_PLAN coverage-by-slice and per-player tables)")
coverage = read(f"{REHEARSAL_FOLD}_interval_coverage.csv")
if coverage is not None:
    c = coverage.set_index("range")
    slice_doc = {
        "pa_lag1 < 300": (105, 0.800, 0.0925),
        "300 <= pa_lag1 < 500": (129, 0.798, 0.0862),
        "pa_lag1 >= 500": (128, 0.844, 0.0842),
        "age < 26": (77, 0.870, 0.0883),
        "26 <= age < 31": (186, 0.780, 0.0872),
        "age >= 31": (99, 0.838, 0.0868),
    }
    for name, (n, cov, width) in slice_doc.items():
        eq(f"{name} n", c.loc[name, "n"], n, tol=0)
        eq(f"{name} coverage", c.loc[name, "coverage"], cov, tol=5e-4)
        eq(f"{name} width", c.loc[name, "width"], width)

players = read(f"{REHEARSAL_FOLD}_players.csv")
if players is not None:
    gap = players["ridge_rel_minus_marcel"]
    eq("per-player gap mean", gap.mean(), -0.0105)
    eq("per-player gap median", gap.median(), -0.0095)
    eq("per-player gap sd", gap.std(), 0.0123)
    eq("per-player gap min", gap.min(), -0.0557)
    eq("per-player gap max", gap.max(), 0.0204)
    eq("players forecast higher than marcel", (gap > 0).sum(), 70, tol=0)
    eq("players forecast lower than marcel", (gap < 0).sum(), 292, tol=0)
    eq("share where ridge_rel is closer", (players["advantage"] > 0).mean(), 0.530, tol=5e-4)
    eq("mean error advantage", players["advantage"].mean(), 0.0023)

# ---- the holdout: every fold 3 number the write-up quotes ----

section(f"{HOLDOUT_FOLD}_*.csv  (README + PROJECT_PLAN + MODEL_CARD holdout tables)")
holdout = read(f"{HOLDOUT_FOLD}_point.csv")
if holdout is not None:
    h = holdout.set_index("model")
    print("\n-- point forecasts, fold 3 extended (unweighted / PA-weighted) --")
    point_doc = {
        # model: (mae, mae_pa, rmse, rmse_pa, bias, bias_pa)
        "last_season": (0.0347, 0.0318, 0.0449, 0.0409, 0.0043, 0.0001),
        "weighted3": (0.0317, 0.0289, 0.0413, 0.0372, 0.0062, 0.0015),
        "marcel": (0.0283, 0.0253, 0.0371, 0.0326, 0.0092, 0.0034),
        "ridge_rel": (0.0272, 0.0247, 0.0347, 0.0311, 0.0012, -0.0039),
    }
    for model, values in point_doc.items():
        row = h.loc[model]
        for column, want in zip(
            ("mae", "mae_pa", "rmse", "rmse_pa", "bias", "bias_pa"), values, strict=True
        ):
            eq(f"{column:8s} fold3 {model}", row[column], want)
    eq("n_train fold3", h.loc["ridge_rel", "n_train"], 3071, tol=0)
    eq("n_test fold3", h.loc["ridge_rel", "n_test"], 353, tol=0)

    print("\n-- the edges the write-up quotes, as differences --")
    eq("fold3 MAE edge over marcel",
       h.loc["marcel", "mae"] - h.loc["ridge_rel", "mae"], 0.0011)
    eq("fold3 PA-weighted edge over marcel",
       h.loc["marcel", "mae_pa"] - h.loc["ridge_rel", "mae_pa"], 0.0006)
    eq("fold3 RMSE edge over marcel",
       h.loc["marcel", "rmse"] - h.loc["ridge_rel", "rmse"], 0.0024)

    print("\n-- 80% intervals, fold 3 --")
    eq("coverage fold3 resid_q_pa", h.loc["resid_q_pa", "coverage"], 0.836, tol=5e-4)
    eq("width    fold3 resid_q_pa", h.loc["resid_q_pa", "width"], 0.0864)

verdict = read(f"{HOLDOUT_FOLD}_verdict.csv")
if verdict is not None:
    print("\n-- the verdict: criterion 2 failed, so the verdict is FAIL --")
    v = verdict.set_index("criterion")
    checks += 1
    passed = [bool(x) for x in v["passed"]]
    if passed != [True, False, True]:
        fails.append(f"the docs lead with a FAIL on criterion 2; artifact says {passed}")
    print(f"  ok  criteria passed {passed} -> verdict {'PASS' if all(passed) else 'FAIL'}")
    for criterion, want in {
        1: "0.0272 vs 0.0283",
        2: "-0.0011 (-0.0024 to +0.0003)",
        3: "0.836",
    }.items():
        checks += 1
        got = str(v.loc[criterion, "value"])
        if got != want:
            fails.append(f"fold3 criterion {criterion}: artifact {got!r} vs doc {want!r}")
        print(f"  {'ok ' if got == want else 'BAD'} criterion {criterion} value {got!r}")

coverage3 = read(f"{HOLDOUT_FOLD}_interval_coverage.csv")
if coverage3 is not None:
    print("\n-- coverage by slice, fold 3 --")
    c3 = coverage3.set_index("range")
    for name, (n, cov, width) in {
        "pa_lag1 < 300": (100, 0.790, 0.0930),
        "300 <= pa_lag1 < 500": (112, 0.848, 0.0867),
        "pa_lag1 >= 500": (141, 0.858, 0.0815),
        "age < 26": (73, 0.849, 0.0871),
        "26 <= age < 31": (171, 0.836, 0.0867),
        "age >= 31": (109, 0.826, 0.0853),
    }.items():
        eq(f"fold3 {name} n", c3.loc[name, "n"], n, tol=0)
        eq(f"fold3 {name} coverage", c3.loc[name, "coverage"], cov, tol=5e-4)
        eq(f"fold3 {name} width", c3.loc[name, "width"], width)
    checks += 1
    # The docs say one slice is outside the band on fold 3, and name which one.
    out = list(c3.loc[(c3["coverage"] < 0.75) | (c3["coverage"] > 0.85)].index)
    if out != ["pa_lag1 >= 500"]:
        fails.append(f"docs name pa_lag1 >= 500 as the one slice outside the band; found {out}")
    print(f"  ok  slices outside 0.75-0.85 {out}")

population3 = read(f"{HOLDOUT_FOLD}_population.csv")
if population3 is not None:
    print("\n-- population accounting, fold 3: 23.4% not scored --")
    eq("fold3 eligible", population3["eligible"].iloc[0], 461, tol=0)
    eq("fold3 0 PA", population3["n"].iloc[0], 53, tol=0)
    eq("fold3 1-99 PA", population3["n"].iloc[1], 55, tol=0)
    eq("fold3 100+ PA (scored)", population3["n"].iloc[2], 353, tol=0)
    eq("fold3 scored share", population3["share"].iloc[2], 0.766, tol=5e-4)
    eq("fold3 excluded count", 461 - 353, 108, tol=0)
    eq("fold3 excluded share", 1 - population3["share"].iloc[2], 0.234, tol=5e-4)

probabilities3 = read(f"{HOLDOUT_FOLD}_probabilities.csv")
if probabilities3 is not None:
    print("\n-- probabilities, fold 3, registered threshold --")
    p3 = probabilities3.set_index("label")
    prob3_doc = {
        "breakout": {
            "n": 353, "base_rate_train": 0.156, "base_rate_test": 0.187,
            "mean_predicted": 0.183, "brier": 0.1221, "brier_base": 0.1530, "skill": 0.202,
            "brier_diff": -0.0309, "ci_low": -0.0451, "ci_high": -0.0160, "ece": 0.040,
            "precision_at_10": 0.800, "precision_at_25": 0.640,
        },
        "regression": {
            "n": 353, "base_rate_train": 0.277, "base_rate_test": 0.272,
            "mean_predicted": 0.253, "brier": 0.1715, "brier_base": 0.1980, "skill": 0.134,
            "brier_diff": -0.0266, "ci_low": -0.0397, "ci_high": -0.0113, "ece": 0.053,
            "precision_at_10": 0.800, "precision_at_25": 0.760,
        },
    }
    for label, wanted in prob3_doc.items():
        for column, want in wanted.items():
            eq(f"fold3 {label} {column}", p3.loc[label, column], want,
               tol=0 if column == "n" else 5e-4)
    checks += 1
    # The docs say both CIs are clear of zero on fold 3; the verdict does not depend on it.
    not_clear = [label for label in prob3_doc if p3.loc[label, "ci_high"] >= 0]
    if not_clear:
        fails.append(f"docs say both fold 3 probability CIs clear zero; these do not: {not_clear}")
    print("  ok  both fold 3 probability CIs are clear of zero")

reliability3 = read(f"{HOLDOUT_FOLD}_reliability.csv")
if reliability3 is not None:
    print("\n-- reliability deciles, fold 3 --")
    r3 = reliability3.set_index(["label", "decile"])
    deciles3 = {
        "breakout": [
            (0.023, 0.000), (0.054, 0.054), (0.081, 0.091), (0.103, 0.111), (0.132, 0.057),
            (0.158, 0.171), (0.196, 0.114), (0.249, 0.343), (0.311, 0.371), (0.523, 0.556),
        ],
        "regression": [
            (0.046, 0.054), (0.100, 0.176), (0.136, 0.194), (0.164, 0.176), (0.195, 0.250),
            (0.235, 0.200), (0.285, 0.343), (0.347, 0.286), (0.421, 0.343), (0.603, 0.694),
        ],
    }
    for label, rows in deciles3.items():
        for decile, (predicted, observed) in enumerate(rows, start=1):
            row = r3.loc[(label, decile)]
            eq(f"fold3 {label} decile {decile} predicted", row["mean_predicted"], predicted,
               tol=5e-4)
            eq(f"fold3 {label} decile {decile} observed", row["observed"], observed, tol=5e-4)
    checks += 1
    counts3 = sorted(set(r3["n"]))
    if min(counts3) < 33 or max(counts3) > 37:
        fails.append(f"fold 3 decile counts outside the documented 33-37: {counts3}")
    print(f"  ok  fold 3 decile n values {counts3}")
    print("\n-- the top-decile breakout gap: +0.153 on fold 2, +0.032 on fold 3 --")
    eq("fold3 top breakout decile gap", r3.loc[("breakout", 10), "gap"], 0.032, tol=5e-4)
    if reliability is not None:
        r2 = reliability.loc[reliability["threshold"] == 0.03].set_index(["label", "decile"])
        eq("fold2 top breakout decile gap", r2.loc[("breakout", 10), "gap"], 0.153, tol=5e-4)

players3 = read(f"{HOLDOUT_FOLD}_players.csv")
if players3 is not None:
    print("\n-- per player, fold 3 --")
    gap3 = players3["ridge_rel_minus_marcel"]
    eq("fold3 per-player gap mean", gap3.mean(), -0.0080)
    eq("fold3 per-player gap median", gap3.median(), -0.0083)
    eq("fold3 per-player gap sd", gap3.std(), 0.0129)
    eq("fold3 per-player gap min", gap3.min(), -0.0551)
    eq("fold3 per-player gap max", gap3.max(), 0.0281)
    eq("fold3 players forecast higher than marcel", (gap3 > 0).sum(), 87, tol=0)
    eq("fold3 players forecast lower than marcel", (gap3 < 0).sum(), 266, tol=0)
    eq("fold3 share where ridge_rel is closer", (players3["advantage"] > 0).mean(), 0.533,
       tol=5e-4)
    eq("fold3 mean error advantage", players3["advantage"].mean(), 0.0011)
    eq("fold3 rows scored", len(players3), 353, tol=0)

# ---- numbers the docs quote that no artifact carries ----

section("recomputed: residual provenance, the Marcel audit, the data snapshot")
df = pd.read_parquet(DATA)
cfg = load_folds_config(CONFIG)
fold2 = next(x for x in cfg.folds if x.name == "fold2")
train, test = fold_frames(df, fold2, cfg, extended=True)

print("\n-- residual sample (DESIGN_FREEZE section 2: 2,709, split 965 / 797 / 947) --")
eq("residuals in the fold 2 extended sample", len(train), 2709, tol=0)
buckets = pa_buckets(train, DEFAULT_PA_EDGES)
for bucket, want in enumerate([965, 797, 947]):
    eq(f"training residuals in bucket {bucket}", int((buckets == bucket).sum()), want, tol=0)

print("\n-- Marcel audit, the regression table (MARCEL_AUDIT.md) --")
LEAGUE = 0.310
for seasons, label, weighted_pa, share, projected in [
    ([(600, 0.360)] * 3, "600/600/600", 7200, 0.143, 0.3529),
    ([(600, 0.360), (0, 0.0), (0, 0.0)], "600/0/0", 3000, 0.286, 0.3457),
    ([(150, 0.360), (0, 0.0), (0, 0.0)], "150/0/0", 750, 0.615, 0.3292),
]:
    den = sum(w * pa for w, (pa, _) in zip((5, 4, 3), seasons, strict=True))
    eq(f"{label} weighted PA", den, weighted_pa, tol=0)
    eq(f"{label} league share", DEFAULT_REGRESSION_PA / (den + DEFAULT_REGRESSION_PA), share,
       tol=5e-4)
    eq(f"{label} .360 hitter projects to", marcel_rate(seasons, 29, LEAGUE), projected)

print("\n-- Marcel audit, the aging table --")
for age, factor, line in [
    (22, 1.0420, 0.3334), (26, 1.0180, 0.3258), (29, 1.0000, 0.3200),
    (32, 0.9910, 0.3171), (35, 0.9820, 0.3142),
]:
    eq(f"age_factor({age})", age_factor(age), factor)
    eq(f"a .320 line at {age}", 0.320 * age_factor(age), line)

print("\n-- Marcel audit difference 7: the 2022 composition break --")
gaps = {}
for season in sorted(df["season"].unique()):
    rows = df.loc[df["season"] == season]
    full = float(np.average(rows["woba"], weights=rows["pa"]))
    qualified = rows.loc[rows["pa"] >= 100]
    gaps[int(season)] = full - float(np.average(qualified["woba"], weights=qualified["pa"]))
before = [g for s, g in gaps.items() if s <= 2021]
after = [g for s, g in gaps.items() if s >= 2022]
eq("pre-2022 gap, smallest magnitude", min(abs(g) for g in before), 0.0076, tol=5e-4)
eq("pre-2022 gap, largest magnitude", max(abs(g) for g in before), 0.0096, tol=5e-4)
eq("2022+ gap, smallest magnitude", min(abs(g) for g in after), 0.0027, tol=5e-4)
eq("2022+ gap, largest magnitude", max(abs(g) for g in after), 0.0032, tol=5e-4)
checks += 1
unreachable = int(((df["pa"] > 0) & df["woba"].isna()).sum())
if unreachable != 0:
    fails.append(f"difference 13 says this path is unreachable, found {unreachable} rows")
print(f"  ok  rows with pa > 0 and a null wOBA (audit difference 13): {unreachable}")

print("\n-- the data snapshot (DESIGN_FREEZE section 2) --")
eq("player-seasons", len(df), 9804, tol=0)
eq("distinct players", df["player_id"].nunique(), 2814, tol=0)
eq("first season", df["season"].min(), 2015, tol=0)
eq("last season", df["season"].max(), 2026, tol=0)
checks += 1
columns = [
    "player_id", "name", "season", "age", "pa", "woba", "xwoba",
    "k_pct", "bb_pct", "avg_ev", "barrel_pct", "hard_hit_pct",
]
if list(df.columns) != columns:
    fails.append(f"snapshot columns differ from DESIGN_FREEZE section 2: {list(df.columns)}")
print(f"  ok  columns {list(df.columns)}")

print("\n-- league wOBA levels quoted in the README (.3184 -> .3102 into 2024) --")
league = df.groupby("season").apply(
    lambda g: np.average(g["woba"], weights=g["pa"]), include_groups=False
)
eq("league wOBA 2023", league.loc[2023], 0.3184, tol=5e-4)
eq("league wOBA 2024", league.loc[2024], 0.3102, tol=5e-4)

print()
print("=" * 78)
print(f"{checks} checks run, {len(fails)} mismatch(es)")
for bad in fails:
    print(f"  MISMATCH {bad}")
print("=" * 78)
raise SystemExit(1 if fails else 0)
