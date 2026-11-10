# DiamondIQ v1 (revised plan)

Audience: baseball-ops hiring. Target: done by ~Jan 1, 2027, hard stop Feb 2027.
Scope: MLB hitters only. Pitchers, minors, contracts, counterfactual simulator, Development Lab,
and park factors are v2 (see section 6 for what the park deferral costs the ablation set).

## 1. What v1 answers
1. Which hitters are likely to outperform what their recent results suggest (buy-low), and which to underperform (sell-high)?
2. What is each hitter's next-season forecast, with calibrated uncertainty?
3. Who are the closest historical comparables and what happened to them?

## 2. Principles (unchanged, plus)
- Every feature is built "as of" a forecast date; a pytest enforces no feature uses later data.
- Success criteria are written down BEFORE running the 2026 fold (see section 6).
- Hyperparameters, imputers, scalers, calibration are fit only inside the training window.
- Honest limitations page in the dashboard.

## 3. Data
- Greenfield build: new repo, no code, data, or helpers reused from GameDay, SwingVision, or Expected Runs.
- Sources: Statcast (via pybaseball), MLB Stats API / Chadwick register for IDs, FanGraphs-style season stats for reference only (check terms before scraping; prefer self-computed metrics).
- Pull seasons 2015-2026 for FEATURES and lag history, even though training folds start at 2023 targets (see section 6).
- Storage: MySQL 8 (Docker) for canonical relational tables, accessed via SQLAlchemy + PyMySQL; raw Statcast pulls cached as Parquet in data/raw/ (re-loadable, keeps ingestion reproducible). No FastAPI in v1. Chosen over DuckDB deliberately for a transferable, widely used SQL skill; DuckDB would be simpler for pure analytics.
- SQL design goals (resume-relevant): schema in versioned .sql files, primary/foreign keys, indexes on (player_id, season), window functions for rolling/lag features, as-of queries for leakage-safe features, bulk loads via LOAD DATA / chunked inserts.
- Practical note: model fitting still happens in pandas/scikit-learn; MySQL is the system of record and feature-query layer, not the training engine.
- Every row keeps source + retrieval date. Missing is never zero.
- Era handling: 2020 short season (down-weight/flag), 2021 ball, 2023 rule changes (pitch clock, shift ban, bigger bases). Normalize rates to league-season (wRC+-style) and test a season-effect feature.
- Tables: players, player_seasons, batted_ball_events, park_context, forecasts (v1 skips pitch_events at scale, contracts, injury, team_context unless ablation justifies).

## 4. Models
- Target: next-season wOBA (rate), plus K%, BB%, and a simple PA projection. WAR deferred; own value metric = runs above average from wOBA x projected PA.
- Baselines: last season; 3-year weighted average; Marcel (weighted 3 yr, regressed to mean, age-adjusted). Archived public projections (Steamer/ZiPS) if obtainable.
- Main model: regularized regression (ridge/elastic net) on lagged skill + Statcast quality-of-contact features + age + PA + park. Gradient boosting only as a comparison.
- Empirical Bayes shrinkage: tested once as a feature/prior; adopted only if it beats plain ridge by a meaningful margin on CV.
- Uncertainty: quantile regression or conformal intervals on the same forecaster.
- Breakout/regression: derived from the forecast distribution, defined as the target season landing
  at least 0.030 wOBA above (breakout) or below (regression) a weighted 3-year baseline. No separate
  classifier, so the probabilities cannot contradict the forecast. Definitions are fixed in section 6
  before anything was computed; threshold sensitivity is reported.
- Similarity: standardized skill/age features + nearest neighbors, show comps' next-season outcomes.
- Undervalued (no contracts): forecast vs Marcel/public projection gap, plus xwOBA-wOBA gap, ranked with uncertainty.

## 5. Dashboard (Streamlit + Plotly, public)
Pages: scouting board (buy-low/sell-high/breakout/regression lists), player profile, player comparison, data coverage + model version, limitations page.

Player profile layout (inspired by Baseball Savant / Baseball-Reference references; own design, no copied assets, logos, or headshots):
- Header card: name, team, pos, bats/throws, age, last-3-season slash line and career row.
- Percentile panel: Savant-style horizontal bars (blue = poor, red = great) vs league for that season, min-PA qualified. Metrics: xwOBA, xBA, xSLG, avg EV, barrel%, hard-hit%, sweet-spot%, chase%, whiff%, K%, BB%, sprint speed; bat speed / squared-up only where tracking exists (flag missing, never zero).
- Season table (BR style): age, team, PA, wOBA, wRC+-style index, K%, BB%, WAR reference; league-leading values in bold.
- Statcast table (heat-shaded cells): barrel%, EV, LA sweet-spot%, xBA/xSLG/wOBA/xwOBA/xwOBAcon, hard-hit%, K%, BB%, with league row.
- Charts: spray chart from hc_x/hc_y, 100-PA rolling xwOBA, pitch-type mix faced by season.
- DiamondIQ panel (the part that is new): next-season wOBA forecast with 80% interval, breakout/stable/regression probabilities, observed-vs-expected gap, top comps and their next-season outcomes, model version + data coverage flags.
- Test case: Vladimir Guerrero Jr. 2026 (wOBA .305 vs xwOBA .331, but barrel% and sweet-spot% sharply down) is a good "unlucky or deteriorating?" example profile.

## 6. Evaluation protocol
Expanding-window folds, model re-initialized from scratch each fold ("clean memory"):

| Fold | Train targets | Test target | Purpose |
|---|---|---|---|
| 1 | 2023 | 2024 | pipeline sanity check |
| 2 | 2023-24 | 2025 | iterate here |
| 3 | 2023-25 | 2026 | FINAL holdout, run once after design freeze |
| Production | all through 2026 | 2027 | frozen, timestamped forecasts |

Notes:
- Folds 1-2 are for debugging and model selection. Fold 3 is run once, after freezing code and config (git tag). Do not tune after seeing it.
- Training on one season is thin. Recommendation: allow earlier seasons (2015-2022) as additional training data in an alternate "extended" configuration and compare against the 2023-only version. Report both.
- Hyperparameters are tuned by chronological CV inside each training window only.
- Metrics: MAE/RMSE vs baselines, calibration of intervals/probabilities, Brier/log loss for breakout probabilities, precision at top 10/25 with bootstrap CIs.
- Pre-registered "success": beats Marcel MAE on folds 2 and 3 with bootstrap CI not crossing zero, and 80% intervals cover 75-85% of outcomes. If not met, report honestly.
- Ablations: remove Statcast contact quality, age, shrinkage prior one at a time.
  Run on fold 2 only (`scripts/run_ablations.py`); results below. Park factors are deferred to
  v2, so the park ablation is not part of v1 and the park row is dropped from this list rather
  than left outstanding.
- Historical decision simulation: top-25 buy-low lists each fold, realized outcome vs baseline.

### Breakout / regression definitions (fixed 2026-11-09, before anything was computed)

Written down and committed before a single probability was computed, so the thresholds
cannot be picked to flatter the result. Nothing below may move without a note in this
section saying what changed and why.

**Baseline.** The PA- and recency-weighted 3-year wOBA average: `models.baselines.Weighted3`,
weights 5/4/3 on lag 1/2/3, PA-weighted inside that, no regression to the mean and no age
adjustment. *Not Marcel* -- section 4 originally said Marcel, and the switch is deliberate.
Question 1 asks who outperforms "what their recent results suggest", and the weighted average
*is* that line: it carries no aging curve and no mean-regression, so a breakout is measured
against the player's own recent record rather than against a projection that has already
predicted part of the move. Measuring against Marcel would also make the label partly a
restatement of the baseline's aging assumptions.

**Labels.** For a player with next-season wOBA `y` and baseline `b`:

| Label | Condition |
|---|---|
| breakout | `y >= b + 0.030` |
| regression | `y <= b - 0.030` |
| stable | everything in between |

0.030 wOBA is a little over one standard error on a ~500-PA season and roughly the gap
between an average and a clearly above-average bat. The three labels are mutually exclusive
and exhaustive, so the probabilities sum to 1 by construction.

**Population.** Exactly the rows the folds already score: at least 100 PA in the lag-1
feature season and at least 100 PA in the target season (`min_pa` in `folds.yaml`). A player
with no 100-PA target season has no outcome to score and is counted on neither side -- which
means these probabilities are conditional on playing again, and the write-up has to say so.

**Predictive distribution.** `F` is the selected configuration's distribution and nothing
else: the `ridge_rel` point forecast on the extended window, plus the empirical residuals of
that same fit taken inside the player's lag-1 PA bucket, with the same bucket edges
(300/500) and the same pooled fallback for a thin bucket as the selected `resid_q_pa`
intervals. The 10th and 90th percentiles of `F` are therefore exactly the published 80%
interval -- the probabilities are a reading of the forecast distribution, not a second model,
so they cannot contradict it. There is no separate classifier.

    P(breakout)   = P(y >= b + 0.030) under F = share of bucket residuals r with point + r >= b + 0.030
    P(regression) = P(y <= b - 0.030) under F = share of bucket residuals r with point + r <= b - 0.030
    P(stable)     = 1 - P(breakout) - P(regression)

The distribution is empirical, so a probability is a count over the bucket's residual
sample and its granularity is 1/n for that bucket. That is a reported property, not a bug
to smooth away before the freeze.

**Evaluation, fold 2 only.** Fold 3 is not run for this and the design is not frozen.
- Brier score for P(breakout) and for P(regression) against the realized labels, each
  compared with a base-rate baseline that predicts the fold's *training-window* base rate for
  every player (the only base rate available at the forecast date). Reported as the Brier
  skill score `1 - BS_model / BS_base`, with a paired bootstrap CI on the per-player
  squared-error difference, same machinery as the MAE comparisons.
- Reliability table over deciles of predicted probability: n, mean predicted, observed rate,
  and the gap. Equal-count deciles rather than fixed 0.1-wide bins, because the predictions
  are skewed toward zero and fixed bins would leave most rows in one bucket.
- The base rates themselves, train and test, for both labels.
- Threshold sensitivity at 0.020 / 0.030 / 0.040, with 0.030 the registered threshold. The
  registered number does not change because a neighbour scores better.

**Pass/fail.** Not pre-registered as a success criterion for the holdout, deliberately: the
pre-registered criteria stay the two in `folds.yaml` (MAE against Marcel, 80% interval
coverage). These probabilities are derived from a distribution those criteria already test,
so a third criterion on the same quantity would be double-counting. Fold 2 reports Brier,
skill against the base rate, and the reliability gap as descriptive evidence.

### Results to date (2026-11-09, folds 1-2 only)

Real 2015-2026 Savant hitter seasons; MAE on next-season wOBA, `base / extended` training window.
Baselines do not train, so the window does not move them.

| Model | fold1 (2024) | fold2 (2025) |
|---|---|---|
| last season | 0.0359 | 0.0325 |
| 3-year weighted | 0.0333 | 0.0306 |
| Marcel | 0.0310 | 0.0285 |
| ridge, absolute features | 0.0322 / 0.0312 | 0.0265 / 0.0289 |
| **ridge_rel** | 0.0310 / **0.0282** | 0.0266 / **0.0263** |

Decisions taken on fold 2, in order, one change at a time:
1. Forecast wOBA relative to the league average (`LeagueRelative` + `woba_rel_lag{k}`), re-levelled
   with `lg_woba_hat`, the prior season's league wOBA. Fold 2 bias +0.0058 -> -0.0017.
2. Extended training window (targets 2016-2019, 2021-2022; 2020 never a target, flagged per lag
   with `short_lag{k}`). Fold 1 0.0333 -> 0.0303; fold 2 roughly flat.
3. Statcast quality of contact (xwOBA, average exit velocity, barrel%, hard-hit%), absolute and
   league-relative lags. Fold 2 0.0271 -> 0.0263, fold 1 0.0303 -> 0.0282.

Selected configuration: `ridge_rel` on the extended window. Against Marcel on fold 2, paired
bootstrap -0.0023 (95% CI -0.0036 to -0.0008), RMSE 0.0335 vs 0.0368. Fold 3 has not been run
and the design is not frozen.

### 80% intervals (`models/intervals.py`), fold 2, extended window

| Approach | Coverage | Mean width | Point MAE |
|---|---|---|---|
| **residual quantiles by PA bucket** | **0.815** | **0.0874** | 0.0263 |
| split conformal (30% held out) | 0.840 | 0.0917 | 0.0264 |
| split conformal, PA buckets | 0.837 | 0.0916 | 0.0264 |
| quantile regression on residuals | 0.801 | 0.0857 | 0.0263 |

Both halves of the pre-registered criterion are now met on fold 2: Marcel beaten with a bootstrap
CI clear of zero, and 80% coverage inside 75-85% for every approach tried.

- Current pick: residual quantiles by lag-1 PA bucket. It keeps the full-data point fit, and its
  per-bucket coverage is flatter (0.800 / 0.798 / 0.844) than the constant conformal width manages
  (0.800 / 0.845 / 0.867), which matters most for part-time hitters.
- Split conformal is the fallback with a finite-sample guarantee. It costs a little point accuracy
  (the forecaster is refitted on 70% of rows) and over-covers regulars.
- Quantile regression is the narrowest at near-nominal coverage but the least stable: on the
  one-season base window of fold 1 it covers 0.480, since 31 features fitted on a few hundred rows
  make its in-sample residual quantiles far too optimistic. Residual quantiles have the same
  weakness in milder form (0.628 on that window). Both need the wide window.

Open points from these runs:
- The absolute-feature ridge degrades on the wider window (fold 2 bias -0.0141) and on synthetic
  two-season windows it blows up (MAE 0.068): near-constant `lg_woba_lag` columns get a large
  standardised coefficient and extrapolate. Decide before the freeze whether it stays as the
  absolute-parameterisation reference or is dropped. The `league_rel` ablation below narrows
  the cause but does not settle it.
- The league level for the target season is a random walk (prior season's league wOBA). A better
  estimator is the obvious next lever on fold-1-style environment shifts, but fitting one on the
  full 2015-2026 series would peek at fold 3's test season.

### Feature-group ablations (`evaluation/ablation.py`), fold 2, extended window

One group dropped at a time from the selected `ridge_rel` configuration, re-fitted from
scratch (`scripts/run_ablations.py`, writes `artifacts/ablation_results.csv`). The CI is a
paired bootstrap of the ablated model's MAE minus the full model's on the same 362 test
rows, so positive means dropping the group costs accuracy. Coverage is `resid_q_pa`, the
selected interval approach, wrapped around the ablated forecaster.

| Group dropped | Cols | MAE | vs full (95% CI) | 80% coverage | Mean width |
|---|---|---|---|---|---|
| none (selected) | 31 | 0.0263 | - | 0.815 | 0.0874 |
| contact quality | -12 | 0.0271 | +0.0008 (-0.0001 to +0.0017) | 0.831 | 0.0899 |
| K% / BB% | -6 | 0.0268 | +0.0006 (**+0.0002 to +0.0009**) | 0.815 | 0.0883 |
| age | -1 | 0.0266 | +0.0003 (-0.0003 to +0.0009) | 0.807 | 0.0876 |
| PA / missing-season flags | -9 | 0.0263 | +0.0000 (-0.0005 to +0.0005) | 0.809 | 0.0865 |
| league-relative rates | swap 15 | 0.0262 | -0.0000 (-0.0002 to +0.0002) | 0.818 | 0.0875 |

No group clearly hurts -- no CI sits entirely below zero -- so the selected configuration is
unchanged. Fold 3 was not run. All six models keep 80% coverage inside the pre-registered
75-85% range, so none of these choices is load-bearing for the interval criterion.

- Only **K% / BB%** has a CI clear of zero. Its MAE gain is not the largest, but it is the
  most consistent across players, which is what a paired bootstrap rewards.
- **Contact quality** costs the most MAE (+0.0008) and yet its CI touches zero
  (-0.0001). This is the same comparison as decision 3 above, and it reproduces it exactly
  (0.0271 without, 0.0263 with). Flagged rather than acted on: the Statcast features were
  adopted on a difference that a bootstrap cannot separate from zero on 362 rows. Dropping
  them also widens the intervals (0.0899) and pushes coverage to 0.831, near the top of the
  range -- the forecast gets noisier without them even where the MAE is ambiguous.
- **PA and the missing-season / 2020 flags** are free to drop: +0.000009 MAE, the tightest
  CI of any group, and slightly *narrower* intervals (0.0865). The flags were added so the
  wide window could carry 2020 lags without being trained on them, and they do that without
  earning anything at the margin. The interval's PA buckets read `pa_lag1` off the feature
  frame, not the model's feature list, so they survive this ablation.
- **Age** alone is worth +0.0003 with a CI spanning zero, and dropping it moves the bias
  most of any group (-0.0013 to -0.0027): the age term is doing level work the aging-curve
  part of Marcel does explicitly.
- **League-relative rates** swapped for absolute ones changes nothing (0.0262 vs 0.0263,
  CI +/-0.0002). `RELATIVE_FEATURES` has no `lg_woba_lag{k}` column to delete -- the rate
  lags carry their season's league level already -- so a literal "drop the league columns"
  is a no-op by construction, and the ablation that means something is this swap. It keeps
  the league-centred target and `lg_woba_hat` re-levelling, so it does not isolate the
  `lg_woba_lag` columns on its own. What it does show: once the target is league-centred,
  absolute rate lags carry the same information as relative ones, which points at the
  league-level columns or the uncentred target -- not the absolute rates -- as what breaks
  the absolute ridge on the wide window. Worth resolving before the freeze.

## 7. Production forecast (2027)
After fold 3 review, retrain on all data through 2026, freeze forecasts, tag the repo and publish before the 2027 season. Score them after 2027 as a prospective test.

## 8. Timeline (~9 weeks + Feb buffer)
- W1 (Oct 27): repo, uv/ruff/pre-commit, MySQL schema (docker compose), ID mapping, ingest 2015-2026, data audit.
- W2: as-of feature builder, leakage pytest, league normalization, Marcel baseline.
- W3: fold harness, ridge forecaster, fold 1-2 results.
- W4: uncertainty intervals, derived breakout probabilities, calibration.
- W5: ablations, empirical Bayes check, extended-vs-2023-only comparison. Design freeze.
- W6: similarity/comps, undervalued ranking.
- W7: dashboard.
- W8: run fold 3 once, write-up, model card, data dictionary.
- W9: 2027 production forecasts, deploy, README polish.
- Feb: buffer / stretch (pitchers v2 start, Statcast pitch-level features).

## 9. Repo
Same layout as the original, minus prospects.py, valuation.py, development_lab.py, and API. Add tests/test_no_leakage.py and configs/folds.yaml.

## 10. Open items
- Hours per week (assumed ~10-12 alongside classes; adjust timeline if lower).
- Confirm licensing/terms for any FanGraphs data before use.
