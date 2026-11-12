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
- Folds 1-2 are for debugging and model selection. Fold 3 is run once, after freezing code and config (git tag). Do not tune after seeing it. The inventory of what the tag pins, the pre-freeze checklist, the run procedure and the pass/fail rules live in [DESIGN_FREEZE.md](DESIGN_FREEZE.md); the freeze has not happened.
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

**Evaluation.** Registered as fold 2 only; the holdout reported the same quantities on fold 3 as
evidence, at the registered threshold only and with no sweep. Both folds are tabulated below.
- Brier score for P(breakout) and for P(regression) against the realized labels, each
  compared with a base-rate baseline that predicts the fold's *training-window* base rate for
  every player (the only base rate available at the forecast date). Reported as the Brier
  skill score `1 - BS_model / BS_base`, with a paired bootstrap CI on the per-player
  squared-error difference, same machinery as the MAE comparisons.
- Reliability table over deciles of predicted probability: n, mean predicted, observed rate,
  and the gap. Equal-count deciles rather than fixed 0.1-wide bins, because the predictions
  are skewed toward zero and fixed bins would leave most rows in one bucket.
- Precision at the top 10 and top 25 by predicted probability, for each label, read against
  the test base rate as the chance level. This is the ranking question the Brier score does
  not answer -- of the players the model is surest about, how many moved -- and it is the
  metric the top-25 buy-low list in the decision simulation above is judged on. Evidence,
  not a criterion.
  No bootstrap CI is attached to it, which is a deviation from the metrics list in section 6
  above: resampling rows changes which rows are in the top 10 at all, so the interval would
  describe a different estimator on each draw. At 362 test rows the honest reading of
  precision@10 is a count out of ten, and it is reported that way.
- The base rates themselves, train and test, for both labels.
- Threshold sensitivity at 0.020 / 0.030 / 0.040, with 0.030 the registered threshold. The
  registered number does not change because a neighbour scores better.

**Pass/fail.** Not pre-registered as a success criterion for the holdout, deliberately: the
pre-registered criteria stay the two in `folds.yaml` (MAE against Marcel, 80% interval
coverage). These probabilities are derived from a distribution those criteria already test,
so a third criterion on the same quantity would be double-counting. Fold 2 reports Brier,
skill against the base rate, and the reliability gap as descriptive evidence.

### The fold 3 holdout (2026): verdict FAIL

Run once, from `scripts/run_holdout.py` on the commit tagged `design-freeze-v1`
(`ab710f79e5c74dd367debb4f66bccec231ce3135`), on the frozen data snapshot
(SHA-256 `b164c087...039cd8`, 9,804 rows). Artifacts: `artifacts/fold3_*.csv`. The full write-up
is [MODEL_CARD.md](MODEL_CARD.md); this section carries the tables so they sit beside the fold 2
ones they are meant to be read with.

| # | Criterion | Fold 2 (2025, selection) | Fold 3 (2026, holdout) | |
|---|---|---|---|---|
| 1 | `ridge_rel` MAE below Marcel MAE | 0.0263 vs 0.0285 | 0.0272 vs 0.0283 | pass |
| 2 | Paired bootstrap of that difference, 95% CI excluding zero | -0.0023 (-0.0036 to -0.0008) | **-0.0011 (-0.0024 to +0.0003)** | **FAIL** |
| 3 | 80% `resid_q_pa` coverage inside 0.75-0.85 | 0.815 | 0.836 | pass |

**Criterion 2 failed, so the verdict is FAIL.** [DESIGN_FREEZE.md](DESIGN_FREEZE.md) section 6
defines pass as all three holding and fail as any one missing. One is missing. **The model is not
shown to beat Marcel on 2026**: the point estimate is 0.0011 of wOBA MAE in its favour and the
95% interval contains zero, which is the pre-registered meaning of "not shown". This is the
failure mode named in advance ("beats Marcel but the CI crosses zero"), it is reported as a fail
and not as a partial success, and nothing was re-fitted, re-tuned, swapped or re-run after the
number was seen.

Fold 3 point forecasts, extended window, 3,071 train rows and 353 test rows, unweighted /
PA-weighted (`artifacts/fold3_point.csv`):

| Model | MAE | RMSE | bias |
|---|---|---|---|
| last season | 0.0347 / 0.0318 | 0.0449 / 0.0409 | +0.0043 / +0.0001 |
| 3-year weighted | 0.0317 / 0.0289 | 0.0413 / 0.0372 | +0.0062 / +0.0015 |
| Marcel | 0.0283 / 0.0253 | 0.0371 / 0.0326 | +0.0092 / +0.0034 |
| **ridge_rel** | **0.0272 / 0.0247** | **0.0347 / 0.0311** | **+0.0012 / -0.0039** |

Fold 2 against fold 3 on everything the holdout reported:

| | Fold 2 (2025) | Fold 3 (2026) |
|---|---|---|
| Train rows / test rows | 2,709 / 362 | 3,071 / 353 |
| `ridge_rel` MAE / Marcel MAE | 0.0263 / 0.0285 | 0.0272 / 0.0283 |
| **Edge over Marcel** | **0.0023** | **0.0011** |
| Paired bootstrap, 95% CI | -0.0023 (-0.0036 to -0.0008) | -0.0011 (-0.0024 to +0.0003) |
| `ridge_rel` RMSE / Marcel RMSE | 0.0335 / 0.0368 | 0.0347 / 0.0371 |
| `ridge_rel` bias / Marcel bias | -0.0013 / +0.0093 | +0.0012 / +0.0092 |
| PA-weighted MAE, model / Marcel | 0.0240 / 0.0248 | 0.0247 / 0.0253 |
| PA-weighted edge | 0.0008 | 0.0006 |
| 80% coverage / mean width | 0.815 / 0.0874 | 0.836 / 0.0864 |
| `ridge_rel` closer than Marcel on | 53.0% of players | 53.3% of players |
| Breakout Brier / skill | 0.1061 / +0.254 | 0.1221 / +0.202 |
| Regression Brier / skill | 0.1679 / +0.123 | 0.1715 / +0.134 |
| Eligible players scored | 362 of 455 (79.6%) | 353 of 461 (76.6%) |

#### The halved margin is a selection effect from fold-2 development

The edge over Marcel is 0.0011 on fold 3 against 0.0023 on fold 2. The explanation is that
**fold 2's 0.0023 is a selected number and fold 3's 0.0011 is not.**

Everything in this document above was decided by looking at fold 2: three model decisions, four
interval approaches, six feature-group ablations, two probability labels at three thresholds
each. Every one of those compared candidates on the same 362 rows and kept whichever scored best
there -- which is what section 1 of DESIGN_FREEZE.md means by "every one of those looks spends a
little of the evidence in fold 2". A maximum over many noisy comparisons on a single sample is
biased upward as an estimate of the winner's performance on a fresh sample: part of the winning
margin is signal that recurs, part is fold-2 noise that does not. Fold 3 is the first measurement
of this configuration on a sample that had no say in building it, so it is the first estimate
without that upward bias, and it is about half the size.

What supports that reading over a 2026-specific explanation:

- **The quantity the selection was run on is the one that shrank.** The PA-weighted edge, never a
  selection target, went 0.0008 -> 0.0006. The per-player win rate went 53.0% -> 53.3%. The mean
  interval width went 0.0874 -> 0.0864. Marcel's own MAE (0.0285 -> 0.0283) and bias (+0.0093 ->
  +0.0092) barely moved, so the comparison's other side is stable across the two seasons.
- **The softest piece of the design is the one flagged in advance.** Contact quality was adopted
  on +0.0008 MAE with a CI touching zero (-0.0001 to +0.0017), recorded in decision 3 below as
  not carried by its MAE case. Fold 3 is consistent with that group contributing less than fold 2
  implied.
- **The other pre-named explanation does not apply.** `lg_woba_hat` as a random walk was named in
  DESIGN_FREEZE section 6 as the likely cause if criterion 1 failed. Criterion 1 passed, and
  `ridge_rel`'s 2026 bias of +0.0012 is the best level of any model on the fold, so the league
  estimate was not the problem on this season. The margin was.

At 353 rows a 0.0011 edge cannot be distinguished from zero, which DESIGN_FREEZE section 6 said
in advance was entirely possible even if the edge is real. The edge may be real. It is not
demonstrated, and the pre-registered consequence of that is a fail.

#### Fold 3 secondary evidence

None of it feeds the verdict; all of it was registered to be reported either way.

Interval coverage by slice (`artifacts/fold3_interval_coverage.csv`), fold 2 beside it:

| Slice | Fold 3 n | Fold 3 coverage | Fold 3 width | Fold 2 coverage |
|---|---|---|---|---|
| pa_lag1 < 300 | 100 | 0.790 | 0.0930 | 0.800 |
| 300 <= pa_lag1 < 500 | 112 | 0.848 | 0.0867 | 0.798 |
| pa_lag1 >= 500 | 141 | **0.858** | 0.0815 | 0.844 |
| age < 26 | 73 | 0.849 | 0.0871 | **0.870** |
| 26 <= age < 31 | 171 | 0.836 | 0.0867 | 0.780 |
| age >= 31 | 109 | 0.826 | 0.0853 | 0.838 |

- One slice is outside the band on each fold and it is a different slice each time: `age < 26` on
  fold 2 (0.870), `pa_lag1 >= 500` on fold 3 (0.858). At 73-141 rows apiece a coverage estimate
  carries roughly +/-0.07 to +/-0.09 at 95%, so neither is clearly outside and the two do not
  describe a common fault. Flagged for v2, as the fold 2 reading already said; nothing changed.
- The widths are near-identical to fold 2's, so the higher aggregate coverage (0.836) comes from
  the 2026 errors being a little smaller relative to the training residual spread, not from the
  intervals having moved.

Probabilities at the registered +/-0.030 threshold, 353 rows
(`artifacts/fold3_probabilities.csv`), with fold 2's row above each:

| Label | Fold | base rate train / test | mean predicted | Brier | base-rate Brier | skill | model - base rate (95% CI) | gap |
|---|---|---|---|---|---|---|---|---|
| breakout | 2 | 0.154 / 0.171 | 0.171 | 0.1061 | 0.1422 | +0.254 | -0.0361 (-0.0501 to -0.0226) | 0.057 |
| breakout | **3** | 0.156 / 0.187 | 0.183 | 0.1221 | 0.1530 | **+0.202** | -0.0309 (-0.0451 to -0.0160) | 0.040 |
| regression | 2 | 0.279 / 0.257 | 0.273 | 0.1679 | 0.1914 | +0.123 | -0.0235 (-0.0373 to -0.0091) | 0.039 |
| regression | **3** | 0.277 / 0.272 | 0.253 | 0.1715 | 0.1980 | **+0.134** | -0.0266 (-0.0397 to -0.0113) | 0.053 |

| Label | Fold 2 @10 | Fold 3 @10 | Fold 2 @25 | Fold 3 @25 | Fold 3 chance |
|---|---|---|---|---|---|
| breakout | 0.900 | 0.800 (8 of 10) | 0.680 | 0.640 (16 of 25) | 0.187 |
| regression | 0.800 | 0.800 (8 of 10) | 0.600 | 0.760 (19 of 25) | 0.272 |

Reliability, equal-count deciles (33-37 rows each), fold 3 beside fold 2
(`artifacts/fold3_reliability.csv`):

| Decile | f2 breakout pred / obs | f3 breakout pred / obs | f2 regression pred / obs | f3 regression pred / obs |
|---|---|---|---|---|
| 1 | 0.019 / 0.000 | 0.023 / 0.000 | 0.056 / 0.054 | 0.046 / 0.054 |
| 2 | 0.051 / 0.000 | 0.054 / 0.054 | 0.107 / 0.111 | 0.100 / 0.176 |
| 3 | 0.074 / 0.056 | 0.081 / 0.091 | 0.142 / 0.250 | 0.136 / 0.194 |
| 4 | 0.097 / 0.111 | 0.103 / 0.111 | 0.179 / 0.108 | 0.164 / 0.176 |
| 5 | 0.116 / 0.079 | 0.132 / 0.057 | 0.224 / 0.171 | 0.195 / 0.250 |
| 6 | 0.150 / 0.029 | 0.158 / 0.171 | 0.262 / 0.243 | 0.235 / 0.200 |
| 7 | 0.186 / 0.139 | 0.196 / 0.114 | 0.312 / 0.257 | 0.285 / 0.343 |
| 8 | 0.228 / 0.333 | 0.249 / 0.343 | 0.375 / 0.361 | 0.347 / 0.286 |
| 9 | 0.289 / 0.306 | 0.311 / 0.371 | 0.442 / 0.444 | 0.421 / 0.343 |
| 10 | **0.496 / 0.649** | **0.523 / 0.556** | 0.626 / 0.568 | 0.603 / 0.694 |

- **Both labels beat their base rate on fold 3 with a paired bootstrap CI clear of zero.** This is
  the part of the holdout that reproduced cleanly. It is evidence, it was registered as not being
  a criterion, and it does not offset the criterion that failed.
- **Fold 2's top-decile breakout under-confidence did not reproduce.** The gap fell from +0.153
  (0.649 observed against 0.496 predicted) to +0.032 (0.556 against 0.523), and the
  count-weighted gap from 0.057 to 0.040. Decision 5 below pre-committed to the reading: if fold
  3 reproduced the shape, a calibration map with its own holdout was the v2 lever; if not,
  36-37 rows a decile was the answer. **It did not reproduce, so that was the answer**, and the
  calibration map comes off the v2 list. Regression's count-weighted gap moved the other way
  (0.039 -> 0.053), consistent with the same sampling noise rather than a fixable bias.
- Breakout skill fell (+0.254 -> +0.202) and regression skill rose (+0.123 -> +0.134). Breakout's
  test base rate rose to 0.187, which lifts the base-rate Brier it is scored against as well as
  the model's own, so the fall is not a clean statement about the model.

Per player (`artifacts/fold3_players.csv`), the same summary the fold 2 table below carries:

| Reading | Fold 2 | Fold 3 |
|---|---|---|
| forecast gap, `ridge_rel` - Marcel | mean -0.0105, median -0.0095, sd 0.0123 | mean -0.0080, median -0.0083, sd 0.0129 |
| gap range | -0.0557 to +0.0204 | -0.0551 to +0.0281 |
| higher than Marcel / lower | 70 / 292 | 87 / 266 |
| `ridge_rel` closer to the outcome | 53.0% | 53.3% |
| mean error advantage | +0.0023 | +0.0011 |

The per-player win rate is the one reading that did not shrink; what shrank is the size of the
wins. Nothing beyond this summary was inspected on fold 3, as DESIGN_FREEZE section 5 requires.

### Fold 1-2 results (the selection runs, 2026-11-09)

Real 2015-2026 Savant hitter seasons; MAE on next-season wOBA, `base / extended` training window.
Baselines do not train, so the window does not move them.

| Model | fold1 (2024) | fold2 (2025) |
|---|---|---|
| last season | 0.0359 | 0.0325 |
| 3-year weighted | 0.0333 | 0.0306 |
| Marcel | 0.0310 | 0.0285 |
| ridge, absolute features | 0.0322 / 0.0312 | 0.0265 / 0.0289 |
| **ridge_rel** | 0.0310 / **0.0282** | 0.0266 / **0.0263** |

Decisions taken on fold 2, in order, one change at a time. *The numbers in this list are
historical* -- see the note under it:
1. Forecast wOBA relative to the league average (`LeagueRelative` + `woba_rel_lag{k}`), re-levelled
   with `lg_woba_hat`, the prior season's league wOBA. Fold 2 bias +0.0058 -> -0.0017 (historical).
2. Extended training window (targets 2016-2019, 2021-2022; 2020 never a target, flagged per lag
   with `short_lag{k}`). Fold 1 0.0333 -> 0.0303 (historical); fold 2 roughly flat.
3. Statcast quality of contact (xwOBA, average exit velocity, barrel%, hard-hit%), absolute and
   league-relative lags. Fold 2 0.0271 -> 0.0263, fold 1 0.0303 -> 0.0282.

**Those three lines are a record of the sequence, not a claim about this commit.** Each arrow
compares two configurations that existed while the decision was being taken, and the left-hand
one no longer exists in the code: step 1's model had no contact-quality features and no extended
window, step 2's had no contact quality. So +0.0058, -0.0017, 0.0333, both 0.0303 and 0.0271
cannot be regenerated from `artifacts/fold_results.csv` at this commit, and the pre-freeze check
on the numbers does not try to. What *is* current and checked: the endpoints 0.0282 (fold 1,
extended) and 0.0263 (fold 2, extended), every cell of the table above, and 0.0271 as the
contact-quality row of the ablation table below, which is the standing evidence for step 3 and
reproduces it exactly. For the record, fold 2 base `ridge_rel` bias at this commit is -0.0034
and fold 2 extended is -0.0013; neither is step 1's -0.0017, because neither model is step 1's.

Selected configuration: `ridge_rel` on the extended window. Against Marcel on fold 2, paired
bootstrap -0.0023 (95% CI -0.0036 to -0.0008), RMSE 0.0335 vs 0.0368. These are the numbers that
chose the configuration, so they are the best of several looks at one 362-row sample; the holdout
section above is the same comparison on a sample that had no say in it, and it failed criterion 2.

### Population accounting (`evaluation/diagnostics.py`), all three folds

Every forecast, interval and probability in this project is conditional on the player taking
100 plate appearances in the target season. That condition was stated when the probability
definitions were registered; this is how big it is. Eligible means what `build_features` means
by it -- at least 100 PA in the season before the target -- and the three buckets are what
those players did next (`scripts/run_folds.py`, writes `artifacts/population_accounting.csv`).

| Fold (target) | Eligible | 0 PA | 1-99 PA | 100+ PA (scored) |
|---|---|---|---|---|
| fold 1 (2024) | 461 | 50 (10.8%) | 53 (11.5%) | **358 (77.7%)** |
| fold 2 (2025) | 455 | 38 (8.4%) | 55 (12.1%) | **362 (79.6%)** |
| fold 3 (2026) | 461 | 53 (11.5%) | 55 (11.9%) | **353 (76.6%)** |

- **About a fifth to a quarter of the eligible population is not scored at all** -- 22.3% on
  fold 1, 20.4% on fold 2 and **23.4% on fold 3**, the holdout's being the largest of the three
  (108 of 461 players) -- and the scored count reconciles exactly with the fold's test row count,
  which is the check the script prints.
- **This is the honest reading of every probability here.** P(breakout) is not the chance a
  hitter breaks out; it is the chance he breaks out *given* he takes 100 PA. The players who
  drop out are disproportionately the ones whose forecast was worst, so the real-world
  distribution of outcomes has a tail this population cannot see. A survival model is a v2
  item, not something to bolt on after the freeze.
- **The attrition is also why the interval criterion is evaluated where it is.** Coverage is
  measured on players who played; a hitter who got hurt has no wOBA to be covered.
- Fold 3's accounting reads 2026 playing time, so it was run only from `run_holdout.py` and only
  at the single look; its row above comes from `artifacts/fold3_population.csv`, not from
  `population_accounting.csv`, because `run_folds.py` skips the holdout fold entirely.

### RMSE and PA-weighted error (`evaluation/harness.py`), both folds

`run_fold` reports MAE, RMSE and bias twice: once per row, and once weighted by the target
season's plate appearances (`y_pa`), which is the outcome's own sample size. One row one vote
is how the pre-registered criterion is stated and is what every table above uses. The weighted
reading answers a different question -- how wrong the forecast is where the plate appearances
actually went -- and on this population the two diverge, because roughly a third of the test
rows are hitters under 300 PA.

Fold 2, extended window, 362 test rows, unweighted / PA-weighted:

| Model | MAE | RMSE | bias |
|---|---|---|---|
| last season | 0.0325 / 0.0298 | 0.0427 / 0.0385 | +0.0015 / -0.0019 |
| 3-year weighted | 0.0306 / 0.0273 | 0.0401 / 0.0358 | +0.0053 / +0.0004 |
| Marcel | 0.0285 / 0.0248 | 0.0368 / 0.0320 | +0.0093 / +0.0034 |
| ridge, absolute features | 0.0289 / 0.0283 | 0.0364 / 0.0355 | -0.0141 / -0.0190 |
| **ridge_rel** | **0.0263 / 0.0240** | **0.0335 / 0.0306** | -0.0013 / -0.0062 |

- **The edge over Marcel is mostly in part-time hitters, and this is the main thing the
  weighting shows.** Unweighted, `ridge_rel` beats Marcel by 0.0023 on fold 2. PA-weighted, by
  0.0008. Both models are more accurate on regulars -- a 650-PA wOBA is a less noisy target
  than a 150-PA one, so everything improves under weighting -- but Marcel improves more
  (0.0285 -> 0.0248) than `ridge_rel` does (0.0263 -> 0.0240). Read plainly: most of what the
  Statcast features and the league-relative target buy is on hitters a club has least reason to
  care about. Fold 1 does not show the same squeeze (0.0248 against Marcel's 0.0275, an edge of
  0.0027 that survives weighting), so one fold of this is not a pattern.
- **The pre-registered criterion does not move.** Criteria 1 and 2 are the unweighted MAE and
  its bootstrap, fixed before fold 2 was run, and they stay that. This table is reported beside
  them, not instead of them, and no PA-weighted criterion is being added now -- picking the
  weighting after seeing which way it cuts is exactly what the freeze exists to prevent. It is
  on the v2 list as a criterion to pre-register properly.
- **`ridge_rel` under-predicts regulars.** Its PA-weighted bias is -0.0062 against -0.0013
  unweighted, so the shortfall is concentrated in the high-PA rows; Marcel over-predicts
  everyone (+0.0093) and less so on regulars (+0.0034). The two errors point in opposite
  directions, which is consistent with the aging-curve-free league-relative fit pulling
  established hitters toward the league.
- **RMSE ranks the models the same way MAE does**, with one exception worth noting: the
  absolute-feature ridge is nearly level with Marcel on RMSE (0.0364 against 0.0368) while
  clearly worse on MAE (0.0289 against 0.0285). Its errors are shaped differently -- fewer
  mid-sized misses, a worse level -- which is the same `lg_woba_lag` extrapolation problem
  recorded above.

### 80% intervals (`models/intervals.py`), fold 2, extended window

| Approach | Coverage | Mean width | Point MAE |
|---|---|---|---|
| **residual quantiles by PA bucket** | **0.815** | **0.0874** | 0.0263 |
| split conformal (30% held out) | 0.840 | 0.0917 | 0.0264 |
| split conformal, PA buckets | 0.837 | 0.0916 | 0.0264 |
| quantile regression on residuals | 0.801 | 0.0857 | 0.0263 |

Both halves of the pre-registered criterion are met on fold 2: Marcel beaten with a bootstrap
CI clear of zero, and 80% coverage inside 75-85% for every approach tried. On fold 3 the coverage
half held (0.836) and the Marcel half did not; see the holdout section above.

- Current pick: residual quantiles by lag-1 PA bucket. It keeps the full-data point fit, and its
  per-bucket coverage is flatter (0.800 / 0.798 / 0.844) than the constant conformal width manages
  (0.800 / 0.845 / 0.867), which matters most for part-time hitters.
- Split conformal is the fallback with a finite-sample guarantee. It costs a little point accuracy
  (the forecaster is refitted on 70% of rows) and over-covers regulars.
- Quantile regression is the narrowest at near-nominal coverage but the least stable: on the
  one-season base window of fold 1 it covers 0.480, since 31 features fitted on a few hundred rows
  make its in-sample residual quantiles far too optimistic. Residual quantiles have the same
  weakness in milder form (0.628 on that window). Both need the wide window.

#### Per player: `ridge_rel` minus Marcel (`evaluation/diagnostics.py`), fold 2, extended window

The MAE comparison is a mean over 362 players, and a mean of that size hides whether the edge is
broad or concentrated. `run_holdout.py` writes a row per scored player -- both forecasts, the
signed gap, each model's absolute error, which came closer, the published interval and whether
it covered -- to `<fold>_players.csv`, and prints this summary:

| Reading | Fold 2, extended |
|---|---|
| forecast gap, `ridge_rel` - Marcel | mean -0.0105, median -0.0095, sd 0.0123 |
| gap range | -0.0557 to +0.0204 |
| higher than Marcel / lower | 70 players / 292 players |
| `ridge_rel` closer to the outcome | **53.0% of players** |
| mean error advantage | +0.0023 (which is the MAE difference) |

- **The edge is narrow, not broad.** `ridge_rel` is closer than Marcel on 53.0% of players --
  near a coin flip -- and still wins the mean by 0.0023, so the advantage comes from the size of
  its wins rather than their frequency. A reader who takes "beats Marcel" to mean "is the better
  forecast for this hitter" would be wrong about 47% of the time.
- **It forecasts lower than Marcel almost everywhere**: 292 of 362 players, mean gap -0.0105.
  That is the aging curve and the mean regression Marcel has and the league-relative ridge does
  not, and it is the same thing the bias column says from the other side (Marcel +0.0093,
  `ridge_rel` -0.0013). The two models disagree in one direction by about ten wOBA points as a
  matter of construction, not case by case.
- **Nothing is selected on this.** It is a reading of the already-pre-registered comparison, and
  the per-player rows are in the CSV so the write-up can quote them deliberately; the holdout
  prints the summary only, because a top-ten disagreement list on fold 3 is the per-player browse
  `DESIGN_FREEZE.md` section 5 rules out.

#### Coverage by slice (`evaluation/diagnostics.py`), fold 2, extended window

The pre-registered criterion is the aggregate coverage, and the aggregate is 0.815. These are
the same bounds sliced two ways -- by the lag-1 PA buckets the intervals are built on, and by
age brackets they are not -- printed by `run_holdout.py` for whichever fold it runs.

| Slice | n | Coverage | Mean width |
|---|---|---|---|
| pa_lag1 < 300 | 105 | 0.800 | 0.0925 |
| 300 <= pa_lag1 < 500 | 129 | 0.798 | 0.0862 |
| pa_lag1 >= 500 | 128 | 0.844 | 0.0842 |
| age < 26 | 77 | **0.870** | 0.0883 |
| 26 <= age < 31 | 186 | 0.780 | 0.0872 |
| age >= 31 | 99 | 0.838 | 0.0868 |

- **The youngest bracket is the one slice outside the 75-85% band**, at 0.870 -- the intervals
  are too wide for hitters under 26. At 77 rows a coverage estimate carries about +/-0.08 at
  95%, so this is not clearly outside; it is flagged, not fixed.
- **The width hardly moves across age** (0.0868 to 0.0883) because the intervals bucket on
  playing time, not age. So the spread in coverage across brackets is a statement about how
  big the errors are by age, not about how the width is being chosen -- which is the opposite
  of the PA slices, where the width does the work (0.0925 down to 0.0842).
- **Nothing is changed in response.** Adding age to the interval bucketing, or letting the
  residual spread depend on features generally, is the v2 lever -- and `quantile_reg` already
  does the general version, at 0.801 aggregate on this fold. Picking it now, on the strength of
  one 77-row bracket seen after the fact, is what the freeze exists to prevent.
- The holdout printed the same two slices for fold 3, and one bucket there is outside the band:
  `pa_lag1 >= 500` at 0.858, while every age bracket is inside. So the two folds put a different
  slice outside the band, on 73-141 rows apiece, which is a finding for v2 and not a change to
  the verdict -- the verdict reads the aggregate, and the fold 3 aggregate of 0.836 passed.

#### The league-average column has a composition break at 2022

Found while auditing Marcel (full treatment in [MARCEL_AUDIT.md](MARCEL_AUDIT.md), difference
7) and recorded here because it reaches further than the baseline. `lg_woba` is the PA-weighted
mean wOBA of every row in the Savant batter leaderboard, pulled with `type=batter, min=1`, and
that leaderboard carries pitchers' batting lines through 2021 and effectively drops them with
the universal DH in 2022. Measured as the gap between the all-rows league wOBA and the same
figure over 100+ PA players: -0.0076 to -0.0096 for 2015-2021, then -0.0027 to -0.0032 from
2022. So the series sits about 5 wOBA points low before 2022 relative to after it.

That column is not only Marcel's regression target: it centres the league-relative target
`ridge_rel` is fitted on (`y_woba - y_lg_woba`) and it supplies `lg_woba_hat`. Training rows
with targets 2016-2019 and 2021 are therefore centred about 5 points differently from the
2024-2026 test rows, which is a systematic inconsistency inside the extended window -- the
window the selected configuration uses. It is **not** being fixed: the league definition is part
of what the tag pins, a correction would rebuild every number in this file, and choosing to
rebuild them while holding a known fold 2 result is exactly the move the freeze forbids. It is
recorded as the strongest candidate for a v2 re-freeze, and the fold 3 write-up should name it
beside the random-walk `lg_woba_hat` as a reason the level could be off.

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

### Breakout / regression probability results (`evaluation/calibration.py`), fold 2

`scripts/run_probabilities.py`, extended window, writing `artifacts/probability_results.csv`
and `artifacts/probability_reliability.csv`. Definitions exactly as registered above, scored
on the same 362 test rows as the MAE and interval tables. The base-rate baseline hands every
player the *training* window's base rate, the only one available at the forecast date.

| Label | base rate train / test | mean predicted | Brier | base-rate Brier | skill | model - base rate (95% CI) | reliability gap |
|---|---|---|---|---|---|---|---|
| **breakout** | 0.154 / 0.171 | 0.171 | **0.1061** | 0.1422 | **+0.254** | -0.0361 (**-0.0501 to -0.0226**) | 0.057 |
| **regression** | 0.279 / 0.257 | 0.273 | **0.1679** | 0.1914 | **+0.123** | -0.0235 (**-0.0373 to -0.0091**) | 0.039 |

Both labels beat their base rate with a paired bootstrap CI clear of zero, so the forecast
distribution carries information about the direction of a move and not only its size.

Precision at the top of the ranking, registered threshold, same 362 rows (`precision_at_10`
and `precision_at_25` in `artifacts/probability_results.csv`, which carries them for the
sensitivity thresholds too):

| Label | precision@10 | precision@25 | chance (test base rate) |
|---|---|---|---|
| **breakout** | **0.900** (9 of 10) | **0.680** (17 of 25) | 0.171 |
| **regression** | **0.800** (8 of 10) | **0.600** (15 of 25) | 0.257 |

The top of the breakout ranking is about five times the base rate at k = 10 and four times at
k = 25, which is the same under-confidence the top decile shows, read as a shortlist instead of
as a calibration curve. It is evidence and not a criterion, and at these counts one player
moves precision@10 by 0.100.

Reliability, equal-count deciles of predicted probability (36-37 rows each):

| Decile | breakout predicted | breakout observed | regression predicted | regression observed |
|---|---|---|---|---|
| 1 | 0.019 | 0.000 | 0.056 | 0.054 |
| 2 | 0.051 | 0.000 | 0.107 | 0.111 |
| 3 | 0.074 | 0.056 | 0.142 | 0.250 |
| 4 | 0.097 | 0.111 | 0.179 | 0.108 |
| 5 | 0.116 | 0.079 | 0.224 | 0.171 |
| 6 | 0.150 | 0.029 | 0.262 | 0.243 |
| 7 | 0.186 | 0.139 | 0.312 | 0.257 |
| 8 | 0.228 | 0.333 | 0.375 | 0.361 |
| 9 | 0.289 | 0.306 | 0.442 | 0.444 |
| 10 | 0.496 | 0.649 | 0.626 | 0.568 |

- **Regression is the commoner label** (0.257 against 0.171 on the test rows) and that is
  the baseline's doing, not the model's. `Weighted3` carries no aging curve and no regression
  to the mean, so a hitter who got lucky, or who is past his peak, sits below his own
  three-year line more often than above it. The asymmetry is a property of the registered
  definition and is exactly why each label is scored against its own base rate rather than
  against 0.5 or against the other label.
- **The distribution is centred about right in aggregate.** Mean predicted P(breakout) is
  0.171 against an observed 0.171; regression is 1.6 points high (0.273 against 0.257). Per
  PA bucket the aggregate gap stays inside 0.04 either way.
- **Breakout separates twice as well as regression** (skill +0.254 against +0.123). Part of
  that is the easier baseline -- a 0.154 base rate leaves more room above it than a 0.279 one
  -- and part is that the model's upside cases are the ones where contact quality and wOBA
  disagree, which is the signal the Statcast features add.
- **The breakout table is not flat, and the shape is under-confidence at the top.** Deciles
  1-7 under-observe, most sharply decile 6 (0.029 observed against 0.150 predicted), and the
  top two over-observe (0.333 against 0.228, then 0.649 against 0.496). The players it is
  surest about break out about 15 points more often than it says. At 36-37 rows a decile
  carries roughly +/-0.08 of sampling noise, so no single row is settled; the count-weighted
  gap is 0.057 for breakout and 0.039 for regression, whose worst row is decile 3 (0.250
  against 0.142).
- **Nothing was fitted to fix that.** A post-hoc calibration map on fold 2 would need its own
  holdout to be worth anything, and fold 3 was not available for it. The under-confidence went
  into the freeze as a reported property, on the pre-committed understanding that reproducing on
  fold 3 would make it the v2 lever. **It did not reproduce** -- the top-decile gap fell from
  +0.153 to +0.032 -- so 36-37 rows a decile was the answer and no calibration map is warranted.
- **Granularity is not the limit.** The PA buckets hold 965 / 797 / 947 training residuals,
  so a probability resolves to about 0.001.
- **Threshold sensitivity.** Skill stays positive for both labels at every threshold:
  breakout +0.225 / **+0.254** / +0.200 and regression +0.147 / **+0.123** / +0.130 at
  0.020 / **0.030** / 0.040. The registered 0.030 is the best of the three for breakout and
  the worst for regression, which at 362 rows is noise in both directions and not a reason to
  move it.
- These probabilities are descriptive, as registered: the pre-registered criteria are still
  the MAE and coverage pair in `folds.yaml`, both already met on fold 2. Nothing in the model,
  its features, or its alpha moved for this work, and fold 3 was not run.

### Design-freeze decisions on the six open items (recorded 2026-11-09, before the tag)

[DESIGN_FREEZE.md](DESIGN_FREEZE.md) section 3 lists six things that had to be settled before
the tag, because an undecided item turns into a temptation to re-open the configuration after
fold 3 is seen. All six are decided below. "Accept and report" is one of the decisions, used
twice. None of them changes the selected configuration: no model, feature, threshold or alpha
moved for this section, and fold 3 was not run.

**1. The absolute-feature ridge stays, as the absolute-parameterisation reference only.**
`ridge` (`models.ridge.RidgeForecaster` on `DEFAULT_FEATURES`) is not a candidate and is not in the
holdout. It stays in `run_folds.py` and in the fold 1-2 table because it is the comparison that
shows the league-relative parameterisation is doing the work: on the extended window it loses
0.0030 of MAE on fold 1 (0.0312 against 0.0282) and 0.0026 on fold 2 (0.0289 against 0.0263),
and it is the one model whose bias moves the wrong way as the window widens (fold 2 -0.0141).
Dropping it would delete that evidence and leave "we centred the target on the league" an
assertion. `run_holdout.py` reports `ridge_rel` only, so keeping it costs the holdout nothing.

**2. The `lg_woba_lag` question is a v2 open question, not investigated further on fold 2.**
The `league_rel` ablation showed that once the target is league-centred, absolute and relative
rate lags carry the same information (-0.0000 MAE, CI +/-0.0002), which points at the
league-level columns or the uncentred target as what breaks the absolute ridge on the wide
window. That is as far as it gets resolved for v1. The reason to stop is that the question is
about a reference model: `RELATIVE_FEATURES` has no `lg_woba_lag{k}` column, so no answer
changes anything the tag pins, and every further look costs fold 2 evidence that is already
thin at 362 rows. Recorded as a v2 question, to be answered on a window that is not the
selection fold.

**3. Contact quality stays, on the reading already written down.** Confirmed as stated: the
group was adopted on +0.0008 MAE whose paired bootstrap CI touches zero (-0.0001 to +0.0017),
so the MAE case alone does not carry it. It stays because dropping it also widens the 80%
intervals (0.0899 against 0.0874) and pushes coverage to 0.831, near the top of the
pre-registered 75-85% range -- the forecast gets noisier without the Statcast features even
where their MAE gain is ambiguous -- and because the observed-versus-expected gap is one of the
two questions v1 exists to answer, which needs xwOBA and contact quality in the model rather
than only on the dashboard. The honest summary for the model card is that this group is
justified by the interval width and the product question, not by a significant MAE difference.

**4. `lg_woba_hat` stays a random walk for v1. A better estimator is v2 and is not attempted
now.** The prior season's league wOBA is what the forecaster knows at the forecast date. Fold 1
shows the cost when the environment moves: bias +0.0111 on the extended window. Any better
estimator -- a trend, a shrunk mean, a state-space filter -- would have to be fitted on the
2015-2026 league series, which contains fold 3's test season, so fitting one now is peeking.
It is therefore named in advance as the most likely explanation if criterion 1 fails on fold 3
(DESIGN_FREEZE section 6), which is an explanation and not a defence, and it is the first item
on the v2 list.

**5. The breakout probabilities ship uncorrected, and the under-confidence is reported.**
The top decile of P(breakout) observed 0.649 against 0.496 predicted on fold 2, with deciles
1-7 under-observing and a count-weighted reliability gap of 0.057. No calibration map is
fitted: one fitted on fold 2 would need its own holdout to be worth anything, fold 3 is the one
look and is not available for it, and a map fitted and then evaluated on the same 362 rows would
report its own fit. The probabilities ship as a direct reading of the forecast distribution, the
under-confidence goes into the model card as a known property, and `run_holdout.py` prints the
fold 3 reliability deciles beside fold 2's so the two are read together. If fold 3 reproduces
the shape, a calibration map with its own holdout is the v2 lever; if it does not, 36-37 rows a
decile was the answer.

**6. Park factors stay deferred to v2.** Nothing to do, and the point of recording it is that
nothing gets quietly re-opened. The limitation is already written in the README: a hitter's wOBA
carries his home park, nothing in the feature set separates the two, and league-relative
normalisation removes the league-wide environment rather than the park. The park ablation moves
to v2 with the features, which is why the ablation list above has five groups and no park row.

**How fold 3 resolved the two conditional decisions.** Written after the holdout; the six
decisions above are the pre-tag record and are not edited.

- **Decision 4 (`lg_woba_hat` as a random walk)** was named in advance as the most likely
  explanation if criterion 1 failed. Criterion 1 passed on fold 3 and `ridge_rel`'s 2026 bias is
  +0.0012, the best level of any model on the fold, so the random walk cost nothing visible on
  this season and is not the explanation for the fail. It stays first on the v2 list on its own
  merits -- fold 1's +0.0111 is still what it costs when the environment moves.
- **Decision 5 (breakout under-confidence)** registered both branches: reproduce on fold 3 and a
  calibration map with its own holdout becomes the v2 lever; do not reproduce and 36-37 rows a
  decile was the answer. The top-decile gap fell from +0.153 to +0.032 and the count-weighted gap
  from 0.057 to 0.040, so **it did not reproduce** and the calibration map comes off the v2 list.
  The probabilities ship uncorrected, as they were going to either way.

## 7. Production forecast (2027)
After fold 3 review, retrain on all data through 2026, freeze forecasts, tag the repo and publish before the 2027 season. Score them after 2027 as a prospective test. The production run uses the frozen configuration unchanged and needs no new tag ([DESIGN_FREEZE.md](DESIGN_FREEZE.md) section 7).

Fold 3 has now been reviewed and the verdict is FAIL, which does not change this plan: a fail is
not retried and the fold 3 number from `design-freeze-v1` stands as the v1 result
([DESIGN_FREEZE.md](DESIGN_FREEZE.md) section 7). The 2027 forecast is the same frozen
configuration published prospectively, and given a holdout that could not distinguish the model's
edge from zero on 353 rows, it is the read worth waiting for rather than any further backtest.
The v2 re-freeze list, in the order the evidence now points, is in
[MODEL_CARD.md](MODEL_CARD.md) section 6.

## 8. Timeline (~9 weeks + Feb buffer)
- W1 (Oct 27): repo, uv/ruff/pre-commit, MySQL schema (docker compose), ID mapping, ingest 2015-2026, data audit.
- W2: as-of feature builder, leakage pytest, league normalization, Marcel baseline.
- W3: fold harness, ridge forecaster, fold 1-2 results.
- W4: uncertainty intervals, derived breakout probabilities, calibration.
- W5: ablations, empirical Bayes check, extended-vs-2023-only comparison. Design freeze.
- W6: similarity/comps, undervalued ranking.
- W7: dashboard.
- W8: run fold 3 once, write-up, model card, data dictionary. *Fold 3 run once and the write-up
  and [MODEL_CARD.md](MODEL_CARD.md) done; verdict FAIL. Data dictionary outstanding.*
- W9: 2027 production forecasts, deploy, README polish.
- Feb: buffer / stretch (pitchers v2 start, Statcast pitch-level features).

## 9. Repo
Same layout as the original, minus prospects.py, valuation.py, development_lab.py, and API. Add tests/test_no_leakage.py and configs/folds.yaml.

## 10. Open items
- Hours per week (assumed ~10-12 alongside classes; adjust timeline if lower).
- Confirm licensing/terms for any FanGraphs data before use.
