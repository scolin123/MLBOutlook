# Model card -- DiamondIQ v1 hitter wOBA forecaster

| | |
|---|---|
| Model | `ridge_rel`: league-relative ridge on 31 lagged skill features, extended training window |
| Uncertainty | `resid_q_pa`: 80% empirical residual quantiles, bucketed by lag-1 PA |
| Task | Next-season wOBA for MLB hitters, plus breakout / regression probabilities |
| Frozen at | tag `design-freeze-v1`, commit `ab710f79e5c74dd367debb4f66bccec231ce3135` |
| Data snapshot | `data/processed/hitter_seasons.parquet`, SHA-256 `b164c087e6186d989f52bebdcc1b1153f25f7de5d8f2f6b8ddb003c906039cd8`, 9,804 player-seasons, 2,814 players, 2015-2026 |
| Holdout | fold 3, target season 2026, run once from the tagged commit |
| **Verdict** | **FAIL** |

## 1. Verdict: FAIL

The pre-registered criteria are the three in `configs/folds.yaml`, written before fold 2 was run
and unchanged since. On fold 3 (2026), **criterion 2 failed**:

| # | Criterion | Fold 3 (2026) | |
|---|---|---|---|
| 1 | `ridge_rel` MAE below `MarcelBaseline` MAE | 0.0272 vs 0.0283 | pass |
| 2 | Paired bootstrap of that difference, 95% CI excluding zero | **-0.0011 (-0.0024 to +0.0003)** | **FAIL** |
| 3 | 80% `resid_q_pa` coverage inside 0.75-0.85 | 0.836 | pass |

[DESIGN_FREEZE.md](DESIGN_FREEZE.md) section 6 defines **pass** as all three criteria holding on
fold 3, and **fail** as any one of them missing. One is missing, so the result is a fail, and it
is the failure mode named in advance as "beats Marcel but the CI crosses zero":

> Criterion 2 fails. Report: the model is not shown to beat Marcel on 2026.

That is the finding. **On the 2026 holdout, `ridge_rel` is not shown to beat Marcel.** The point
estimate is 0.0011 of wOBA MAE in the model's favour, the 95% paired-bootstrap interval runs
from -0.0024 to +0.0003, and an interval containing zero is the pre-registered definition of
"not shown". This is not a partial success, a near miss, or a pass on two of three: the design
registered one verdict over three criteria, and that verdict is FAIL.

Nothing was re-fitted, re-tuned, swapped or re-run after the number was seen. The holdout ran
once, from `scripts/run_holdout.py` on the tagged commit with a clean tree, and this write-up is
the only thing produced afterwards.

## 2. Fold 2 (selection) vs fold 3 (holdout), side by side

Fold 2 is the season the configuration was chosen on. Fold 3 is the one season nothing was
fitted to, chosen on, or argued about. Both use the extended training window and the same frozen
configuration; they are meant to be read together.

| | Fold 2 (2025) | Fold 3 (2026) |
|---|---|---|
| Train rows / test rows | 2,709 / 362 | 3,071 / 353 |
| `ridge_rel` MAE | 0.0263 | 0.0272 |
| Marcel MAE | 0.0285 | 0.0283 |
| **Edge over Marcel** | **0.0023** | **0.0011** |
| Paired bootstrap, 95% CI | -0.0023 (-0.0036 to -0.0008) | **-0.0011 (-0.0024 to +0.0003)** |
| Criterion 2 | clear of zero | **spans zero** |
| `ridge_rel` RMSE / Marcel RMSE | 0.0335 / 0.0368 | 0.0347 / 0.0371 |
| `ridge_rel` bias / Marcel bias | -0.0013 / +0.0093 | +0.0012 / +0.0092 |
| PA-weighted MAE, model / Marcel | 0.0240 / 0.0248 | 0.0247 / 0.0253 |
| PA-weighted edge | 0.0008 | 0.0006 |
| 80% interval coverage | 0.815 | 0.836 |
| Mean interval width | 0.0874 | 0.0864 |
| `ridge_rel` closer than Marcel on | 53.0% of players | 53.3% of players |
| Breakout Brier / skill | 0.1061 / +0.254 | 0.1221 / +0.202 |
| Regression Brier / skill | 0.1679 / +0.123 | 0.1715 / +0.134 |
| Eligible players scored | 362 of 455 (79.6%) | 353 of 461 (76.6%) |

### Why the margin halved: selection effects from fold-2 development

The edge over Marcel is 0.0011 on fold 3 against 0.0023 on fold 2 -- a little under half. The
explanation is that **fold 2's 0.0023 is a selected number and fold 3's 0.0011 is not.**

Every decision in this model was taken by looking at fold 2: three model decisions (the
league-relative target, the extended window, the contact-quality group), four interval
approaches, six feature-group ablations, and two probability labels at three thresholds each.
Each of those looks compared candidates on the same 362 rows and kept whichever scored best
there. A maximum over many noisy comparisons on one sample is biased upward as an estimate of
how the winning choice performs on a fresh sample: part of what the winner won by is signal that
will reappear, and part is fold-2 noise that will not. Fold 3 is the first measurement of this
configuration on a sample that had no say in building it, so it is the first estimate without
that upward bias -- and it comes in at roughly half.

Two details support reading it that way rather than as something peculiar to 2026:

- **The thinnest-evidence parts of the design shrank the most, and the unselected reading barely
  moved.** The contact-quality group was adopted on +0.0008 MAE whose CI touched zero (-0.0001
  to +0.0017), flagged in `PROJECT_PLAN.md` section 6 decision 3 as not carried by its MAE case.
  Meanwhile the PA-weighted edge, which was never a selection target, was 0.0008 on fold 2 and is
  0.0006 on fold 3. What shrank by half is precisely the quantity the selection was run on.
- **Nothing else about either model moved between the folds.** Marcel's MAE (0.0285 -> 0.0283)
  and bias (+0.0093 -> +0.0092), the model's share of per-player wins (53.0% -> 53.3%), the mean
  interval width (0.0874 -> 0.0864) and the sign and size of the model-minus-Marcel forecast gap
  are all within noise of each other across the two seasons. 2026 was not a strange season for
  this population; the model simply has a smaller true edge than fold 2 suggested.

The honest reading is that the margin was measured on the data that chose it, the holdout
measured it on data that did not, and at 353 rows a 0.0011 edge cannot be distinguished from
zero. `DESIGN_FREEZE.md` section 6 anticipated exactly this -- "362-ish test rows make this
entirely possible even if the edge is real; that is a known property of the design, not a reason
to widen the test." The edge may well be real. It is not demonstrated.

The other explanation named in advance, `lg_woba_hat` as a random walk, is **not** the cause
here: criterion 1 passed and the model's 2026 bias is +0.0012, the smallest of any model on the
fold. The level was fine on 2026; the margin was the problem.

## 3. Secondary evidence

None of this feeds the verdict. It is registered as evidence and is reported whether or not it
flatters the model.

### Bias and RMSE

Fold 3, extended window, 353 test rows, unweighted / PA-weighted
(`artifacts/fold3_point.csv`):

| Model | MAE | RMSE | bias |
|---|---|---|---|
| last season | 0.0347 / 0.0318 | 0.0449 / 0.0409 | +0.0043 / +0.0001 |
| 3-year weighted | 0.0317 / 0.0289 | 0.0413 / 0.0372 | +0.0062 / +0.0015 |
| Marcel | 0.0283 / 0.0253 | 0.0371 / 0.0326 | +0.0092 / +0.0034 |
| **`ridge_rel`** | **0.0272 / 0.0247** | **0.0347 / 0.0311** | **+0.0012 / -0.0039** |

- **`ridge_rel` has the best level of the four models on 2026** (+0.0012 against Marcel's
  +0.0092), and the league-relative parameterisation is what buys that. It over-predicts
  slightly on 2026 where it under-predicted slightly on 2025 (-0.0013), so across the two
  seasons its level is roughly centred while every baseline over-predicts in both.
- **The RMSE gap survives better than the MAE gap**: 0.0347 against Marcel's 0.0371, a gap of
  0.0024 against the MAE gap of 0.0011. No bootstrap is attached -- RMSE is not a registered
  criterion, and attaching an interval to it after the fact would be choosing the metric that
  looks best after seeing which one does.
- **PA-weighted, the edge is 0.0006** (0.0247 against 0.0253), continuing fold 2's squeeze: most
  of what this model buys is on hitters under 300 PA. The PA-weighted bias of -0.0039 against
  -0.0062 on fold 2 says the same thing from the other side -- it still under-predicts regulars.
- The ordering of the four models is identical on both folds and under both weightings.

### Interval coverage, aggregate and by slice

Aggregate 80% coverage is **0.836** on fold 3 (0.815 on fold 2), inside the pre-registered
0.75-0.85 band, so criterion 3 passes. Sliced (`artifacts/fold3_interval_coverage.csv`), with
fold 2 beside it:

| Slice | Fold 3 n | Fold 3 coverage | Fold 3 width | Fold 2 coverage |
|---|---|---|---|---|
| pa_lag1 < 300 | 100 | 0.790 | 0.0930 | 0.800 |
| 300 <= pa_lag1 < 500 | 112 | 0.848 | 0.0867 | 0.798 |
| pa_lag1 >= 500 | 141 | **0.858** | 0.0815 | 0.844 |
| age < 26 | 73 | 0.849 | 0.0871 | **0.870** |
| 26 <= age < 31 | 171 | 0.836 | 0.0867 | 0.780 |
| age >= 31 | 109 | 0.826 | 0.0853 | 0.838 |

- **One slice is outside the band on each fold, and it is a different slice each time**: hitters
  under 26 on fold 2 (0.870), hitters with 500+ lag-1 PA on fold 3 (0.858). At 73-141 rows a
  coverage estimate carries roughly +/-0.07 to +/-0.09 at 95%, so neither is clearly outside, and
  the two do not point at a common structural fault. Per `DESIGN_FREEZE.md` section 6 this is a
  finding for v2, not a change to the verdict, and nothing was altered in response.
- **The intervals are a little wide on 2026 overall** (0.836 against a 0.80 target) while the
  widths are near-identical to fold 2's -- so the model's 2026 errors were slightly smaller
  relative to the training-window residual spread, rather than the widths having moved.
- The in-sample-residual weakness documented on fold 1's narrow window (coverage 0.628) does not
  appear here. Both uncertainty objects need the wide training window, and fold 3 used it.

### Probability skill and precision@k

Breakout and regression probabilities are read off the same predictive distribution, at the
registered +/-0.030 threshold, on the same 353 rows (`artifacts/fold3_probabilities.csv`). They
are **not** a pass/fail criterion, by design: they are a reading of a distribution that criteria
1-3 already test, so a fourth criterion on the same quantity would be double-counting.

| Label | Fold | base rate train / test | mean predicted | Brier | base-rate Brier | skill | model - base rate (95% CI) | gap |
|---|---|---|---|---|---|---|---|---|
| breakout | 2 | 0.154 / 0.171 | 0.171 | 0.1061 | 0.1422 | +0.254 | -0.0361 (-0.0501 to -0.0226) | 0.057 |
| breakout | **3** | 0.156 / 0.187 | 0.183 | 0.1221 | 0.1530 | **+0.202** | -0.0309 (**-0.0451 to -0.0160**) | 0.040 |
| regression | 2 | 0.279 / 0.257 | 0.273 | 0.1679 | 0.1914 | +0.123 | -0.0235 (-0.0373 to -0.0091) | 0.039 |
| regression | **3** | 0.277 / 0.272 | 0.253 | 0.1715 | 0.1980 | **+0.134** | -0.0266 (**-0.0397 to -0.0113**) | 0.053 |

- **Both labels beat their base rate on fold 3 with a paired bootstrap CI clear of zero**, so the
  forecast distribution carries information about the direction of a move and not only its size.
  This is the part of the holdout that reproduced cleanly. It is reported as evidence; it does
  not offset the failed criterion, and it was never registered as able to.
- Breakout skill fell (+0.254 -> +0.202) and regression skill rose (+0.123 -> +0.134). Breakout's
  test base rate rose to 0.187, which lifts the base-rate Brier it is scored against as well as
  the model's own, so the drop is not a clean statement about either.
- **Fold 2's top-decile under-confidence did not reproduce.** On fold 2 the top breakout decile
  observed 0.649 against 0.496 predicted, a gap of +0.153; on fold 3 it observed 0.556 against
  0.523, a gap of +0.032, and the count-weighted gap fell from 0.057 to 0.040. Design-freeze
  decision 5 pre-committed to the reading: "if fold 3 reproduces the shape, a calibration map
  with its own holdout is the v2 lever; if it does not, 36-37 rows a decile was the answer." It
  did not reproduce, so **36-37 rows a decile was the answer**, and no calibration map is
  warranted on this evidence. Regression's count-weighted gap moved the other way (0.039 ->
  0.053), which points at the same sampling noise rather than at a fixable miscalibration.

Precision at the top of the ranking, with the fold's own base rate as the chance level:

| Label | Fold 2 @10 | Fold 3 @10 | Fold 2 @25 | Fold 3 @25 | Fold 3 chance |
|---|---|---|---|---|---|
| breakout | 0.900 | 0.800 (8 of 10) | 0.680 | 0.640 (16 of 25) | 0.187 |
| regression | 0.800 | 0.800 (8 of 10) | 0.600 | 0.760 (19 of 25) | 0.272 |

The shortlist reading holds on 2026: the top of the breakout ranking is about four times the base
rate at k = 10 and three and a half times at k = 25. No bootstrap CI is attached, deliberately --
resampling rows changes which rows are in the top 10 at all -- and at these counts one player
moves precision@10 by 0.100. Evidence, not a criterion.

### Population accounting: 23.4% of eligible players are not scored

`artifacts/fold3_population.csv`. Eligible means at least 100 PA in the season before the
target; the buckets are what those players did in the target season.

| Fold (target) | Eligible | 0 PA | 1-99 PA | 100+ PA (scored) |
|---|---|---|---|---|
| fold 1 (2024) | 461 | 50 (10.8%) | 53 (11.5%) | 358 (77.7%) |
| fold 2 (2025) | 455 | 38 (8.4%) | 55 (12.1%) | 362 (79.6%) |
| **fold 3 (2026)** | **461** | **53 (11.5%)** | **55 (11.9%)** | **353 (76.6%)** |

- **23.4% of the eligible population is not scored on fold 3** -- 108 of 461 players -- the
  largest share of the three folds. The scored count reconciles exactly with the fold's 353 test
  rows, which is the check the script prints.
- **Every number in this card is conditional on the player taking 100 PA in 2026.** P(breakout)
  is not the chance a hitter breaks out; it is the chance he breaks out *given* that he plays.
  The players who drop out are disproportionately the ones whose forecast was worst, so the
  real-world distribution of outcomes has a tail this population cannot see.
- Coverage is measured on players who played, for the same reason: a hitter who got hurt has no
  wOBA to be covered. A survival model is a v2 item and was not bolted on after the freeze.

### Per player

`artifacts/fold3_players.csv` carries a row per scored player -- both forecasts, the signed gap,
each model's absolute error, which came closer, the published interval and whether it covered.

| Reading | Fold 2 | Fold 3 |
|---|---|---|
| forecast gap, `ridge_rel` - Marcel | mean -0.0105, median -0.0095, sd 0.0123 | mean -0.0080, median -0.0083, sd 0.0129 |
| gap range | -0.0557 to +0.0204 | -0.0551 to +0.0281 |
| higher than Marcel / lower | 70 / 292 | 87 / 266 |
| `ridge_rel` closer to the outcome | 53.0% | **53.3%** |
| mean error advantage | +0.0023 | **+0.0011** |

`ridge_rel` is closer than Marcel on 53.3% of players -- near a coin flip, and almost exactly
fold 2's 53.0%. The per-player win *rate* is the one comparison that did not shrink; what shrank
is the size of the wins. A reader who takes "beats Marcel" to mean "is the better forecast for
this hitter" would be wrong about 47% of the time on both folds. It still forecasts lower than
Marcel almost everywhere (266 of 353 players, mean gap -0.0080), which is the aging curve and the
mean regression Marcel has and the league-relative ridge does not. No per-player leaderboard was
inspected on fold 3 beyond this summary, as `DESIGN_FREEZE.md` section 5 requires.

## 4. What the model is

Pinned in full by `DESIGN_FREEZE.md` section 2; in brief:

- **Forecaster.** `models.relative.relative_ridge`: a `LeagueRelative` wrapper around
  `RidgeForecaster`, `alpha = 10.0`, fitted on `y_woba - y_lg_woba` and re-levelled on predict
  with `lg_woba_hat` (the prior season's league wOBA, a random walk). Median imputer, standard
  scaler and ridge, all fitted inside the training window only.
- **Features.** `models.ridge.RELATIVE_FEATURES`, 31 columns: `woba_rel`, `xwoba_rel`,
  `avg_ev_rel`, `barrel_pct_rel`, `hard_hit_pct_rel`, `pa`, `k_pct`, `bb_pct`, `miss`, `short`
  at lags 1-3, plus `age_target`. No park features. No empirical Bayes prior.
- **Training window.** Extended: for fold 3, target seasons 2016-2019 and 2021-2025. 2020 is
  never a target and is flagged per lag.
- **Population.** At least 100 PA in the lag-1 feature season and at least 100 PA in the target
  season. Features for target season T come only from seasons strictly before T
  (`tests/test_no_leakage.py`).
- **Intervals.** `ResidualQuantileIntervals`: empirical residual quantiles of the training fit,
  bucketed by `pa_lag1` at edges (300, 500), `min_bucket = 50` with a pooled fallback, level
  0.80. The residual sample is one residual per training row and never touches a test row, which
  `tests/test_residual_provenance.py` proves three ways.
- **Probabilities.** `BucketedResidualDistribution`, same factory and edges; breakout is a target
  season at least 0.030 wOBA above `Weighted3` (weights 5/4/3, PA-weighted, no mean regression,
  no age adjustment), regression at least 0.030 below. The 10th and 90th percentiles of that
  distribution are exactly the published 80% interval, so the probabilities cannot contradict the
  forecast; there is no separate classifier.
- **Baselines.** `LastSeason`, `Weighted3`, `MarcelBaseline` (weights 5/4/3,
  `regression_pa = 1200`, `age_factor` 0.006/0.003 either side of 29).

## 5. Intended use and limitations

**Intended use.** A decision aid for baseball-operations questions about established MLB hitters:
a next-season wOBA forecast with an 80% interval, the probability of a +/-0.030 move against the
hitter's own recent record, and historical comparables. The forecast is a statistical estimate,
not a guarantee, and the labels describe model expectations, not causal claims.

**Not shown to beat Marcel on unseen data.** This is the headline limitation and it follows from
section 1. On the one season nothing was fitted to, the model's MAE edge over Marcel is 0.0011
with a 95% interval of -0.0024 to +0.0003. Anyone deciding whether to carry this model instead of
a public Marcel implementation should read that as "not demonstrated", and should weigh what *is*
demonstrated separately: the better level (bias +0.0012 against +0.0092), the RMSE gap, 80%
intervals inside the pre-registered band on both folds, and probability skill clear of its base
rate on both folds. The intervals and the probabilities are the part of v1 that Marcel does not
offer at all.

**Our Marcel is not Tango's Marcel.** It projects wOBA as one rate rather than per component, and
it is only asked about hitters with 100+ PA in the prior season -- the population it handles best.
All thirteen differences are audited in [MARCEL_AUDIT.md](MARCEL_AUDIT.md), written before the
holdout ran so that none of it could be produced afterwards as an excuse. Criterion 1 should be
read as "beats that baseline on established hitters".

**The league-average column has a composition break at 2022.** `lg_woba` is the PA-weighted mean
of the Savant batter leaderboard, which carries pitchers' batting lines through 2021 and
effectively drops them with the universal DH in 2022, so the series sits about 5 wOBA points low
before 2022. That column centres the target the model is fitted on, so training targets from
2016-2021 are centred differently from the 2024-2026 test rows. Found during the Marcel audit,
reported rather than fixed -- the league definition is part of the frozen configuration -- and the
strongest candidate for a v2 re-freeze.

**`lg_woba_hat` is a random walk.** The forecast-date league estimate is simply last season's
league wOBA. Fold 1 showed the cost when the environment moves (bias +0.0111, as league wOBA fell
.3184 to .3102 into 2024). It cost nothing visible on 2026.

**Conditional on playing.** Everything is scored on players who reached 100 PA in the target
season: 76.6% of the eligible population on fold 3. See section 3.

**Most of the edge is in part-timers.** PA-weighted, the edge over Marcel is 0.0006 on fold 3 and
0.0008 on fold 2. Most of what the Statcast features and the league-relative target buy is on
hitters a club has least reason to care about.

**No park factors.** Deferred to v2. A hitter's wOBA carries his home park and nothing in the
feature set separates the two, so a forecast for a hitter who changes parks -- or whose park is
re-dimensioned -- absorbs that as a skill change. League-relative normalisation removes the
league-wide environment, not the park.

**Contact quality is justified by the intervals and the product question, not by MAE.** The group
was adopted on +0.0008 MAE whose paired bootstrap CI touched zero (-0.0001 to +0.0017). It stays
because dropping it widens the 80% intervals (0.0899 against 0.0874) and pushes coverage to
0.831, and because the observed-versus-expected gap is one of the two questions v1 exists to
answer. Fold 3 is consistent with that group being the softest part of the design.

**MLB hitters only.** No pitchers, no minor leagues, no contracts, no team-change
counterfactuals. Public data only: Statcast via `pybaseball`, the MLB Stats API, and the Chadwick
register for IDs.

## 6. What happens next

Per `DESIGN_FREEZE.md` section 7, a fail is not retried and the fold 3 number from
`design-freeze-v1` stands as the v1 result. What is still allowed without a new tag:

- **The 2027 production forecast.** The same frozen configuration, retrained on everything
  through 2026, timestamped and published before the season, scored after it. That is a genuine
  prospective test and the only honest way to get a second out-of-sample read; given this
  holdout, it is the thing to wait for rather than a second backtest.
- Comps, the ranking, the dashboard, this card, the data dictionary and the write-up.

The v2 re-freeze list, in the order the evidence points:

1. **The `lg_woba` composition break at 2022** -- a defensible league definition applied
   consistently across the whole series.
2. **A league-level estimator better than a random walk**, fitted on a window that does not
   contain the test season.
3. **Park factors**, with the ablation that was deferred along with them.
4. **A pre-registered PA-weighted criterion**, since the unweighted / weighted split is now the
   most informative thing about where this model's edge lives -- and picking the weighting after
   seeing which way it cuts is exactly what the freeze exists to prevent.
5. **A survival model** for the 23.4% who are not scored.
6. **Interval bucketing on more than lag-1 PA** -- `quantile_reg` already does the general
   version and covered 0.801 on fold 2.

A calibration map for the breakout probabilities comes **off** the v2 list on this evidence: the
fold-2 under-confidence it would have corrected did not reproduce on fold 3.

## 7. Reproducing the holdout

```bash
git checkout design-freeze-v1
python -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('data/processed/hitter_seasons.parquet').read_bytes()).hexdigest())"
# must print b164c087e6186d989f52bebdcc1b1153f25f7de5d8f2f6b8ddb003c906039cd8
uv run python scripts/run_holdout.py
```

`run_holdout.py` refuses to run fold 3 unless `fold3.frozen` is `true`, HEAD is tagged exactly
`design-freeze-v1`, and the working tree is clean. The artifacts from the single look are
committed as `artifacts/fold3_*.csv`, and `scripts/check_docs.py` matches every number this card
quotes against them.
