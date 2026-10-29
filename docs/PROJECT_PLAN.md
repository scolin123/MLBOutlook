# DiamondIQ v1 (revised plan)

Audience: baseball-ops hiring. Target: done by ~Jan 1, 2027, hard stop Feb 2027.
Scope: MLB hitters only. Pitchers, minors, contracts, counterfactual simulator, and Development Lab are v2.

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
- Breakout/regression: derived from the forecast distribution, defined as actual minus baseline (Marcel) expectation. No separate classifier, so outputs cannot contradict. Report sensitivity to thresholds.
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
- Ablations: remove park, Statcast contact quality, age, shrinkage prior one at a time.
- Historical decision simulation: top-25 buy-low lists each fold, realized outcome vs baseline.

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
