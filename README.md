# DiamondIQ

*(working title)*

MLB hitter forecasting for baseball operations: next-season projections with calibrated uncertainty, breakout/regression probabilities derived from the forecast distribution, historical comparables, and a buy-low / sell-high scouting dashboard.

**Status:** v1 in development. Scope is MLB hitters only. Folds 1-2 run on real 2015-2026
Savant data; fold 3 (2026) stays untouched until the design freeze, which has not happened --
the checklist and the pass/fail rules are in [DESIGN_FREEZE.md](docs/DESIGN_FREEZE.md). See
[Results so far](#results-so-far).

## Questions v1 answers
1. Which hitters are likely to outperform what their recent results suggest (buy-low), and which to underperform (sell-high)?
2. What is each hitter's next-season forecast, with calibrated uncertainty?
3. Who are the closest historical comparables, and what happened to them next?

## Approach
- **Target:** next-season wOBA (plus K%, BB%, and a simple PA projection). Own value metric: runs above average from wOBA x projected PA.
- **Baselines:** last season, 3-year weighted average, and Marcel (weighted 3-year, regressed to the mean, age-adjusted). Archived public projections (Steamer/ZiPS) if obtainable.
- **Model:** regularized regression (ridge / elastic net) on lagged skill, Statcast contact-quality features, age, PA, and park. Gradient boosting as a comparison only. Empirical Bayes shrinkage is tested once and adopted only if it clearly beats plain ridge.
- **Uncertainty:** quantile regression or conformal intervals on the same forecaster.
- **Breakout / regression:** read off the forecast distribution, so they cannot contradict the forecast. Breakout is a target season at least 0.030 wOBA above a weighted 3-year baseline, regression at least 0.030 below; the definitions were fixed and committed before anything was computed ([PROJECT_PLAN.md](docs/PROJECT_PLAN.md), section 6). Threshold sensitivity is reported.
- **Similarity:** standardized skill/age features with nearest neighbors, showing comps' next-season outcomes.
- **Undervalued (no contract data):** forecast vs baseline/public projection gap, plus the xwOBA-wOBA gap, ranked with uncertainty.
- **Era handling:** 2020 short season flagged or down-weighted; 2021 ball and 2023 rule changes handled by league-season normalization and tested season effects.

## Evaluation protocol
Expanding-window folds; the model is re-initialized from scratch for each fold. Defined in `configs/folds.yaml`.

| Fold | Train targets | Test target | Purpose |
|---|---|---|---|
| 1 | 2023 | 2024 | pipeline sanity check |
| 2 | 2023-24 | 2025 | iterate and select models |
| 3 | 2023-25 | 2026 | final holdout, run once after design freeze (git tag) |
| Production | 2023-26 | 2027 | frozen, timestamped forecasts, scored after the 2027 season |

- An "extended" configuration adds 2015-2022 training seasons and is compared against the 2023-only version.
- Hyperparameters, imputers, scalers, and calibration are fit only inside each training window.
- Success criteria are fixed before the 2026 fold is run: beat Marcel MAE on folds 2 and 3 with a bootstrap CI excluding zero, and 80% intervals covering 75-85% of outcomes. If they are not met, that is reported as-is.
- Ablations remove park, contact-quality, age, and shrinkage features one at a time.
- A leakage test (`tests/test_no_leakage.py`) asserts no feature row uses data from the target season or later.

### Results so far

MAE on next-season wOBA, from `scripts/run_folds.py`, which writes `artifacts/fold_results.csv`.
`ridge_rel` forecasts wOBA relative to the league average and puts the level back with the prior
season's league wOBA; "extended" is the wider training window from `folds.yaml` (targets 2016-2019
and 2021-2022 added, 2020 flagged as a lag season rather than trained on).

| Model | fold1 MAE (2024) | fold2 MAE (2025) |
|---|---|---|
| last season | 0.0359 | 0.0325 |
| 3-year weighted | 0.0333 | 0.0306 |
| Marcel | 0.0310 | 0.0285 |
| ridge, absolute features | 0.0322 / 0.0312 | 0.0265 / 0.0289 |
| **ridge_rel** | 0.0310 / **0.0282** | 0.0266 / **0.0263** |

`base / extended` training window. The baselines do not train, so both windows give them the same
number. Fold 2 is the selection fold; fold 1 is a sanity check only.

- Selected on fold 2: `ridge_rel` on the extended window. MAE 0.0263 against Marcel's 0.0285,
  paired bootstrap -0.0023 (95% CI -0.0036 to -0.0008), RMSE 0.0335 against 0.0368, bias -0.0013.
- Both folds over-predict less than Marcel does, but fold 1 still over-predicts (bias +0.0111,
  Marcel +0.0151): league wOBA fell .3184 to .3102 going into 2024, and a forecast-date estimate
  built from prior seasons cannot see that coming.
- The absolute-feature ridge gets *worse* on the wider window (fold 2 bias -0.0141). Its
  league-level lag columns are near-constant inside a fold, which gives them a large standardised
  coefficient that extrapolates badly; it is kept as the comparison point for the relative
  parameterisation.
- Fold 3 has not been run, and the design is not frozen.

80% prediction intervals (`src/mlb_outlook/models/intervals.py`), coverage and mean width on
fold 2, extended window. Each wraps the same point forecaster, so these compare uncertainty
estimates rather than forecasts:

| Approach | Coverage | Mean width |
|---|---|---|
| **residual quantiles by PA bucket** | **0.815** | **0.0874** |
| split conformal | 0.840 | 0.0917 |
| split conformal, PA buckets | 0.837 | 0.0916 |
| quantile regression on residuals | 0.801 | 0.0857 |

All four land inside the pre-registered 75-85% range on fold 2. Residual quantiles by PA bucket
are the current pick: they keep the full-data point fit, and per-bucket coverage is flatter
(0.800 / 0.798 / 0.844 across the three PA buckets) than a constant width gives
(0.800 / 0.845 / 0.867). Both in-sample approaches need the wide training window -- on fold 1's
one-season base window, residual quantiles cover 0.628 and quantile regression 0.480.

Feature-group ablations on the selected configuration (`scripts/run_ablations.py`, writes
`artifacts/ablation_results.csv`), fold 2 only, one group dropped at a time and the model
re-fitted from scratch. The CI is a paired bootstrap of ablated minus full MAE on the same
362 test rows, so positive means the group was worth keeping.

| Group dropped | Cols | MAE | vs full (95% CI) | 80% coverage |
|---|---|---|---|---|
| none (selected) | 31 | 0.0263 | - | 0.815 |
| contact quality | -12 | 0.0271 | +0.0008 (-0.0001 to +0.0017) | 0.831 |
| K% / BB% | -6 | 0.0268 | +0.0006 (**+0.0002 to +0.0009**) | 0.815 |
| age | -1 | 0.0266 | +0.0003 (-0.0003 to +0.0009) | 0.807 |
| PA / missing-season flags | -9 | 0.0263 | +0.0000 (-0.0005 to +0.0005) | 0.809 |
| league-relative rates | swap 15 | 0.0262 | -0.0000 (-0.0002 to +0.0002) | 0.818 |

No group clearly hurts, so the selected configuration stands. K%/BB% is the only group whose
CI clears zero. Contact quality costs the most MAE but its CI touches zero, which is the
same comparison that got it adopted; it is flagged, not acted on. PA and the missing-season
flags are free to drop. Swapping the league-relative rate lags for absolute ones changes
nothing -- `RELATIVE_FEATURES` carries no `lg_woba_lag` column to drop in the first place.
Every ablation stays inside the 75-85% coverage range. Park factors are deferred to v2, so
the park ablation is not part of v1. See
[PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the per-group reading.

Breakout and regression probabilities, read off the same predictive distribution
(`src/mlb_outlook/models/probabilities.py`, scored by `evaluation/calibration.py` via
`scripts/run_probabilities.py`). Breakout is a target season at least 0.030 wOBA above the
weighted 3-year baseline, regression at least 0.030 below; both definitions were committed
before anything was computed. Fold 2 only, extended window, the same 362 test rows as above.
The baseline forecast hands every player the training window's base rate.

| Label | base rate train / test | mean predicted | Brier | base-rate Brier | skill | model - base rate (95% CI) |
|---|---|---|---|---|---|---|
| **breakout** | 0.154 / 0.171 | 0.171 | **0.1061** | 0.1422 | **+0.254** | -0.0361 (**-0.0501 to -0.0226**) |
| **regression** | 0.279 / 0.257 | 0.273 | **0.1679** | 0.1914 | **+0.123** | -0.0235 (**-0.0373 to -0.0091**) |

Both beat their base rate with a paired bootstrap CI clear of zero. The 10th and 90th
percentiles of this distribution are exactly the published 80% interval, so the
probabilities cannot contradict the forecast -- there is no separate classifier.

- Regression is the commoner label (0.257 vs 0.171) because the registered baseline carries
  no aging curve and no regression to the mean, so a hitter who got lucky or is past his peak
  sits below his own three-year line more often than above it. Each label is therefore scored
  against its own base rate.
- Mean predicted P(breakout) matches the observed rate to three decimals; regression runs 1.6
  points high. Per-decile reliability is another matter: the breakout table under-observes in
  deciles 1-7 and over-observes in the top two (0.649 observed against 0.496 predicted), so
  the surest calls break out more often than stated. Count-weighted gaps are 0.057 and 0.039.
  Nothing was fitted to correct this -- a calibration map tuned on fold 2 would need its own
  holdout. It goes into the freeze as a reported property.
- At the top of the ranking, precision@10 is 0.900 for breakout and 0.800 for regression,
  precision@25 is 0.680 and 0.600, against chance levels of 0.171 and 0.257. That is the
  shortlist reading of the same under-confidence the top decile shows. It is evidence rather
  than a criterion, and no bootstrap CI is attached: resampling rows changes which rows are
  in the top 10 at all. One player moves precision@10 by 0.100.
- Skill stays positive at thresholds of 0.020 and 0.040 for both labels. The registered 0.030
  does not move because a neighbour scores better.

Per-decile tables and the per-threshold reading are in
[PROJECT_PLAN.md](docs/PROJECT_PLAN.md); fold 3 has not been run and the design is not frozen.

## Data
- Statcast via `pybaseball`, MLB Stats API and the Chadwick register for player IDs. Seasons 2015-2026 are pulled for features and lag history.
- FanGraphs-style figures are reference only; licensing is checked before any scraping, and metrics are self-computed where possible.
- Every record keeps its source and retrieval date. Missing values are never treated as zero. Bat-tracking fields (2023+) are optional features with explicit missing flags.

## Stack
Python, pandas, scikit-learn, MySQL 8 (Docker) via SQLAlchemy + PyMySQL, Parquet for raw Statcast caches, Streamlit + Plotly for the dashboard, pytest, ruff, pre-commit.

MySQL is the system of record and feature-query layer (versioned `.sql` schema, keys and indexes, window functions for rolling/lag features, as-of queries). Model fitting happens in pandas/scikit-learn.

## Dashboard (planned)
Scouting board (buy-low / sell-high / breakout / regression lists), player profile, player comparison, data coverage and model version, and a limitations page. Player profiles show a percentile panel, season and Statcast tables, spray chart, rolling xwOBA, pitch-mix faced, and a DiamondIQ panel with the forecast and 80% interval, breakout/stable/regression probabilities, observed-vs-expected gap, and comps. The layout is custom; no third-party assets, logos, or headshots are used.

## Setup
```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,app]"
cp .env.example .env
docker compose up -d      # MySQL 8
pytest
```

## Repo layout
```
app/            Streamlit dashboard
configs/        folds.yaml, data and model configs
data/           raw/ and processed/ (git-ignored)
docs/           PROJECT_PLAN.md (scope, evaluation, timeline), DESIGN_FREEZE.md (what the tag pins, how fold 3 is run, pass/fail), PLAN.md (overnight phases), REPORT_TEMPLATE.md
notebooks/      data audit and feature analysis
reports/        overnight run reports
scripts/        run_folds.py, run_ablations.py, run_probabilities.py, night_run.sh (overnight driver), verify_phase.sh, check_bootstrap.sh
sql/            versioned schema
src/mlb_outlook/  ingestion, validation, features, context, models, evaluation, reporting
tests/
```

Project scope, models, and evaluation design: [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md). What the design freeze pins and how the 2026 holdout gets its one run: [docs/DESIGN_FREEZE.md](docs/DESIGN_FREEZE.md). Phased plan for unattended overnight runs: [docs/PLAN.md](docs/PLAN.md), driven by `scripts/night_run.sh`.

## Roadmap
Weeks 1-2: data audit, MySQL schema, ID mapping, as-of feature builder, leakage test, Marcel baseline. Weeks 3-5: fold harness, ridge forecaster, intervals, derived breakout probabilities, ablations, design freeze. Weeks 6-7: comps, undervalued ranking, dashboard. Week 8: run fold 3 once, write-up, model card. Week 9: 2027 production forecasts and deploy. Target completion around Jan 1, 2027, with February as buffer.

Out of scope for v1 (planned for v2): park factors, pitchers, minor-league projection, contracts and acquisition value, counterfactual team-change simulation, development lab.

## Limitations
Public data only. Forecasts are statistical estimates, not guarantees; breakout and undervalued labels describe model expectations, not causal claims. Historical simulations show predictive usefulness, not that a trade would have produced the same result.

- **No park factors in v1.** Park is deferred to v2. A hitter's wOBA carries his home park, and nothing in the feature set separates the two, so a forecast for a hitter who changes parks -- or whose park is re-dimensioned -- absorbs that as a skill change. League-relative normalisation removes the league-wide environment, not the park, and the planned park ablation moves to v2 with the features.
- **Breakout and regression probabilities are conditional on playing again.** They are scored only on players with at least 100 PA in the target season, matching the fold population, so they say nothing about the chance of losing playing time or missing the season.

---

## Project bootstrap spec (for Claude Code)

If the files below do not exist yet, create them exactly as written, create the folder layout above (add `.gitkeep` to empty folders: `data/raw`, `data/processed`, `artifacts`, `sql`, `notebooks`, `app/pages`, `app/components`, and each `src/mlb_outlook/*` subpackage gets an `__init__.py`), then commit. Do not install dependencies or pull data until asked.

### `pyproject.toml`
```toml
[project]
name = "mlb-outlook"
version = "0.1.0"
description = "MLB hitter forecasting, breakout/regression probabilities, and buy-low/sell-high scouting."
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "sqlalchemy>=2.0",
    "pymysql>=1.1",
    "cryptography>=42",
    "pyarrow>=15",
    "pybaseball>=2.2",
    "pydantic>=2.6",
    "pyyaml>=6.0",
    "requests>=2.31",
    "scikit-learn>=1.4",
]

[project.optional-dependencies]
app = ["streamlit>=1.35", "plotly>=5.20"]
dev = ["pytest>=8.0", "ruff>=0.5", "pre-commit>=3.7"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mlb_outlook"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### `configs/folds.yaml`
```yaml
# Expanding-window evaluation folds. Each fold builds a fresh model from scratch.
# "train_targets" are the seasons whose outcomes the model learns from;
# features for a target season come only from seasons strictly before it.
forecast_date_rule: "end of regular season preceding the target season"

folds:
  - name: fold1
    train_targets: [2023]
    test_target: 2024
    purpose: pipeline sanity check
  - name: fold2
    train_targets: [2023, 2024]
    test_target: 2025
    purpose: iterate and select models here
  - name: fold3
    train_targets: [2023, 2024, 2025]
    test_target: 2026
    purpose: FINAL holdout; run once after design freeze (git tag)
    frozen: false

production:
  name: prod2027
  train_targets: [2023, 2024, 2025, 2026]
  forecast_target: 2027

# Alternate configuration: extra pre-2023 training seasons, compared against the above.
extended:
  extra_train_targets: [2016, 2017, 2018, 2019, 2021, 2022]
  flag_seasons: [2020]  # short season; down-weight or flag

min_pa:
  feature_season: 100
  target_season: 100

success_criteria:
  beats_marcel_mae_on: [fold2, fold3]
  bootstrap_ci_excludes_zero: true
  interval_80_coverage_range: [0.75, 0.85]
```

### `docker-compose.yml`
```yaml
services:
  mysql:
    image: mysql:8.4
    container_name: diamondiq-mysql
    restart: unless-stopped
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD}
      MYSQL_DATABASE: ${MYSQL_DATABASE}
      MYSQL_USER: ${MYSQL_USER}
      MYSQL_PASSWORD: ${MYSQL_PASSWORD}
    ports:
      - "3306:3306"
    command: ["--local-infile=1", "--character-set-server=utf8mb4"]
    volumes:
      - ./mysql_data:/var/lib/mysql
      - ./sql:/docker-entrypoint-initdb.d:ro
```

### `.env.example` (copy to `.env`; never commit `.env`)
```
MYSQL_ROOT_PASSWORD=change_me_root
MYSQL_DATABASE=diamondiq
MYSQL_USER=diamondiq
MYSQL_PASSWORD=change_me
DATABASE_URL=mysql+pymysql://diamondiq:change_me@127.0.0.1:3306/diamondiq
```

### `.gitignore`
```
__pycache__/
*.pyc
.venv/
.env
.ruff_cache/
.pytest_cache/
.ipynb_checkpoints/
mysql_data/
.night/

# data and model artifacts stay out of git
data/raw/*
data/processed/*
artifacts/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!artifacts/.gitkeep
```

### `tests/test_no_leakage.py` (placeholder until the feature builder exists)
```python
"""Leakage guard.

Contract: every feature row for target season T is derived only from data dated
before the forecast date for T. The feature builder must expose, per row,
`max_source_season` (latest season of any input); this test asserts
max_source_season < target_season for every row.
"""

import pytest


@pytest.mark.skip(reason="feature builder not implemented yet (week 2)")
def test_features_use_only_prior_seasons():
    raise NotImplementedError
```

### Next task after bootstrap
Data audit: design the MySQL schema in `sql/` (players, player_seasons, batted_ball_events, park_context, forecasts), build the player ID mapping (Chadwick register), ingest 2015-2026 Statcast and season data via pybaseball into Parquet then MySQL, and write a notebook measuring coverage and missingness by metric, player, and season. Before relying on any source, confirm its terms allow automated access.
