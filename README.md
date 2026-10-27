# MLB Outlook

*(working title)*

MLB hitter forecasting for baseball operations: next-season projections with calibrated uncertainty, breakout/regression probabilities derived from the forecast distribution, historical comparables, and a buy-low / sell-high scouting dashboard.

**Status:** v1 in development. Scope is MLB hitters only. No results yet; this README will be updated as evaluation folds are run.

## Questions v1 answers
1. Which hitters are likely to outperform what their recent results suggest (buy-low), and which to underperform (sell-high)?
2. What is each hitter's next-season forecast, with calibrated uncertainty?
3. Who are the closest historical comparables, and what happened to them next?

## Approach
- **Target:** next-season wOBA (plus K%, BB%, and a simple PA projection). Own value metric: runs above average from wOBA x projected PA.
- **Baselines:** last season, 3-year weighted average, and Marcel (weighted 3-year, regressed to the mean, age-adjusted). Archived public projections (Steamer/ZiPS) if obtainable.
- **Model:** regularized regression (ridge / elastic net) on lagged skill, Statcast contact-quality features, age, PA, and park. Gradient boosting as a comparison only. Empirical Bayes shrinkage is tested once and adopted only if it clearly beats plain ridge.
- **Uncertainty:** quantile regression or conformal intervals on the same forecaster.
- **Breakout / regression:** derived from the forecast distribution and defined as actual minus baseline (Marcel) expectation, so it cannot contradict the forecast. Sensitivity to thresholds is reported.
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

## Data
- Statcast via `pybaseball`, MLB Stats API and the Chadwick register for player IDs. Seasons 2015-2026 are pulled for features and lag history.
- FanGraphs-style figures are reference only; licensing is checked before any scraping, and metrics are self-computed where possible.
- Every record keeps its source and retrieval date. Missing values are never treated as zero. Bat-tracking fields (2023+) are optional features with explicit missing flags.

## Stack
Python, pandas, scikit-learn, MySQL 8 (Docker) via SQLAlchemy + PyMySQL, Parquet for raw Statcast caches, Streamlit + Plotly for the dashboard, pytest, ruff, pre-commit.

MySQL is the system of record and feature-query layer (versioned `.sql` schema, keys and indexes, window functions for rolling/lag features, as-of queries). Model fitting happens in pandas/scikit-learn.

## Dashboard (planned)
Scouting board (buy-low / sell-high / breakout / regression lists), player profile, player comparison, data coverage and model version, and a limitations page. Player profiles show a percentile panel, season and Statcast tables, spray chart, rolling xwOBA, pitch-mix faced, and a MLB Outlook panel with the forecast and 80% interval, breakout/stable/regression probabilities, observed-vs-expected gap, and comps. The layout is custom; no third-party assets, logos, or headshots are used.

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
docs/           PLAN.md, methodology, data dictionary, model card
notebooks/      data audit and feature analysis
sql/            versioned schema
src/mlb_outlook/  ingestion, validation, features, context, models, evaluation, reporting
tests/
```

## Roadmap
Weeks 1-2: data audit, MySQL schema, ID mapping, as-of feature builder, leakage test, Marcel baseline. Weeks 3-5: fold harness, ridge forecaster, intervals, derived breakout probabilities, ablations, design freeze. Weeks 6-7: comps, undervalued ranking, dashboard. Week 8: run fold 3 once, write-up, model card. Week 9: 2027 production forecasts and deploy. Target completion around Jan 1, 2027, with February as buffer.

Out of scope for v1 (planned for v2): pitchers, minor-league projection, contracts and acquisition value, counterfactual team-change simulation, development lab.

## Limitations
Public data only. Forecasts are statistical estimates, not guarantees; breakout and undervalued labels describe model expectations, not causal claims. Historical simulations show predictive usefulness, not that a trade would have produced the same result.

---

## Project bootstrap spec (for Claude Code)

If the files below do not exist yet, create them exactly as written, create the folder layout above (add `.gitkeep` to empty folders: `data/raw`, `data/processed`, `artifacts`, `sql`, `notebooks`, `app/pages`, `app/components`, and each `src/mlb_outlook/*` subpackage gets an `__init__.py`), then commit. Do not install dependencies or pull data until asked.

### `pyproject.toml`
```toml
[project]
name = "mlb_outlook"
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
    container_name: mlb-outlook-mysql
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
MYSQL_DATABASE=mlb_outlook
MYSQL_USER=mlb_outlook
MYSQL_PASSWORD=change_me
DATABASE_URL=mysql+pymysql://mlb_outlook:change_me@127.0.0.1:3306/mlb_outlook
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