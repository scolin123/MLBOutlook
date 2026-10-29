# Overnight plan, night 2

Drives `NIGHT_PLAN=n2 bash scripts/night_run.sh`. Project scope and evaluation design live in [PROJECT_PLAN.md](PROJECT_PLAN.md). Night 1 (`docs/PLAN.md`) built the folds config validator (`src/mlb_outlook/config.py`) and the Marcel baseline (`src/mlb_outlook/models/marcel.py`). This plan builds on both. Everything is offline: synthetic data only.

## RULES (read first, apply to every phase)

**Git**
1. Work only on branch `night/<NIGHT_DATE>`. The driver passes `NIGHT_DATE` (it may carry a suffix such as `2026-10-28-n2`); use that exact value. Never touch `main`. Never push. No deploys. Never run `git push`, `git merge`, `git rebase`, `git reset`, or change remotes.
2. A phase is complete only when ALL its acceptance commands pass, verified by actually running them and reading the output. Then commit with exactly the message given for the phase (`n2 phase N: <name>`), including the updated `NIGHT_LOG.md` and report in that commit.
3. Before every commit run `git status` and `git diff --cached --name-only`. If `.env`, anything under `data/raw/`, `data/processed/`, `artifacts/`, `mysql_data/`, or `.night/` is staged, unstage it and fix `.gitignore`. No secrets in commits.

**Time** (the clock crosses midnight, so never compare clock strings)
4. Before starting each phase run `date +%s`. If the epoch is >= `PHASE_CUTOFF_EPOCH` (in the prompt and `.night/state.env`; 03:15 local), do NOT start the phase: skip to the report. The driver hard-stops the run at 04:00 local.

**Failure handling**
5. If a phase fails, make up to 3 fix attempts. If it still fails: stop building, log the exact error output, do NOT start later phases, write `FAIL` to `.night/status/phase-N`, commit ONLY `NIGHT_LOG.md` and the report with message `night: phase N failed (log only)`, leave partial code uncommitted, and go to the report.
6. Never build on a failing phase. Never skip, weaken, xfail, or delete a test to make it pass. Never edit an acceptance command. Fixing a test input that contradicts the written spec is allowed only if you record the spec line and the correction under Decisions.
7. Do only the phase you were invoked for. Do not read ahead.

**Scope**
8. Offline only: no data pulls, no network, no pybaseball data calls, no new dependencies (use what `pyproject.toml` already lists). Do not edit `configs/folds.yaml`, `README.md`, or any file under `docs/` except the report files.
9. Earlier code (`config.py`, `models/marcel.py`, existing tests) may be reused but not weakened. Existing tests must keep passing.
10. Decisions the plan does not settle: choose the simplest reasonable option and record it under "Decisions".

**Working style (this makes the unattended run reliable)**
11. Run ONE simple command per Bash call. No pipes, subshells, heredocs, `;` or `&&` chains, or output redirection. Create and edit files with the Write and Edit tools, never with `cat >`/`echo >`. Use `uv run --no-sync <tool>` for ruff, pytest and python. Lint line length is 100.

**Logging (after EVERY phase, pass or fail)**
12. Append to `NIGHT_LOG.md`: phase, start/end time, commands run, pass/fail, decisions, blockers.
13. Create or update `reports/<NIGHT_DATE>-overnight.md` using the structure of `docs/REPORT_TEMPLATE.md`, but with the phase table listing THIS plan's six phases (leakage test and as-of builder core, lag features, baselines, evaluation harness, ridge forecaster, morning report).
14. Write the single word `PASS` or `FAIL` to `.night/status/phase-N`. The driver reads it and independently re-runs `scripts/verify_n2.sh N`; both must agree.

## Shared data model

Long table `player_seasons` (pandas DataFrame), one row per player-season:
`player_id` (int), `season` (int), `age` (int, age during that season), `pa` (int), `woba` (float), `k_pct` (float, may be NaN), `bb_pct` (float, may be NaN).

Fold and config objects come from `mlb_outlook.config.load_folds_config('configs/folds.yaml')`: read `src/mlb_outlook/config.py` to see the exact field names (fold name, `train_targets`, `test_target`, optional `frozen`, `min_pa.feature_season`, `min_pa.target_season`).

---

## Phase 1: leakage test and as-of builder core
Commit message: `n2 phase 1: leakage test and as-of builder core`

**Tasks**
1. Create `tests/synthetic.py` with `make_seasons(seasons=range(2018, 2027), n_players=300, seed=7) -> DataFrame` in the long format above, and a fixture `seasons_df` in `tests/conftest.py`. Generator model (fixed constants, seeded `numpy.random.default_rng`): each player has talent `T ~ Normal(0.320, 0.025)`, a debut season drawn uniformly from the available seasons, a career length uniform on 3..9 seasons (cut at the last season), and debut age uniform on 21..28 (age grows by 1 per season). Per season: `pa = randint(80, 650)`; `woba = T + aging(age) + Normal(0, 0.045 * sqrt(300 / pa))` with `aging(age) = 0.001 * (27 - age)` for age <= 27 and `-0.0015 * (age - 27)` for age > 27; `k_pct = 0.22 + 0.5 * (0.320 - T) + Normal(0, 0.02)`; `bb_pct = 0.08 + 0.4 * (T - 0.320) + Normal(0, 0.01)`; set about 3% of `k_pct` and `bb_pct` values to NaN. Same seed must give an identical frame.
2. Create `src/mlb_outlook/features/asof.py` with `build_asof_rows(df, target_season) -> DataFrame`: one row per player that has at least one row with `season < target_season`; columns `player_id`, `target_season`, `max_source_season` (latest season among that player's rows with `season < target_season`), `n_source_seasons`. Rows with `season >= target_season` must never influence the output.
3. Replace the skipped placeholder in `tests/test_no_leakage.py` (keep the contract docstring) with real tests, none skipped:
   - for every `target_season` in 2020..2026 on `seasons_df`: `(rows.max_source_season < rows.target_season).all()`;
   - poison test: append rows with `season >= target_season` and absurd values (`woba=9.99`, `pa=10**6`); output must equal the output without them (`pandas.testing.assert_frame_equal`);
   - a helper `assert_no_leakage(rows)` that raises `AssertionError` if any `max_source_season >= target_season`; negative control: a deliberately leaky frame must make it raise (`pytest.raises`);
   - a player with only future rows is excluded; a `target_season` earlier than all data returns an empty frame that still has all four columns;
   - determinism: `make_seasons` called twice with the same seed gives equal frames.
4. Update log and report; write status; commit.

**Acceptance commands** are the Phase 1 block of `scripts/verify_n2.sh` (ruff clean; `tests/test_no_leakage.py` passes with zero skips; the import works; full pytest green).

**Done:** the leakage contract is enforced by real tests including a negative control, committed.

---

## Phase 2: lag features
Commit message: `n2 phase 2: lag features`

**Tasks**
1. In `src/mlb_outlook/features/asof.py` add `build_features(df, target_season, *, lags=3, min_pa_feature=100, min_pa_target=100, include_target=True) -> DataFrame`. Keep `build_asof_rows` working.
   - Row eligibility: the player has a row in `target_season - 1` with `pa >= min_pa_feature`.
   - Columns: `player_id`, `target_season`, `max_source_season`, `age_target` (age in `target_season - 1` plus 1), and for each `k` in `1..lags`: `woba_lag{k}`, `pa_lag{k}`, `k_pct_lag{k}`, `bb_pct_lag{k}` (values from season `target_season - k`), `lg_woba_lag{k}` (league wOBA for season `target_season - k`: the PA-weighted mean of `woba` over ALL players that season, NaN if the season has no rows), and `miss_lag{k}` (1 if the player has no row in that season, else 0).
   - A missing lag season gives NaN for `woba_lag`, `k_pct_lag`, `bb_pct_lag`, `pa_lag = 0`, `miss_lag = 1`.
   - If `include_target`: add `y_woba` and `y_pa` (from `target_season`) and keep only players with a `target_season` row with `pa >= min_pa_target`. If not (production mode): no target columns and no target filter.
   - `max_source_season` is the latest season actually used, including league rates; it must always be `< target_season`.
2. Create `tests/test_features_lags.py` with a tiny hand-built fixture (3 players, seasons 2020..2024) and expected values written out as arithmetic in comments: lag alignment (lag1 is the previous season), missing-season flags and NaN/0 fill, `min_pa_feature` and `min_pa_target` filtering (one player excluded by each), the league rate computed by hand for one season, `age_target`, `include_target=False` (no `y_*` columns, a player without a target row is kept), and column order/names. Add the same poison and `max_source_season` leakage tests from Phase 1 for `build_features` (extend `tests/test_no_leakage.py`, still zero skips).
3. Update log and report; write status; commit.

**Acceptance commands** are the Phase 2 block of `scripts/verify_n2.sh`.

**Done:** features are lag-aligned, leakage-tested, and hand-verified on a fixture, committed.

---

## Phase 3: baselines
Commit message: `n2 phase 3: baselines`

**Spec.** All models share one interface (put it in `src/mlb_outlook/models/baselines.py`): a `Forecaster` `typing.Protocol` with `fit(X, y=None) -> Forecaster` and `predict(X) -> pandas.Series` (indexed like `X`). `X` is the Phase 2 feature frame.
- `LastSeason`: predicts `woba_lag1`.
- `Weighted3(weights=(5, 4, 3))`: `sum_k w_k * pa_k * woba_k / sum_k w_k * pa_k` over lag seasons with `pa_k > 0` and non-NaN `woba_k`; no regression and no age adjustment.
- `MarcelBaseline(weights=(5, 4, 3), regression_pa=1200.0)`: for each row build `seasons = [(pa_lag1, woba_lag1), (pa_lag2, woba_lag2), (pa_lag3, woba_lag3)]` (missing seasons already have `pa = 0`), compute the league rate as the weighted mean of the non-NaN `lg_woba_lag{k}` using the weights renormalised over available seasons, then call the existing `marcel_rate(seasons, age_target, league_rate, weights=..., regression_pa=...)` from `models/marcel.py`. Reuse it; do not reimplement the formula.
- `fit` is a no-op that returns `self` for all three.

**Tasks**
1. Implement the three classes.
2. Create `tests/test_baselines.py` with a hand-built feature-frame fixture and hand-computed values. Required (league rate 0.320 in all three lag seasons): row with `pa_lag = (600, 500, 400)`, `woba_lag = (.400, .360, .320)`:
   - `LastSeason` = 0.400;
   - `Weighted3` = (5*600*.400 + 4*500*.360 + 3*400*.320) / (5*600 + 4*500 + 3*400) = 2304 / 6200 = 0.371613;
   - `MarcelBaseline` with `age_target = 29` = 2688 / 7400 = 0.363243; with `age_target = 25` = 0.371961; with `age_target = 33` = 0.358884.
   - A row with seasons 2 and 3 missing (`pa_lag = (300, 0, 0)`, `woba_lag1 = .350`, lag2/lag3 NaN, `miss_lag` = 0,1,1, age 24): `Weighted3` = 0.350; `MarcelBaseline` = 0.346767.
   - `predict` keeps the input index; `fit` returns `self`; all three satisfy the `Forecaster` protocol (use `isinstance` with `runtime_checkable` or a structural check). Use `pytest.approx(..., abs=1e-6)`. Re-derive each expected number yourself before trusting it; fix the test and log a decision if one is wrong.
3. Update log and report; write status; commit.

**Acceptance commands** are the Phase 3 block of `scripts/verify_n2.sh`.

**Done:** three baselines on one interface, matching hand-computed values, committed.

---

## Phase 4: evaluation harness
Commit message: `n2 phase 4: evaluation harness`

**Tasks**
1. Create `src/mlb_outlook/evaluation/harness.py`:
   - `fold_frames(df, fold, cfg) -> tuple[DataFrame, DataFrame]`: train = concatenation of `build_features(df, t, include_target=True, ...)` for each `t` in `fold.train_targets`; test = `build_features(df, fold.test_target, include_target=True, ...)`; `min_pa` values come from `cfg.min_pa`. Train rows must all have `target_season` in `train_targets`; test rows must all have `target_season == test_target`.
   - `run_fold(df, fold, cfg, factories, *, allow_unfrozen=False) -> DataFrame`: `factories` is `dict[str, Callable[[], Forecaster]]`. For each model it calls the factory inside `run_fold` (a fresh model per fold), fits on `(train_X, train_y = train["y_woba"])`, predicts on the test frame, and returns one row per model with `model`, `fold`, `n_train`, `n_test`, `mae` (mean absolute error vs `y_woba`), `bias` (mean of prediction minus truth). Guard: if the fold's `frozen` field is explicitly `False` and `allow_unfrozen` is false, raise `FoldNotFrozenError` (define it); this protects the final holdout fold from being run before design freeze.
   - `run_all(df, cfg, factories, *, allow_unfrozen=False) -> DataFrame`: the concatenation over `cfg.folds` (not production).
   - `paired_bootstrap_mae_diff(errors_a, errors_b, *, n_boot=1000, seed=0) -> tuple[float, float, float]`: given two aligned arrays of absolute errors, returns `(mean_diff, ci_low, ci_high)` where `diff = mean(errors_a) - mean(errors_b)` over resampled pairs and the CI is the 2.5th and 97.5th percentiles.
2. Create `tests/test_harness.py` (synthetic data via `seasons_df`; no network):
   - `fold_frames` target-season membership for each fold in `configs/folds.yaml`;
   - leakage through the harness: for each fold, build the train frame from the full data and again from data truncated to `season < fold.test_target`; the two train frames must be equal (the test season can never influence training);
   - a spy forecaster records the `target_season` values it was fit on: assert it saw only `train_targets` and never `test_target`, and that the factory was called exactly once per fold per model;
   - a `LastSeason` run on a hand-built mini dataset gives the hand-computed `mae` and `bias`;
   - `FoldNotFrozenError` is raised for a fold with `frozen: False` and not raised with `allow_unfrozen=True`;
   - bootstrap: when A has strictly smaller errors than B on every pair the CI excludes zero and the point estimate is negative; for identical arrays the diff is exactly 0 and the CI contains 0; same seed gives the same result.
3. Update log and report; write status; commit.

**Acceptance commands** are the Phase 4 block of `scripts/verify_n2.sh`.

**Done:** the harness evaluates any `Forecaster` per fold with fresh models, no leakage, and a guarded holdout, committed.

---

## Phase 5: ridge forecaster
Commit message: `n2 phase 5: ridge forecaster`

**Tasks**
1. Create `src/mlb_outlook/models/ridge.py` with `RidgeForecaster(alpha=10.0, feature_cols=None)` following the `Forecaster` protocol. Default features: `woba_lag{1,2,3}`, `pa_lag{1,2,3}`, `k_pct_lag{1,2,3}`, `bb_pct_lag{1,2,3}`, `lg_woba_lag{1,2,3}`, `miss_lag{1,2,3}`, `age_target`. Internally a scikit-learn `Pipeline`: `SimpleImputer(strategy="median")`, `StandardScaler`, `Ridge(alpha=alpha)`. The imputer and scaler are fit only inside `fit` (on the training rows passed in). `predict` before `fit` raises a clear error. Return a `Series` indexed like `X`.
2. Create `tests/test_ridge.py`:
   - fitted scaler `mean_` equals the training column means (computed in the test); calling `predict` on a test frame with extreme values does not change the fitted scaler/imputer/ridge parameters;
   - fit then predict twice gives identical output; two independent fits on the same data give identical output;
   - NaN lags (missing seasons) are handled without error; unfitted `predict` raises;
   - a tiny exact case: with perfectly linear data and `alpha` close to 0 the fit recovers the line;
   - integration on `seasons_df` through `run_all(..., allow_unfrozen=True)` with factories `last_season`, `weighted3`, `marcel`, `ridge`: mean `mae` across the three folds satisfies `marcel < last_season` and `ridge < last_season`. If the synthetic data cannot show this, adjust the generator's noise or signal constants in `tests/synthetic.py` (data, not the assertion), re-run all earlier tests, and log the change as a decision;
   - harness-level leakage: truncating the data to `season < test_target` for each fold does not change that fold's ridge predictions on the test frame.
3. Update log and report; write status; commit.

**Acceptance commands** are the Phase 5 block of `scripts/verify_n2.sh`.

**Done:** a leakage-safe ridge forecaster that beats last-season on synthetic data, committed.

---

## Phase 6 (final): morning report
Commit message: `n2 phase 6: morning report`

Runs even if earlier phases failed or were skipped. No build work.

**Tasks**
1. Follow `docs/REPORT_TEMPLATE.md`, with the six-phase table of this plan. Build the report from real data: re-run `VERIFY_SKIP_COMMIT=1 bash scripts/verify_n2.sh N` for each phase that claims PASS and use that output; take hashes and messages from `git log --oneline`, file changes from `git log --stat`; read `NIGHT_LOG.md` and `.night/status/`.
2. Mark phases that did not run as `skipped` or `blocked` with the reason the driver gave.
3. List uncommitted leftovers under Problems; do not commit them.
4. Commit only the report and log files.

**Acceptance commands**
```bash
test -s "reports/${NIGHT_DATE}-overnight.md"
git log -1 --format=%s
```

**Done:** a complete report exists on the night branch, built from command output and git history, committed as `n2 phase 6: morning report`.
