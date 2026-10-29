# Night log

Append one entry per phase (pass or fail): phase, start/end time, commands run, pass/fail, decisions made, blockers. See `docs/PLAN.md` for the rules.

<!-- entries below -->

## Phase 1: bootstrap verification — PASS
- Start: 2026-10-28 22:07 (epoch 1793239656, before cutoff 1793258100). End: 22:09.
- Branch confirmed: night/2026-10-28.
- Commands: `bash scripts/check_bootstrap.sh` (bootstrap OK, nothing missing); `uv venv` (created .venv); `uv pip install -e ".[dev,app]"`; `VERIFY_SKIP_COMMIT=1 bash scripts/verify_phase.sh 1` (import check, ruff "All checks passed!", pytest "1 skipped", exit 0).
- Fix attempts: 0. No files needed to be created.
- Decisions: none beyond plan. Note pytest's only test is the intentionally skipped leakage placeholder.
- Blockers: none.

## Phase 2: folds config validator — PASS
- Start: 2026-10-28 22:10 (epoch 1793239783, before cutoff 1793258100). End: 22:13.
- Branch confirmed: night/2026-10-28.
- Commands: `VERIFY_SKIP_COMMIT=1 bash scripts/verify_phase.sh 2` (ruff clean; 25 passed in test_folds_config.py; real folds.yaml loads; full pytest 25 passed, 1 skipped).
- Fix attempts: 0. A first attempt to write files via a shell heredoc failed on a quoting error before writing anything; files were then created with the file-write tool.
- Decisions: `interval_80_coverage_range` is `tuple[float, float]`; `purpose`/`frozen` optional on Fold; `forecast_date_rule` required string. `configs/folds.yaml` unchanged (it is valid).
- Blockers: none.

## Phase 3: Marcel baseline — PASS
- Start: 2026-10-28 22:11 (epoch 1793239869, before cutoff 1793258100). End: 22:13.
- Branch confirmed: night/2026-10-28.
- Commands: `VERIFY_SKIP_COMMIT=1 bash scripts/verify_phase.sh 3` (ruff clean; test_marcel.py 15 passed; full pytest 40 passed, 1 skipped).
- Fix attempts: 0 formal attempts needed beyond first-run corrections: ruff E501 line lengths, and one of my own tests had a wrong expectation (player aged 29 in 2024 is 30 in 2025; fixed the test input, not the implementation).
- Spec values re-derived by hand (0.363243, 0.371961, 0.358884, 0.346767, 0.32384): all correct as written.
- Decisions: wrapper's league rate = weighted mean over lookback seasons T-1..T-3 that have a league rate (error if none); target age is taken from the player's last observed season before T (even if outside the 3-year lookback, in which case no PA is used); players with no rows before T are omitted from the output; missing lookback season counts as pa=0; `marcel_rate` ignores seasons beyond len(weights).
- Blockers: none.

## Phase 4: morning report — PASS
- Start: 2026-10-28 22:13 (epoch 1793239994, before cutoff 1793258100). End: 22:14.
- Commands: `VERIFY_SKIP_COMMIT=1 bash scripts/verify_phase.sh 1`, `... 2`, `... 3` (all exit 0; 40 passed, 1 skipped); `git log --stat`; `git status --short` (clean).
- Wrote reports/2026-10-28-overnight.md. Stop reason: all build phases passed.
- Fix attempts: 0. Decisions: none. Blockers: none.

## Night 2 (branch night/2026-10-28-n2)

## N2 Phase 1: leakage test and as-of builder core — PASS
- Start: epoch 1793240518 (before cutoff 1793258100). End: epoch ~1793240600.
- Branch confirmed: night/2026-10-28-n2.
- Commands: `uv run --no-sync ruff check --fix .` (fixed 2 import-order errors, I001, in new test files); `VERIFY_SKIP_COMMIT=1 bash scripts/verify_n2.sh 1` (ruff clean; test_no_leakage.py 19 passed, 0 skipped; import OK; full pytest 59 passed).
- Fix attempts: 1 (ruff import sorting only).
- Decisions: tests import the generator as `from synthetic import make_seasons` (tests/ has no `__init__`, pytest puts it on sys.path). Generator draws the career with `rng.integers(3, 10)` (3..9) and truncates at the last season. NaN masking uses one `rng.random` draw per row per column over the whole frame. `build_asof_rows` returns int64 columns sorted by `player_id`, with a fresh RangeIndex.
- Blockers: none.

## N2 Phase 2: lag features — PASS
- Start: epoch 1793240593 (before cutoff 1793258100). End: epoch 1793240658.
- Branch confirmed: night/2026-10-28-n2.
- Commands: `VERIFY_SKIP_COMMIT=1 bash scripts/verify_n2.sh 2` (ruff clean; test_no_leakage.py 47 passed, 0 skipped; test_features_lags.py 10 passed; import OK; full pytest 97 passed).
- Fix attempts: 0.
- Decisions: `max_source_season` is the latest lag season with league data (always `target-1` for eligible rows). `age_target` is taken from the player's `target-1` row plus 1. Output sorted by `player_id`; duplicate player-season rows collapse to the first. Added a `feature_columns(lags)` helper for column order. Spec-compliant empty frames keep all columns.
- Blockers: none.

## N2 Phase 3: baselines — PASS
- Start: epoch 1793240700 (before cutoff 1793258100). End: epoch ~1793240740.
- Branch confirmed: night/2026-10-28-n2.
- Commands: `VERIFY_SKIP_COMMIT=1 bash scripts/verify_n2.sh 3` (ruff clean; test_baselines.py 9 passed; import OK; full pytest 106 passed).
- Fix attempts: 0. Spec values (0.371613, 0.363243, 0.371961, 0.358884, 0.346767) re-derived by hand and correct as written.
- Decisions: no usable league rate or no history gives NaN from `Weighted3`/`LastSeason`; `MarcelBaseline` with no PA regresses fully to the league rate (NaN only if all `lg_woba_lag` are NaN). A NaN lag wOBA is passed to `marcel_rate` as rate 0 (its PA is 0 for frames from `build_features`, so it has no effect). Two extra tests beyond the spec (no history 0.319040; renormalised league rate 0.362162, hand-derived).
- Blockers: none.

## N2 Phase 4: evaluation harness — PASS
- Start: epoch 1793240772 (before cutoff 1793258100). End: epoch 1793240821.
- Branch confirmed: night/2026-10-28-n2.
- Commands: `VERIFY_SKIP_COMMIT=1 bash scripts/verify_n2.sh 4` (ruff clean; test_harness.py 17 passed; import OK; full pytest 123 passed).
- Fix attempts: 1 (ruff E501 on the `fold_frames` signature; wrapped the line).
- Decisions: `FoldNotFrozenError` subclasses `RuntimeError`. Only `frozen is False` triggers the guard (`None`/`True` run freely). `run_fold` returns one row per model in factory order; `run_all` uses a fresh index. Bootstrap resamples pair indices with `default_rng(seed)` and raises `ValueError` on misaligned or empty input. Hand-computed LastSeason mini case: errors +.02, -.04, +.03 give mae 0.03, bias 0.01/3; n_train 2, n_test 3. Extra tests: `run_all` calls each factory once per fold, and `run_all` blocks the unfrozen holdout by default.
- Blockers: none.

## N2 Phase 5: ridge forecaster � PASS
- Start: epoch 1793240865 (before cutoff 1793258100). End: epoch ~1793240960.
- Branch confirmed: night/2026-10-28-n2.
- Commands: `VERIFY_SKIP_COMMIT=1 bash scripts/verify_n2.sh 5` (ruff clean; test_ridge.py 11 passed; import OK; full pytest 134 passed).
- Fix attempts: 0 (one ruff import-order autofix in the new test).
- Decisions: `fit` without `y` raises ValueError; unfitted `predict` raises RuntimeError ("not fitted"). Synthetic generator unchanged (marcel and ridge already beat last_season). Truncation leakage test compares ridge predictions on `include_target=False` test features from full vs truncated data (truncated data has no test-season targets).
- Blockers: none.
