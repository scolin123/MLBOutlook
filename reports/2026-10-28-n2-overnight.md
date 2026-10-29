# Overnight report: 2026-10-28-n2

> Built from actual command output (all five verify scripts re-run in phase 6 with `VERIFY_SKIP_COMMIT=1`) and from `git log --stat`.

## Summary
All five build phases passed and were committed on `night/2026-10-28-n2`; `main` was not touched and nothing was pushed. The branch now has leakage tests with a negative control, lag features, three baselines on one `Forecaster` interface, an evaluation harness with a holdout guard, and a leakage-safe ridge forecaster. The full suite is 134 tests, all passing, ruff clean. Look first at the Decisions section, since several behaviors (NaN handling, the unfrozen-fold guard) were my choices, and at "What was NOT verified": everything uses synthetic data only.

## Phase table
| Phase | Name | Status (done / partial / skipped / blocked) | Start | End | Commit hash |
|---|---|---|---|---|---|
| 1 | leakage test and as-of builder core | done | epoch 1793240518 | epoch ~1793240600 | 4ea7be8 |
| 2 | lag features | done | epoch 1793240593 | epoch 1793240658 | b681f6c |
| 3 | baselines | done | epoch 1793240700 | epoch ~1793240740 | 531d8fa |
| 4 | evaluation harness | done | epoch 1793240772 | epoch 1793240821 | 20ad334 |
| 5 | ridge forecaster | done | epoch 1793240865 | epoch ~1793240960 | abf83fc |
| 6 | morning report | done | epoch 1793241235 | see commit | `n2 phase 6: morning report` (see `git log`) |

Start/end times are the epochs the build phases logged in `NIGHT_LOG.md`. The phase 1 and 2 epochs overlap slightly, as logged. All phases started before `PHASE_CUTOFF_EPOCH` (1793258100). Build loop ended because all build phases passed.

## Test results per phase
Re-run during phase 6. Each ran `VERIFY_SKIP_COMMIT=1 bash scripts/verify_n2.sh N`. The final full-suite line is the same for every phase because it runs the whole repo at the phase-5 state.

### Phase 1
- **Result:** pass. `All checks passed!`; `47 passed` in tests/test_no_leakage.py (no skips); import OK; full pytest `134 passed`; `phase 1 verified`.
- **Fix attempts:** 1 (ruff I001 import ordering, `ruff check --fix`). Note: phase 1 originally had 19 tests in `test_no_leakage.py`; phase 2 extended the file to 47.

### Phase 2
- **Result:** pass. ruff clean; `47 passed` (no leakage skips); `10 passed` in tests/test_features_lags.py; import OK; `134 passed`; `phase 2 verified`.
- **Fix attempts:** 0.

### Phase 3
- **Result:** pass. ruff clean; `9 passed` in tests/test_baselines.py; import OK; `134 passed`; `phase 3 verified`.
- **Fix attempts:** 0.

### Phase 4
- **Result:** pass. ruff clean; `17 passed` in tests/test_harness.py; import OK; `134 passed`; `phase 4 verified`.
- **Fix attempts:** 1 (ruff E501 on the `fold_frames` signature; line wrapped).

### Phase 5
- **Result:** pass. ruff clean; `11 passed` in tests/test_ridge.py; import OK; `134 passed`; `phase 5 verified`.
- **Fix attempts:** 0 (one ruff import-order autofix in the new test).

### Phase 6
- **Acceptance:** `test -s reports/2026-10-28-n2-overnight.md` and `git log -1 --format=%s`; see the commit.

## Changes
From `git log --stat main..HEAD`:
- `abf83fc` n2 phase 5: ridge forecaster. `src/mlb_outlook/models/ridge.py` (+46), `tests/test_ridge.py` (+130), `NIGHT_LOG.md`, report. `RidgeForecaster`: median impute, standard scale, Ridge, all fit only inside `fit`.
- `20ad334` n2 phase 4: evaluation harness. `src/mlb_outlook/evaluation/harness.py` (+95), `tests/test_harness.py` (+137), log, report. `fold_frames`, `run_fold`, `run_all`, `paired_bootstrap_mae_diff`, `FoldNotFrozenError`.
- `531d8fa` n2 phase 3: baselines. `src/mlb_outlook/models/baselines.py` (+102), `tests/test_baselines.py` (+89), log, report. `Forecaster` protocol, `LastSeason`, `Weighted3`, `MarcelBaseline` (reuses `marcel_rate`).
- `b681f6c` n2 phase 2: lag features. `src/mlb_outlook/features/asof.py` (+74), `tests/test_features_lags.py` (+144), `tests/test_no_leakage.py` (+55), log, report. `build_features` with lag alignment, missing-season flags, league rates.
- `4ea7be8` n2 phase 1: leakage test and as-of builder core. `tests/synthetic.py` (+38), `tests/conftest.py` (+8), `src/mlb_outlook/features/asof.py` (+23), `tests/test_no_leakage.py` (rewritten, replacing the skip), log, report. `make_seasons` and `build_asof_rows`.

## Decisions made without me, and why
Taken from `NIGHT_LOG.md`; each was a choice the plan did not settle.
- **Phase 1:** career length via `rng.integers(3, 10)` for 3..9 inclusive, truncated at the last season. NaN masking uses one `rng.random` draw per row per column. `build_asof_rows` returns int64 columns sorted by `player_id` with a fresh index. Tests import the generator as `from synthetic import make_seasons`.
- **Phase 2:** `max_source_season` is the latest lag season with league data (always `target-1` for eligible rows). `age_target` comes from the player's `target-1` row plus 1. Output sorted by `player_id`; duplicate player-season rows collapse to the first. Added a `feature_columns(lags)` helper. Empty frames keep all columns.
- **Phase 3:** no usable league rate or no history gives NaN from `Weighted3`/`LastSeason`; `MarcelBaseline` with no PA regresses fully to the league rate (NaN only if all `lg_woba_lag` are NaN). A NaN lag wOBA reaches `marcel_rate` as rate 0, which has no effect because its PA is 0 for `build_features` output. Two extra tests beyond the spec (0.319040 and 0.362162, hand-derived). The spec's expected values were re-derived and correct as written.
- **Phase 4:** `FoldNotFrozenError` subclasses `RuntimeError`. Only `frozen is False` triggers the guard. `run_fold` returns rows in factory order. Bootstrap resamples pair indices with `default_rng(seed)` and raises `ValueError` on misaligned or empty input. Two extra tests (factory called once per fold in `run_all`; `run_all` blocks the unfrozen holdout by default).
- **Phase 5:** `fit` without `y` raises `ValueError`; unfitted `predict` raises `RuntimeError`. Generator constants unchanged because marcel and ridge already beat last_season. The truncation leakage test compares predictions on `include_target=False` test features, since truncated data has no test-season targets.

## Problems and blockers
- No failed phases and no fix attempt exceeded the limit.
- Working tree was clean (`git status --short` empty) when phase 6 started, so there are no uncommitted leftovers.
- Cosmetic: the `NIGHT_LOG.md` heading for N2 Phase 5 contains a garbled character (a mis-encoded dash). Left as is.
- `NIGHT_LOG.md` also holds the night 1 entries, and their branch name is `night/2026-10-28`.

## What was NOT verified
- Everything runs on synthetic data. Nothing was checked against real player data, and the generator's statistical properties (talent, aging) are tested only for determinism.
- The "ridge and marcel beat last_season" result is on synthetic data, so it says nothing about real-world accuracy.
- The final holdout fold was only run with `allow_unfrozen=True` on synthetic data, for the integration test.
- Marcel constants (weights 5/4/3, 1200 PA regression) are the plan's, not validated.
- I did not run the driver's independent verification. Only the verify scripts were re-run, each once.
- Cross-OS behavior (this ran on Windows under Git Bash) was not checked.

## Suggested next steps
1. Review the Decisions list, especially NaN handling in the baselines and the `frozen is False` guard semantics.
2. Merge the branch (or open a PR) after review; this run did not touch `main`.
3. Fix the garbled character in the `NIGHT_LOG.md` Phase 5 heading.
4. Add a real-data loader (separate session) and run the harness on the non-holdout folds only.
5. Add a bootstrap comparison of ridge vs marcel using `paired_bootstrap_mae_diff` in a small script.
