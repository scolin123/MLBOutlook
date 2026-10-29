# Overnight report: 2026-10-28-n2

> Built from actual command output. Updated after every phase.

## Summary
Phase 1 (leakage test and as-of builder core) passed. The skipped leakage placeholder is now 19 real tests, including a poison test and a negative control. Branch: `night/2026-10-28-n2`; `main` untouched. Phase 2 (lag features) passed: `build_features` added and hand-verified on a 3-player fixture. Phase 3 (baselines) passed: `LastSeason`, `Weighted3` and `MarcelBaseline` share a `Forecaster` protocol and match the hand-computed values. Phases 4-6 have not run yet.

## Phase table
| Phase | Name | Status (done / partial / skipped / blocked) | Start | End | Commit hash |
|---|---|---|---|---|---|
| 1 | leakage test and as-of builder core | done | epoch 1793240518 | epoch ~1793240600 | see `git log` (`n2 phase 1`) |
| 2 | lag features | done | epoch 1793240593 | epoch 1793240658 | see `git log` (`n2 phase 2`) |
| 3 | baselines | done | epoch 1793240700 | epoch ~1793240740 | see `git log` (`n2 phase 3`) |
| 4 | evaluation harness | not run | | | |
| 5 | ridge forecaster | not run | | | |
| 6 | morning report | not run | | | |

## Test results per phase
### Phase 1
- **Command:** `VERIFY_SKIP_COMMIT=1 bash scripts/verify_n2.sh 1`
- **Result:** pass. `All checks passed!`; `19 passed` in tests/test_no_leakage.py (no skips); import OK; full pytest `59 passed`; `phase 1 verified`.
- **Fix attempts:** 1 (ruff I001 import ordering in `tests/conftest.py` and `tests/test_no_leakage.py`, fixed with `ruff check --fix`).

### Phase 2
- **Command:** `VERIFY_SKIP_COMMIT=1 bash scripts/verify_n2.sh 2`
- **Result:** pass. ruff clean; `47 passed` in tests/test_no_leakage.py (no skips); `10 passed` in tests/test_features_lags.py; import OK; full pytest `97 passed`; `phase 2 verified`.
- **Fix attempts:** 0.

### Phase 3
- **Command:** `VERIFY_SKIP_COMMIT=1 bash scripts/verify_n2.sh 3`
- **Result:** pass. ruff `All checks passed!`; `9 passed` in tests/test_baselines.py; import OK; full pytest `106 passed`; `phase 3 verified`.
- **Fix attempts:** 0.

## Changes
- Phase 3: `src/mlb_outlook/models/baselines.py` (`Forecaster`, `LastSeason`, `Weighted3`, `MarcelBaseline`), `tests/test_baselines.py` (new).
- Phase 2: `src/mlb_outlook/features/asof.py` (`build_features`, `feature_columns`), `tests/test_features_lags.py` (new), `tests/test_no_leakage.py` (extended with `build_features` leakage and poison tests).
- `tests/synthetic.py`: seeded `make_seasons` generator.
- `tests/conftest.py`: `seasons_df` fixture.
- `src/mlb_outlook/features/asof.py`: `build_asof_rows`.
- `tests/test_no_leakage.py`: real leakage tests replacing the skip.

## Decisions made without me, and why
- Generator career length uses `integers(3, 10)` for 3..9 inclusive; truncated at the last season.
- `build_asof_rows` output is int64, sorted by `player_id`, with a reset index.
- Phase 2: `max_source_season` is the latest lag season (of `target-1..target-lags`) that has league data; `age_target` comes from the player's `target-1` row. `build_features` output is sorted by `player_id`, int64 ids/PA/miss flags. Duplicate player-season rows are collapsed to the first.

- Phase 3: a row with no usable league rate gives NaN from `Weighted3`/`MarcelBaseline` rather than raising; NaN wOBA with `pa > 0` is treated as 0 PA in Marcel's input (pa kept, rate 0) only when the feature frame is inconsistent, which `build_features` never produces. `LastSeason`/`Weighted3`/`MarcelBaseline` return NaN for players with no history in `Weighted3`/`LastSeason`; Marcel regresses fully to the league rate. Extra tests added for no-history and renormalised league rate (hand-derived: 0.319040, 0.362162).

## Problems and blockers
None.

## What was NOT verified
- Phases 2-6 not run. The generator's statistical properties (talent, aging) are not tested beyond determinism.

## Suggested next steps
1. Run phase 4 (evaluation harness).
