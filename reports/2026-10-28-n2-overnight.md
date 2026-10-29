# Overnight report: 2026-10-28-n2

> Built from actual command output. Updated after every phase.

## Summary
Phase 1 (leakage test and as-of builder core) passed. The skipped leakage placeholder is now 19 real tests, including a poison test and a negative control. Branch: `night/2026-10-28-n2`; `main` untouched. Phases 2-6 have not run yet.

## Phase table
| Phase | Name | Status (done / partial / skipped / blocked) | Start | End | Commit hash |
|---|---|---|---|---|---|
| 1 | leakage test and as-of builder core | done | epoch 1793240518 | epoch ~1793240600 | see `git log` (`n2 phase 1`) |
| 2 | lag features | not run | | | |
| 3 | baselines | not run | | | |
| 4 | evaluation harness | not run | | | |
| 5 | ridge forecaster | not run | | | |
| 6 | morning report | not run | | | |

## Test results per phase
### Phase 1
- **Command:** `VERIFY_SKIP_COMMIT=1 bash scripts/verify_n2.sh 1`
- **Result:** pass. `All checks passed!`; `19 passed` in tests/test_no_leakage.py (no skips); import OK; full pytest `59 passed`; `phase 1 verified`.
- **Fix attempts:** 1 (ruff I001 import ordering in `tests/conftest.py` and `tests/test_no_leakage.py`, fixed with `ruff check --fix`).

## Changes
- `tests/synthetic.py`: seeded `make_seasons` generator.
- `tests/conftest.py`: `seasons_df` fixture.
- `src/mlb_outlook/features/asof.py`: `build_asof_rows`.
- `tests/test_no_leakage.py`: real leakage tests replacing the skip.

## Decisions made without me, and why
- Generator career length uses `integers(3, 10)` for 3..9 inclusive; truncated at the last season.
- `build_asof_rows` output is int64, sorted by `player_id`, with a reset index.

## Problems and blockers
None.

## What was NOT verified
- Phases 2-6 not run. The generator's statistical properties (talent, aging) are not tested beyond determinism.

## Suggested next steps
1. Run phase 2 (lag features).
