# Overnight report: 2026-10-28

## Summary
Phases 1 (bootstrap verification), 2 (folds config validator) and 3 (Marcel baseline) passed; phase 4 has not run yet. Work is on branch `night/2026-10-28`; `main` is untouched. Nothing was pushed.

## Phase table
| Phase | Name | Status (done / partial / skipped / blocked) | Start | End | Commit hash |
|---|---|---|---|---|---|
| 1 | bootstrap verification | done | 22:07 | 22:09 | filled in by Phase 4 |
| 2 | folds config validator | done | 22:10 | 22:13 | filled in by Phase 4 |
| 3 | marcel baseline | done | 22:11 | 22:13 | filled in by Phase 4 |
| 4 | morning report | not run | | | |

## Test results per phase
### Phase 1
- **Command:** `VERIFY_SKIP_COMMIT=1 bash scripts/verify_phase.sh 1`
- **Result:** pass. `bootstrap OK`; imports OK; `All checks passed!` (ruff); `1 skipped in 0.02s` (pytest, exit 0); `phase 1 verified`.
- **Fix attempts:** 0

### Phase 2
- **Command:** `VERIFY_SKIP_COMMIT=1 bash scripts/verify_phase.sh 2`
- **Result:** pass. `All checks passed!` (ruff); `25 passed` (tests/test_folds_config.py); real `configs/folds.yaml` loads; full suite `25 passed, 1 skipped`; `phase 2 verified`.
- **Fix attempts:** 0

### Phase 3
- **Command:** `VERIFY_SKIP_COMMIT=1 bash scripts/verify_phase.sh 3`
- **Result:** pass. `All checks passed!` (ruff); `15 passed` (tests/test_marcel.py); full suite `40 passed, 1 skipped`; `phase 3 verified`.
- **Fix attempts:** 0 formal; first run had ruff E501 line-length errors and one wrong test expectation of mine (age 29 in 2024 is 30 in 2025); fixed the lines and the test input.

## Changes
Phase 1 commit adds only NIGHT_LOG.md and this report; the bootstrap files already existed. `.venv` created locally (ignored).
Phase 2 commit adds `src/mlb_outlook/config.py` (Pydantic v2 models, `extra="forbid"`, all folds rules, `load_folds_config`) and `tests/test_folds_config.py` (25 tests). `configs/folds.yaml` unchanged.
Phase 3 commit adds `src/mlb_outlook/models/marcel.py` (`age_factor`, `marcel_rate`, `project_marcel`) and `tests/test_marcel.py` (15 tests, hand-computed values in comments).

## Decisions made without me, and why
- Phase 2: `interval_80_coverage_range` typed as `tuple[float, float]` (rejects wrong length); `Fold.purpose` and `Fold.frozen` are optional since only some folds in the real file set them; `forecast_date_rule` is a required top-level string. Alternative: leave them as untyped extras, rejected because of `extra="forbid"`.
- Phase 3: wrapper league rate is the weighted mean over lookback seasons that have a league rate; target age comes from the player's last observed pre-target season; players with no pre-target rows are omitted; missing season = 0 PA. Alternative: emit league-rate rows for such players, rejected as no age is known.

## Problems and blockers
None.

## What was NOT verified
The only test is the intentionally skipped leakage placeholder, so pytest passing says little about behavior. Requirements were installed from the network via uv (allowed).

## Suggested next steps
1. Run Phase 4 (morning report).
