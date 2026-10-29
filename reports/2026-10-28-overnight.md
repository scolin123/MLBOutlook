# Overnight report: 2026-10-28

> Built from re-run `VERIFY_SKIP_COMMIT=1 bash scripts/verify_phase.sh N` output (run during Phase 4) and from `git log`, not from memory of the run.

## Summary
All three build phases (bootstrap verification, folds config validator, Marcel baseline) finished and were committed on branch `night/2026-10-28`. The build loop ended because all build phases passed; none were skipped or blocked. `main` is untouched (still at `6821c8a`, the `MAIN_SHA` in `.night/state.env`), and nothing was pushed. Look first at the Marcel wrapper decisions below, since the plan left those open. The working tree was clean when this report was started.

## Phase table
| Phase | Name | Status | Start | End | Commit hash |
|---|---|---|---|---|---|
| 1 | bootstrap verification | done | 22:07 | 22:09 | 41e60fa |
| 2 | folds config validator | done | 22:10 | 22:13 (log) / commit 22:10 | 7bb5d42 |
| 3 | marcel baseline | done | 22:11 | 22:13 (log) / commit 22:12 | 2b2850b |
| 4 | morning report | done (this commit) | 22:13 | 22:13 | see `git log` (a commit cannot contain its own hash) |

Times are from `NIGHT_LOG.md` plus `git log` author times. The logged start/end times for phases 2 and 3 do not line up with the commit times (e.g. phase 2 logged end 22:13 but committed 22:10), so treat them as approximate. All phases started before `PHASE_CUTOFF_EPOCH` (1793258100); Phase 4 started at epoch 1793239994.

## Test results per phase
Re-run in Phase 4 with `VERIFY_SKIP_COMMIT=1 bash scripts/verify_phase.sh N`; all exited 0. Status files `.night/status/phase-1..3` each contain `PASS`.

- **Phase 1** — Command: `VERIFY_SKIP_COMMIT=1 bash scripts/verify_phase.sh 1`
  - Result: pass. `bootstrap OK`; import check ok; `All checks passed!`; `40 passed, 1 skipped in 2.17s`; `phase 1 verified`.
  - Fix attempts: 0.
- **Phase 2** — Command: `VERIFY_SKIP_COMMIT=1 bash scripts/verify_phase.sh 2`
  - Result: pass. ruff clean; `25 passed in 0.47s` (test_folds_config.py); real `configs/folds.yaml` loads; full suite `40 passed, 1 skipped in 2.14s`; `phase 2 verified`.
  - Fix attempts: 0 (log notes a first heredoc write failed on a quoting error before writing anything; files were then created with the file-write tool).
- **Phase 3** — Command: `VERIFY_SKIP_COMMIT=1 bash scripts/verify_phase.sh 3`
  - Result: pass. ruff clean; `15 passed in 1.63s` (test_marcel.py); full suite `40 passed, 1 skipped in 2.10s`; `phase 3 verified`.
  - Fix attempts: 0 formal. Per the log, first-run corrections were ruff E501 line lengths and a wrong expectation in one of the new tests (a player aged 29 in 2024 is 30 in 2025); the test input was fixed, not the implementation.

The one skipped test is the intentional leakage placeholder from the bootstrap.

## Changes
- `41e60fa phase 1: bootstrap verification` — `NIGHT_LOG.md` (+8), `reports/2026-10-28-overnight.md` (+34). Bootstrap check found nothing missing, so only log and report were added; venv created and project installed.
- `7bb5d42 phase 2: folds config validator` — `src/mlb_outlook/config.py` (+115), `tests/test_folds_config.py` (+183), log and report. Pydantic v2 models (`extra="forbid"`) and `load_folds_config`, with all the ordering, overlap, forecast-target and success-criteria rules from the plan, each with passing and failing tests plus an integration test on the real config.
- `2b2850b phase 3: marcel baseline` — `src/mlb_outlook/models/marcel.py` (+77), `tests/test_marcel.py` (+113), log and report. Implements `age_factor`, `marcel_rate`, and the `project_marcel` DataFrame wrapper with a leakage guard (seasons >= target ignored).

## Decisions made without me, and why
From `NIGHT_LOG.md`:
- Phase 2: `interval_80_coverage_range` is `tuple[float, float]`; `purpose` and `frozen` are optional on `Fold`; `forecast_date_rule` is a required string. `configs/folds.yaml` was not changed (it is valid).
- Phase 3: the wrapper's league rate is the weighted mean over lookback seasons T-1..T-3 that have a league rate (error if none). Target age comes from the player's last observed season before T, even if that is outside the 3-year lookback (then no PA is used). Players with no rows before T are omitted from the output. A missing lookback season counts as `pa=0`. `marcel_rate` ignores seasons beyond `len(weights)`.
- Phase 3: the spec's hand-computed values (0.363243, 0.371961, 0.358884, 0.346767, 0.32384) were re-derived and found correct as written.

## Problems and blockers
- None blocking. No failed phases, no uncommitted leftovers (`git status --short` was empty before this report).
- Minor: the phase 2/3 start/end times in `NIGHT_LOG.md` overlap and do not match commit times (see above).

## What was NOT verified
- Marcel constants (weights 5/4/3, K=1200, age factors) are taken from the plan spec and only checked against it, not against outside sources.
- `project_marcel` behavior on real data, including edge cases the plan did not settle (no history, gaps outside the lookback window), is tested only on small synthetic fixtures.
- Phase 1 pytest coverage is thin: the only non-config, non-Marcel test is the skipped leakage placeholder.
- Tests were run on this machine only (Windows); nothing was tested on another OS or against a database/network.
- Each verify script was re-run once in Phase 4; test runs are otherwise as logged by earlier phases.

## Suggested next steps
1. Review the Phase 3 wrapper decisions (age source, omitted players, league-rate averaging) and confirm or change them.
2. Replace the skipped leakage placeholder test with a real test that uses `project_marcel`.
3. Add a Marcel evaluation against the folds from `load_folds_config` (MAE per fold) to start on the success criteria.
4. Fix the time logging in `NIGHT_LOG.md` entries (record actual `date` at start and end).
