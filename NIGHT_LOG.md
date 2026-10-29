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
