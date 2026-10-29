# Overnight plan

This file drives unattended overnight runs (`scripts/night_run.sh`). The project's scope, models, and evaluation design live in [PROJECT_PLAN.md](PROJECT_PLAN.md); this file only covers the overnight phases.

## RULES (read first, apply to every phase)

**Git**
1. Work only on branch `night/YYYY-MM-DD`, where the date is the day the run STARTED (the driver passes it as `NIGHT_DATE`; use that value even if the clock has passed midnight). Never touch `main`. Never push anywhere. No deploys. Never run `git push`, `git merge`, `git rebase`, `git reset`, or change remotes.
2. A phase is complete only when ALL its acceptance commands pass, verified by actually running them and reading the output. Then commit with exactly the message given for the phase (`phase N: <name>`), including the updated `NIGHT_LOG.md` and report in that commit.
3. Before every commit run `git status` and `git diff --cached --name-only`. If `.env`, anything under `data/raw/`, `data/processed/`, `artifacts/`, `mysql_data/`, or `.night/` is staged, unstage it and fix `.gitignore`. No secrets in commits. Never commit `.env`.

**Time** (the clock crosses midnight, so never compare clock strings)
4. Before starting each phase run `date` and `date +%s`. If the epoch is >= `PHASE_CUTOFF_EPOCH` (given in the prompt and in `.night/state.env`; this is 03:15 local), do NOT start the phase: skip to the report. The driver hard-stops the whole run at 04:00 local.

**Failure handling**
5. If a phase fails, make up to 3 fix attempts. If it still fails: stop building, log the exact error output, do NOT start later phases, write `FAIL` to `.night/status/phase-N`, commit ONLY `NIGHT_LOG.md` and the report with message `night: phase N failed (log only)`, leave any partial code uncommitted and untouched, and go to the report.
6. Never build on a failing phase. Never skip, weaken, xfail, or delete a test to make it pass. Never edit an acceptance command to make it pass.

**Scope**
7. Do only the phase you were invoked for. Do not read ahead and start later phases.
8. Tonight's phases are offline-friendly. No data pulls and no network ingestion (no pybaseball data calls, no Statcast/MLB/FanGraphs requests, no curl/wget, no web fetch/search) unless a phase in this file explicitly calls for it. `uv`/`pip` installs, `ruff`, and `pytest` are allowed.
9. Decisions that the plan does not settle: choose the simplest reasonable option, and record it under "Decisions" in the log and report.

**Logging (after EVERY phase, pass or fail, not only at the end)**
10. Append to `NIGHT_LOG.md`: phase, start/end time, commands run, pass/fail, decisions made, blockers.
11. Create or update `reports/YYYY-MM-DD-overnight.md` using `docs/REPORT_TEMPLATE.md`, so a cutoff at any moment still leaves a usable report. The commit hash of the phase's own commit is filled in by the Phase 4 report pass (a commit cannot contain its own hash).
12. Write the single word `PASS` or `FAIL` to `.night/status/phase-N` after the phase (create the folder if needed). The driver reads this file and independently re-runs `scripts/verify_phase.sh N`; both must agree.

**Commands**: use `uv run --no-sync <tool>` for ruff, pytest, and python so the commands behave the same on every OS. The acceptance commands below are exactly what `scripts/verify_phase.sh` runs.

---

## Phase 1: bootstrap verification
Commit message: `phase 1: bootstrap verification`

**Tasks**
1. Run `bash scripts/check_bootstrap.sh`. If anything is missing, create it exactly as specified in the README section "Project bootstrap spec" (files) and the README layout (folders; every `src/mlb_outlook/*` subpackage needs an `__init__.py`; empty folders need `.gitkeep`). Re-run until it passes.
2. Create the virtualenv if absent and install the project with dev and app extras.
3. Run ruff and pytest; fix lint problems in files the bootstrap created (do not loosen ruff rules).
4. Update `NIGHT_LOG.md` and the report; write the status file; commit.

**Acceptance commands (all must exit 0)**
```bash
bash scripts/check_bootstrap.sh
test -d .venv || uv venv
uv pip install -e ".[dev,app]"
uv run --no-sync python -c "import mlb_outlook, pandas, numpy, sklearn, pydantic, yaml, sqlalchemy, pymysql, pyarrow, streamlit, plotly, pybaseball"
uv run --no-sync ruff check .
uv run --no-sync pytest -q
```

**Done:** all bootstrap files and folders exist, dependencies import, ruff is clean, pytest exits 0, and the work is committed as `phase 1: bootstrap verification`.

---

## Phase 2: folds config loader and validator
Commit message: `phase 2: folds config validator`

**Tasks**
1. Create `src/mlb_outlook/config.py` with Pydantic v2 models (all with `extra="forbid"`) for `configs/folds.yaml`: top-level config, fold, production, extended, min_pa, success_criteria. Expose `load_folds_config(path) -> FoldsConfig` (reads YAML with `yaml.safe_load`, validates, returns the model).
2. Validation rules, each raising a clear `ValueError`/`ValidationError` message:
   - fold names are unique and folds are listed in strictly increasing `test_target` order;
   - each fold's `train_targets` is non-empty, strictly increasing, and has no duplicates;
   - no train/test overlap: a fold's `test_target` is not in its `train_targets`;
   - `test_target` is strictly greater than `max(train_targets)`;
   - `production.forecast_target` is strictly greater than `max(production.train_targets)`;
   - `success_criteria.beats_marcel_mae_on` is a non-empty list of fold names that exist in `folds`; `bootstrap_ci_excludes_zero` is a bool; `interval_80_coverage_range` is two floats with `0 <= lo < hi <= 1`.
3. Write `tests/test_folds_config.py` with focused tests for every rule above, both the passing and failing case (build configs as Python dicts or `tmp_path` YAML files), plus one integration test that loads the real `configs/folds.yaml`. Include: shuffled fold order fails; equal and earlier test target fail; overlapping train/test fails; unknown fold name in success criteria fails; coverage range `[0.9, 0.8]` fails; missing `success_criteria` key fails.
4. Do not change `configs/folds.yaml` to make tests pass unless it is genuinely invalid; if so, log the change as a decision.
5. Update log and report; write status; commit.

**Acceptance commands (all must exit 0)**
```bash
uv run --no-sync ruff check .
uv run --no-sync pytest tests/test_folds_config.py -q
uv run --no-sync python -c "from mlb_outlook.config import load_folds_config; load_folds_config('configs/folds.yaml')"
uv run --no-sync pytest -q
```

**Done:** the real folds file loads and every rule has a passing and failing test, all green, committed as `phase 2: folds config validator`.

---

## Phase 3: Marcel baseline
Commit message: `phase 3: marcel baseline`

**Spec (simplified, Marcel-style; constants are configurable parameters)**
For target season T, use the three seasons T-1, T-2, T-3 (most recent first) with weights `w = (5, 4, 3)`. Each season has plate appearances `pa` and a rate `r` (for example wOBA). Seasons at or after T must be ignored (leakage guard). A missing season counts as `pa = 0`.

```
regression_pa K = 1200
lg  = weighted mean of the league rates of the available lookback seasons, using weights w
      (renormalised over the seasons that have a league rate)
rate_regressed = ( sum_i w_i * pa_i * r_i  +  K * lg ) / ( sum_i w_i * pa_i  +  K )
age_factor(a)  = 1 + (29 - a) * 0.006   if a < 29
               = 1 + (29 - a) * 0.003   if a >= 29        (so a = 29 gives 1)
marcel_rate    = rate_regressed * age_factor(age_in_target_season)
```
`age_in_target_season = age_in_last_observed_season + (T - last_observed_season)`.
These constants reflect Tango's Marcel as commonly described; they are defined here so tests check the implementation against this spec, not against outside sources. Do not add a playing-time projection tonight.

**Tasks**
1. Create `src/mlb_outlook/models/marcel.py` with: `age_factor(age)`; a pure function `marcel_rate(seasons, target_age, league_rate, *, weights=(5, 4, 3), regression_pa=1200.0)` where `seasons` is a most-recent-first sequence of `(pa, rate)` pairs; and a DataFrame wrapper `project_marcel(df, target_season, league_rates)` for a long table with columns `player_id, season, age, pa, rate` and a `{season: league_rate}` mapping, returning one row per player with `player_id, target_season, marcel_rate`.
2. Write `tests/test_marcel.py` using a small synthetic fixture (no files, no network) with hand-computed expected values written out as arithmetic in comments. Required cases (verified values for league rate 0.320):
   - Three seasons, most recent first: (600 PA, .400), (500, .360), (400, .320). Weighted numerator = 5*600*.400 + 4*500*.360 + 3*400*.320 = 2304; weighted PA = 6200; add K*lg = 384 and K = 1200, so the regressed rate = 2688 / 7400 = 0.363243.
   - Same player at age 29: 0.363243. At age 25: x 1.024 = 0.371961. At age 33: x 0.988 = 0.358884.
   - Rookie with one season (300 PA, .350), age 24: (5*300*.350 + 384) / (1500 + 1200) = 909 / 2700 = 0.336667, x 1.03 = 0.346767.
   - No history, age 27: lg * 1.012 = 0.32384.
   - The DataFrame wrapper ignores rows with `season >= target_season` (add a future season with an absurd rate and show the output does not change), handles two players, and a player missing the oldest season.
   - `age_factor` boundary values: 20, 28, 29, 30, 35.
   Use `pytest.approx(..., abs=1e-6)`. Re-derive each expected value yourself before trusting the numbers above; if one is wrong, fix the test and log it as a decision.
3. Update log and report; write status; commit.

**Acceptance commands (all must exit 0)**
```bash
uv run --no-sync ruff check .
uv run --no-sync pytest tests/test_marcel.py -q
uv run --no-sync pytest -q
```

**Done:** Marcel matches every hand-computed case, ignores future seasons, all tests and ruff green, committed as `phase 3: marcel baseline`.

---

## Phase 4 (final): morning report
Commit message: `phase 4: morning report`

Runs even if earlier phases failed or were skipped. No build work.

**Tasks**
1. Follow `docs/REPORT_TEMPLATE.md`. Generate the report from real data, not from memory of the run: re-run `bash scripts/verify_phase.sh N` (with `VERIFY_SKIP_COMMIT=1`) for each phase that claims PASS and use that output; take commit hashes and messages from `git log --oneline` and file changes from `git diff --stat`/`git log --stat`; read `NIGHT_LOG.md` and `.night/status/`.
2. Mark phases that did not run as `skipped` or `blocked` with the reason given in the prompt (the driver supplies the stop reason).
3. If the working tree has uncommitted changes from a failed or interrupted phase, list them under Problems and do not commit them.
4. Commit only the report and log files. Write the final `reports/YYYY-MM-DD-overnight.md`.

**Acceptance commands**
```bash
test -s "reports/${NIGHT_DATE}-overnight.md"
git log -1 --format=%s
```

**Done:** a complete report exists on the night branch, built from command output and git history, committed as `phase 4: morning report`.
