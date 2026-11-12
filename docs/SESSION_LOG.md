# Session log

Unattended session, 2026-11-11. Hard rules followed throughout: fold 3 never run, `frozen`
never flipped, no tags created or moved, nothing pushed, `main` untouched, frozen modules and
`configs/folds.yaml` unchanged, data not re-pulled.

## Part 1 -- pre-freeze (branch `real-data`)

Items 1-3 B-G were already committed by an earlier session when this one started; they were
checked, not redone.

| Step | Commit |
|---|---|
| 1. Three decision-sequence numbers annotated as historical | `39b135f` |
| 2. `run_holdout.py` refuses a dirty tree | `3f492f3` |
| B. Residual provenance documented + `tests/test_residual_provenance.py` | `a50b0f3` |
| C. PA-weighted MAE / RMSE / bias in `run_folds.py` | `9319375` |
| D. Population accounting per fold | `cc2a16f` |
| E. Coverage by age bracket | `d3aab3c` |
| F. `docs/MARCEL_AUDIT.md` | `5e05ba4` |
| G. Per-player `ridge_rel` minus Marcel | `548a941` |
| C in `run_holdout.py` (D, E, G were already in it): RMSE + PA-weighted trio printed | `59e55c6` |
| `scripts/check_docs.py`: docs vs artifacts, as a script | `2d4a307` |

Re-verification, run on `2d4a307`:

- Regenerated `run_folds`, `run_ablations`, `run_probabilities`, and
  `run_holdout.py --rehearse fold2`. The rehearsal reproduces the published fold 2 numbers
  (MAE 0.0263 vs Marcel 0.0285, bootstrap -0.0023, CI -0.0036 to -0.0008, coverage 0.815).
- `scripts/check_docs.py`: 260 checks, 0 mismatches.
- `pytest`: 366 passed. `ruff check src scripts tests`: clean. `git status`: clean.
- Data: `b164c087e6186d989f52bebdcc1b1153f25f7de5d8f2f6b8ddb003c906039cd8`, 9,804 rows,
  unchanged.
- `run_holdout.py` with no arguments refuses (`frozen: False`). No tag exists.

## 4. Freeze candidate

The freeze candidate is **the tip of `real-data`**: the commit that adds this file, directly
on top of `2d4a307`, and docs-only. A commit cannot name its own hash, so the hash is recorded
in this log's copy on `v1-app` and in the summary there. Nothing from Part 2 is on `real-data`.

What's left for you, by hand (DESIGN_FREEZE.md section 4): flip `fold3.frozen` to `true` in a
commit on top of the candidate, with nothing else changed, recording the SHA-256 above and the
row count in that commit's message; then `git tag -a design-freeze-v1` on that flip commit.

## Part 2 -- branch `v1-app`, from the freeze candidate `dfc9901db62dca98b41164ed164bbab7d0b66727`

### 5. MySQL -- `5ec3b6e`

- `sql/001_schema.sql`: `data_versions` (SHA-256), `players`, `player_seasons` (keyed by data
  version, so a re-pull sits beside the old snapshot), `model_runs`, `predictions` (point,
  80% bounds, Marcel and Weighted3, the three probabilities, outcome columns; model version
  and data version on every row), `evaluation_results` (long format).
- `mlb_outlook.db.loader` (`uv run python -m mlb_outlook.db.loader`): idempotent; refuses if
  the parquet's hash differs from the one the fold 2 artifacts were built from.
- No artifact held per-player probabilities, so `scripts/export_fold2_forecasts.py` computes
  them by calling the frozen code read-only, on fold 2 only, and writes
  `artifacts/fold2_forecasts.csv` + `fold2_manifest.json`. It checks that its point fit
  equals `fold2_players.csv`'s `ridge_rel` to 1e-9. The mean P(breakout) is 0.1705, the
  same as the registered fold 2 figure.
- Model version = `ridge_rel+resid_q_pa@<last commit touching the frozen paths>`
  (currently `@548a941ff747`), so app and docs commits don't mint new model versions.
- **Docker: ran it.** Docker Desktop was installed but stopped; I started it. Port 3306 was
  already taken by a local `mysqld` (PID 5388, not Docker), which I did not touch. The
  compose host port is now `${MYSQL_PORT:-3306}` (the default is unchanged), and this
  machine's `.env` sets 3307. The load against MySQL 8.4.13 gave 9,804 player-seasons,
  2,814 players, 362 predictions, and 252 evaluation rows. The fold 2 MAE recomputed in SQL
  is 0.0263. The integration tests (4) pass, and they skip when no server is reachable.
- `.env` was created locally with random passwords. It is git-ignored and was not committed.
- Unrelated: two containers from another project (`whichllm-postgres-1`, `whichllm-redis-1`)
  show as exited. I left them alone.

### 6. Historical comps -- see the commit "historical comps"

- `mlb_outlook.comps.find_comps(df, player_id, season, k=10)`: the k nearest player-seasons of
  *other* players, on z-scored age plus league-relative wOBA, xwOBA, K%, BB%, average exit
  velocity, barrel% and hard-hit%. Each comp comes with its following season (PA, wOBA, and
  the change in league-relative wOBA). `comps_summary` condenses that to one line.
- **Decision (most conservative reading):** "use only seasons before the query season" is
  applied to the comps' *outcomes* too. A comp season is therefore at most `season - 2`, so
  its following year is still before the query season. This costs the most recent comp year.
  If you'd rather allow `season - 1` comps, whose outcome is the query season itself (known at
  the forecast date), change one line in `comp_pool`.
- **Decision:** K% and BB% are league-relative too, not raw, so seasons from different
  eras compare on the same footing.
- **Decision:** 2020 is excluded as a comp season (60-game rates). Comps who didn't play
  the following year stay in the list with `next_pa = 0`, so the comps' futures aren't
  survivor-biased. A query missing any feature is refused rather than imputed.
- Tests (12): every comp and outcome season is before the query; poisoning or deleting later
  seasons leaves the answer unchanged; a constant shift of the query season leaves it
  unchanged; the player is never his own comp; a constructed twin is nearest.
- Sanity check on real data: Judge's 2024 season returns Alvarez 2022, Trout 2021, and
  Harper 2021.
