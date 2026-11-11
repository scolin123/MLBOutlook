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
