# Overnight report: YYYY-MM-DD

> Generation rule: build this report from actual test output (re-run `scripts/verify_phase.sh N` with `VERIFY_SKIP_COMMIT=1`) and from `git log` / `git diff --stat`, never from memory of the run. Update it after every phase so a cutoff still leaves a usable report. Anything you did not verify goes under "What was NOT verified". Quote error output exactly.

## Summary
Three or four sentences: what finished, what did not, and the single most important thing to look at first. State the branch name and whether `main` is untouched.

## Phase table
| Phase | Name | Status (done / partial / skipped / blocked) | Start | End | Commit hash |
|---|---|---|---|---|---|
| 1 | bootstrap verification | | | | |
| 2 | folds config validator | | | | |
| 3 | marcel baseline | | | | |
| 4 | morning report | | | | |

## Test results per phase
For each phase:
- **Command:** exact command
- **Result:** pass or fail, with the last relevant lines of output
- **Fix attempts:** number (0-3) and what was changed each time

## Changes
Per commit (from `git log`): hash, message, files touched (from `git show --stat`), and a plain-language description of what changed and why.

## Decisions made without me, and why
Each decision the plan did not settle, with the alternative considered.

## Problems and blockers
Exact error messages and commands. List any uncommitted partial work left in the working tree and where it is.

## What was NOT verified
Be specific: code paths without tests, behavior that depends on the OS or network, assumptions in the Marcel constants, anything run only once.

## Suggested next steps
Ordered, concrete, each small enough to be one session.
