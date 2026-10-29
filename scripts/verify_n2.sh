#!/usr/bin/env bash
# Acceptance commands for docs/PLAN_N2.md (night 2). Exits non-zero on the first failure.
# Set VERIFY_SKIP_COMMIT=1 to skip the "latest commit message" check (used when re-verifying for the report).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

phase="${1:?usage: scripts/verify_n2.sh <1|2|3|4|5|6>}"

run() { echo "+ $*"; "$@"; }

# No skipped tests allowed in the named test file (the old leakage placeholder was a skip).
no_skips() {
  local out
  echo "+ pytest $1 (no skips allowed)"
  out="$(uv run --no-sync pytest "$1" -q 2>&1)" || { echo "$out"; return 1; }
  echo "$out"
  if echo "$out" | grep -qi "skipped"; then echo "FAIL: skipped tests in $1" >&2; return 1; fi
  if ! echo "$out" | grep -q "passed"; then echo "FAIL: no tests passed in $1" >&2; return 1; fi
}

case "$phase" in
  1)
    run uv run --no-sync ruff check .
    no_skips tests/test_no_leakage.py
    run uv run --no-sync python -c "from mlb_outlook.features.asof import build_asof_rows"
    run uv run --no-sync pytest -q
    expected="n2 phase 1:"
    ;;
  2)
    run uv run --no-sync ruff check .
    no_skips tests/test_no_leakage.py
    run uv run --no-sync pytest tests/test_features_lags.py -q
    run uv run --no-sync python -c "from mlb_outlook.features.asof import build_features"
    run uv run --no-sync pytest -q
    expected="n2 phase 2:"
    ;;
  3)
    run uv run --no-sync ruff check .
    run uv run --no-sync pytest tests/test_baselines.py -q
    run uv run --no-sync python -c "from mlb_outlook.models.baselines import LastSeason, Weighted3, MarcelBaseline"
    run uv run --no-sync pytest -q
    expected="n2 phase 3:"
    ;;
  4)
    run uv run --no-sync ruff check .
    run uv run --no-sync pytest tests/test_harness.py -q
    run uv run --no-sync python -c "from mlb_outlook.evaluation.harness import fold_frames, run_fold, run_all, paired_bootstrap_mae_diff"
    run uv run --no-sync pytest -q
    expected="n2 phase 4:"
    ;;
  5)
    run uv run --no-sync ruff check .
    run uv run --no-sync pytest tests/test_ridge.py -q
    run uv run --no-sync python -c "from mlb_outlook.models.ridge import RidgeForecaster"
    run uv run --no-sync pytest -q
    expected="n2 phase 5:"
    ;;
  6)
    : "${NIGHT_DATE:?NIGHT_DATE must be set for phase 6}"
    run test -s "reports/${NIGHT_DATE}-overnight.md"
    expected="n2 phase 6:"
    ;;
  *)
    echo "unknown phase: $phase" >&2
    exit 2
    ;;
esac

if [ "${VERIFY_SKIP_COMMIT:-0}" != "1" ]; then
  subject="$(git log -1 --format=%s)"
  case "$subject" in
    "$expected"*) echo "commit message OK: $subject" ;;
    *) echo "FAIL: latest commit is '$subject', expected it to start with '$expected'" >&2; exit 1 ;;
  esac
fi

echo "phase $phase verified"
