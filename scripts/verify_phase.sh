#!/usr/bin/env bash
# Runs the exact acceptance commands for a phase (see docs/PLAN.md). Exits non-zero on the first failure.
# Set VERIFY_SKIP_COMMIT=1 to skip the "latest commit message" check (used when re-verifying for the report).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

phase="${1:?usage: scripts/verify_phase.sh <1|2|3|4>}"

run() { echo "+ $*"; "$@"; }

case "$phase" in
  1)
    run bash scripts/check_bootstrap.sh
    run uv run --no-sync python -c "import mlb_outlook, pandas, numpy, sklearn, pydantic, yaml, sqlalchemy, pymysql, pyarrow, streamlit, plotly, pybaseball"
    run uv run --no-sync ruff check .
    run uv run --no-sync pytest -q
    expected="phase 1:"
    ;;
  2)
    run uv run --no-sync ruff check .
    run uv run --no-sync pytest tests/test_folds_config.py -q
    run uv run --no-sync python -c "from mlb_outlook.config import load_folds_config; load_folds_config('configs/folds.yaml')"
    run uv run --no-sync pytest -q
    expected="phase 2:"
    ;;
  3)
    run uv run --no-sync ruff check .
    run uv run --no-sync pytest tests/test_marcel.py -q
    run uv run --no-sync pytest -q
    expected="phase 3:"
    ;;
  4)
    : "${NIGHT_DATE:?NIGHT_DATE must be set for phase 4}"
    run test -s "reports/${NIGHT_DATE}-overnight.md"
    expected="phase 4:"
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
