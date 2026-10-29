#!/usr/bin/env bash
# Verifies that the bootstrap files and folders from the README exist.
# Exits non-zero and lists everything missing.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

missing=0
need_file() { [ -f "$1" ] || { echo "MISSING file: $1"; missing=1; }; }
need_dir()  { [ -d "$1" ] || { echo "MISSING dir:  $1"; missing=1; }; }

for f in pyproject.toml README.md .gitignore .gitattributes .env.example docker-compose.yml \
         configs/folds.yaml tests/test_no_leakage.py docs/PLAN.md docs/PROJECT_PLAN.md \
         docs/REPORT_TEMPLATE.md NIGHT_LOG.md src/mlb_outlook/__init__.py; do
  need_file "$f"
done

for d in app/pages app/components configs data/raw data/processed notebooks sql artifacts \
         reports scripts tests docs; do
  need_dir "$d"
done

for sub in ingestion validation features context models evaluation reporting; do
  need_file "src/mlb_outlook/$sub/__init__.py"
done

# .gitkeep in folders that would otherwise be empty
for d in data/raw data/processed artifacts sql notebooks app/pages app/components reports; do
  need_file "$d/.gitkeep"
done

# .env must be ignored and never tracked
if [ -d .git ]; then
  git check-ignore -q .env || { echo "PROBLEM: .env is not git-ignored"; missing=1; }
  if git ls-files --error-unmatch .env >/dev/null 2>&1; then
    echo "PROBLEM: .env is tracked by git"; missing=1
  fi
fi

if [ "$missing" -eq 0 ]; then
  echo "bootstrap OK"
fi
exit "$missing"
