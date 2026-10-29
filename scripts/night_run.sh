#!/usr/bin/env bash
# Overnight driver: one fresh headless Claude Code invocation per phase of docs/PLAN.md.
#
#   bash scripts/night_run.sh
#
# Stages (this one file, re-executed):
#   setup  create/checkout branch night/YYYY-MM-DD, compute deadlines, write .night/state.env
#   loop   phases 1-3, wrapped in `timeout` so the build work ends before the report window
#   report phase 4, ALWAYS runs afterwards (also after a failure or a timeout)
#
# Deadlines (local clock): no new phase after 03:15; everything, including the report, ends by 04:00.
# Needs: bash, GNU date and timeout (Linux, WSL, or Git Bash on Windows), git, uv, claude.
# Never pushes. Never touches main (the driver verifies main's commit is unchanged after every phase).

unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$REPO_ROOT/scripts/night_run.sh"
cd "$REPO_ROOT" || exit 1

STATE_DIR=".night"
STATE_FILE="$STATE_DIR/state.env"
LOG_DIR="$STATE_DIR/logs"
STATUS_DIR="$STATE_DIR/status"

CUTOFF_CLOCK="03:15"        # no new phase after this
HARD_STOP_CLOCK="04:00"     # everything ends by this
REPORT_RESERVE_SECS=600     # time reserved at the end for the report pass
PHASE_TIMEOUT_SECS="${PHASE_TIMEOUT_SECS:-3000}"       # 50 min cap per phase
PERMISSION_MODE="${PERMISSION_MODE:-dontAsk}"          # dontAsk = anything not allowed below is denied
CLAUDE_MODEL="${CLAUDE_MODEL:-}"                       # optional model alias or name

# Tools the headless agent may use without asking. Everything else is denied (dontAsk).
ALLOWED_TOOLS=(
  "Read" "Edit" "Write" "Glob" "Grep"
  "Bash(git *)"
  "Bash(uv *)" "Bash(ruff *)" "Bash(pytest *)"
  "Bash(python *)" "Bash(python3 *)"
  "Bash(bash scripts/*)"
  "Bash(date)" "Bash(date *)"
  "Bash(ls *)" "Bash(mkdir *)" "Bash(test *)" "Bash(pwd)" "Bash(wc *)"
)
# Denied even though some are covered by the allow list above (deny wins).
DENIED_TOOLS=(
  "WebFetch" "WebSearch"
  "Bash(git push *)" "Bash(git remote *)" "Bash(git merge *)" "Bash(git rebase *)"
  "Bash(git reset *)" "Bash(git clean *)" "Bash(git config *)"
  "Bash(git checkout main*)" "Bash(git switch main*)"
  "Bash(git branch -D *)" "Bash(git branch -d *)" "Bash(git branch -f *)" "Bash(git branch -M *)"
  "Bash(curl *)" "Bash(wget *)" "Bash(ssh *)" "Bash(scp *)" "Bash(sudo *)"
  "Bash(rm -rf *)" "Bash(rm -r *)"
  "Read(.env)" "Read(**/.env)" "Edit(.env)" "Edit(**/.env)" "Write(.env)" "Write(**/.env)"
)

log() {
  local line
  line="$(date '+%F %T') $*"
  echo "$line"
  [ -d "$STATE_DIR" ] && echo "$line" >>"$STATE_DIR/driver.log"
  return 0
}

die() { echo "night_run: $*" >&2; exit 1; }

phase_name() {
  case "$1" in
    1) echo "bootstrap verification" ;;
    2) echo "folds config validator" ;;
    3) echo "marcel baseline" ;;
    4) echo "morning report" ;;
  esac
}

# Epoch of the next occurrence of a local HH:MM clock time (today if still ahead, else tomorrow).
next_epoch() {
  local now t
  now="$(date +%s)"
  t="$(date -d "today $1" +%s)"
  if [ "$t" -le "$now" ]; then t="$(date -d "tomorrow $1" +%s)"; fi
  echo "$t"
}

# run_claude <label> <prompt> <timeout_secs>; the prompt goes in on stdin so the variadic tool flags cannot swallow it.
run_claude() {
  local label="$1" prompt="$2" secs="$3"
  local args=(-p --permission-mode "$PERMISSION_MODE" --permission-prompts none
              --output-format json --name "night-$label")
  [ -n "$CLAUDE_MODEL" ] && args+=(--model "$CLAUDE_MODEL")
  args+=(--allowedTools "${ALLOWED_TOOLS[@]}" --disallowedTools "${DENIED_TOOLS[@]}")
  printf '%s\n' "$prompt" \
    | timeout --kill-after=30 "$secs" claude "${args[@]}" \
        >"$LOG_DIR/$label.json" 2>"$LOG_DIR/$label.err"
}

run_facts() {
  echo "Run facts: NIGHT_DATE=$NIGHT_DATE, branch=$BRANCH, PHASE_CUTOFF_EPOCH=$PHASE_CUTOFF_EPOCH, HARD_STOP_EPOCH=$HARD_STOP_EPOCH (also in .night/state.env)."
}

phase_prompt() {
  cat <<EOF
You are running unattended as part of an overnight run. Nobody will answer questions, so follow the plan exactly.
Read docs/PLAN.md (the RULES section first) and NIGHT_LOG.md. Execute ONLY Phase $1 ("$(phase_name "$1")") as written, following the rules.
$(run_facts)
Confirm you are on branch $BRANCH before changing anything. When the phase ends, update NIGHT_LOG.md and reports/${NIGHT_DATE}-overnight.md, write PASS or FAIL to .night/status/phase-$1, and commit as the plan specifies.
Do not start any other phase.
EOF
}

report_prompt() {
  cat <<EOF
You are running unattended as part of an overnight run. Execute ONLY Phase 4 ("morning report") from docs/PLAN.md, following the rules and docs/REPORT_TEMPLATE.md. Do no build work.
$(run_facts)
Why the build loop ended: $STOP_REASON
Generate the report from actual command output (re-run: VERIFY_SKIP_COMMIT=1 bash scripts/verify_phase.sh N for each phase that claims PASS) and from git log, not from memory. Phases that did not run are skipped or blocked, with that reason. List uncommitted leftovers under Problems; do not commit them.
Write reports/${NIGHT_DATE}-overnight.md, update NIGHT_LOG.md, write PASS or FAIL to .night/status/phase-4, and commit only the report and log as "phase 4: morning report".
EOF
}

# gate <phase>: the agent's PASS must agree with an independent re-run of the acceptance commands.
GATE_REASON=""
gate() {
  local n="$1" st
  st="$(tr -d '[:space:]' <"$STATUS_DIR/phase-$n" 2>/dev/null)"
  if [ "$st" != "PASS" ]; then
    GATE_REASON="phase $n: status file says '${st:-<missing>}'"; return 1
  fi
  if ! bash scripts/verify_phase.sh "$n" >"$LOG_DIR/verify-$n.txt" 2>&1; then
    GATE_REASON="phase $n: independent verification failed (see $LOG_DIR/verify-$n.txt)"; return 1
  fi
  if [ "$(git rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]; then
    GATE_REASON="phase $n: HEAD is not on $BRANCH"; return 1
  fi
  if [ "$(git rev-parse refs/heads/main)" != "$MAIN_SHA" ]; then
    GATE_REASON="phase $n: main moved (expected $MAIN_SHA)"; return 1
  fi
  if git ls-files | grep -Eq '(^|/)\.env$'; then
    GATE_REASON="phase $n: a .env file is tracked by git"; return 1
  fi
  return 0
}

stage_loop() {
  # shellcheck disable=SC1090
  . "$STATE_FILE"
  export NIGHT_DATE
  local n now rc
  for n in 1 2 3; do
    echo "$n" >"$STATE_DIR/current_phase"
    now="$(date +%s)"
    if [ "$now" -ge "$PHASE_CUTOFF_EPOCH" ]; then
      echo "time cutoff ($CUTOFF_CLOCK) reached before phase $n" >"$STATE_DIR/stop_reason"
      return 0
    fi
    rm -f "$STATUS_DIR/phase-$n"
    log "phase $n ($(phase_name "$n")): starting"
    run_claude "phase-$n" "$(phase_prompt "$n")" "$PHASE_TIMEOUT_SECS"
    rc=$?
    log "phase $n: claude exited with code $rc (124 = per-phase timeout)"
    if ! gate "$n"; then
      echo "$GATE_REASON (claude exit code $rc)" >"$STATE_DIR/stop_reason"
      log "phase $n: FAILED - $GATE_REASON"
      return 1
    fi
    log "phase $n: passed"
  done
  echo "all build phases passed" >"$STATE_DIR/stop_reason"
  return 0
}

# Written by the driver only if the report pass failed or produced nothing.
fallback_report() {
  local f="reports/${NIGHT_DATE}-overnight.md" n
  {
    echo
    echo "## Driver fallback summary (written by scripts/night_run.sh, not by Claude)"
    echo
    echo "- Stop reason: $STOP_REASON"
    echo "- Branch: $BRANCH"
    echo "- main unchanged: $([ "$(git rev-parse refs/heads/main)" = "$MAIN_SHA" ] && echo yes || echo NO)"
    for n in 1 2 3 4; do
      echo "- phase $n status file: $(tr -d '[:space:]' <"$STATUS_DIR/phase-$n" 2>/dev/null || true)"
    done
    echo
    echo "Commits on the night branch:"
    echo '```'
    git log --oneline "main..$BRANCH" 2>/dev/null
    echo '```'
    echo
    echo "Uncommitted changes:"
    echo '```'
    git status --short
    echo '```'
    echo
    echo "Raw logs: $LOG_DIR/ (local, not committed)."
  } >>"$f"
  git add "$f" NIGHT_LOG.md >/dev/null 2>&1
  git commit -q -m "night: driver fallback report" >/dev/null 2>&1 || true
}

stage_setup() {
  command -v claude >/dev/null 2>&1 || die "claude CLI not found on PATH"
  command -v uv >/dev/null 2>&1 || die "uv not found on PATH"
  command -v timeout >/dev/null 2>&1 || die "GNU timeout not found (use Linux, WSL, or Git Bash)"
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not inside a git repository"
  git show-ref --verify --quiet refs/heads/main || die "no local branch named main"

  NIGHT_DATE="${NIGHT_DATE:-$(date +%F)}"
  BRANCH="night/$NIGHT_DATE"

  local now loop_secs
  now="$(date +%s)"
  HARD_STOP_EPOCH="$(next_epoch "$HARD_STOP_CLOCK")"
  if [ $((HARD_STOP_EPOCH - now)) -gt 43200 ]; then
    die "refusing to start: the next $HARD_STOP_CLOCK is more than 12 hours away. Start this in the evening or overnight."
  fi
  PHASE_CUTOFF_EPOCH="$(next_epoch "$CUTOFF_CLOCK")"
  # If the next 03:15 is after the next 04:00, we are already between 03:15 and 04:00: no new phases.
  if [ "$PHASE_CUTOFF_EPOCH" -gt "$HARD_STOP_EPOCH" ]; then PHASE_CUTOFF_EPOCH=0; fi
  loop_secs=$((HARD_STOP_EPOCH - REPORT_RESERVE_SECS - now))

  [ -z "$(git status --porcelain)" ] || die "working tree is not clean; commit or stash first"
  MAIN_SHA="$(git rev-parse refs/heads/main)"
  if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git checkout -q "$BRANCH" || die "could not check out $BRANCH"
  else
    [ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] || die "start from main (the night branch is created from it)"
    git checkout -q -b "$BRANCH" || die "could not create $BRANCH"
  fi

  mkdir -p "$LOG_DIR" "$STATUS_DIR" reports
  rm -f "$STATUS_DIR"/phase-* "$STATE_DIR/stop_reason" "$STATE_DIR/current_phase"
  cat >"$STATE_FILE" <<EOF
NIGHT_DATE=$NIGHT_DATE
BRANCH=$BRANCH
MAIN_SHA=$MAIN_SHA
PHASE_CUTOFF_EPOCH=$PHASE_CUTOFF_EPOCH
HARD_STOP_EPOCH=$HARD_STOP_EPOCH
EOF
  export NIGHT_DATE

  log "branch $BRANCH, main at $MAIN_SHA"
  if [ "$PHASE_CUTOFF_EPOCH" -eq 0 ]; then
    log "already past $CUTOFF_CLOCK: no build phase will start; hard stop $(date -d "@$HARD_STOP_EPOCH" '+%F %T')"
  else
    log "no new phase after $(date -d "@$PHASE_CUTOFF_EPOCH" '+%F %T'); hard stop $(date -d "@$HARD_STOP_EPOCH" '+%F %T')"
  fi

  # --- build loop, hard-capped by timeout so build work ends before the report window ---
  local loop_rc=0
  if [ "$loop_secs" -ge 60 ]; then
    timeout --kill-after=30 "$loop_secs" env NIGHT_STAGE=loop bash "$SCRIPT_PATH"
    loop_rc=$?
  else
    echo "too close to the hard stop to start any phase" >"$STATE_DIR/stop_reason"
  fi

  STOP_REASON="$(cat "$STATE_DIR/stop_reason" 2>/dev/null || true)"
  if [ "$loop_rc" -eq 124 ] || [ "$loop_rc" -eq 137 ]; then
    STOP_REASON="hard stop: the time limit was reached while phase $(cat "$STATE_DIR/current_phase" 2>/dev/null || echo '?') was running"
  elif [ -z "$STOP_REASON" ]; then
    STOP_REASON="build loop ended unexpectedly (exit code $loop_rc)"
  fi
  log "build loop ended: $STOP_REASON"

  # --- final report: always runs ---
  if [ "$(git rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]; then
    git checkout -q "$BRANCH" 2>/dev/null || log "WARNING: could not return to $BRANCH before the report"
  fi
  local now2 report_secs rc
  now2="$(date +%s)"
  report_secs=$((HARD_STOP_EPOCH - now2))
  [ "$report_secs" -gt "$REPORT_RESERVE_SECS" ] && report_secs=$REPORT_RESERVE_SECS
  [ "$report_secs" -lt 120 ] && report_secs=120
  rm -f "$STATUS_DIR/phase-4"
  log "phase 4 (morning report): starting (limit ${report_secs}s)"
  run_claude "phase-4" "$(report_prompt)" "$report_secs"
  rc=$?
  log "phase 4: claude exited with code $rc"
  if [ "$rc" -ne 0 ] || [ ! -s "reports/${NIGHT_DATE}-overnight.md" ]; then
    log "report pass failed or empty; writing driver fallback report"
    fallback_report
  fi

  log "done. Branch: $BRANCH. Report: reports/${NIGHT_DATE}-overnight.md. Nothing was pushed."
  log "Reason the build loop stopped: $STOP_REASON"
}

case "${NIGHT_STAGE:-setup}" in
  loop) stage_loop; exit $? ;;
  *)    stage_setup ;;
esac
