#!/usr/bin/env bash
# cron_push.sh — Compile claims, verify, commit, push to GitHub.
#
# Runs every 3 hours via cron. Only pushes if:
#   1. New data exists (JSONL files changed since last commit)
#   2. compile_claims.py succeeds
#   3. verify_from_raw.py passes (all claims match)
#   4. No egress violations (banned terms)
#
# Usage:
#   ./cron_push.sh                    # normal run
#   ./cron_push.sh --dry-run          # skip push
#   0 */3 * * * /path/to/cron_push.sh >> /path/to/cron_push.log 2>&1
#
# Requires: git push access (deploy key or token)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PAPER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PAPER_DIR/data"
VENV="${VENV:-$HOME/ml_test/venv}"
LOG_PREFIX="[$(date -u '+%Y-%m-%d %H:%M:%S UTC')]"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

log() { echo "$LOG_PREFIX $*"; }
die() { log "FATAL: $*"; exit 1; }

# ---------------------------------------------------------------------------
# 0. Activate venv
# ---------------------------------------------------------------------------
if [[ -f "$VENV/bin/activate" ]]; then
    source "$VENV/bin/activate"
fi

# ---------------------------------------------------------------------------
# 1. Check for new data
# ---------------------------------------------------------------------------
cd "$PAPER_DIR"

# Count changed JSONL files since last commit
CHANGED=$(git diff --name-only HEAD -- data/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
UNTRACKED=$(git ls-files --others --exclude-standard -- data/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
TOTAL_NEW=$((CHANGED + UNTRACKED))

if [[ "$TOTAL_NEW" -eq 0 ]]; then
    log "No new data. Nothing to push."
    exit 0
fi

log "Found $TOTAL_NEW changed/new JSONL files"

# ---------------------------------------------------------------------------
# 2. Compile claims
# ---------------------------------------------------------------------------
log "Compiling claims..."
python3 "$PAPER_DIR/compile_claims.py" || die "compile_claims.py failed"
log "Claims compiled successfully"

# ---------------------------------------------------------------------------
# 3. Verify claims from raw data
# ---------------------------------------------------------------------------
log "Verifying claims from raw data..."
VERIFY_OUTPUT=$(python3 "$PAPER_DIR/verify_from_raw.py" 2>&1)
echo "$VERIFY_OUTPUT"

if echo "$VERIFY_OUTPUT" | grep -q "FAIL"; then
    die "verify_from_raw.py has FAILing claims — NOT pushing"
fi

log "All claims verified"

# ---------------------------------------------------------------------------
# 4. Egress check (banned terms)
# ---------------------------------------------------------------------------
BANNED_TERMS=(
    "beast"
    "moat"
    "vault"
    "auditor council"
    "mlwork"
    "agents/modeler"
    "private/"
    "smithy"
)

log "Running egress scan on changed files..."
EGRESS_FAIL=false
for term in "${BANNED_TERMS[@]}"; do
    HITS=$(grep -ril "$term" data/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$HITS" -gt 0 ]]; then
        log "EGRESS VIOLATION: '$term' found in $HITS files"
        EGRESS_FAIL=true
    fi
done

if [[ "$EGRESS_FAIL" == "true" ]]; then
    die "Egress violations found — NOT pushing"
fi

log "Egress check passed"

# ---------------------------------------------------------------------------
# 5. Pull latest from GitHub (never overwrite remote data)
# ---------------------------------------------------------------------------
log "Pulling latest from origin (rebase to preserve remote data)..."
git pull --rebase origin HEAD || die "git pull --rebase failed — resolve manually"
log "Up to date with remote"

# ---------------------------------------------------------------------------
# 6. Commit
# ---------------------------------------------------------------------------
log "Staging data files..."
git add data/*.jsonl
git add claims.json 2>/dev/null || true

# Build commit message with data summary
SUMMARY=""
for f in data/*.jsonl; do
    if [[ -f "$f" ]]; then
        COUNT=$(wc -l < "$f" | tr -d ' ')
        BASENAME=$(basename "$f")
        SUMMARY="${SUMMARY}  ${BASENAME}: ${COUNT} rows\n"
    fi
done

COMMIT_MSG="data: continuous landscape update $(date -u '+%Y-%m-%d %H:%M UTC')

Data counts:
$(echo -e "$SUMMARY")
All claims verified. Egress clean."

log "Committing..."
git commit -m "$COMMIT_MSG" || die "git commit failed"

# ---------------------------------------------------------------------------
# 7. Safety check: no JSONL file should have FEWER rows than on remote
# ---------------------------------------------------------------------------
log "Checking no data was lost..."
DATA_LOSS=false
for f in data/*.jsonl; do
    if [[ ! -f "$f" ]]; then continue; fi
    BASENAME=$(basename "$f")
    LOCAL_COUNT=$(wc -l < "$f" | tr -d ' ')
    # Get remote count (0 if file doesn't exist on remote)
    REMOTE_COUNT=$(git show "origin/HEAD:data/$BASENAME" 2>/dev/null | wc -l | tr -d ' ' || echo 0)
    if [[ "$LOCAL_COUNT" -lt "$REMOTE_COUNT" ]]; then
        log "DATA LOSS DETECTED: $BASENAME has $LOCAL_COUNT rows locally but $REMOTE_COUNT on remote"
        DATA_LOSS=true
    fi
done

if [[ "$DATA_LOSS" == "true" ]]; then
    git reset HEAD~1  # undo the commit
    die "Data loss detected — NOT pushing. Investigate manually."
fi

# ---------------------------------------------------------------------------
# 8. Push
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == "true" ]]; then
    log "DRY RUN — skipping push"
    log "Would push: $(git log --oneline -1)"
else
    log "Pushing to origin..."
    git push origin HEAD || die "git push failed"
    log "Push successful: $(git log --oneline -1)"
fi

log "Done."
