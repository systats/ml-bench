#!/usr/bin/env bash
# cron_push.sh — Run experiments, commit new data, push to GitHub.
#
# Usage:
#   ./cron_push.sh                    # normal run
#   ./cron_push.sh --dry-run          # skip push
#   0 */3 * * * /home/simon/ml-bench/experiments/cron_push.sh >> /home/simon/ml-bench/cron.log 2>&1

set -euo pipefail

REPO_DIR="$HOME/ml-bench"
DATA_DIR="$REPO_DIR/data"
EXPERIMENTS_DIR="$REPO_DIR/experiments"
VENV="$HOME/ml_test/venv"
LOG_PREFIX="[$(date -u "+%Y-%m-%d %H:%M:%S UTC")]"
DRY_RUN=false
BATCH_SIZE=50  # cells per cron cycle

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
# 1. Pull latest (never overwrite remote data)
# ---------------------------------------------------------------------------
cd "$REPO_DIR"
log "Pulling latest..."
git pull --rebase origin main || die "git pull failed"

# ---------------------------------------------------------------------------
# 2. Run experiments (batch of N cells)
# ---------------------------------------------------------------------------
log "Running $BATCH_SIZE experiment cells..."
cd "$EXPERIMENTS_DIR"
python3 continuous_runner.py \
    --data-dir "$DATA_DIR" \
    --experiments-dir "$EXPERIMENTS_DIR" \
    --limit "$BATCH_SIZE" \
    --nice 15 \
    2>&1 || log "Runner exited with error (partial results may exist)"

# ---------------------------------------------------------------------------
# 3. Check for new data
# ---------------------------------------------------------------------------
cd "$REPO_DIR"
CHANGED=$(git diff --name-only -- data/*.jsonl 2>/dev/null | wc -l | tr -d " ")
UNTRACKED=$(git ls-files --others --exclude-standard -- data/*.jsonl 2>/dev/null | wc -l | tr -d " ")
TOTAL_NEW=$((CHANGED + UNTRACKED))

if [[ "$TOTAL_NEW" -eq 0 ]]; then
    log "No new data. Nothing to push."
    exit 0
fi

log "Found $TOTAL_NEW changed/new JSONL files"

# ---------------------------------------------------------------------------
# 4. Egress check (banned terms)
# ---------------------------------------------------------------------------
BANNED_TERMS=("beast" "moat" "vault" "auditor council" "mlwork" "agents/modeler" "private/" "smithy")

log "Running egress scan..."
EGRESS_FAIL=false
for term in "${BANNED_TERMS[@]}"; do
    HITS=$(grep -ril "$term" data/*.jsonl 2>/dev/null | wc -l | tr -d " ")
    if [[ "$HITS" -gt 0 ]]; then
        log "EGRESS VIOLATION:  found in $HITS files"
        EGRESS_FAIL=true
    fi
done

if [[ "$EGRESS_FAIL" == "true" ]]; then
    die "Egress violations — NOT pushing"
fi

# ---------------------------------------------------------------------------
# 5. Safety: no file should shrink
# ---------------------------------------------------------------------------
DATA_LOSS=false
for f in data/*.jsonl; do
    [[ -f "$f" ]] || continue
    BASENAME=$(basename "$f")
    LOCAL_COUNT=$(wc -l < "$f" | tr -d " ")
    REMOTE_COUNT=$(git show "origin/main:data/$BASENAME" 2>/dev/null | wc -l | tr -d " " || echo 0)
    if [[ "$LOCAL_COUNT" -lt "$REMOTE_COUNT" ]]; then
        log "DATA LOSS: $BASENAME local=$LOCAL_COUNT remote=$REMOTE_COUNT"
        DATA_LOSS=true
    fi
done

if [[ "$DATA_LOSS" == "true" ]]; then
    die "Data loss detected — NOT pushing"
fi

# ---------------------------------------------------------------------------
# 6. Commit
# ---------------------------------------------------------------------------
SUMMARY=""
for f in data/*.jsonl; do
    [[ -f "$f" ]] || continue
    COUNT=$(wc -l < "$f" | tr -d " ")
    SUMMARY="${SUMMARY}  $(basename "$f"): ${COUNT}\n"
done

git add data/*.jsonl
git add data/continuous_checkpoint.db 2>/dev/null || true

git commit -m "data: +${BATCH_SIZE} cells $(date -u "+%Y-%m-%d %H:%M UTC")

$(echo -e "$SUMMARY")" || die "Nothing to commit"

# ---------------------------------------------------------------------------
# 7. Push
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == "true" ]]; then
    log "DRY RUN — skipping push"
else
    git push origin main || die "git push failed"
    log "Pushed: $(git log --oneline -1)"
fi

log "Done."
