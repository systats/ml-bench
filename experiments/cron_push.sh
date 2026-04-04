#!/usr/bin/env bash
# cron_push.sh — Run experiments, commit new data, push to GitHub.
#
# 0 */3 * * * /home/simon/ml-bench/experiments/cron_push.sh >> /home/simon/ml-bench/cron.log 2>&1

set -euo pipefail

REPO_DIR="$HOME/ml-bench"
DATA_DIR="$REPO_DIR/data"
EXPERIMENTS_DIR="$REPO_DIR/experiments"
VENV="$HOME/ml_test/venv"
LOG_PREFIX="[$(date -u "+%Y-%m-%d %H:%M:%S UTC")]"
DRY_RUN=false
BATCH_SIZE=50

if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=true; fi

log() { echo "$LOG_PREFIX $*"; }
die() { log "FATAL: $*"; exit 1; }

# Activate venv
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"

cd "$REPO_DIR"

# Stash any local changes, pull, pop
git stash -q 2>/dev/null || true
log "Pulling latest..."
git pull --rebase origin main || die "git pull failed"
git stash pop -q 2>/dev/null || true

# Run experiments
log "Running $BATCH_SIZE experiment cells..."
cd "$EXPERIMENTS_DIR"
python3 continuous_runner.py \
    --data-dir "$DATA_DIR" \
    --experiments-dir "$EXPERIMENTS_DIR" \
    --limit "$BATCH_SIZE" \
    --nice 15 \
    2>&1 || log "Runner exited with error (partial results may exist)"

# Check for changes
cd "$REPO_DIR"
CHANGED=$(git diff --name-only -- "data/" 2>/dev/null | wc -l | tr -d " ")
UNTRACKED=$(git ls-files --others --exclude-standard -- "data/" 2>/dev/null | wc -l | tr -d " ")
TOTAL_NEW=$((CHANGED + UNTRACKED))

if [[ "$TOTAL_NEW" -eq 0 ]]; then
    log "No new data. Done."
    exit 0
fi
log "Found $TOTAL_NEW changed/new files"

# Egress check
BANNED_TERMS=("beast" "moat" "vault" "auditor council" "mlwork" "agents/modeler" "private/" "smithy")
EGRESS_FAIL=false
for term in "${BANNED_TERMS[@]}"; do
    HITS=$(grep -ril "$term" data/*.jsonl 2>/dev/null | wc -l | tr -d " ")
    if [[ "$HITS" -gt 0 ]]; then
        log "EGRESS:  in $HITS files"
        EGRESS_FAIL=true
    fi
done
[[ "$EGRESS_FAIL" == "true" ]] && die "Egress violations"

# Safety: no file shrinks
for f in data/*.jsonl; do
    [[ -f "$f" ]] || continue
    LOCAL=$(wc -l < "$f" | tr -d " ")
    REMOTE=$(git show "origin/main:$(git ls-files --full-name "$f" 2>/dev/null || echo "$f")" 2>/dev/null | wc -l | tr -d " " || echo 0)
    if [[ "$LOCAL" -lt "$REMOTE" ]]; then
        die "DATA LOSS: $(basename "$f") local=$LOCAL remote=$REMOTE"
    fi
done

# Commit
SUMMARY=""
TOTAL_ROWS=0
for f in data/*.jsonl; do
    [[ -f "$f" ]] || continue
    N=$(wc -l < "$f" | tr -d " ")
    TOTAL_ROWS=$((TOTAL_ROWS + N))
    SUMMARY="${SUMMARY}  $(basename "$f"): ${N}\n"
done

git add data/
echo "__pycache__/" >> .gitignore 2>/dev/null
git add .gitignore 2>/dev/null || true

git commit -m "data: ${TOTAL_ROWS} total rows $(date -u "+%Y-%m-%d %H:%M UTC")

$(echo -e "$SUMMARY")" || { log "Nothing to commit"; exit 0; }

# Push
if [[ "$DRY_RUN" == "true" ]]; then
    log "DRY RUN — $(git log --oneline -1)"
else
    git push origin main || die "push failed"
    log "Pushed: $(git log --oneline -1)"
fi

log "Done."
