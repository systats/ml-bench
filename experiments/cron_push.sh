#!/usr/bin/env bash
# cron_push.sh — Run experiments, commit new data, push to GitHub.
# 0 */3 * * * /home/simon/ml-bench/experiments/cron_push.sh >> /home/simon/ml-bench/cron.log 2>&1

set -uo pipefail  # no -e: we handle errors explicitly

REPO_DIR="$HOME/ml-bench"
DATA_DIR="$REPO_DIR/data"
EXPERIMENTS_DIR="$REPO_DIR/experiments"
VENV="$HOME/ml_test/venv"
DRY_RUN=false
BATCH_SIZE=50

[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

log() { echo "[$(date -u "+%Y-%m-%d %H:%M:%S UTC")] $*"; }
die() { log "FATAL: $*"; exit 1; }

# 0. Venv
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"

# 1. Pull (stash dirty state)
cd "$REPO_DIR"
git stash -q 2>/dev/null || true
git pull --rebase origin main || die "git pull failed"
git stash pop -q 2>/dev/null || true

# 2. Run experiments
log "Running $BATCH_SIZE cells..."
cd "$EXPERIMENTS_DIR"
python3 continuous_runner.py \
    --data-dir "$DATA_DIR" \
    --experiments-dir "$EXPERIMENTS_DIR" \
    --limit "$BATCH_SIZE" \
    --nice 15 2>&1 || log "Runner had errors (partial results ok)"

# 3. Check for changes
cd "$REPO_DIR"
git add data/
STAGED=$(git diff --cached --name-only | wc -l | tr -d " ")
if [[ "$STAGED" -eq 0 ]]; then
    log "No new data."
    exit 0
fi
log "$STAGED files staged"

# 4. Egress check
BANNED=("beast" "moat" "vault" "auditor council" "mlwork" "agents/modeler" "private/" "smithy")
for term in "${BANNED[@]}"; do
    if grep -qri "$term" data/*.jsonl 2>/dev/null; then
        git reset HEAD -- data/ >/dev/null
        die "EGRESS:  found"
    fi
done

# 5. Safety: no file shrinks
for f in data/*.jsonl; do
    [[ -f "$f" ]] || continue
    LOCAL=$(wc -l < "$f" | tr -d " ")
    REMOTE=$(git show "origin/main:$f" 2>/dev/null | wc -l | tr -d " ")
    REMOTE=${REMOTE:-0}
    if [[ "$LOCAL" -lt "$REMOTE" ]]; then
        git reset HEAD -- data/ >/dev/null
        die "SHRINK: $(basename "$f") $LOCAL < $REMOTE"
    fi
done

# 6. Commit
TOTAL=0
for f in data/*.jsonl; do
    [[ -f "$f" ]] || continue
    N=$(wc -l < "$f" | tr -d " ")
    TOTAL=$((TOTAL + N))
done

git commit -m "data: ${TOTAL} total rows $(date -u "+%m-%d %H:%M")" \
    || { log "Nothing to commit"; exit 0; }

# 7. Push
if [[ "$DRY_RUN" == "true" ]]; then
    log "DRY RUN: $(git log --oneline -1)"
else
    git push origin main || die "push failed"
    log "Pushed: $(git log --oneline -1)"
fi
