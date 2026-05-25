#!/bin/bash
# ============================================================================
#  push_and_run.sh
#  One-command workflow:
#    1. Commit all local changes to Git
#    2. Push to GitHub (origin/main)
#    3. SSH into all TPU v5e-16 workers, pull the code
#    4. Install dependencies (if requested)
#    5. Launch Shor's algorithm simulation across all workers
#
#  Run from your local machine (WSL2 / Git Bash / Cloud Shell):
#    bash push_and_run.sh
#
#  Or with flags:
#    bash push_and_run.sh --skip-install    # Skip pip install step
#    bash push_and_run.sh --dry-run         # Print commands, don't execute
# ============================================================================

set -e   # exit immediately on any error

# ─── Defaults ────────────────────────────────────────────────────────────────
SKIP_INSTALL=false
DRY_RUN=false
BRANCH="main"
COMMIT_MSG="Add Shor's algorithm 33-qubit TPU simulation [auto]"

# ─── Parse flags ─────────────────────────────────────────────────────────────
for arg in "$@"; do
    case $arg in
        --skip-install) SKIP_INSTALL=true ;;
        --dry-run)      DRY_RUN=true      ;;
        --branch=*)     BRANCH="${arg#*=}" ;;
        --message=*)    COMMIT_MSG="${arg#*=}" ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

RUN() {
    if $DRY_RUN; then
        echo "  [DRY-RUN] $*"
    else
        eval "$@"
    fi
}

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Shor's Algorithm — GitHub Push + TPU v5e-16 Deploy        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─── Step 1: Git commit & push ───────────────────────────────────────────────
echo "── Step 1/3 ─ Git Commit & Push ──────────────────────────────"
echo ""

# Stage all changes
RUN git add -A

# Check if there is anything to commit
if git diff --cached --quiet; then
    echo "  ℹ️  Nothing to commit — working tree clean."
else
    echo "  📝  Committing changes..."
    echo "  Message: \"$COMMIT_MSG\""
    RUN git commit -m "\"$COMMIT_MSG\""
fi

echo "  ⬆️  Pushing to origin/$BRANCH ..."
RUN git push origin "$BRANCH"
echo "  ✅  Push complete."
echo ""

# ─── Step 2: Install dependencies on all TPU workers (optional) ──────────────
if ! $SKIP_INSTALL; then
    echo "── Step 2/3 ─ Install Dependencies on All TPU Workers ────────"
    echo ""
    RUN bash tpu/run_shor_tpu.sh 1
    echo ""
else
    echo "── Step 2/3 ─ [SKIPPED] Dependency install (--skip-install) ──"
    echo ""
fi

# ─── Step 3: Pull latest code and launch simulation on all workers ────────────
echo "── Step 3/3 ─ Pull & Launch Shor's Simulation ────────────────"
echo ""
RUN bash tpu/run_shor_tpu.sh 2
echo ""

# ─── Done ────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅  All steps complete!                                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Monitor progress  :  bash tpu/run_shor_tpu.sh 3            ║"
echo "║  Download results  :  bash tpu/run_shor_tpu.sh 4            ║"
echo "║  Clean workers     :  bash tpu/run_shor_tpu.sh 5            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
