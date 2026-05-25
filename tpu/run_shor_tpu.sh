#!/bin/bash
# ============================================================================
#  run_shor_tpu.sh
#  Shor's Algorithm 33-Qubit Simulation — TPU v5e-16 Cluster Launcher
#
#  Designed to run from Google Cloud Shell.
#  Manages all 16 TPU workers simultaneously using --worker=all.
#
#  Usage:
#    bash tpu/run_shor_tpu.sh [action]
#
#  Actions:
#    1) INSTALL  : Install Python dependencies on all workers
#    2) RUN      : Pull latest code and run Shor's simulation on all workers
#    3) STATUS   : Check if the simulation is still running
#    4) DOWNLOAD : Package and download results + plots
#    5) CLEAN    : Remove generated files from all workers
#    6) FULL     : Install + Run (end-to-end, no interaction needed)
# ============================================================================

# ─── Configuration ───────────────────────────────────────────────────────────
ZONE="us-central1-a"
TPU_NAME="tpu-16chip-worker"
REPO_DIR="~/jax-quantum-research"
ENV_ACTIVATE="~/tpu_env/bin/activate"
SCRIPT="tpu/shors_algorithm_33q.py"
REQS="requirements_tpu.txt"

echo "============================================================"
echo "  Shor's Algorithm 33-Qubit Simulation"
echo "  Google Cloud TPU v5e-16  (16 chips × 16 GB HBM = 256 GB)"
echo "  TPU: $TPU_NAME  |  Zone: $ZONE"
echo "============================================================"

# ─── Determine action ────────────────────────────────────────────────────────
if [ -n "$1" ]; then
    choice="$1"
else
    echo ""
    echo "  Choose an action:"
    echo "  1) INSTALL  : Install dependencies on all workers"
    echo "  2) RUN      : Pull code + run Shor's simulation (all workers)"
    echo "  3) STATUS   : Check if simulation is still running"
    echo "  4) DOWNLOAD : Download latest results and plots"
    echo "  5) CLEAN    : Remove generated results from all workers"
    echo "  6) FULL     : INSTALL + RUN (non-interactive end-to-end)"
    echo ""
    read -p "  Enter choice [1-6]: " choice
fi

# ─── Helper: run command on all workers with clear per-worker output ──────────
run_all_workers() {
    local CMD="$1"
    echo "--> Running on all $TPU_NAME workers: $CMD"
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" \
        --worker=all \
        --command="$CMD"
}

# ─── Actions ─────────────────────────────────────────────────────────────────
case $choice in

    1|install)
        # ── Install jax[tpu] + numpy + matplotlib on every worker ──
        echo ""
        echo "--> [1/1] Installing Python dependencies on all workers..."
        run_all_workers "
            set -e
            # Create or reuse virtual environment
            if [ ! -d ~/tpu_env ]; then
                python3 -m venv ~/tpu_env
                echo 'Created fresh virtual environment ~/tpu_env'
            fi
            source ~/tpu_env/bin/activate
            pip install --upgrade pip --quiet

            # Install JAX with TPU backend (libtpu will be pulled automatically)
            pip install 'jax[tpu]>=0.5.0' \
                --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \
                --quiet

            # Install remaining requirements
            pip install -r $REPO_DIR/$REQS --quiet
            echo '✅  All packages installed successfully on \$(hostname)'

            # Quick sanity check
            python3 -c \"
import jax
print('JAX version :', jax.__version__)
print('Backend     :', jax.default_backend())
print('Devices     :', jax.devices())
\"
        "
        echo ""
        echo "✅  Installation complete on all workers."
        ;;

    2|run)
        # ── Pull latest code from GitHub and launch Shor's simulation ──
        echo ""
        echo "--> [1/2] Pulling latest code from GitHub on all workers..."
        run_all_workers "
            cd $REPO_DIR && git pull origin main
            echo 'Git pull done on \$(hostname)'
        "

        echo ""
        echo "--> [2/2] Launching Shor's algorithm simulation (all workers, background)..."
        # Launch with nohup so the SSH session can close without killing the job.
        # Output is captured by the Tee class inside the Python script
        # (tpu/results/shors_33q_<timestamp>.txt).
        run_all_workers "
            source $ENV_ACTIVATE
            cd $REPO_DIR
            nohup python3 $SCRIPT \
                > /tmp/shors_latest.log 2>&1 &
            echo \"🚀  Launched on \$(hostname) — PID \$!\"
            echo \"    Tail log: tail -f /tmp/shors_latest.log\"
        "
        echo ""
        echo "  ✅  Shor's simulation launched on all workers."
        echo "  📋  To monitor:   bash tpu/run_shor_tpu.sh 3"
        echo "  📥  To download:  bash tpu/run_shor_tpu.sh 4"
        ;;

    3|status)
        # ── Check if simulation is still running ──
        echo ""
        echo "--> Checking simulation status on all workers..."
        run_all_workers "
            echo '── \$(hostname) ──'
            if pgrep -f shors_algorithm_33q.py > /dev/null; then
                echo '  🟢 RUNNING'
                echo '  Recent output:'
                tail -n 10 /tmp/shors_latest.log 2>/dev/null || echo '  (no log yet)'
            else
                echo '  🔴 NOT RUNNING'
                echo '  Last output:'
                tail -n 5 /tmp/shors_latest.log 2>/dev/null || echo '  (no log found)'
            fi
        "
        ;;

    4|download)
        # ── Package results and download to Cloud Shell / PC ──
        echo ""
        read -p "  Enter run timestamp (e.g. 20260525_140000) or press Enter for latest: " TS_IN
        if [ -z "$TS_IN" ]; then
            echo "--> Finding latest timestamp on worker 0..."
            TS_IN=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
                --zone="$ZONE" --worker=0 \
                --command="ls $REPO_DIR/tpu/results/shors_33q_*.json 2>/dev/null \
                           | sort | tail -1 \
                           | grep -oP '\d{8}_\d{6}'" 2>/dev/null)
            if [ -z "$TS_IN" ]; then
                echo "  ❌  No results found. Run the simulation first (option 2)."
                exit 1
            fi
            echo "  Found timestamp: $TS_IN"
        fi

        echo "--> Packing results from worker 0..."
        gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
            --zone="$ZONE" --worker=0 \
            --command="
                cd $REPO_DIR
                tar -czf ~/shors_run_${TS_IN}.tar.gz \
                    tpu/results/*${TS_IN}* \
                    tpu/plots/*${TS_IN}* \
                    2>/dev/null
                echo 'Archive created: ~/shors_run_${TS_IN}.tar.gz'
                ls -lh ~/shors_run_${TS_IN}.tar.gz
            "

        echo "--> Downloading archive to Cloud Shell..."
        gcloud compute tpus tpu-vm scp \
            "$TPU_NAME":~/shors_run_${TS_IN}.tar.gz \
            ~/shors_run_${TS_IN}.tar.gz \
            --zone="$ZONE" --worker=0

        echo "--> Triggering browser download..."
        cloudshell download ~/shors_run_${TS_IN}.tar.gz

        echo "  ✅  Download complete: shors_run_${TS_IN}.tar.gz"
        ;;

    5|clean)
        # ── Remove generated files from all workers ──
        echo ""
        echo "--> Cleaning results and plots on all workers..."
        run_all_workers "
            rm -rf $REPO_DIR/tpu/results/shors_* \
                   $REPO_DIR/tpu/plots/shors_* \
                   ~/shors_run_*.tar.gz \
                   /tmp/shors_latest.log
            echo '🗑  Cleaned on \$(hostname)'
        "
        echo "  ✅  Clean complete."
        ;;

    6|full)
        # ── Full non-interactive end-to-end: install + run ──
        echo ""
        echo "--> Running FULL pipeline: install dependencies + launch simulation"
        bash "$0" 1   # Install
        echo ""
        bash "$0" 2   # Run
        echo ""
        echo "  ✅  Full pipeline complete."
        echo "  📋  Monitor:   bash tpu/run_shor_tpu.sh 3"
        echo "  📥  Download:  bash tpu/run_shor_tpu.sh 4"
        ;;

    *)
        echo "  ❌  Invalid choice: $choice"
        echo "  Usage: bash tpu/run_shor_tpu.sh [1-6]"
        exit 1
        ;;
esac
