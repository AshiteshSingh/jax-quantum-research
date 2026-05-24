#!/bin/bash
# ============================================================
#  run_tpu.sh - Cloud TPU v5e-16 Cluster Management Launcher
#  Run this inside Google Cloud Shell to manage and run
#  the JAX Quantum Research Suite on your TPU VMs.
# ============================================================

ZONE="us-central1-a"
TPU_NAME="tpu-16chip-worker"
REPO_DIR="~/jax-quantum-research"
ENV_ACTIVATE="~/tpu_env/bin/activate"

echo "==========================================="
echo "  Google Cloud TPU VM Manager & Launcher"
echo "  TPU Name: $TPU_NAME | Zone: $ZONE"
echo "==========================================="
echo "  Choose an action to perform:"
echo "  1) TERMINATE: Kill zombie Python processes (all workers)"
echo "  2) SYNC & RUN: Pull git updates & run suite (all workers)"
echo "  3) DOWNLOAD: Package & download latest plots/results to PC"
echo "  4) CLEANUP: Delete generated results/plots on TPU VM"
echo "==========================================="
echo ""

read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo "--> Terminating all active Python processes across all workers..."
        gcloud compute tpus tpu-vm ssh $TPU_NAME \
            --zone=$ZONE \
            --worker=all \
            --command="sudo pkill -9 -f python3; sleep 2; echo 'Done. All Python processes terminated.'"
        ;;
    2)
        echo "--> Pulling latest repository updates on all workers..."
        gcloud compute tpus tpu-vm ssh $TPU_NAME \
            --zone=$ZONE \
            --worker=all \
            --command="cd $REPO_DIR && git pull"
        
        echo "--> Starting JAX Quantum Research Suite (8 experiments)..."
        gcloud compute tpus tpu-vm ssh $TPU_NAME \
            --zone=$ZONE \
            --worker=all \
            --command="source $ENV_ACTIVATE && cd $REPO_DIR && python3 tpu/tpu_quantum_scale.py"
        ;;
    3)
        read -p "Enter the run timestamp (e.g. 20260524_110111): " ts
        if [ -z "$ts" ]; then
            echo "Error: Timestamp is required."
            exit 1
        fi
        echo "--> Packing files with timestamp: $ts..."
        gcloud compute tpus tpu-vm ssh $TPU_NAME \
            --zone=$ZONE \
            --worker=0 \
            --command="cd $REPO_DIR && tar -czf ~/run_$ts.tar.gz tpu/results/*$ts* tpu/plots/*$ts*"
        
        echo "--> Downloading packed archive to Cloud Shell..."
        gcloud compute tpus tpu-vm scp $TPU_NAME:~/run_$ts.tar.gz ~/run_$ts.tar.gz \
            --zone=$ZONE \
            --worker=0
        
        echo "--> Triggering browser download to your PC..."
        cloudshell download ~/run_$ts.tar.gz
        ;;
    4)
        echo "--> Clearing results and plots on all TPU workers..."
        gcloud compute tpus tpu-vm ssh $TPU_NAME \
            --zone=$ZONE \
            --worker=all \
            --command="rm -rf $REPO_DIR/tpu/results/* $REPO_DIR/tpu/plots/* ~/run_*.tar.gz; echo 'Cleaned successfully.'"
        ;;
    *)
        echo "Invalid choice."
        ;;
esac

