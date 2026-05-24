#!/bin/bash
# ============================================================
#  run_gpu.sh - GPU-Accelerated JAX Quantum Simulator Launcher
#  Run this inside WSL2 to use your NVIDIA GeForce RTX 2050 GPU
# ============================================================

# Activate the GPU-enabled virtual environment
source ~/jax_gpu_env/bin/activate

# Set the project directory
cd "/mnt/c/Users/mswuk/Desktop/qauntum machine learning"

# Add GPU directory to python path so jax_qsim is importable
export PYTHONPATH="$PYTHONPATH:/mnt/c/Users/mswuk/Desktop/qauntum machine learning/gpu"

# Verify GPU is being used by JAX
echo "==========================================="
echo "  JAX GPU Verification"
echo "==========================================="
python3 -c "
import jax
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
print('Default backend:', jax.default_backend())
d = jax.devices('gpu')
print('GPU devices:', d)
"

echo ""
echo "==========================================="
echo "  Choose a Quantum Research Experiment to run:"
echo "  1) GHZ State Preparation             (ghz_state_preparation.py)"
echo "  2) Variational Quantum Classifier    (variational_quantum_classifier_xor.py)"
echo "  3) GPU Qubit & VRAM Scaling Benchmark (gpu_vram_and_qubit_scaling_benchmark.py)"
echo "  4) Run Test Suite                    (pytest tests/)"
echo "==========================================="
echo ""

read -p "Enter choice [1-4]: " choice

case $choice in
    1) python3 gpu/quantum_research/ghz_state_preparation.py ;;
    2) python3 gpu/quantum_research/variational_quantum_classifier_xor.py ;;
    3) python3 gpu/quantum_research/gpu_vram_and_qubit_scaling_benchmark.py ;;
    4) python3 -m pytest tests/ -v ;;
    *) echo "Invalid choice." ;;
esac

