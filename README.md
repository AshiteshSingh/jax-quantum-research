# JAX Quantum Research Suite — Dual GPU & TPU Accelerated Architectures

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![JAX](https://img.shields.io/badge/JAX-0.4%2B-orange?style=for-the-badge&logo=google)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green?style=for-the-badge&logo=nvidia)
![TPU](https://img.shields.io/badge/TPU-v5e--16-purple?style=for-the-badge&logo=google-cloud)
![Platform](https://img.shields.io/badge/Platform-GPU_|_TPU_|_CPU-blueviolet?style=for-the-badge)

**A high-performance, research-grade quantum state-vector simulator built purely in JAX.  
Execute differentiable, noise-resilient, and large-scale quantum circuits accelerated on local NVIDIA GPUs and multi-worker Google Cloud TPU clusters.**

</div>

---

## 🌟 Co-Existing Architectures

This repository is bifurcated into two specialized architectures optimized for different hardware scales:

### 1. 🎮 GPU Architecture (Modular & Differentiable Simulator)
Designed for local development, custom circuit composition, and interactive research using consumer or datacenter **NVIDIA GPUs** via CUDA / WSL2.
* **Core:** The modular `jax_qsim/` engine. It implements high-performance gate application using `jnp.tensordot` axis contractions and inverse permutations.
* **Workflow:** Ideal for rapid design of quantum neural networks (QML), quantum chemistry ansatzes (VQE), and custom quantum noise models.

### 2. ⚡ TPU Architecture (High-Performance Distributed Scaling Suite)
Tailored to high-memory scaling experiments on multi-worker distributed clusters (e.g., **Google Cloud TPU v5e-16** / **v5litepod-16** clusters, total 256 GB HBM2e).
* **Core:** `tpu_quantum_scale.py` — A self-contained, multi-worker optimized executable running 8 unified quantum experiments.
* **Optimizations:** Eliminates compiler graph-bloat via exact XLA parameter boundaries, replaces massive tensor operations with lightweight JAX `lax.fori_loop` state transitions, and splits massive $2^N$ state vectors across physical nodes utilizing multi-device `PositionalSharding`.

---

## 🏗 Directory & Architecture Layout

```
qauntum machine learning/
├── jax_qsim/                     # === GPU MODULAR SIMULATOR ===
│   ├── __init__.py               
│   ├── core.py                   ← Tensor contraction engine (tensordot + transpose)
│   ├── ops.py                    ← Standard unitary & parameter-driven gates
│   ├── observables.py            ← Pauli strings, expectation values, sampling
│   └── noise.py                  ← Quantum noise Kraus channel stochastic applying
│
├── examples/                     # === GPU RESEARCH SAMPLES ===
│   ├── 01_state_preparation.py   ← GHZ State learning using JAX grad & Adam
│   ├── 02_vqc_classification.py  ← VQC XOR boundary resolution (jax.vmap batched)
│   ├── 03_benchmarks.py          ← Local GPU VRAM & qubit scaling scaling
│   ├── 04_vqe_h2_molecule.py     ← VQE ground-state chemical accuracy VQE
│   ├── 05_qaoa_maxcut.py         ← QAOA MaxCut graph optimiser (p=1..5)
│   └── 06_barren_plateaus.py     ← Barren plateaus gradient vanishing analysis
│
├── tpu_quantum_scale.py          # === TPU DISTRIBUTED SCALE SUITE (All 8 Exps) ===
│                                 # Self-contained executable with multi-device PositionalSharding, 
│                                 # flat lax.fori_loop complexity & Tee logs.
│
├── run_gpu.sh                    ← Local WSL2 GPU example launcher
├── run_tpu.sh                    ← Remote Cloud Shell TPU cluster automation controller
├── tests/                        ← Pytest verification suite (gates, AD gradients)
└── results/                      ← Generated JSON outputs, CSVs, and logs
```

---

## 🛠 GPU Getting Started Guide (WSL2 / Linux PC)

For Windows systems with NVIDIA GPUs, JAX requires **WSL2** (Windows Subsystem for Linux) to run GPU acceleration.

### 1. Set Up WSL2 & Create Virtual Environment
In Windows PowerShell (as Administrator), enable WSL2 if you haven't already:
```powershell
wsl --install
```
Then open your WSL2 Linux terminal, create, and activate a virtual environment:
```bash
python3 -m venv ~/jax_gpu_env
source ~/jax_gpu_env/bin/activate
pip install --upgrade pip
```

### 2. Install CUDA-Enabled JAX & Dependencies
```bash
# Install CUDA 12 support
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install physics, testing, and charting packages
pip install matplotlib pytest numpy
```

### 3. Clone & Verify GPU Execution
```bash
git clone https://github.com/AshiteshSingh/jax-quantum-research.git
cd jax-quantum-research

# Run JAX device check
python3 -c "import jax; print('Backend:', jax.default_backend()); print('Devices:', jax.devices())"
```
*Expected Output:* `Backend: gpu` along with your local `CudaDevice`.

### 4. Run Modular GPU Examples
Launch the interactive GPU shell helper:
```bash
chmod +x run_gpu.sh
./run_gpu.sh
```

---

## 🚀 TPU Getting Started Guide (Google Cloud TPU v5e-16)

For high-end scaling experiments, run the suite on a **16-chip Cloud TPU VM cluster** (256 GB aggregate HBM2e memory).

### 1. SSH into the TPU VM Cluster
From your local Google Cloud Shell, authenticate and open a connection into the distributed TPU VM cluster (this targets all 4 workers in a 16-chip mesh):
```bash
gcloud compute tpus tpu-vm ssh tpu-16chip-worker \
  --zone=us-central1-a \
  --worker=all
```

### 2. Configure Virtual Environment & Packages (All Workers)
Inside the SSH session (configured for all workers), run:
```bash
# Create and activate Python virtual environment
python3 -m venv ~/tpu_env
source ~/tpu_env/bin/activate
pip install --upgrade pip

# Install JAX with official Google TPU support & Matplotlib
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install matplotlib numpy
```

### 3. Initialize Repository on TPU VM Mesh
Still inside the mesh SSH session, clone the repository to all physical hosts:
```bash
git clone https://github.com/AshiteshSingh/jax-quantum-research.git
```

### 4. Run & Control TPU Execution via Cloud Shell
Exit the TPU VM SSH session to return to your **Cloud Shell console**. We have created an automation controller script `run_tpu.sh` to make managing the cluster easy.

Run the launcher from your Cloud Shell:
```bash
chmod +x run_tpu.sh
./run_tpu.sh
```
The script provides interactive options:
* **`1` (TERMINATE):** Instantly kills any zombie Python processes locked on `libtpu.so` across all workers (crucial if a previous run crashed or hung).
* **`2` (SYNC & RUN):** Syncs all workers with your latest git commit, compiles, and runs the entire 8-experiment suite.
* **`3` (DOWNLOAD):** Archives only the CSV/JSON results and high-res PNG plots generated from the run and pulls them to your local PC.
* **`4` (CLEANUP):** Clears output directories on the cluster to reset storage space.

---

## 🔬 Unified Research Suite (8 Experiments)

Both platforms cover high-fidelity experiments illustrating advanced physics phenomena:

| Exp | Name | Core Physics Concepts | JAX/Hardware Operations |
|:---:|---|---|---|
| **1** | **GHZ State Preparation** | Quantum Entanglement $|\text{GHZ}\rangle = \frac{|000\rangle+|111\rangle}{\sqrt{2}}$ | Reverse-mode Auto-Diff, Adam optimizer |
| **2** | **VQC XOR Classifier** | Variational Classifiers, Quantum Feature Mapping | `jax.vmap` high-speed batch evaluation |
| **3** | **VQE $H_2$ Ground State** | Molecular Orbitals, STO-3G potential energy surfaces | Jordan-Wigner Hamiltonian transformations |
| **4** | **QAOA MaxCut** | Discrete Combinatorial Optimization, Graph cuts | Parametric gate compilation & JIT |
| **5** | **Quantum Noise Simulation** | System-bath interactions, Kraus maps | Stochastic Monte Carlo Trajectory method |
| **6** | **Noisy NISQ Simulation** | Gate errors, physical decay | Depolarizing quantum gate channels |
| **7** | **Barren Plateau Study** | Parameterized Quantum Circuits (PQCs) | Exponential gradient decay fits, 2D Loss Landscapes |
| **8** | **High-Perf Scaling Benchmark** | State-vector scaling limit limits (up to **33 qubits/64 GB**) | TPU multi-device `PositionalSharding` & `lax.fori_loop` |

---

## 📊 Hardware Benchmarks & Performance Comparison

### Local GPU (RTX 2050 4 GB VRAM)
* **Max Qubits:** 29 qubits ($2^{29} \times 8$ bytes $\approx$ 4.29 GB VRAM saturation limit).
* **JIT Speedup:** Up to **400× faster** compared to uncompiled Python loops.
* **Output Plots:** Saves detailed convergence plots to `examples/plots/`.

### Distributed Cloud TPU (v5e-16 Cluster, 256 GB HBM2e)
* **Max Qubits:** **33 qubits** successfully benchmarked ($2^{33} \times 8$ bytes $\approx$ 64.00 GB distributed state vector).
* **Scaling speed:** Scales seamlessly up to 33 qubits in **154 seconds** total run time due to high-performance `lax.fori_loop` vectorizations.
* **Watermarked Graphs:** The benchmark suite saves a 6-panel performance plot (`tpu_benchmark_[timestamp].png`) containing exact scaling fit laws directly in `examples/plots/`.

---

## 📝 TPU Results Download Guide
When you run the TPU suite, it outputs files with a unique run timestamp (e.g. `20260524_110111`). You can easily download them by running:
```bash
./run_tpu.sh
```
Select **Option 3**, enter your run timestamp `20260524_110111`, and the script will automatically pack the results (`.csv`, `.json`, `.png` plot, and the full console log `.txt` file) and trigger a browser download popup.

---

## 📄 License
This JAX research suite is licensed under the MIT License.

<div align="center">
Built with ❤️ by JAX Quantum Computing Researchers
</div>
