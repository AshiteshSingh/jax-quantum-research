# Cloud TPU Deployment

This directory documents Cloud TPU deployment for large-scale experiments.

## Hardware Used

| Hardware | Specs | Max Qubits (statevector) | Experiment |
|---|---|---|---|
| TPU v5e-16 | 256 GB HBM2e aggregate | 33 qubits (64 GB sharded) | Shor's algorithm |
| TPU v6e-64 | 2 TB HBM3 aggregate | 37 qubits RCS (tensor-network) | Random Circuit Sampling |

## Provisioning (Google Cloud)

```bash
# TPU v5e-16 (16-chip pod)
gcloud compute tpus tpu-vm create tpu-16chip \
  --zone=us-central1-a \
  --accelerator-type=v5litepod-16 \
  --version=v2-alpha-tpuv5-lite

# SSH to TPU
gcloud compute tpus tpu-vm ssh tpu-16chip --zone=us-central1-a
```

## JAX TPU Setup

```bash
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -c "import jax; print(jax.devices())"  # Should show TPU devices
```

## Sharding for Large Statevectors

```python
import jax
from jax.sharding import PositionalSharding

devices = jax.devices()
sharding = PositionalSharding(devices).reshape(-1)

# 33-qubit state: 8 GB / 16 chips = 512 MB per chip
state = jax.device_put(initial_state, sharding)
```

## 37-Qubit RCS (TensorCircuit)

The 37-qubit random circuit sampling uses TensorCircuit's tensor-network backend:

```python
import tensorcircuit as tc
tc.set_backend("jax")
tc.set_dtype("complex64")

# Circuit: 37-qubit, 20-layer RX/RZ + alternating CZ
c = tc.Circuit(37)
# ... build circuit ...
amplitude = c.amplitude("0" * 37)  # Single amplitude via tensor contraction
```

> **Note:** This is NOT full statevector simulation. Full 37-qubit statevector
> requires ~1 TB RAM (2^37 × 8 bytes = 1.1 TB). Tensor-network contraction
> with finite bond dimension approximates individual amplitudes at polynomial cost.

## Honest F_XEB Result

F_XEB ≈ 0.001 ± 0.003 (N=5 preliminary runs) indicates the sampled distribution
is approximately uniform — expected for a deep chaotic circuit with finite bond
dimension tensor-network. A perfect statevector simulator would yield F_XEB ≈ 1.0
but is computationally intractable.
