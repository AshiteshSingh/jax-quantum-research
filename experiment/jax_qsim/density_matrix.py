"""
Density matrix simulator implemented in pure, differentiable JAX.
Supports noisy channels via Kraus operators and fully analytic gradients of noisy circuits.
"""

import jax
import jax.numpy as jnp
from . import gates
from .statevector import apply_gate as sv_apply_gate
from .observables import PauliString, Hamiltonian

def zero_state(num_qubits):
    """
    Returns the computational basis density matrix |00...0><00...0| 
    as a JAX array of shape (2,)*2N.
    """
    rho = jnp.zeros((2,) * (2 * num_qubits), dtype=jnp.complex64)
    zero_idx = (0,) * (2 * num_qubits)
    return rho.at[zero_idx].set(1.0)

def apply_gate(rho, gate, target_qubits):
    r"""
    Applies a quantum gate to a density matrix: rho -> U * rho * U^\dagger.
    """
    n = rho.ndim // 2
    rho = sv_apply_gate(rho, gate, target_qubits)
    target_cols = [q + n for q in target_qubits]
    rho = sv_apply_gate(rho, jnp.conj(gate), target_cols)
    return rho

# ==============================================================================
# Quantum Channels (Noise Models) via Kraus Operators
# ==============================================================================

def depolarizing_kraus(p):
    """Kraus operators for the single-qubit depolarizing channel."""
    s = jnp.sqrt(p / 3.0)
    K0 = jnp.sqrt(1.0 - p) * jnp.eye(2, dtype=jnp.complex64)
    K1 = s * gates.X()
    K2 = s * gates.Y()
    K3 = s * gates.Z()
    return [K0, K1, K2, K3]

def amplitude_damping_kraus(gamma):
    """Kraus operators for the single-qubit amplitude damping channel."""
    K0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1.0 - gamma)]], dtype=jnp.complex64)
    K1 = jnp.array([[0.0, jnp.sqrt(gamma)], [0.0, 0.0]], dtype=jnp.complex64)
    return [K0, K1]

def phase_damping_kraus(gamma):
    """Kraus operators for the single-qubit phase damping channel."""
    K0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1.0 - gamma)]], dtype=jnp.complex64)
    K1 = jnp.array([[0.0, 0.0], [0.0, jnp.sqrt(gamma)]], dtype=jnp.complex64)
    return [K0, K1]

def bit_flip_kraus(p):
    """Kraus operators for the single-qubit bit flip channel."""
    K0 = jnp.sqrt(1.0 - p) * jnp.eye(2, dtype=jnp.complex64)
    K1 = jnp.sqrt(p) * gates.X()
    return [K0, K1]

def phase_flip_kraus(p):
    """Kraus operators for the single-qubit phase flip channel."""
    K0 = jnp.sqrt(1.0 - p) * jnp.eye(2, dtype=jnp.complex64)
    K1 = jnp.sqrt(p) * gates.Z()
    return [K0, K1]
