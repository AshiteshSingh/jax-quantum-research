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

def apply_channel_1q(rho, kraus_ops, qubit):
    r"""
    Applies a single-qubit channel to a density matrix: rho -> sum_i K_i * rho * K_i^\dagger.
    """
    n = rho.ndim // 2
    out = jnp.zeros_like(rho)
    for K in kraus_ops:
        temp = sv_apply_gate(rho, K, [qubit])
        temp = sv_apply_gate(temp, jnp.conj(K), [qubit + n])
        out = out + temp
    return out

# ==============================================================================
# Expectation Values
# ==============================================================================

def expectation_pauli_string(rho, pauli_string):
    """
    Computes the expectation value Tr(P * rho) for a PauliString P.
    """
    n = rho.ndim // 2
    phi = rho
    for q, op in pauli_string.paulis.items():
        if op == 'X':
            phi = sv_apply_gate(phi, gates.X(), [q])
        elif op == 'Y':
            phi = sv_apply_gate(phi, gates.Y(), [q])
        elif op == 'Z':
            phi = sv_apply_gate(phi, gates.Z(), [q])
            
    phi_mat = phi.reshape((2**n, 2**n))
    return jnp.real(jnp.trace(phi_mat))

def expectation_hamiltonian(rho, hamiltonian):
    """
    Computes the expectation value Tr(H * rho) for a Hamiltonian H.
    """
    exp_val = 0.0
    for coeff, pauli_string in zip(hamiltonian.coeffs, hamiltonian.pauli_strings):
        exp_val += coeff * expectation_pauli_string(rho, pauli_string)
    return exp_val

class DensityMatrix:
    """
    A user-friendly class wrapper around functional density matrix JAX routines.
    """
    def __init__(self, num_qubits, data=None):
        self.num_qubits = num_qubits
        self.data = data if data is not None else zero_state(num_qubits)
        
    def apply_gate(self, gate, target_qubits):
        self.data = apply_gate(self.data, gate, target_qubits)
        return self
        
    def apply_channel(self, kraus_ops, qubit):
        self.data = apply_channel_1q(self.data, kraus_ops, qubit)
        return self
        
    def expectation(self, observable):
        if isinstance(observable, PauliString):
            return expectation_pauli_string(self.data, observable)
        elif isinstance(observable, Hamiltonian):
            return expectation_hamiltonian(self.data, observable)
        else:
            raise TypeError("Observable must be PauliString or Hamiltonian")
