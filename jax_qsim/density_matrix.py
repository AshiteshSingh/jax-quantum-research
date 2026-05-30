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
    rho = jnp.zeros(2**(2 * num_qubits), dtype=jnp.complex64)
    rho = rho.at[0].set(1.0)
    return rho.reshape((2,) * (2 * num_qubits))

def apply_gate(rho, gate, target_qubits):
    r"""
    Applies a quantum gate to a density matrix: rho -> U * rho * U^\dagger.
    
    Args:
        rho: JAX array of shape (2,)*2N representing the density matrix.
        gate: JAX array representing the gate unitary. For a k-qubit gate, 
              shape must be (2**k, 2**k) or (2,)*2k.
        target_qubits: List or tuple of integer qubit indices.
        
    Returns:
        Updated density matrix as a JAX array of shape (2,)*2N.
    """
    n = rho.ndim // 2
    # Apply U on the left (row indices, 0 to n-1)
    rho = sv_apply_gate(rho, gate, target_qubits)
    # Apply U* on the right (column indices, n to 2n-1)
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
    
    Args:
        rho: JAX array of shape (2,)*2N.
        kraus_ops: List of 2x2 JAX arrays representing the Kraus operators.
        qubit: Integer qubit index the channel acts on.
    """
    n = rho.ndim // 2
    out = jnp.zeros_like(rho)
    for K in kraus_ops:
        # Apply K on row indices
        temp = sv_apply_gate(rho, K, [qubit])
        # Apply K* on column indices
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
    # Apply Pauli operators to row indices
    for q, op in pauli_string.paulis.items():
        if op == 'X':
            phi = sv_apply_gate(phi, gates.X(), [q])
        elif op == 'Y':
            phi = sv_apply_gate(phi, gates.Y(), [q])
        elif op == 'Z':
            phi = sv_apply_gate(phi, gates.Z(), [q])
            
    # Trace is sum of diagonal elements of phi (reshaped to 2^n x 2^n)
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
