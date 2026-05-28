"""
Statevector simulator implemented in pure, differentiable JAX.
"""

import jax
import jax.numpy as jnp
from . import gates
from .observables import PauliString, Hamiltonian

def zero_state(num_qubits):
    """
    Returns the computational basis state |00...0> as a JAX array of shape (2,)*num_qubits.
    """
    state = jnp.zeros((2,) * num_qubits, dtype=jnp.complex64)
    return state.at[(0,) * num_qubits].set(1.0)

def apply_gate(state, gate, target_qubits):
    """
    Applies a quantum gate to a state vector.
    """
    n = state.ndim
    k = len(target_qubits)
    
    if gate.ndim != 2 * k:
        gate = gate.reshape((2,) * (2 * k))
        
    gate_axes = tuple(range(k, 2 * k))
    state_axes = tuple(target_qubits)
    
    out = jnp.tensordot(gate, state, axes=(gate_axes, state_axes))
    uncontracted = [i for i in range(n) if i not in target_qubits]
    
    dest = [None] * n
    for i, q in enumerate(target_qubits):
        dest[q] = i
    curr = k
    for q in uncontracted:
        dest[q] = curr
        curr += 1
        
    return jnp.transpose(out, dest)

def expectation_pauli_string(state, pauli_string):
    """
    Computes the expectation value <psi| P |psi> for a PauliString P.
    """
    n = state.ndim
    phi = state
    
    for q, op in pauli_string.paulis.items():
        if op == 'X':
            phi = apply_gate(phi, gates.X(), [q])
        elif op == 'Y':
            phi = apply_gate(phi, gates.Y(), [q])
        elif op == 'Z':
            phi = apply_gate(phi, gates.Z(), [q])
            
    return jnp.real(jnp.vdot(state, phi))

def expectation_hamiltonian(state, hamiltonian):
    """
    Computes the expectation value <psi| H |psi> for a Hamiltonian H.
    """
    exp_val = 0.0
    for coeff, pauli_string in zip(hamiltonian.coeffs, hamiltonian.pauli_strings):
        exp_val += coeff * expectation_pauli_string(state, pauli_string)
    return exp_val
