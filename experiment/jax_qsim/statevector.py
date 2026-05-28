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

def sample(state, num_shots, key):
    """
    Samples computational basis bitstrings from the state vector.
    """
    n = state.ndim
    probs = jnp.abs(state.reshape(-1)) ** 2
    sampled_indices = jax.random.choice(key, 2**n, shape=(num_shots,), p=probs)
    powers = 2 ** jnp.arange(n - 1, -1, -1)
    bitstrings = (sampled_indices[:, None] // powers[None, :]) % 2
    return bitstrings

def measure(state, qubit, key):
    """
    Projects the specified qubit, measuring it in the computational basis.
    """
    n = state.ndim
    probs = jnp.abs(state) ** 2
    axes = tuple(i for i in range(n) if i != qubit)
    marginal_prob = jnp.sum(probs, axis=axes)
    prob_0 = marginal_prob[0]
    
    r = jax.random.uniform(key)
    measured_bit = jnp.where(r < prob_0, 0, 1)
    
    collapsed = jnp.where(
        measured_bit == 0,
        state.at[tuple(slice(None) if i != qubit else 0 for i in range(n))].get(),
        state.at[tuple(slice(None) if i != qubit else 1 for i in range(n))].get()
    )
    
    new_state = jnp.zeros_like(state)
    new_state = jnp.where(
        measured_bit == 0,
        new_state.at[tuple(slice(None) if i != qubit else 0 for i in range(n))].set(collapsed),
        new_state.at[tuple(slice(None) if i != qubit else 1 for i in range(n))].set(collapsed)
    )
    
    norm = jnp.sqrt(jnp.sum(jnp.abs(new_state) ** 2))
    return measured_bit, new_state / norm
