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
    state = jnp.zeros(2**num_qubits, dtype=jnp.complex64)
    state = state.at[0].set(1.0)
    return state.reshape((2,) * num_qubits)

def apply_gate(state, gate, target_qubits):
    """
    Applies a quantum gate to a state vector.
    
    Args:
        state: JAX array of shape (2,)*N representing the wavefunction.
        gate: JAX array representing the gate unitary. For a k-qubit gate, 
              shape must be (2**k, 2**k) or (2,)*2k.
        target_qubits: List or tuple of integer qubit indices the gate acts on.
        
    Returns:
        New JAX array representing the updated state vector.
    """
    n = state.ndim
    k = len(target_qubits)
    
    # Flatten the state vector
    state_flat = state.reshape(-1)
    
    if k == 1:
        # 1-qubit gate: 3D transpose optimization
        q = target_qubits[0]
        gate_2x2 = gate.reshape(2, 2)
        
        shape_3d = (1 << q, 2, 1 << (n - q - 1))
        state_3d = state_flat.reshape(shape_3d)
        state_transposed = jnp.transpose(state_3d, (1, 0, 2))
        state_2d = state_transposed.reshape(2, -1)
        out_2d = gate_2x2 @ state_2d
        out_transposed = out_2d.reshape(2, 1 << q, 1 << (n - q - 1))
        out_3d = jnp.transpose(out_transposed, (1, 0, 2))
        res_flat = out_3d.reshape(-1)
    elif k == 2:
        # 2-qubit gate: 5D transpose optimization
        q1, q2 = target_qubits
        gate_4x4 = gate.reshape(4, 4)
        
        if q1 > q2:
            gate_tensor = gate_4x4.reshape(2, 2, 2, 2)
            gate_tensor = jnp.transpose(gate_tensor, (1, 0, 3, 2))
            gate_4x4 = gate_tensor.reshape(4, 4)
            left_q, right_q = q2, q1
        else:
            left_q, right_q = q1, q2
            
        shape_5d = (
            1 << left_q,
            2,
            1 << (right_q - left_q - 1),
            2,
            1 << (n - right_q - 1)
        )
        state_5d = state_flat.reshape(shape_5d)
        state_transposed = jnp.transpose(state_5d, (1, 3, 0, 2, 4))
        state_2d = state_transposed.reshape(4, -1)
        out_2d = gate_4x4 @ state_2d
        out_transposed = out_2d.reshape(2, 2, 1 << left_q, 1 << (right_q - left_q - 1), 1 << (n - right_q - 1))
        out_5d = jnp.transpose(out_transposed, (2, 0, 3, 1, 4))
        res_flat = out_5d.reshape(-1)
    else:
        # Fallback to standard multi-dimensional tensor contractions for k > 2
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
        
    return res_flat.reshape((2,) * n)

def expectation_pauli_string(state, pauli_string):
    """
    Computes the expectation value <psi| P |psi> for a PauliString P.
    """
    n = state.ndim
    phi = state
    
    # Apply Pauli operators
    for q, op in pauli_string.paulis.items():
        if op == 'X':
            phi = apply_gate(phi, gates.X(), [q])
        elif op == 'Y':
            phi = apply_gate(phi, gates.Y(), [q])
        elif op == 'Z':
            phi = apply_gate(phi, gates.Z(), [q])
            
    # Compute inner product <psi|phi>
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
    
    Returns:
        JAX array of shape (num_shots, num_qubits) containing sampled bits.
    """
    n = state.ndim
    probs = jnp.abs(state.reshape(-1)) ** 2
    
    # Sample integer indices [0, 2**n - 1]
    sampled_indices = jax.random.choice(key, 2**n, shape=(num_shots,), p=probs)
    
    # Convert integer indices to bitstrings
    # We use bitwise shifting to extract bits
    powers = 2 ** jnp.arange(n - 1, -1, -1)
    bitstrings = (sampled_indices[:, None] // powers[None, :]) % 2
    return bitstrings

def measure(state, qubit, key):
    """
    Projects the specified qubit, measuring it in the computational basis.
    
    Returns:
        measured_bit: 0 or 1 (as JAX int)
        new_state: Normalized collapsed state vector.
    """
    n = state.ndim
    
    # Compute probability of measuring 0
    # Sum over all qubit dimensions except the measured qubit
    probs = jnp.abs(state) ** 2
    axes = tuple(i for i in range(n) if i != qubit)
    marginal_prob = jnp.sum(probs, axis=axes)
    prob_0 = marginal_prob[0]
    
    # Roll a random number
    r = jax.random.uniform(key)
    measured_bit = jnp.where(r < prob_0, 0, 1)
    
    # Collapsed state vector
    # Project qubit to the measured bit slice
    # Slice dynamic update
    collapsed = jnp.where(
        measured_bit == 0,
        state.at[tuple(slice(None) if i != qubit else 0 for i in range(n))].get(),
        state.at[tuple(slice(None) if i != qubit else 1 for i in range(n))].get()
    )
    
    # Re-expand the slice back to original shape with zeros for the other slice
    new_state = jnp.zeros_like(state)
    new_state = jnp.where(
        measured_bit == 0,
        new_state.at[tuple(slice(None) if i != qubit else 0 for i in range(n))].set(collapsed),
        new_state.at[tuple(slice(None) if i != qubit else 1 for i in range(n))].set(collapsed)
    )
    
    # Re-normalize
    norm = jnp.sqrt(jnp.sum(jnp.abs(new_state) ** 2))
    return measured_bit, new_state / norm

class Statevector:
    """
    A user-friendly class wrapper around functional statevector JAX routines.
    """
    def __init__(self, num_qubits, data=None):
        self.num_qubits = num_qubits
        self.data = data if data is not None else zero_state(num_qubits)
        
    def apply_gate(self, gate, target_qubits):
        self.data = apply_gate(self.data, gate, target_qubits)
        return self
        
    def expectation(self, observable):
        if isinstance(observable, PauliString):
            return expectation_pauli_string(self.data, observable)
        elif isinstance(observable, Hamiltonian):
            return expectation_hamiltonian(self.data, observable)
        else:
            raise TypeError("Observable must be PauliString or Hamiltonian")
            
    def sample(self, num_shots, key):
        return sample(self.data, num_shots, key)
        
    def measure(self, qubit, key):
        bit, new_data = measure(self.data, qubit, key)
        self.data = new_data
        return bit
