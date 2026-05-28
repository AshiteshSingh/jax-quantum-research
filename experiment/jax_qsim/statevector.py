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
