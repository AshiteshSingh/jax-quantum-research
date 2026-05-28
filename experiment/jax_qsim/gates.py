"""
Quantum gates implemented in pure JAX.
All gates return complex64 JAX arrays.
"""

import jax.numpy as jnp

# ==============================================================================
# Non-parametric Gates (Single-qubit)
# ==============================================================================

def H():
    """Hadamard gate."""
    return jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex64) / jnp.sqrt(2.0)

def X():
    """Pauli-X (NOT) gate."""
    return jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex64)

def Y():
    """Pauli-Y gate."""
    return jnp.array([[0.0, -1j], [1j, 0.0]], dtype=jnp.complex64)

def Z():
    """Pauli-Z gate."""
    return jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex64)

def S():
    """S (Phase) gate."""
    return jnp.array([[1.0, 0.0], [0.0, 1j]], dtype=jnp.complex64)

def T():
    """T gate (pi/8 gate)."""
    return jnp.array([[1.0, 0.0], [0.0, jnp.exp(1j * jnp.pi / 4.0)]], dtype=jnp.complex64)

# ==============================================================================
# Parametric Gates (Single-qubit)
# ==============================================================================

def RX(theta):
    """Rotation around X-axis by angle theta."""
    c = jnp.cos(theta / 2.0)
    s = -1j * jnp.sin(theta / 2.0)
    return jnp.array([[c, s], [s, c]], dtype=jnp.complex64)

def RY(theta):
    """Rotation around Y-axis by angle theta."""
    c = jnp.cos(theta / 2.0)
    s = jnp.sin(theta / 2.0)
    return jnp.array([[c, -s], [s, c]], dtype=jnp.complex64)

def RZ(theta):
    """Rotation around Z-axis by angle theta."""
    e = jnp.exp(-1j * theta / 2.0)
    return jnp.array([[e, 0.0], [0.0, jnp.conj(e)]], dtype=jnp.complex64)
