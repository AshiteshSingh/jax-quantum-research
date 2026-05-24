import jax
import jax.numpy as jnp

# --- 1-Qubit Non-Parameterized Gates ---

# Pauli-X (NOT)
X = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex64)

# Pauli-Y
Y = jnp.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=jnp.complex64)

# Pauli-Z
Z = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex64)

# Hadamard
H = (1.0 / jnp.sqrt(2.0)) * jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex64)

# Identity
I = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.complex64)

# S gate (Phase gate)
S = jnp.array([[1.0, 0.0], [0.0, 1.0j]], dtype=jnp.complex64)

# T gate (pi/8 gate)
T = jnp.array([[1.0, 0.0], [0.0, jnp.exp(1.0j * jnp.pi / 4.0)]], dtype=jnp.complex64)


# --- 1-Qubit Parameterized Gates ---

def rx(theta: float) -> jnp.ndarray:
    """Rotation around the X axis by angle theta."""
    cos = jnp.cos(theta / 2.0)
    sin = jnp.sin(theta / 2.0)
    return jnp.array([
        [cos, -1.0j * sin],
        [-1.0j * sin, cos]
    ], dtype=jnp.complex64)

def ry(theta: float) -> jnp.ndarray:
    """Rotation around the Y axis by angle theta."""
    cos = jnp.cos(theta / 2.0)
    sin = jnp.sin(theta / 2.0)
    return jnp.array([
        [cos, -sin],
        [sin, cos]
    ], dtype=jnp.complex64)

def rz(theta: float) -> jnp.ndarray:
    """Rotation around the Z axis by angle theta."""
    exp_neg = jnp.exp(-1.0j * theta / 2.0)
    exp_pos = jnp.exp(1.0j * theta / 2.0)
    return jnp.array([
        [exp_neg, 0.0],
        [0.0, exp_pos]
    ], dtype=jnp.complex64)

def phase_shift(phi: float) -> jnp.ndarray:
    """Phase shift gate by angle phi."""
    return jnp.array([
        [1.0, 0.0],
        [0.0, jnp.exp(1.0j * phi)]
    ], dtype=jnp.complex64)


# --- Multi-Qubit Construction Helpers ---

def controlled_gate(U: jnp.ndarray) -> jnp.ndarray:
    """Construct a 2-qubit controlled version of a 1-qubit gate U.
    
    The first qubit is the control and the second is the target.
    
    Returns a (4, 4) complex matrix.
    """
    zero = jnp.zeros((2, 2), dtype=jnp.complex64)
    # top-left block is Identity (when control is |0>)
    # bottom-right block is U (when control is |1>)
    row1 = jnp.hstack([I, zero])
    row2 = jnp.hstack([zero, U])
    return jnp.vstack([row1, row2])


# --- 2-Qubit Non-Parameterized Gates ---

# Controlled-X (CNOT)
CNOT = controlled_gate(X)

# Controlled-Z (CZ)
CZ = controlled_gate(Z)

# SWAP Gate
SWAP = jnp.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
], dtype=jnp.complex64)


# --- 2-Qubit Parameterized Gates ---

def crx(theta: float) -> jnp.ndarray:
    """Controlled rotation around the X axis by angle theta."""
    return controlled_gate(rx(theta))

def cry(theta: float) -> jnp.ndarray:
    """Controlled rotation around the Y axis by angle theta."""
    return controlled_gate(ry(theta))

def crz(theta: float) -> jnp.ndarray:
    """Controlled rotation around the Z axis by angle theta."""
    return controlled_gate(rz(theta))

def cphase(phi: float) -> jnp.ndarray:
    """Controlled phase shift by angle phi."""
    return controlled_gate(phase_shift(phi))
