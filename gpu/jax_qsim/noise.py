import jax
import jax.numpy as jnp
from functools import partial
from jax_qsim.core import apply_gate
import jax_qsim.ops as ops

# --- Noise Channel Kraus Operator Generators ---

@jax.jit
def depolarizing_channel(p: float) -> jnp.ndarray:
    """Generate Kraus operators for a 1-qubit depolarizing channel with rate p.
    
    Returns an array of shape (4, 2, 2).
    """
    K0 = jnp.sqrt(1.0 - p) * ops.I
    K1 = jnp.sqrt(p / 3.0) * ops.X
    K2 = jnp.sqrt(p / 3.0) * ops.Y
    K3 = jnp.sqrt(p / 3.0) * ops.Z
    return jnp.stack([K0, K1, K2, K3])

@jax.jit
def amplitude_damping_channel(gamma: float) -> jnp.ndarray:
    """Generate Kraus operators for a 1-qubit amplitude damping channel with relaxation rate gamma.
    
    Returns an array of shape (2, 2, 2).
    """
    K0 = jnp.array([
        [1.0, 0.0],
        [0.0, jnp.sqrt(1.0 - gamma)]
    ], dtype=jnp.complex64)
    K1 = jnp.array([
        [0.0, jnp.sqrt(gamma)],
        [0.0, 0.0]
    ], dtype=jnp.complex64)
    return jnp.stack([K0, K1])

@jax.jit
def phase_damping_channel(gamma: float) -> jnp.ndarray:
    """Generate Kraus operators for a 1-qubit phase damping channel with rate gamma.
    
    Returns an array of shape (2, 2, 2).
    """
    K0 = jnp.array([
        [1.0, 0.0],
        [0.0, jnp.sqrt(1.0 - gamma)]
    ], dtype=jnp.complex64)
    K1 = jnp.array([
        [0.0, 0.0],
        [0.0, jnp.sqrt(gamma)]
    ], dtype=jnp.complex64)
    return jnp.stack([K0, K1])


# --- Trajectory Noise Application ---

@partial(jax.jit, static_argnums=(2,))
def _apply_channel_jit(state: jnp.ndarray, kraus_ops: jnp.ndarray, targets: tuple[int, ...], key: jax.random.PRNGKey) -> tuple[jnp.ndarray, jax.random.PRNGKey]:
    """Pure JAX/XLA JIT compiled quantum trajectory noise channel application."""
    num_kraus = kraus_ops.shape[0]
    
    # 1. Apply each Kraus operator in a pure JAX vectorized way using vmap
    # Convert static targets tuple to a list for apply_gate compatibility
    targets_list = list(targets)
    applied_states = jax.vmap(apply_gate, in_axes=(None, 0, None))(state, kraus_ops, targets_list)
    
    # 2. Compute probabilities: p_j = <psi| K_j^dag K_j |psi> = norm(K_j |psi>)^2
    # Reshape applied_states to (num_kraus, -1) to sum over state dimensions
    norms_sq = jnp.sum(jnp.abs(applied_states.reshape(num_kraus, -1)) ** 2, axis=-1)
    
    # 3. Sample which Kraus operator to apply
    log_probs = jnp.log(jnp.maximum(norms_sq, 1e-12))
    sub_key, new_key = jax.random.split(key)
    sampled_idx = jax.random.categorical(sub_key, log_probs)
    
    # 4. Select the corresponding state and re-normalize it
    selected_state = applied_states[sampled_idx]
    norm = jnp.sqrt(norms_sq[sampled_idx])
    new_state = selected_state / (norm + 1e-12)
    
    return new_state, new_key

def apply_channel(state: jnp.ndarray, kraus_operators: jnp.ndarray, targets: list[int], key: jax.random.PRNGKey) -> tuple[jnp.ndarray, jax.random.PRNGKey]:
    """Apply a noise channel to a state vector using quantum trajectories.
    
    Args:
        state: State-vector of shape (2, ..., 2)
        kraus_operators: Kraus operators of shape (num_kraus, 2**k, 2**k) or list of matrices.
        targets: Target qubits of length k.
        key: JAX PRNGKey.
        
    Returns:
        A tuple of (new_state, new_key).
    """
    kraus_ops = jnp.array(kraus_operators, dtype=jnp.complex64)
    # Convert targets to a hashable tuple to support dynamic compile-time hashing under JIT
    targets_tuple = tuple(targets)
    return _apply_channel_jit(state, kraus_ops, targets_tuple, key)
