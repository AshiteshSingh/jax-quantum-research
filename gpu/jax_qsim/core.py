import jax
import jax.numpy as jnp

def zero_state(num_qubits: int) -> jnp.ndarray:
    """Initialize the |00...0> state for num_qubits qubits.
    
    Returns a complex64 JAX array of shape (2, 2, ..., 2).
    """
    state = jnp.zeros(2**num_qubits, dtype=jnp.complex64)
    state = state.at[0].set(1.0 + 0.0j)
    return state.reshape((2,) * num_qubits)

def apply_gate(state: jnp.ndarray, gate_matrix: jnp.ndarray, targets: list[int]) -> jnp.ndarray:
    """Apply a k-qubit gate to target qubits of a state-vector.
    
    Args:
        state: State-vector of shape (2, ..., 2) for N qubits.
        gate_matrix: Gate matrix of shape (2**k, 2**k).
        targets: List of length k containing the target qubit indices.
        
    Returns:
        New state-vector of shape (2, ..., 2).
    """
    num_qubits = state.ndim
    k = len(targets)
    
    # Reshape the gate matrix to a tensor of shape (2,)*k + (2,)*k
    # First k axes are the output indices, next k axes are the input indices
    gate_tensor = gate_matrix.reshape((2,) * (2 * k))
    
    # Contract the input axes of the gate tensor with the target axes of the state vector
    gate_contract_axes = list(range(k, 2 * k))
    state_contract_axes = list(targets)
    
    res = jnp.tensordot(gate_tensor, state, axes=(gate_contract_axes, state_contract_axes))
    
    # After tensordot, the new state has axes:
    # [new_targets[0], ..., new_targets[k-1], ...uncontracted axes in relative order...]
    # We must permute these axes back to the original order [0, 1, ..., num_qubits - 1]
    uncontracted = [i for i in range(num_qubits) if i not in targets]
    
    # Construct the inverse permutation:
    # inv_perm[u] will hold the current index in res that should go to original position u
    inv_perm = [0] * num_qubits
    for i, t in enumerate(targets):
        inv_perm[t] = i
    for j, u in enumerate(uncontracted):
        inv_perm[u] = k + j
        
    return jnp.transpose(res, inv_perm)

def state_vector_flat(state: jnp.ndarray) -> jnp.ndarray:
    """Flatten the multi-dimensional state-vector into a 1D vector of shape (2**N,)."""
    return state.reshape(-1)
