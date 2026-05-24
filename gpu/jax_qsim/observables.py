import jax
import jax.numpy as jnp
import jax_qsim.ops as ops
from jax_qsim.core import apply_gate

class PauliString:
    """Represents a tensor product of Pauli operators, e.g., X0 * Y1 * Z3."""
    
    def __init__(self, term: dict[int, str]):
        """
        Args:
            term: Dict mapping qubit index to Pauli operator string ('X', 'Y', or 'Z').
                  e.g., {0: 'X', 1: 'Y'}
        """
        self.term = term

    def apply(self, state: jnp.ndarray) -> jnp.ndarray:
        """Apply the Pauli string to a state vector."""
        out_state = state
        for qubit, op_char in sorted(self.term.items()):
            if op_char == 'X':
                out_state = apply_gate(out_state, ops.X, [qubit])
            elif op_char == 'Y':
                out_state = apply_gate(out_state, ops.Y, [qubit])
            elif op_char == 'Z':
                out_state = apply_gate(out_state, ops.Z, [qubit])
            elif op_char == 'I':
                pass
            else:
                raise ValueError(f"Unknown Pauli operator: {op_char}")
        return out_state

    def __repr__(self) -> str:
        ops_str = " * ".join(f"{op}{i}" for i, op in sorted(self.term.items()))
        return f"PauliString({ops_str if ops_str else 'Identity'})"


def expectation(state: jnp.ndarray, observable: PauliString) -> float:
    """Compute the expectation value <psi| O |psi> of a PauliString observable.
    
    Args:
        state: State-vector of shape (2, ..., 2)
        observable: A PauliString instance.
        
    Returns:
        Expectation value as a float.
    """
    # Apply operator to state: O|psi>
    o_state = observable.apply(state)
    # Complex inner product: <psi| O|psi>
    val = jnp.vdot(state.reshape(-1), o_state.reshape(-1))
    return jnp.real(val)


class Hamiltonian:
    """Represents a sum of Pauli strings with coefficients, e.g. H = sum_i c_i P_i."""
    
    def __init__(self, coeffs: jnp.ndarray, paulis: list[PauliString]):
        """
        Args:
            coeffs: Array of coefficients.
            paulis: List of PauliString instances.
        """
        self.coeffs = jnp.array(coeffs, dtype=jnp.float32)
        self.paulis = paulis

    def expectation(self, state: jnp.ndarray) -> float:
        """Compute the expectation value of the Hamiltonian <psi| H |psi>."""
        vals = jnp.stack([expectation(state, p) for p in self.paulis])
        return jnp.dot(self.coeffs, vals)

    def __repr__(self) -> str:
        terms = [f"{c:.4f} * {p}" for c, p in zip(self.coeffs, self.paulis)]
        return " + ".join(terms)


def sample(state: jnp.ndarray, num_samples: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    """Sample computational basis bitstrings from the state vector.
    
    Args:
        state: State-vector of shape (2, ..., 2)
        num_samples: Number of samples to draw.
        key: JAX PRNGKey.
        
    Returns:
        Array of shape (num_samples, num_qubits) containing 0s and 1s.
    """
    num_qubits = state.ndim
    # Compute probabilities: p(x) = |psi(x)|^2
    probs = jnp.abs(state.reshape(-1)) ** 2
    
    # Draw indices from categorical distribution
    log_probs = jnp.log(jnp.maximum(probs, 1e-12))
    indices = jax.random.categorical(key, log_probs, shape=(num_samples,))
    
    # Convert flat indices back to bitstrings
    powers = 2 ** jnp.arange(num_qubits - 1, -1, -1)
    bitstrings = (indices[:, None] // powers) % 2
    return bitstrings
