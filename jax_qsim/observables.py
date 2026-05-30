"""
Quantum observables (Pauli strings and Hamiltonians) in JAX.
"""

import jax.numpy as jnp
from . import gates

class PauliString:
    """
    Represents a product of Pauli operators acting on specific qubits.
    Example: PauliString({0: 'X', 2: 'Y', 3: 'Z'}) represents X_0 * Y_2 * Z_3.
    """
    def __init__(self, paulis=None):
        # Maps qubit index -> 'X', 'Y', or 'Z'
        self.paulis = paulis if paulis is not None else {}
        
    def __repr__(self):
        if not self.paulis:
            return "Identity"
        terms = [f"{op}_{q}" for q, op in sorted(self.paulis.items())]
        return " * ".join(terms)

class Hamiltonian:
    """
    Represents a linear combination of Pauli strings.
    Example: H = 0.5 * X_0 + 0.8 * Z_1
    """
    def __init__(self, coeffs, pauli_strings):
        self.coeffs = jnp.array(coeffs, dtype=jnp.float32)
        self.pauli_strings = pauli_strings
        
    def __repr__(self):
        terms = []
        for c, p in zip(self.coeffs, self.pauli_strings):
            terms.append(f"{c:+.4f} * ({p})")
        return " + ".join(terms)
