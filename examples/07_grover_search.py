"""
Example 07 — Grover's Search Algorithm
========================================
Demonstrates Grover's amplitude amplification on a small number of qubits.
For n qubits searching for a target bitstring, Grover's algorithm finds the
target in O(sqrt(2^n)) oracle calls vs O(2^n) classically.

This implementation runs cleanly on CPU with jax_qsim for up to ~16 qubits.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
from jax_qsim.statevector import Statevector
import math


def grover_oracle(state: jnp.ndarray, target: int, n: int) -> jnp.ndarray:
    """Phase oracle: flips sign of |target> amplitude.

    Args:
        state: (2,)*n statevector tensor.
        target: Target basis state index.
        n: Number of qubits.

    Returns:
        State with |target> amplitude negated.
    """
    flat = state.reshape(-1)
    flat = flat.at[target].multiply(-1.0 + 0j)
    return flat.reshape(state.shape)


def grover_diffuser(state: jnp.ndarray) -> jnp.ndarray:
    """Grover diffusion operator: 2|s><s| - I, where |s> is uniform superposition.

    Args:
        state: (2,)*n statevector tensor.

    Returns:
        State after diffusion operator.
    """
    flat = state.reshape(-1)
    mean_amp = jnp.mean(flat)
    flat = 2 * mean_amp - flat  # Inversion about mean
    return flat.reshape(state.shape)


def run_grover(n: int, target: int) -> dict:
    """Run Grover's algorithm for n qubits searching for target.

    Args:
        n: Number of qubits (search space size = 2^n).
        target: Target basis state index (0 <= target < 2^n).

    Returns:
        Dict with success probability and number of iterations used.
    """
    N = 2 ** n
    n_iterations = max(1, int(math.pi / 4 * math.sqrt(N)))

    # Initialize uniform superposition
    sv = Statevector(n)
    state = sv.zero_state()
    # Apply H to all qubits
    H = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) / jnp.sqrt(2)
    for q in range(n):
        state = sv.apply_gate(state, H, [q])

    # Grover iterations
    for _ in range(n_iterations):
        state = grover_oracle(state, target, n)
        state = grover_diffuser(state)

    flat = state.reshape(-1)
    probs = jnp.abs(flat) ** 2
    success_prob = float(probs[target])

    return {
        "n_qubits": n,
        "N": N,
        "target": target,
        "n_iterations": n_iterations,
        "success_probability": success_prob,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  Grover's Search Algorithm — jax_qsim Implementation")
    print("=" * 60)

    test_cases = [
        (3,  5),
        (4,  11),
        (5,  23),
        (6,  41),
        (8,  127),
        (10, 512),
    ]

    for n, target in test_cases:
        result = run_grover(n, target)
        print(f"  n={n:2d}  N={result['N']:6d}  target={target:4d}  "
              f"iters={result['n_iterations']:3d}  "
              f"P(success)={result['success_probability']:.4f}")

    print()
    print("  Classical brute-force: O(N) calls on average")
    print("  Grover: O(sqrt(N)) oracle calls — quadratic quantum speedup")
