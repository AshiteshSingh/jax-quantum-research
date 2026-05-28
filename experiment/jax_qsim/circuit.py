"""
High-level compiled Circuit builder for the JAX Quantum Circuit Simulation Suite.
"""

import functools
import jax
import jax.numpy as jnp
from . import gates
from . import statevector as sv
from . import density_matrix as dm

@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def _run_circuit_functional(params, num_qubits, ops, state_type):
    """
    Pure functional evaluator for quantum circuits.
    """
    if state_type == 'statevector':
        state = sv.zero_state(num_qubits)
    elif state_type == 'density_matrix':
        state = dm.zero_state(num_qubits)
    else:
        raise ValueError("state_type must be 'statevector' or 'density_matrix'")
        
    for op_name, qubits, p_val in ops:
        if op_name == 'h':
            u = gates.H()
        elif op_name == 'x':
            u = gates.X()
        elif op_name == 'y':
            u = gates.Y()
        elif op_name == 'z':
            u = gates.Z()
        elif op_name == 's':
            u = gates.S()
        elif op_name == 't':
            u = gates.T()
        elif op_name == 'rx':
            u = gates.RX(params[p_val])
        elif op_name == 'ry':
            u = gates.RY(params[p_val])
        elif op_name == 'rz':
            u = gates.RZ(params[p_val])
        elif op_name == 'phase_shift':
            u = gates.PhaseShift(params[p_val])
        elif op_name == 'cnot':
            u = gates.CNOT()
        elif op_name == 'cz':
            u = gates.CZ()
        elif op_name == 'swap':
            u = gates.SWAP()
        elif op_name == 'toffoli':
            u = gates.Toffoli()
        elif op_name == 'crx':
            u = gates.CRX(params[p_val])
        elif op_name == 'cry':
            u = gates.CRY(params[p_val])
        elif op_name == 'crz':
            u = gates.CRZ(params[p_val])
        elif op_name == 'cp':
            u = gates.CP(params[p_val])
        elif op_name == 'noise_depol':
            if state_type == 'density_matrix':
                kraus = dm.depolarizing_kraus(p_val)
                state = dm.apply_channel_1q(state, kraus, qubits[0])
            continue
        elif op_name == 'noise_amp_damp':
            if state_type == 'density_matrix':
                kraus = dm.amplitude_damping_kraus(p_val)
                state = dm.apply_channel_1q(state, kraus, qubits[0])
            continue
        elif op_name == 'noise_phase_damp':
            if state_type == 'density_matrix':
                kraus = dm.phase_damping_kraus(p_val)
                state = dm.apply_channel_1q(state, kraus, qubits[0])
            continue
        else:
            raise ValueError(f"Unknown operation: {op_name}")
            
        if state_type == 'statevector':
            state = sv.apply_gate(state, u, qubits)
        else:
            state = dm.apply_gate(state, u, qubits)
            
    return state
