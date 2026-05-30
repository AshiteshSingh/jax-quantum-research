"""
High-level compiled Circuit builder for the JAX Quantum Circuit Simulation Suite.
"""

import functools
import jax
import jax.numpy as jnp
from . import gates
from . import statevector as sv
from . import density_matrix as dm

@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def _run_circuit_functional(params, initial_state, num_qubits, ops, state_type):
    """
    Pure functional evaluator for quantum circuits.
    
    Since the structure of the circuit (ops) is marked as a static argument,
    JAX will fully compile and unroll the loop, producing a highly optimized 
    XLA graph of pure tensor contractions.
    
    Args:
        params: JAX array of gate parameters.
        initial_state: Optional initial JAX state array. If None, initialized to zero state.
        num_qubits: Static integer number of qubits.
        ops: Static tuple of operations: ((op_name, qubits_tuple, param_idx_or_noise_val), ...)
        state_type: Static string 'statevector' or 'density_matrix'.
    """
    if initial_state is not None:
        state = initial_state
    elif state_type == 'statevector':
        state = sv.zero_state(num_qubits)
    elif state_type == 'density_matrix':
        state = dm.zero_state(num_qubits)
    else:
        raise ValueError("state_type must be 'statevector' or 'density_matrix'")
        
    for op_name, qubits, p_val in ops:
        # Resolve gate unitary / noise channels
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
            # p_val is noise probability
            if state_type == 'density_matrix':
                kraus = dm.depolarizing_kraus(p_val)
                state = dm.apply_channel_1q(state, kraus, qubits[0])
            continue
        elif op_name == 'noise_amp_damp':
            # p_val is damping rate
            if state_type == 'density_matrix':
                kraus = dm.amplitude_damping_kraus(p_val)
                state = dm.apply_channel_1q(state, kraus, qubits[0])
            continue
        elif op_name == 'noise_phase_damp':
            # p_val is damping rate
            if state_type == 'density_matrix':
                kraus = dm.phase_damping_kraus(p_val)
                state = dm.apply_channel_1q(state, kraus, qubits[0])
            continue
        else:
            raise ValueError(f"Unknown operation: {op_name}")
            
        # Apply gate
        if state_type == 'statevector':
            state = sv.apply_gate(state, u, qubits)
        else:
            state = dm.apply_gate(state, u, qubits)
            
    return state

class Circuit:
    """
    A quantum circuit builder that compiles to optimized, differentiable JAX code.
    """
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.ops = []
        self.num_params = 0
        
    # ==============================================================================
    # Single-qubit Gates
    # ==============================================================================
    def h(self, q):
        self.ops.append(('h', (q,), None))
        return self
        
    def x(self, q):
        self.ops.append(('x', (q,), None))
        return self
        
    def y(self, q):
        self.ops.append(('y', (q,), None))
        return self
        
    def z(self, q):
        self.ops.append(('z', (q,), None))
        return self
        
    def s(self, q):
        self.ops.append(('s', (q,), None))
        return self
        
    def t(self, q):
        self.ops.append(('t', (q,), None))
        return self
        
    def rx(self, q, param_index):
        self.ops.append(('rx', (q,), param_index))
        self.num_params = max(self.num_params, param_index + 1)
        return self
        
    def ry(self, q, param_index):
        self.ops.append(('ry', (q,), param_index))
        self.num_params = max(self.num_params, param_index + 1)
        return self
        
    def rz(self, q, param_index):
        self.ops.append(('rz', (q,), param_index))
        self.num_params = max(self.num_params, param_index + 1)
        return self
        
    def phase_shift(self, q, param_index):
        self.ops.append(('phase_shift', (q,), param_index))
        self.num_params = max(self.num_params, param_index + 1)
        return self
        
    # ==============================================================================
    # Two-qubit Gates
    # ==============================================================================
    def cnot(self, c, t):
        self.ops.append(('cnot', (c, t), None))
        return self
        
    def cz(self, c, t):
        self.ops.append(('cz', (c, t), None))
        return self
        
    def swap(self, q1, q2):
        self.ops.append(('swap', (q1, q2), None))
        return self
        
    # ==============================================================================
    # Three-qubit Gates
    # ==============================================================================
    def toffoli(self, c1, c2, t):
        self.ops.append(('toffoli', (c1, c2, t), None))
        return self
        
    # ==============================================================================
    # Controlled Parametric Gates
    # ==============================================================================
    def crx(self, c, t, param_index):
        self.ops.append(('crx', (c, t), param_index))
        self.num_params = max(self.num_params, param_index + 1)
        return self
        
    def cry(self, c, t, param_index):
        self.ops.append(('cry', (c, t), param_index))
        self.num_params = max(self.num_params, param_index + 1)
        return self
        
    def crz(self, c, t, param_index):
        self.ops.append(('crz', (c, t), param_index))
        self.num_params = max(self.num_params, param_index + 1)
        return self
        
    def cp(self, c, t, param_index):
        self.ops.append(('cp', (c, t), param_index))
        self.num_params = max(self.num_params, param_index + 1)
        return self
        
    # ==============================================================================
    # Noise Channels (Applicable only in Density Matrix mode)
    # ==============================================================================
    def noise_depolarizing(self, q, p):
        self.ops.append(('noise_depol', (q,), p))
        return self
        
    def noise_amplitude_damping(self, q, gamma):
        self.ops.append(('noise_amp_damp', (q,), gamma))
        return self
        
    def noise_phase_damping(self, q, gamma):
        self.ops.append(('noise_phase_damp', (q,), gamma))
        return self
        
    # ==============================================================================
    # Execution
    # ==============================================================================
    def run(self, params, state_type='statevector', initial_state=None):
        """
        Executes the circuit on the specified JAX backend using the provided parameters.
        
        Args:
            params: JAX array containing values for all parametric gates in the circuit.
            state_type: String 'statevector' or 'density_matrix'.
            initial_state: Optional initial JAX state array. If None, initialized to zero state.
            
        Returns:
            The final simulated state as a JAX array.
        """
        # Convert operations list to a hashable tuple of tuples for JAX JIT static arg compatibility
        ops_tuple = tuple(self.ops)
        return _run_circuit_functional(params, initial_state, self.num_qubits, ops_tuple, state_type)
