import jax
import jax.numpy as jnp
from jax_qsim.core import zero_state, apply_gate
import jax_qsim.ops as ops

class Circuit:
    """An elegant quantum circuit builder designed for JAX.
    
    Accumulates gates and compiles circuit execution into a pure JAX function
    that is fully compatible with jax.jit, jax.grad, jax.vmap, etc.
    """
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.gates = []
        self.num_params = 0

    def _register_param(self, param_index: int):
        """Track the number of parameters needed for the circuit."""
        if param_index is not None:
            self.num_params = max(self.num_params, param_index + 1)

    # --- 1-Qubit Static Gates ---

    def h(self, qubit: int):
        """Hadamard gate on target qubit."""
        self.gates.append({'type': 'static', 'matrix': ops.H, 'targets': [qubit]})
        return self

    def x(self, qubit: int):
        """Pauli-X (NOT) gate on target qubit."""
        self.gates.append({'type': 'static', 'matrix': ops.X, 'targets': [qubit]})
        return self

    def y(self, qubit: int):
        """Pauli-Y gate on target qubit."""
        self.gates.append({'type': 'static', 'matrix': ops.Y, 'targets': [qubit]})
        return self

    def z(self, qubit: int):
        """Pauli-Z gate on target qubit."""
        self.gates.append({'type': 'static', 'matrix': ops.Z, 'targets': [qubit]})
        return self

    def s(self, qubit: int):
        """S (phase) gate on target qubit."""
        self.gates.append({'type': 'static', 'matrix': ops.S, 'targets': [qubit]})
        return self

    def t(self, qubit: int):
        """T gate on target qubit."""
        self.gates.append({'type': 'static', 'matrix': ops.T, 'targets': [qubit]})
        return self

    # --- 2-Qubit Static Gates ---

    def cnot(self, control: int, target: int):
        """CNOT (CX) gate with control and target qubits."""
        self.gates.append({'type': 'static', 'matrix': ops.CNOT, 'targets': [control, target]})
        return self

    def cz(self, control: int, target: int):
        """CZ gate with control and target qubits."""
        self.gates.append({'type': 'static', 'matrix': ops.CZ, 'targets': [control, target]})
        return self

    def swap(self, qubit1: int, qubit2: int):
        """SWAP gate between two qubits."""
        self.gates.append({'type': 'static', 'matrix': ops.SWAP, 'targets': [qubit1, qubit2]})
        return self

    # --- 1-Qubit Parameterized Gates ---

    def rx(self, qubit: int, param_index: int):
        """Rotation around X axis by parameter at param_index."""
        self._register_param(param_index)
        self.gates.append({
            'type': 'parameterized',
            'gate_func': ops.rx,
            'targets': [qubit],
            'param_index': param_index
        })
        return self

    def ry(self, qubit: int, param_index: int):
        """Rotation around Y axis by parameter at param_index."""
        self._register_param(param_index)
        self.gates.append({
            'type': 'parameterized',
            'gate_func': ops.ry,
            'targets': [qubit],
            'param_index': param_index
        })
        return self

    def rz(self, qubit: int, param_index: int):
        """Rotation around Z axis by parameter at param_index."""
        self._register_param(param_index)
        self.gates.append({
            'type': 'parameterized',
            'gate_func': ops.rz,
            'targets': [qubit],
            'param_index': param_index
        })
        return self

    def phase_shift(self, qubit: int, param_index: int):
        """Phase shift by parameter at param_index."""
        self._register_param(param_index)
        self.gates.append({
            'type': 'parameterized',
            'gate_func': ops.phase_shift,
            'targets': [qubit],
            'param_index': param_index
        })
        return self

    # --- 2-Qubit Parameterized Gates ---

    def crx(self, control: int, target: int, param_index: int):
        """Controlled rotation around X axis by parameter at param_index."""
        self._register_param(param_index)
        self.gates.append({
            'type': 'parameterized',
            'gate_func': ops.crx,
            'targets': [control, target],
            'param_index': param_index
        })
        return self

    def cry(self, control: int, target: int, param_index: int):
        """Controlled rotation around Y axis by parameter at param_index."""
        self._register_param(param_index)
        self.gates.append({
            'type': 'parameterized',
            'gate_func': ops.cry,
            'targets': [control, target],
            'param_index': param_index
        })
        return self

    def crz(self, control: int, target: int, param_index: int):
        """Controlled rotation around Z axis by parameter at param_index."""
        self._register_param(param_index)
        self.gates.append({
            'type': 'parameterized',
            'gate_func': ops.crz,
            'targets': [control, target],
            'param_index': param_index
        })
        return self

    def cphase(self, control: int, target: int, param_index: int):
        """Controlled phase shift by parameter at param_index."""
        self._register_param(param_index)
        self.gates.append({
            'type': 'parameterized',
            'gate_func': ops.cphase,
            'targets': [control, target],
            'param_index': param_index
        })
        return self

    # --- Execution and Compilation ---

    def run(self, params: jnp.ndarray, initial_state: jnp.ndarray = None) -> jnp.ndarray:
        """Execute the circuit on an initial state or zero-state.
        
        This is a pure JAX function, fully JIT-compilable and differentiable.
        
        Args:
            params: Flat JAX array containing parameter values.
            initial_state: Optional JAX array of shape (2, ..., 2).
                           Defaults to the zero-state.
                           
        Returns:
            The final state-vector of shape (2, ..., 2).
        """
        if initial_state is None:
            state = zero_state(self.num_qubits)
        else:
            state = initial_state
            
        for gate in self.gates:
            if gate['type'] == 'static':
                state = apply_gate(state, gate['matrix'], gate['targets'])
            elif gate['type'] == 'parameterized':
                val = params[gate['param_index']]
                matrix = gate['gate_func'](val)
                state = apply_gate(state, matrix, gate['targets'])
                
        return state

    def compile(self):
        """Return a JIT-compiled version of the circuit's run function."""
        return jax.jit(self.run)

    def __repr__(self) -> str:
        return f"Circuit(num_qubits={self.num_qubits}, gates={len(self.gates)}, params={self.num_params})"
