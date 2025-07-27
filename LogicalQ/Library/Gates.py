import numpy as np

from qiskit.circuit.library import UGate as IBMUGate
from qiskit.circuit.library import HGate, XGate, YGate, ZGate, SGate, TGate, CXGate, CYGate, CZGate, RXGate, RYGate, RZGate, RXXGate, RYYGate, RZZGate

# Gates

class UGate(IBMUGate):
    def __init__(self, theta, phi, label=None):
        """Create new U gate dependent on theta and phi, following Quantinuum standard."""
        super().__init__(theta=theta, phi=phi-np.pi/2, lam=np.pi/2-phi, label=label)

    def __eq__(self, other):
        if isinstance(other, UGate):
            return self._compare_parameters(other)
        return False

class ZZGate(RZZGate):
    _standard_gate = None

    def __init__(self, label=None):
        """Create new ZZ gate."""
        super().__init__(theta=np.pi/2, label=label)
        self.name = "zz"

        assert (len(self.params) == 1) and (np.isclose(self.params[0], np.pi/2))

    def __eq__(self, other):
        if isinstance(other, ZZGate):
            return self._compare_parameters(other)
        return False

# Mapping from gate names to classes
# NOTE - Use the robust string_to_gate_class function in code
gates = {
    "h": UGate,
    "h": HGate,
    "x": XGate,
    "y": YGate,
    "z": ZGate,
    "s": SGate,
    "t": TGate,
    "cx": CXGate,
    "cy": CYGate,
    "cz": CZGate,
    "rx": RXGate,
    "ry": RYGate,
    "rz": RZGate,
    "rxx": RXXGate,
    "ryy": RYYGate,
    "rzz": RZZGate,
}

def string_to_gate_class(_gate_string):
    gate_string = _gate_string.lower()
    if gate_string in gates.keys():
        return gates[gate_string]
    else:
        raise KeyError(f"Input {_gate_string} is not a valid gate name.")

# Gate sets

clifford_gates = {
    "strings": ["h", "x", "y", "z", "s", "cx", "cy", "cz"],
    "classes": [HGate, XGate, YGate, ZGate, SGate, CXGate, CYGate, CZGate]
}
clifford_gates_1q = {
    "strings": ["h", "x", "y", "z", "s"],
    "classes": [HGate, XGate, YGate, ZGate, SGate]
}
pauli_gates = {
    "strings": ["x", "y", "z"],
    "classes": [XGate, YGate, ZGate]
}
gates_1q = {
    "strings": ["u", "h", "x", "y", "z", "s", "rx", "ry", "rz"],
    "classes": [UGate, HGate, XGate, YGate, ZGate, SGate, TGate, RXGate, RYGate, RZGate]
}
gates_2q = {
    "strings": ["cx", "cy", "cz", "rxx", "ryy", "rzz", "zz"],
    "classes": [CXGate, CYGate, CZGate, RXXGate, RYYGate, RZZGate, ZZGate]
}

# Mapping from gate set names to objects
# NOTE - Use the robust string_to_gate_set function in code
gate_sets = {
    "clifford_gates": clifford_gates,
    "clifford_gates_1q": clifford_gates_1q,
    "pauli_gates": pauli_gates,
    "gates_1q": gates_1q,
    "gates_2q": gates_2q,
}

def string_to_gate_set(_gate_set_string):
    gate_set_string = _gate_set_string.lower()
    if gate_set_string in gate_sets.keys():
        return gate_sets[gate_set_string]
    else:
        raise KeyError(f"Input {_gate_set_string} is not a valid gate set name.")

