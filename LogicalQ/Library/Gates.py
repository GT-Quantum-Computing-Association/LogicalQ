import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import HGate, XGate, YGate, ZGate, SGate, TGate, CXGate, CYGate, CZGate, RGate, RXGate, RYGate, RZGate, RXXGate, RYYGate, RZZGate
from qiskit.circuit.library.standard_gates.equivalence_library import StandardEquivalenceLibrary

# Gates

UGate = RGate

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

# Add custom gates to the Standard Equivalence Library (SEL)

# ZZGate
zzgate = ZZGate()
ZZGate_def = QuantumCircuit(2)
ZZGate_def.append(zzgate, range(2))
StandardEquivalenceLibrary.add_equivalence(zzgate, ZZGate_def)

# Mapping from gate names to classes
# NOTE - Use the robust string_to_gate_class function in code
gates = {
    "r": RGate,
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
    "classes": [HGate, XGate, YGate, ZGate, SGate, CXGate, CYGate, CZGate],
    "num_params": [0, 0, 0, 0, 0, 0, 0, 0],
}
clifford_gates_1q = {
    "strings": ["h", "x", "y", "z", "s"],
    "classes": [HGate, XGate, YGate, ZGate, SGate],
    "num_params": [0, 0, 0, 0, 0],
}
pauli_gates = {
    "strings": ["x", "y", "z"],
    "classes": [XGate, YGate, ZGate],
    "num_params": [0, 0, 0],
}
gates_1q = {
    "strings": ["r", "h", "x", "y", "z", "s", "rx", "ry", "rz"],
    "classes": [RGate, HGate, XGate, YGate, ZGate, SGate, TGate, RXGate, RYGate, RZGate],
    "num_params": [1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
}

gates_2q = {
    "strings": ["cx", "cy", "cz", "rxx", "ryy", "rzz", "zz"],
    "classes": [CXGate, CYGate, CZGate, RXXGate, RYYGate, RZZGate, ZZGate],
    "num_params": [0, 0, 0, 1, 1, 1, 0]
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

def get_num_params(gate_class):
    if gate_class in [RXGate, RYGate, RZGate, RXXGate, RYYGate, RZZGate]:
        return 1
    elif gate_class in [RGate]:
        return 2
    else:
        return 0

