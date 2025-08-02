import itertools

from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit import Parameter

from qiskit.circuit.library import Measure, Reset, RZGate, RZZGate
from qiskit.circuit.controlflow import IfElseOp
from .Gates import UGate, ZZGate

"""
    Constructs a transpiler Target representing a generic ion trap, with physical parameters similar to state-of-the-art devices.
"""
def construct_target_generic_iontrap(n_qubits=32):
    # Prepare data
    qubit_list = range(n_qubits)

    gates = {
        "measure": {
            "n_qubits": 1,
            "instruction": Measure,
            "parameters": [],
            "duration": 25E-6,
            "error": 1E-4,
        },
        "reset": {
            "n_qubits": 1,
            "instruction": Reset,
            "parameters": [],
            "duration": 25E-6,
            "error": 1E-4,
        },
        "if_else": {
            "n_qubits": -1,
            "instruction": IfElseOp,
            "parameters": ["condition", "true_body", "false_body"],
            "duration": 0.00,
            "error": 0.00,
        },
        "u": {
            "n_qubits": 1,
            "instruction": UGate,
            "parameters": ["theta", "phi"],
            "duration": 5E-6,
            "error": 1E-5,
        },
        "rz": {
            "n_qubits": 1,
            "instruction": RZGate,
            "parameters": ["phi"],
            "duration": 5E-6,
            "error": 1E-5,
        },
        "rzz": {
            "n_qubits": 2,
            "instruction": RZZGate,
            "parameters": ["theta"],
            "duration": 50E-6,
            "error": 1E-4,
        },
        "zz": {
            "n_qubits": 2,
            "instruction": ZZGate,
            "parameters": [],
            "duration": 50E-6,
            "error": 1E-4,
        },
        # @TODO - implement Rxxyyzz gate
    }

    # Construct Target
    target = Target()

    for gate_name, gate_data in gates.items():
        parameters = []
        for parameter_name in gate_data["parameters"]:
            parameters.append(Parameter(parameter_name))

        if gate_data["n_qubits"] == -1:
            target.add_instruction(gate_data["instruction"]())
        else:
            qubit_target_list = list(itertools.product(*([qubit_list]*gate_data["n_qubits"])))

            gate_properties = {}
            for qubit_targets in qubit_target_list:
                gate_properties[qubit_targets] = InstructionProperties(duration=gate_data["error"], error=gate_data["error"])

            target.add_instruction(gate_data["instruction"](*parameters), gate_properties)

    return target

