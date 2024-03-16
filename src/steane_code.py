from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

def encode_gate():
    encoding_circuit = QuantumCircuit(7)
    encoding_circuit.h(0)
    encoding_circuit.h(4)
    encoding_circuit.h(6)
    encoding_circuit.cx(0, 1)
    encoding_circuit.cx(4, 5)
    encoding_circuit.cx(6, 3)
    encoding_circuit.cx(4, 2)
    encoding_circuit.cx(6, 5)
    encoding_circuit.cx(0, 3)
    encoding_circuit.cx(4, 1)
    encoding_circuit.cx(3, 2)

    return encoding_circuit.to_gate(label="$U_\\text{enc}$")
