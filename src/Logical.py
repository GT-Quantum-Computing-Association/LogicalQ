from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Pauli
class LogicalQubit:
    # @TODO - Add a unique identifier of some sorts for multi-logical qubit circuits
    qreg = QuantumRegister(7, "logical_quantum")
    creg = ClassicalRegister(3, "logical_classical")
    qcirc = QuantumCircuit(qreg)

    def __init__(self, stabilizers: list, initial_state=0):
        N_stab = len(stabilizers)
        N_q = len(stabilizers[0])
        self.stabilizer_list = stabilizers
        self.initial_state = initial_state
        self.qreg = QuantumRegister(N_q, "logical_quantum")
        self.creg = ClassicalRegister(3, "logical_classical")
        self.qcirc = QuantumCircuit(self.qreg)

    # @TODO - allow for QECC selection
    def encode(self, initial_state=0, draw=False):

        # qcirc = QuantumCircuit(self.qreg)
        # qcirc.h(self.qreg[0])
        # qcirc.h(self.qreg[1])
        # qcirc.h(self.qreg[3])
        # qcirc.cx(self.qreg[0], self.qreg[2])
        # qcirc.cx(self.qreg[3], self.qreg[5])
        # qcirc.cx(self.qreg[1], self.qreg[6])
        # qcirc.cx(self.qreg[0], self.qreg[4])
        # qcirc.cx(self.qreg[3], self.qreg[6])
        # qcirc.cx(self.qreg[1], self.qreg[5])
        # qcirc.cx(self.qreg[0], self.qreg[6])
        # qcirc.cx(self.qreg[1], self.qreg[2])
        # qcirc.cx(self.qreg[3], self.qreg[4])

        if initial_state == 1:
            qcirc.x(self.qreg)

        if draw:
            qcirc.draw('mpl')

        return

    def measure_syndrome():
        # @TODO - Measure syndromes
        pass

class LogicalRegister:
    qubits = None
    bits = None

    def __init__(self, n_qubits, n_bits):
        self.n_qubits = n_qubits
        self.n_bits = n_bits

        self.qubits = []
        for q in range(self.n_qubits):
            self.qubits.append(LogicalQubit())

        self.bits = []
        for b in range(self.n_bits):
            # @TODO - Handle classical bits
            continue

class LogicalCircuit:
    qcirc = None

    def __init__(self, logical_register):
        # @TODO - Integrate logical registers into the quantum circuit

        self.logical_register = logical_register
        self.qcirc = QuantumCircuit(self.logical_register)

    def x(logical_qubit_idx):
        # each logical qubit consists of 7 physical qubits
        physical_qubit_idx = logical_qubit_idx * 7
