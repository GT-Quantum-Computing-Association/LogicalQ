import random
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, XGate, YGate, ZGate, SGate, TGate, CXGate, CYGate, CZGate, RXGate, RYGate, RZGate
from qiskit_experiments.library import StandardRB

# Gate sub-populations for benchmarking circuit generation
clifford_gates = [
    HGate, XGate, YGate, ZGate, SGate, CXGate, CYGate, CZGate
]
clifford_sq_gates = [
    HGate, XGate, YGate, ZGate, SGate
]
pauli_gates = [
    XGate, YGate, ZGate
]
sq_gates = [
    HGate, XGate, YGate, ZGate, SGate, TGate, RXGate, RYGate, RZGate
]

# mirror benchmarking uses circuits composed of various Clifford gates and the full inverse, such that the composite operation is the identity if no errors occur
def mirror_benchmarking(n_qubits=None, qubits=None, circuit_length=2, gate_sample=None):
    if n_qubits is None and qubits is None:
        qubits = [0]
        n_qubits = 1
    if n_qubits is not None and qubits is None:
        qubits = range(n_qubits)

    if gate_sample is None:
        gate_sample = clifford_sq_gates if n_qubits == 1 else clifford_gates
    
    mb_circuit = QuantumCircuit(n_qubits)

    # verify that all gates in sample are Clifford gates (necessary condition for MB to work)
    for gate in gate_sample:
        if gate not in clifford_gates:
            raise ValueError(f"Gate {gate.__name__} is not a Clifford gate")
        if gate.num_qubits > n_qubits:
            raise ValueError(f"Gate {gate.__name__} requires more qubits than available")

    # random shuffled sample of gates used for circuit
    gate_sample = np.random.choice(gate_sample, circuit_length//2)
    shuffled_gates = np.random.permutation(gate_sample)

    # append original gates to circuit, targeting random qubits
    for gate in shuffled_gates:
        target_qubits = np.random.choice(qubits, gate.num_qubits)
        mb_circuit.append(gate(), [0])

    # append inverse of current circuit so that final state is left unchanged under no errors
    mb_circuit = mb_circuit.compose(mb_circuit.inverse())

    return mb_circuit

"""
Parameters:
    n_qubits (int): Index of the qubit to benchmark. Defaults to 0.
    circuit_length (int): RB sequence length. Defaults to [2,16,64,128].
    circuit_lengths (list): List of RB sequence lengths. Defaults to [2,16,64,128].
    num_samples (int): Number of random samples to run. Defaults to 10.
    seed (int): Random seed for reproducibility. Defaults to 1234.
"""
def randomized_benchmarking(n_qubits=None, circuit_length=None, circuit_lengths=None, num_samples=10, seed=1234):
    if qubits is None:
        qubits = [0]

    if circuit_length is not None and circuit_lengths is None:
        circuit_lengths = [circuit_length]
    if circuit_length is None and circuit_lengths is None:
        circuit_lengths = [2, 16, 64, 128]

    experiment = StandardRB(qubits=qubits, lengths=circuit_lengths, num_samples=num_samples, seed=seed)
    
    rb_circuits = experiment.circuits()

    if len(circuit_lengths) == 0:
        return rb_circuits[0]
    else:
        return rb_circuits

# @TODO - implement different algorithms for utility benchmarking
def quantum_teleportation():
    raise NotImplementedError

def ghz_state_generation():
    raise NotImplementedError

# @TODO - implement other methods of benchmarking
#       - implement methods as described in https://github.com/CQCL/quantinuum-hardware-specifications/blob/main/notebooks/Loading%20Experimental%20Data.ipynb