import random
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, XGate, YGate, ZGate, SGate, TGate, CXGate, CYGate, CZGate, RXGate, RYGate, RZGate
from qiskit_experiments.library import StandardRB, QuantumVolume

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

"""
    Constructs circuits composed of various Clifford gates and the full inverse, such that the composite operation is the identity if no errors occur.
"""
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
    Constructs circuits composed of various Clifford gates, such that the composite operation is the identity if no errors occur.

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

    if circuit_length is None:
        if circuit_lengths is None:
            circuit_lengths = [2, 16, 64, 128]
    else:
        circuit_lengths = [circuit_length]

    experiment = StandardRB(qubits=qubits, lengths=circuit_lengths, num_samples=num_samples, seed=seed)

    rb_circuits = experiment.circuits()

    if len(circuit_lengths) == 0:
        return rb_circuits[0]
    else:
        return rb_circuits

"""
    Generate quantum volume benchmark circuits.

    Parameters:
        n_qubits (int): Number of qubits available for the benchmark (sets a maximum iteration limit on the circuit width).
        circuit_length: Not used.
        trials: Number of trials to run for each qubit count.
        seed (int): Random seed for reproducibility.
        backend: Backend to be used for simulation.
"""
def quantum_volume(n_qubits=1, circuit_length=None, trials=100, seed=1234, backend=None):
    qv_experiment = QuantumVolume(num_qubits=n_qubits, trials=trials, seed=seed, simulation_backend=backend)

    qv_circuits = qv_experiment.circuits()

    return qv_circuits

"""

    Generate a quantum teleportation circuit.

    Parameters:
        statevector: Arbitrary input state
        n_qubits: Not used.
        circuit_length: Not used.

"""
def generate_quantum_teleportation_circuit(statevector, n_qubits=None, circuit_length=None, barriers=False):
    creg0 = ClassicalRegister(1, 'cr0')
    creg1 = ClassicalRegister(1, 'cr1')
    creg2 = ClassicalRegister(1, 'cr2')

    qc = QuantumCircuit(3, creg0, creg1, creg2)

    # Initialize state to be teleported
    qc.initialize(statevector, 0)
    if barriers: qc.barrier()

    # Entangled pair between qubits 1 and 2
    qc.h(1)
    qc.cx(1, 2)

    # Bell measurement on qubits 0 and 1
    qc.cx(0, 1)
    qc.h(0)

    qc.measure(0, creg0)
    qc.measure(1, creg1)

    if barriers: qc.barrier()

    # Apply corrections on qubit 2 based on measurement outcomes
    #   - 00 -> I
    #   - 01 -> X
    #   - 10 -> Z
    #   - 11 -> ZX
    qc.x(2).c_if(creg1, 1)
    qc.z(2).c_if(creg0, 1)

    # Measure
    qc.measure(2, creg2)

    return qc

"""
    Generate an n-qubit GHZ state generation circuit.
    Parameters:
        n_qubits (int): Number of qubits in the GHZ state.
        circuit_length: Not used.
"""
def n_qubit_ghz_generation(n_qubits=3, circuit_length=None, barriers=False):
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Apply Hadamard to qubit 0 to generate superposition
    qc.h(0)

    # Apply chain of CNOTS after qubit 0 to spread superposition through entanglement
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    if barriers: qc.barrier()

    # Measure
    qc.measure(range(n_qubits), range(n_qubits))

    return qc

# @TODO - implement other methods of benchmarking
#       - implement methods as described in https://github.com/CQCL/quantinuum-hardware-specifications/blob/main/notebooks/Loading%20Experimental%20Data.ipynb
