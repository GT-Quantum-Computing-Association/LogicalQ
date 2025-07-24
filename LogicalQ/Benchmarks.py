import random
import numpy as np
import warnings
from typing import Optional, Sequence

from qiskit.quantum_info import Operator
from qiskit.quantum_info.random import random_clifford
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit, ClassicalRegister
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
def mirror_benchmarking(n_qubits=None, qubits=None, circuit_length=2, gate_sample=None,
                        measure=False):
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
        if gate().num_qubits > n_qubits:
            raise ValueError(f"Gate {gate.__name__} requires more qubits than available")

    # random shuffled sample of gates used for circuit
    gate_sample = np.random.choice(gate_sample, circuit_length//2)
    shuffled_gates = np.random.permutation(gate_sample)

    # append original gates to circuit, targeting random qubits
    for gate in shuffled_gates:
        target_qubits = np.random.choice(qubits, size=gate().num_qubits, replace=False)
        mb_circuit.append(gate(), list(target_qubits))

    # append inverse of current circuit so that final state is left unchanged under no errors
    mb_circuit = mb_circuit.compose(mb_circuit.inverse())

    if measure: mb_circuit.measure_all()

    return mb_circuit

def randomized_benchmarking(n_qubits=None, qubits=None, circuit_length=2, gate_sample=None, measure=False):
    if qubits is None:
        if n_qubits is None:
            raise ValueError("Specify at least one of n_qubits or qubits.")
        qubits = list(range(n_qubits))
    else:
        n_qubits = max(qubits) + 1

    rb_circuit = QuantumCircuit(n_qubits)
    rng = np.random.default_rng()
    
    if gate_sample is None:
        if n_qubits == 1:
            basis = clifford_sq_gates
            if not basis:
                raise ValueError("No Clifford basis defined for this qubit count.")
            forward = list(rng.choice(basis, size=circuit_length, replace=True))
        else:
            # true n‑qubit Cliffords for >1 qubit
            forward = [random_clifford(n_qubits) for _ in range(circuit_length)]
    else:
        for item in gate_sample:
            # get a name for messages
            name = item.__name__ if isinstance(item, type) else item.__class__.__name__
            # warn if it’s not in your 1‑qubit basis
            if item not in clifford_gates:
                warnings.warn(f"Gate {name} is not a Clifford gate", UserWarning)
            # figure out how many qubits it acts on
            needed = item().num_qubits if isinstance(item, type) else item.num_qubits
            if needed > n_qubits:
                raise ValueError(f"Gate {name} requires more qubits than available")

        forward = [random.choice(gate_sample) for _ in range(circuit_length)]

    for gate in forward:
        # if it's a Gate class, instantiate it
        if isinstance(gate, type):
            inst = gate()
        else:
            # otherwise assume it's already a Clifford object
            inst = gate.to_instruction()
            
        rb_circuit.append(inst, qubits)

    # Compose single net inverse Clifford and append
    matrix = Operator(rb_circuit).data
    inv_matrix = np.linalg.inv(matrix)
    inv_gate = UnitaryGate(inv_matrix, label="U_inv")
    rb_circuit.append(inv_gate, qubits)

    if measure:
        rb_circuit.measure_all()

    return rb_circuit

"""
    Generate quantum volume benchmark circuits.

    Parameters:
        n_qubits (int): Number of qubits available for the benchmark (sets a maximum iteration limit on the circuit width).
        trials: Number of trials to run for each qubit count. Defaults to 1.
        seed (int): Random seed for reproducibility. Defaults to 1234.
        backend: Backend to be used for simulation.
"""
def quantum_volume(n_qubits=1, trials=100, seed=1234, backend=None):
    qv_experiment = QuantumVolume(num_qubits=n_qubits, trials=1, seed=seed, simulation_backend=backend)

    qv_circuits = qv_experiment.circuits()

    return qv_circuits

"""
    Generate a quantum teleportation circuit.

    Parameters:
        statevector: Arbitrary input state
"""
def generate_quantum_teleportation_circuit(statevector, barriers=False):
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
"""
def n_qubit_ghz_generation(n_qubits=3, barriers=False):
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
