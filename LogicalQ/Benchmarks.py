import numpy as np

from LogicalQ.Library.Gates import clifford_gates, clifford_gates_1q, string_to_gate_class, string_to_gate_set

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit_experiments.library import QuantumVolume

def mirror_benchmarking(n_qubits=None, qubits=None, circuit_length=2, gate_sample=None, measure=False, seed=None):
    """
    Constructs circuits composed of various Clifford gates and the full inverse, such that the composite operation is the identity if no errors occur.
    """

    if seed is not None:
        np.random.seed(seed)

    if n_qubits is not None:
        qubits = list(range(n_qubits))

    if qubits is None:
        if n_qubits is None:
            qubits = [0]
            n_qubits = 1
        else:
            qubits = list(range(n_qubits))
    elif not hasattr(qubits, "__iter__"):
        raise TypeError(f"Invalid type for qubits input; must either be None (if n_qubits is specified) or iterable, not {type(qubits)}.")
    else:
        n_qubits = len(list(qubits))

    if gate_sample is None:
        gate_sample = clifford_gates_1q["classes"] if n_qubits == 1 else clifford_gates["classes"]
    else:
        if hasattr(gate_sample, "__iter__"):
            for g, gate in enumerate(gate_sample):
                if isinstance(gate, str):
                    gate = string_to_gate_class(gate)
                elif not issubclass(gate, Gate):
                    raise TypeError(f"Invalid type for gate input at index {g}: {gate}; must be subclass Gate")

                if gate().num_qubits > n_qubits:
                    raise ValueError(f"Gate {gate.__name__} requires more qubits than are available")
        elif isinstance(gate_sample, str):
            gate_sample = string_to_gate_set(gate_sample)["classes"]
        else:
            raise TypeError(f"Invalid type for gate_sample input: {type(gate_sample)}; must either be None (default, samples from Clifford gates), an iterable containing gate names or classes, or a string corresponding to a gate set.")

    # @TODO - use custom logging
    # if any([gate.__name__ not in clifford_gates["strings"] for gate in gate_sample]):
    #     print(f"WARNING - gate_sample contains non-Clifford gate(s)")

    mb_circuit = QuantumCircuit(n_qubits)

    # Random shuffled sample of gates used for circuit
    gate_selection = np.random.choice(gate_sample, circuit_length//2)

    # Append original gates to circuit, targeting random qubits
    for gate in gate_selection:
        gate_obj = gate()
        target_qubits = list(np.random.choice(qubits, gate_obj.num_qubits, replace=False))
        mb_circuit.append(gate_obj, target_qubits)

    # Append inverse of current circuit so that final state is left unchanged under no errors
    mb_circuit.compose(mb_circuit.inverse(), inplace=True)

    if measure: mb_circuit.measure_all()

    return mb_circuit

def randomized_benchmarking(n_qubits=None, qubits=None, circuit_length=2, gate_sample=None, measure=False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if qubits is None:
        if n_qubits is None:
            qubits = [0]
            n_qubits = 1
        else:
            qubits = list(range(n_qubits))
    elif not hasattr(qubits, "__iter__"):
        raise TypeError(f"Parameter qubits must either be None (if n_qubits is specified) or iterable, not {type(qubits)}")
    else:
        n_qubits = len(list(qubits))

    if gate_sample is None:
        gate_sample = clifford_gates_1q["classes"] if n_qubits == 1 else clifford_gates["classes"]
    else:
        if hasattr(gate_sample, "__iter__"):
            for g, gate in enumerate(gate_sample):
                if isinstance(gate, str):
                    gate = string_to_gate_class(gate)
                elif not issubclass(gate, Gate):
                    raise TypeError(f"Invalid type for gate input at index {g}: {gate}; must be subclass of Gate")

                if gate().num_qubits > n_qubits:
                    raise ValueError(f"Gate {gate.__name__} requires more qubits than are available")
        elif isinstance(gate_sample, str):
            gate_sample = string_to_gate_set(gate_sample)["classes"]
        else:
            raise TypeError(f"Invalid type for gate_sample input: {type(gate_sample)}; must either be None (default, samples from Clifford gates), an iterable containing gate names or classes, or a string corresponding to a gate set.")

    # @TODO - use custom logging
    # if any([gate.__name__ not in clifford_gates["strings"] for gate in gate_sample]):
    #     print(f"WARNING - gate_sample contains non-Clifford gate(s)")

    rb_circuit = QuantumCircuit(n_qubits)

    # Random shuffled sample of gates used for circuit
    gate_selection = np.random.choice(gate_sample, circuit_length-1)

    # Append original gates to circuit, targeting random qubits
    for gate in gate_selection:
        gate_obj = gate()
        target_qubits = list(np.random.choice(qubits, gate_obj.num_qubits, replace=False))
        rb_circuit.append(gate_obj, target_qubits)

    # Construct a single inverse gate and append (really only works if the inverse gate is a basis gate)
    # @TODO - find a better way of doing this
    rb_circuit_inverse_matrix = Operator(rb_circuit.inverse()).data
    rb_circuit_inverse_gate = UnitaryGate(rb_circuit_inverse_matrix)
    rb_circuit.append(rb_circuit_inverse_gate, qubits)

    if measure: rb_circuit.measure_all()

    return rb_circuit

def quantum_volume(n_qubits=1, seed=None):
    """
    Generate quantum volume benchmark circuits.

    Parameters
    ----------
    n_qubits : int
        Number of qubits available for the benchmark (sets a maximum iteration limit on the circuit width).
    seed : int
        Random seed for reproducibility. Defaults to 1234.
    """

    qubits = list(range(n_qubits))
    qv_experiment = QuantumVolume(physical_qubits=qubits, trials=1, seed=seed)

    qv_circuits = qv_experiment.circuits()

    return qv_circuits

def quantum_teleportation(statevector, barriers=False):
    """
    Generate a quantum teleportation circuit.

    Parameters
    ----------
    statevector : iterable
        Arbitrary input state
    """

    creg0 = ClassicalRegister(1, 'cr0')
    creg1 = ClassicalRegister(1, 'cr1')
    creg2 = ClassicalRegister(1, 'cr2')

    qc = QuantumCircuit(3, creg0, creg1, creg2)

    # Initialize state to be teleported
    qc.initialize(statevector, [0])
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

def n_qubit_ghz_generation(n_qubits=3, barriers=False):
    """
    Generate an n-qubit GHZ state generation circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the GHZ state.
    barriers : boolean
        If True, apply barriers between sections of the circuit.
    """

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

