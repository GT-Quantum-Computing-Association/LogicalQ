import copy
import itertools
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import RGate, HGate, XGate, YGate, ZGate, TGate
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity

from .Logical import LogicalCircuit, LogicalStatevector, logical_state_fidelity
from .Experiments import execute_circuits
from .Library.Gates import string_to_gate_set

from typing import TYPE_CHECKING, Callable
from typing import Iterable
from .Typing import Number, QuantumBackend

def compute_constraint_model(
    hardware_model: dict,
    label: Iterable[int],
    stabilizer_tableau: Iterable[str],
    optimizer: bool = True,
    effective_threshold: Number | None = None,
    gadget_costs: dict | None = None,
    constraint_model: dict | None = None,
) -> dict:
    """Compute constraint model for a given quantum error correction code.
    """

    if constraint_model is None:
        constraint_model = {}

    if effective_threshold is None:
        if "effective_threshold" in constraint_model:
            effective_threshold = constraint_model["effective_threshold"]
        constraint_model["effective_threshold"] = effective_threshold

    # Step 1: Add single-component gadget costs
    # @TODO - assumes errors are all-qubit errors, needs to be able to handle qubit-specific errors
    for qubit_noise_params in hardware_model["noise_params"].values():
        for noise_param_key, noise_param_data in qubit_noise_params.items():
            if noise_param_key in ["depolarizing_error", "amplitude_damping_error"]:
                for param_n_qubits, n_qubit_error_data in noise_param_data.items():
                    for gate, gate_error_value in n_qubit_error_data.items():
                        if gate == "all":
                            constraint_model[f"cost_ops_{param_n_qubits}q"] = constraint_model.get(f"cost_ops_{param_n_qubits}q", 0) + gate_error_value
                        elif isinstance(gate, str):
                            constraint_model[f"cost_{gate}"] = constraint_model.get(f"cost_{gate}", 0) + gate_error_value

    # Step 2: Add multi-component gadget costs
    if gadget_costs is not None:
        for order, order_gadgets in gadget_costs.items():
            for n_qubits, n_qubit_gadgets in order_gadgets.items():
                for gadget, gadget_cost in n_qubit_gadgets.items():
                    constraint_key = "cost_" + "-".join(component[0]().name for component in gadget)
                    constraint_model[constraint_key] = gadget_cost

    return constraint_model

def compute_effective_threshold(
    hardware_model: dict,
    label: Iterable[int],
    stabilizer_tableau: Iterable[str],
    initial_states: Iterable[str] | None = None,
    min_theta: Number = 0,
    max_theta: Number = np.pi/2,
    n_theta: int = 16,
    min_phi: Number = 0,
    max_phi: Number = np.pi,
    n_phi: int = 32,
    max_n_qec_cycles: int = 1,
    shots: Number = 1E4,
    threshold_conditions: Iterable[Callable[[float, float], bool]] | None = None,
) -> tuple[dict, dict, float]:
    """Compute effective threshold for a given quantum error correction code.
    """

    if initial_states is None:
        initial_states = ["0"]
    else:
        initial_states = [initial_state.lower() for initial_state in initial_states]

    state_prep_gates = []
    for initial_state in initial_states:
        if initial_state in ["0", "z"]:
            state_prep_gates.append(ZGate())
        elif initial_state in ["1", "x"]:
            state_prep_gates.append(XGate())
        elif initial_state in ["i", "y"]:
            state_prep_gates.append(YGate())
        elif initial_state in ["+", "h"]:
            state_prep_gates.append(HGate())
        elif initial_state in ["t"]:
            state_prep_gates.append(TGate())

    if threshold_conditions is None:
        # By default, check whether infidelity ratio is greater than 1.0
        threshold_conditions = [
            lambda fidelity_noqec, fidelity_qec : (1-fidelity_noqec)/(1-fidelity_qec) >= 1.0,
        ]

    qc_list = []
    lqc_list = []
    sv_list = []

    interior_points = {}
    fidelities = {}

    for initial_state, state_prep_gate in zip(initial_states, state_prep_gates):
        qc = QuantumCircuit(1)
        qc.append(state_prep_gate, [0])

        sv = Statevector(qc)

        lqc = LogicalCircuit.from_physical_circuit(qc, label, stabilizer_tableau)

        qc_list.append(qc)
        lqc_list.append(lqc)
        sv_list.append(sv)

        fidelities[initial_state] = {}

    scan_angles = itertools.product(np.linspace(min_theta, max_theta, n_theta), np.linspace(min_phi, max_phi, n_phi))

    for theta, phi in scan_angles:
        rgate = RGate(theta=theta, phi=phi)

        for initial_state, qc, _lqc, sv in zip(initial_states, qc_list, lqc_list, sv_list):
            lqc = copy.deepcopy(_lqc)

            k = label[1]
            d = label[2]
            data_qubits = np.random.choice(lqc.logical_qregs[0], d)
            for data_qubit in data_qubits:
                lqc._append(rgate, [data_qubit])

            corrected = False
            fidelity_noqec = fidelity_qec = None
            for n_qec_cycles in range(max_n_qec_cycles):
                # Without QEC
                lqc_noqec = lqc.copy()
                lqc_noqec.measure_all()

                result_noqec = execute_circuits(lqc_noqec, backend="aer_simulator", method="statevector", shots=shots)[0]

                lsv_noqec = LogicalStatevector.from_counts(result_noqec.get_counts(), k, label, stabilizer_tableau)

                # With QEC
                lqc_qec = lqc.copy()
                for _ in range(n_qec_cycles): lqc_qec.append_qec_cycle()
                lqc_qec.measure_all()

                # @TODO - I feel like density-matrix sims are preferable
                result_qec = execute_circuits(lqc_qec, backend="aer_simulator", method="statevector", shots=shots)[0]

                # @TODO - use a saved statevector instead
                lsv_qec = LogicalStatevector.from_counts(result_qec.get_counts(), k, label, stabilizer_tableau)

                fidelity_noqec = logical_state_fidelity(sv, lsv_noqec)
                fidelity_qec = logical_state_fidelity(sv, lsv_qec)

                for threshold_condition in threshold_conditions:
                    if threshold_condition(fidelity_noqec, fidelity_qec):
                        corrected = True
                        break

            if corrected:
                interior_points[initial_state] = interior_points.get(initial_state, []) + [(theta, phi)]

            fidelities[initial_state][(theta,phi)] = (fidelity_noqec, fidelity_qec)

    # @TODO - make this support scans with multiple initial_states based on threshold_conditions
    # @TODO - verify that this computation makes sense
    if interior_points:
        angles_list = interior_points[initial_states[0]]
        thetas = np.array([angles[0] for angles in angles_list])
        effective_threshold_theta = np.max(thetas)

        effective_threshold = effective_threshold_theta/np.pi
    else:
        print("WARNING - No interior points found!")

        effective_threshold = 0.0

    return interior_points, fidelities, effective_threshold

def compute_gadget_costs(
    gadgets_library: dict | None = None,
    backend: QuantumBackend | None = None,
    hardware_model: dict | None = None,
) -> tuple[dict, dict, dict]:
    """Compute the costs of a gadgets library based on a hardware model or a backend.
    """

    if gadgets_library is None:
        gadgets_library = build_gadgets_library(min_depth=2, max_depth=2+1, step_depth=1, min_n_qubits=1, max_n_qubits=1+1, step_n_qubits=1)

    if hardware_model is None and backend is None:
        raise ValueError("At least one of hardware_model or backend must be specified.")

    gadget_infidelities = {}
    gadget_costs = {}
    for depth, depth_gadgets in gadgets_library.items():
        gadget_infidelities[depth] = {}
        gadget_costs[depth] = {}

        for n_qubits, n_qubits_gadgets in depth_gadgets.items():
            gadget_infidelities[depth][n_qubits] = {}
            gadget_costs[depth][n_qubits] = {}

            for g, gadget in enumerate(n_qubits_gadgets):
                # Construct a quantum circuit with all parts of the gadget
                qc_full = QuantumCircuit(n_qubits)
                for (gate, qubit) in gadget:
                    qc_full.append(gate(), [qubit])

                # Construct single-component quantum circuits
                qc_component_list = []
                for (gate, qubit) in gadget:
                    qc_component = QuantumCircuit(n_qubits)
                    qc_component.append(gate(), [qubit])
                    qc_component_list.append(qc_component)

                # Construct list of all quantum circuits for ease of use
                qc_list = [qc_full] + qc_component_list

                # Compute exact density matrices
                density_matrices_exact_all = [Statevector(qc) for qc in qc_list]
                density_matrix_exact_full = density_matrices_exact_all[0]
                density_matrices_exact_component_list = density_matrices_exact_all[1:]

                # Execute circuits on noisy backends
                qc_full.save_density_matrix()
                qc_full.measure_all()
                for qc_component in qc_component_list:
                    qc_component.save_density_matrix()
                    qc_component.measure_all()

                results = execute_circuits(qc_list, backend=backend, hardware_model=hardware_model, method="density_matrix")

                density_matrices_noisy_all = [result.data()["density_matrix"] for result in results]
                density_matrix_noisy_full = density_matrices_noisy_all[0]
                density_matrices_noisy_component_list = density_matrices_noisy_all[1:]

                # Compute fidelity and infidelity of the full quantum circuit
                fidelity_full = state_fidelity(density_matrix_exact_full, density_matrix_noisy_full)
                infidelity_full = 1-fidelity_full

                # Compute fidelities and infidelities of the component quantum circuits
                fidelities_component = np.array([state_fidelity(density_matrix_exact_component, density_matrix_noisy_component) for density_matrix_exact_component, density_matrix_noisy_component in zip(density_matrices_exact_component_list, density_matrices_noisy_component_list)])
                infidelities_component = 1-fidelities_component

                # The punchline - compute the total infidelity of each component circuit
                infidelity_full_reconstructed = float(np.sum(infidelities_component))

                # Store data
                gadget_infidelities[depth][n_qubits][gadget] = (infidelity_full, infidelity_full_reconstructed)
                gadget_costs[depth][n_qubits][gadget] = infidelity_full - infidelity_full_reconstructed

    return gadgets_library, gadget_infidelities, gadget_costs

def build_gadgets_library(
    min_depth: int,
    max_depth: int,
    step_depth: int,
    min_n_qubits: int,
    max_n_qubits: int,
    step_n_qubits: int
) -> dict:
    """Build a library of gadgets within the specified depth and qubit count parameters.
    """

    gadgets_library = {}
    
    for depth in range(min_depth, max_depth, step_depth):
        gadgets_library[depth] = {}

        gate_set = []
        for n_qubits in range(min_n_qubits, max_n_qubits, step_n_qubits):
            gadgets_library[depth][n_qubits] = []

            if n_qubits > 1:
                qubits = list(range(1, n_qubits))

            new_gate_set = string_to_gate_set(f"gates_{n_qubits}q")
            for gate_class, num_params in zip(new_gate_set["classes"], new_gate_set["num_params"]):
                if num_params == 0:
                    gate_set.append(gate_class)

            gate_tuples = [gate_tuple for gate_tuple in itertools.product(gate_set, repeat=depth) if gate_tuple[0] != gate_tuple[1]]

            for gate_tuple in gate_tuples:
                gadget = []
                for gate in gate_tuple:
                    if n_qubits == 1:
                        target_qubits = (0,)
                    else:
                        target_qubits = list(np.random.choice(qubits, gate().num_qubits-1))
                        target_qubits.append(0)
                        target_qubits = tuple(int(val) for val in np.random.permutation(target_qubits))

                    gadget.append((gate, target_qubits))

                gadget = tuple(gadget)

                gadgets_library[depth][n_qubits].append(gadget)

    return gadgets_library

