import copy
import itertools
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RGate, HGate, XGate, YGate, ZGate, TGate
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity

from qiskit_aer import AerSimulator
from qiskit.providers import Backend
from qiskit_aer.noise import NoiseModel

from .Logical import LogicalCircuit, LogicalStatevector, logical_state_fidelity
from .Experiments import execute_circuits
from .Library.Gates import string_to_gate_set, get_num_params

def compute_effective_threshold(
    hardware_model,
    label, stabilizer_tableau,
    initial_states=None,
    min_theta=0, max_theta=np.pi/2, n_theta=16,
    min_phi=0, max_phi=np.pi/2, n_phi=16,
    max_n_qec_cycles=1,
    threshold_conditions=None
):
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
        elif initial_state in ["magic", "t"]:
            state_prep_gates.append(TGate())

    qc_list = []
    lqc_list = []
    sv_list = []

    interior_points = {}
    fidelities = {}

    for initial_state, state_prep_gate in zip(initial_states, state_prep_gates):
        qc = QuantumCircuit(1)
        qc.append(state_prep_gate, [0])

        lqc = LogicalCircuit.from_physical_circuit(qc, label, stabilizer_tableau)

        sv = Statevector(qc)

        qc_list.append(qc)
        lqc_list.append(lqc)
        sv_list.append(sv)

        fidelities[initial_state] = {}

    scan_angles = itertools.product(np.linspace(min_theta, max_theta, n_theta), np.linspace(min_phi, max_phi, n_phi))

    for theta, phi in scan_angles:
        rgate = RGate(theta=theta, phi=phi)

        for initial_state, qc, _lqc, sv in zip(initial_states, qc_list, lqc_list, sv_list):
            lqc = copy.deepcopy(_lqc)

            n = label[0]
            d = label[2]
            data_qubits = np.random.choice(lqc.logical_qregs[0], d)
            for data_qubit in data_qubits:
                lqc._append(rgate, [data_qubit])

            corrected = False
            for i in range(max_n_qec_cycles):
                lqc.append_qec_cycle()

                # @TODO - use the hardware model
                lsv = LogicalStatevector(lqc)

                fidelity = logical_state_fidelity(sv, lsv)
                if np.isclose(fidelity, 1.0, atol=1E-2):
                    corrected = True
                    break

            if corrected:
                interior_points[initial_state] = interior_points.get(initial_state, []) + [(theta, phi)]

            fidelities[initial_state][(theta,phi)] = fidelity

    # @TODO - make this support scans with multiple initial_states based on threshold_conditions
    # @TODO - verify that this computation makes sense
    angles_list = list(fidelities[initial_state[0]].keys())
    thetas = np.array([angles[0] for angles in angles_list])
    phis = np.array([angles[1] for angles in angles_list])
    ys = np.sin(phis) * np.sin(thetas)
    effective_threshold_angular_idx = np.argmin(ys)
    effective_threshold_theta = thetas[effective_threshold_angular_idx]
    effective_threshold_phi = phis[effective_threshold_angular_idx]

    ds = np.sqrt(effective_threshold_phi**2 + np.sin(effective_threshold_phi)**2 * effective_threshold_theta**2)

    return interior_points, fidelities

def compute_gadget_costs(
    gadgets_library=None,
    backend=None,
    hardware_model=None,
):
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
                for qc_component in qc_component_list:
                    qc_component.save_density_matrix()

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

def build_gadgets_library(min_depth, max_depth, step_depth, min_n_qubits, max_n_qubits, step_n_qubits):
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

