from __future__ import annotations

import itertools
import numpy as np

from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error, ReadoutError

from qiskit.circuit import Parameter
from qiskit.circuit.library import Measure

from .Library.Gates import gates_1q, gates_2q, get_num_params
from .Library.HardwareModels import hardware_models_Quantinuum

from typing import TYPE_CHECKING
from typing import Iterable

# General function for constructing a Qiskit NoiseModel
def construct_noise_model(
        basis_gates: list[str],
        n_qubits: int = None,
        qubits: Iterable[int] = None,
        ignore_qubits: Iterable[int] = None,
        **noise_params
    ) -> NoiseModel:
    if qubits is None and n_qubits is None:
        qubits = [0]
        n_qubits = 1
    elif qubits is None and n_qubits is not None:
        qubits = range(n_qubits)
    elif qubits is not None and n_qubits is None:
        n_qubits = len(qubits)

    ignore_qubits = set(ignore_qubits or [])
    used_qubits   = sorted(set(qubits) - ignore_qubits)

    noise_model = NoiseModel(basis_gates=basis_gates) # @todo - check if basis gates are really being used

    # Depolarizing errors: Simulates decay into random mixed state
    for gate in ["x", "y", "z", "h"]:
        if f"depolarizing_error_{gate}" in noise_params and "depolarizing_error_1q" in noise_params:
            depolarizing_error_gate = depolarizing_error(noise_params[f"depolarizing_error_{gate}"], 1)
            for q in used_qubits:
                noise_model.add_quantum_error(depolarizing_error_gate, [gate], [q], warnings=False)

        if f"depolarizing_error_c{gate}" in noise_params:
            depolarizing_error_cgate = depolarizing_error(noise_params[f"depolarizing_error_c{gate}"], 2)
            for q in used_qubits:
                noise_model.add_quantum_error(depolarizing_error_cgate, [f"c{gate}"], [q], warnings=False)

    if "depolarizing_error_1q" in noise_params:
        depolarizing_error_1q = depolarizing_error(noise_params[f"depolarizing_error_1q"], 1)
        for q in used_qubits:
            noise_model.add_quantum_error(depolarizing_error_1q, [gate for gate in basis_gates if type(gate) in gates_1q], [q], warnings=False)

    if "depolarizing_error_2q" in noise_params:
        depolarizing_error_2q = depolarizing_error(noise_params[f"depolarizing_error_2q"], 2)
        for q1 in used_qubits:
            for q2 in used_qubits:
                if q1 != q2:
                    noise_model.add_quantum_error(depolarizing_error_2q, [gate for gate in basis_gates if type(gate) in gates_2q], [q1, q2], warnings=False)

    # Readout errors: models errors in qubit measurement.
    if "readout_error_0|1" and "readout_error_1|0" in noise_params:
        p0given1 = noise_params["readout_error_0|1"]
        p1given0 = noise_params["readout_error_1|0"]

        readout_error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
        for q in used_qubits:
            noise_model.add_quantum_error(readout_error, [Measure], [q], warnings=False)

    # Thermal relaxation error: Error from releasing energy and settling back to the ground state
    if "thermal_relaxation_error" in noise_params:
        if "t1" in noise_params and "t2" in noise_params and "gate_time_1q" in noise_params and "gate_time_2q" in noise_params:
            T1 = noise_params["t1"]
            T2 = noise_params["t2"]
            gate_time_1q = noise_params["gate_time_1q"]
            gate_time_2q = noise_params["gate_time_2q"]

            thermal_relaxation_error = thermal_relaxation_error(T1, T2, gate_time_1q)
            thermal_relaxation_error_id = thermal_relaxation_error(T1, T2, 0)
            thermal_relaxation_error_1q = thermal_relaxation_error(T1, T2, gate_time_1q)
            thermal_relaxation_error_2q = thermal_relaxation_error(T1, T2, gate_time_2q)

            for q in used_qubits:
                noise_model.add_quantum_error(thermal_relaxation_error_id, ["id"], [q], warnings=False)

            for gate in gates_1q:
                for q in used_qubits:
                    noise_model.add_quantum_error(thermal_relaxation_error, [gate], [q], warnings=False)

            for gate in gates_2q:
                for q1 in used_qubits:
                    for q2 in used_qubits:
                        if q1 != q2:
                            noise_model.add_quantum_error(thermal_relaxation_error.tensor(thermal_relaxation_error), [gate], [q1, q2], warnings=False)

    # Amplituded damping error: Simulates error due to energy dissipation (e.g. spontaneous emission, thermal equilibrium)
    if "amplitude_damping_error_1q" in noise_params:
        amplitude_damping_error_1q = amplitude_damping_error(noise_params["amplitude_damping_error_1q"], 1)
        for q in used_qubits:
            noise_model.add_quantum_error(amplitude_damping_error_1q, ["x", "y", "z", "h", "s", "t", "rx", "ry", "rz"], [q], warnings=False)

    # @TODO - incorporate the missing error types

    return noise_model

# General function for constructing a Qiskit NoiseModel from a HardwareModel
def construct_noise_model_from_hardware_model(hardware_model, ignore_qubits=None):
    n_qubits = hardware_model["device_info"]["n_qubits"]
    all_qubits = range(n_qubits)

    ignore_qubits = set(ignore_qubits or [])
    noisy_qubits   = sorted(set(all_qubits) - ignore_qubits)

    qubit_tuple_lists = [[qubit_tuple for qubit_tuple in list(itertools.product(*([noisy_qubits]*(_n_qubits)))) if len(qubit_tuple) == len(set(qubit_tuple))] for _n_qubits in range(1,3+1)]

    basis_gates = hardware_model["device_info"]["basis_gates"]

    noise_model = NoiseModel(basis_gates=list(basis_gates.keys()))

    p = 0
    for qubit_specifier, qubit_noise_params in hardware_model["noise_params"].items():
        if qubit_specifier == "all_qubit":
            def add_error(error, gates, param_n_qubits, **kwargs):
                if isinstance(error, ReadoutError):
                    for qubit_tuple in qubit_tuple_lists[param_n_qubits-1]:
                        noise_model.add_readout_error(
                            error,
                            qubit_tuple,
                            warnings=False,
                            **kwargs
                        )
                else:
                    for qubit_tuple in qubit_tuple_lists[param_n_qubits-1]:
                        noise_model.add_quantum_error(
                            error,
                            gates,
                            qubit_tuple,
                            warnings=False,
                            **kwargs
                        )
        else:
            if qubit_specifier in noisy_qubits:
                add_error = lambda error, gates, param_n_qubits, **kwargs : noise_model.add_quantum_error(
                    error,
                    gates,
                    [qubit_tuple for qubit_tuple in list(itertools.permutations([qubit_specifier], *([noisy_qubits]*(param_n_qubits-1)))) if list(qubit_tuple) == list(set(qubit_tuple))],
                    warnings=False,
                    **kwargs
                )
            else:
                add_error = lambda *args : None

        if "T1" in qubit_noise_params and "T2" in qubit_noise_params and "gate_time" in qubit_noise_params:
            T1 = qubit_noise_params["T1"]
            T2 = qubit_noise_params["T2"]

            thermal_relaxation_error_id = thermal_relaxation_error(T1, T2, 0)
            add_error(thermal_relaxation_error_id, ["id"], 1)

            for n_qubits, n_qubits_data in qubit_noise_params["gate_time"].items():
                for gate, gate_time in n_qubits_data.items():
                    if gates == "all":
                        gates = []
                        for gate_name, gate_class in basis_gates.items():
                            parameter_list = [Parameter(f"p{p}") for _ in range(get_num_params(gate_class))]
                            if gate_class(*parameter_list).num_qubits == param_n_qubits:
                                gates.append(gate_name)
                    elif isinstance(gates, str):
                        gates = [gates]

                    thermal_relaxation_error = thermal_relaxation_error(T1, T2, gate_time)

                    add_error(thermal_relaxation_error, gates, n_qubits)

        for noise_param_key, noise_param_data in qubit_noise_params.items():
            apply_layered_error = False
            if noise_param_key == "depolarizing_error":
                error_method = depolarizing_error
                apply_layered_error = True
            elif noise_param_key == "amplitude_damping_error":
                error_method = amplitude_damping_error
                apply_layered_error = True
            elif noise_param_key == "phase_damping_error" or noise_param_key == "dephasing_error":
                error_method = phase_damping_error
                apply_layered_error = True

            if apply_layered_error:
                for param_n_qubits, n_qubit_error_data in noise_param_data.items():
                    for gates, gate_error_value in n_qubit_error_data.items():
                        if gates == "all":
                            gates = []
                            for gate_name, gate_class in basis_gates.items():
                                parameter_list = [Parameter(f"p{p}") for _ in range(get_num_params(gate_class))]
                                if gate_class(*parameter_list).num_qubits == param_n_qubits:
                                    gates.append(gate_name)
                        elif isinstance(gates, str):
                            gates = [gates]

                        error_object = error_method(gate_error_value, param_n_qubits)
                        add_error(error_object, gates, param_n_qubits)
            elif noise_param_key == "readout_error":
                for param_n_qubits, n_qubit_error_data in noise_param_data.items():
                    transition_matrix = np.zeros((2**param_n_qubits, 2**param_n_qubits))

                    for transition_tuple, transition_probability in n_qubit_error_data.items():
                        measured_bin, actual_bin = transition_tuple
                        measured_int = int(measured_bin, 2)
                        actual_int = int(actual_bin, 2)

                        transition_matrix[actual_int][measured_int] = transition_probability

                    for i in range(2**param_n_qubits):
                        if not np.isclose(np.sum(transition_matrix[i, :]), 1.0):
                            transition_matrix[i][i] = 1 - np.sum(transition_matrix[i, :])

                    readout_error = ReadoutError(transition_matrix)
                    add_error(readout_error, None, param_n_qubits)

    # @TODO - incorporate missing noise params

    return noise_model

# Quantinuum H1-1
def construct_noise_model_QuantinuumH1_1(n_qubits=None, qubits=None, ignore_qubits=None):
    if n_qubits is not None or qubits is not None:
        print("WARNING - n_qubits and qubits are no longer valid arguments for this method and are not in use; they will be removed from the function signature soon, so code should be adapted to not rely on these parameters.")

    return construct_noise_model_from_hardware_model(hardware_models_Quantinuum["H1-1"], ignore_qubits=ignore_qubits)

# Quantinuum H2-1
def construct_noise_model_QuantinuumH2_1(n_qubits=None, qubits=None, ignore_qubits=None):
    if n_qubits is not None or qubits is not None:
        print("WARNING - n_qubits and qubits are no longer valid arguments for this method and are not in use; they will be removed from the function signature soon, so code should be adapted to not rely on these parameters.")

    return construct_noise_model_from_hardware_model(hardware_models_Quantinuum["H2-1"], ignore_qubits=ignore_qubits)

# Quantinuum H2-2
def construct_noise_model_QuantinuumH2_2(n_qubits=None, qubits=None, ignore_qubits=None):
    if n_qubits is not None or qubits is not None:
        print("WARNING - n_qubits and qubits are no longer valid arguments for this method and are not in use; they will be removed from the function signature soon, so code should be adapted to not rely on these parameters.")

    return construct_noise_model_from_hardware_model(hardware_models_Quantinuum["H2-2"], ignore_qubits=ignore_qubits)

