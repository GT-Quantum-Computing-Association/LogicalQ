from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError

gates_1q = ["x", "y", "z", "h", "s", "t", "rx", "ry", "rz"]
gates_2q = ["cx", "cy", "cz", "ch"]

# General function for constructing a Qiskit NoiseModel
def construct_noise_model(basis_gates, n_qubits=None, qubits=None, ignore_qubits=None, **noise_params):
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
            noise_model.add_quantum_error(depolarizing_error_1q, gates_1q, [q], warnings=False)

    if "depolarizing_error_2q" in noise_params:
        depolarizing_error_2q = depolarizing_error(noise_params[f"depolarizing_error_2q"], 2)
        for q1 in used_qubits:
            for q2 in used_qubits:
                if q1 != q2:
                    noise_model.add_quantum_error(depolarizing_error_2q, gates_2q, [q1, q2], warnings=False)

    # Readout errors: models errors in qubit measurement.
    if "readout_error_01" in noise_params:
        p0given1 = noise_params["readout_error_0|1"]
        p1given0 = noise_params["readout_error_1|1"]

        readout_error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
        for q in used_qubits:
            noise_model.add_quantum_error(readout_error, [q], warnings=False)

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

    # @TODO - implement gate-specific thermal relaxation erorrs

    # Pauli error: Simulates error on Pauli gates

    # Coherent unitary error: Simulates general coherent unitary error

    # Amplituded damping error: Simulates error due to energy dissipation (e.g. spontaneous emission, thermal equilibrium)
    if "amplitude_damping_error_1q" in noise_params:
        amplitude_damping_error_1q = depolarizing_error(noise_params["amplitude_damping_error_1q"], 1)
        for q in used_qubits:
            noise_model.add_quantum_error(amplitude_damping_error_1q, ["x", "y", "z", "h", "s", "t", "rx", "ry", "rz"], [q], warnings=False)

    if "amplitude_damping_error_2q" in noise_params:
        # @TODO - not sure if Qiskit supports these
        pass

    # @TODO - implement gate-specific amplitude damping erorrs

    # Pauli-Lindblad error
    # Reset error
    # Kraus error
    # Phase damping error
    # Phase amplitude damping error

    # @TODO - incorporate the missing error types

    return noise_model

# @TODO - construct pre-made noise models for specific hardware
#       - wishlist:
#           - Quantinuum H1-1 (WIP)
#           - Quantinuum H1-2
#           - Quantinuum H2-1
#           - Harvard/MIT/QuEra collaboration (e.g. papers by Vuletic, Lukin, Bluvstein, Evered, Levine, Kalinowski, Li)

# Quantinuum H1-1 (02-May-2025 Calibration):
def construct_noise_model_QuantinuumH1_1(n_qubits=None, qubits=None, ignore_qubits=None):
    basis_gates = ["u", "rz", "zz", "rzz"] # @TODO - missing RXXYYZZ, not sure if ZZ is valid, and need to verify that angle conventions are correct

    noise_params = {
        "depolarizing_error_1q": 1.80e-5, # single-qubit fault probability
        "depolarizing_error_2q": 9.73e-4, # two-qubit fault probability
        "readout_error_0|1": 1.22e-3,
        "readout_error_1|0": 3.43e-3,
        "crosstalk_measure": 1.45e-5,
        "memory_error": 2.22e-4,
        "amplitude_damping_error_1q" : 0.54 * 1.80e-5, # calculated as a fraction of single-qubit fault probability
        "amplitude_damping_error_2q" : 0.43 * 9.73e-4, # calculated as a fraction of single-qubit fault probability
        "t1": 60e9, # converted from seconds to nanoseconds
        "t2": 4e9, # converted from seconds to nanoseconds
        "gate_time_1q": 10e3, # converted from microseconds to nanoseconds (pessimistic)
        "gate_time_2q": 300e3, # converted from microseconds to nanoseconds (pessimistic)
    }

    return construct_noise_model(n_qubits=n_qubits, basis_gates=basis_gates, ignore_qubits=ignore_qubits, **noise_params)

# Quantinuum H2-1 (30-Apr-2025 Calibration):
def construct_noise_model_QuantinuumH2_1(n_qubits=None, qubits=None, ignore_qubits=None):
    basis_gates = ["u", "rz", "zz", "rzz"]

    noise_params = {
        "depolarizing_error_1q": 1.89e-5, # single-qubit fault probability
        "depolarizing_error_2q": 1.05e-3, # two-qubit fault probability
        "readout_error_0|1": 6.00e-4,
        "readout_error_1|0": 1.39e-3,
        "crosstalk_measure": 6.65e-6,
        "memory_error": 2.03e-4,
        "amplitude_damping_error_1q": 0.54 * 1.89e-5, # calculated as a fraction of single-qubit fault probability
        "amplitude_damping_error_2q": 0.43 * 1.05e-3, # calculated as a fraction of single-qubit fault probability
        "t1": 60e9, # converted from seconds to nanoseconds
        "t2": 4e9, # converted from seconds to nanoseconds
        "gate_time_1q": 10e3, # converted from microseconds to nanoseconds (pessimistic)
        "gate_time_2q": 300e3, # converted from microseconds to nanoseconds (pessimistic)
    }

    return construct_noise_model(n_qubits=n_qubits, basis_gates=basis_gates, ignore_qubits=ignore_qubits, **noise_params)

# Quantinuum H2-2 (31-May-2024 Calibration):
def construct_noise_model_QuantinuumH2_2(n_qubits=None, qubits=None, ignore_qubits=None):
    basis_gates = ["u", "rz", "zz", "rzz"]

    noise_params = {
        "depolarizing_error_1q": 7.30e-5, # single-qubit fault probability
        "depolarizing_error_2q": 1.29e-3, # two-qubit fault probability
        "readout_error_0|1": 9.00e-4,
        "readout_error_1|0": 1.80e-3,
        "crosstalk_measure": 8.80e-6,
        "memory_error": 5.00e-4,
        "amplitude_damping_error_1q": 0.54 * 7.30e-5, # calculated as a fraction of single-qubit fault probability
        "amplitude_damping_error_2q": 0.43 * 1.29e-3, # calculated as a fraction of single-qubit fault probability
        "t1": 60e9, # converted from seconds to nanoseconds
        "t2": 4e9, # converted from seconds to nanoseconds
        "gate_time_1q": 10e3, # converted from microseconds to nanoseconds (pessimistic)
        "gate_time_2q": 300e3, # converted from microseconds to nanoseconds (pessimistic)
    }

    return construct_noise_model(n_qubits=n_qubits, basis_gates=basis_gates, ignore_qubits=ignore_qubits, **noise_params)

