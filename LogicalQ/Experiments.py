import os
import time
import copy
import atexit
import pickle
import itertools
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor as Pool

from .Logical import LogicalCircuit, LogicalStatevector, LogicalDensityMatrix
from .NoiseModel import construct_noise_model, construct_noise_model_from_hardware_model

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qiskit import transpile
from qiskit.transpiler import PassManager
from .Transpilation.UnBox import UnBox

from qiskit.providers import Backend
from qiskit_ibm_runtime import QiskitRuntimeService
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.qiskit import qiskit_to_tk
from qbraid.runtime.native.device import QbraidDevice

DEFAULT = object()

"""
    General function to execute a circuit with smart handling of parameters, especially for circuits with QEC.

    The parameters target, backend, and hardware_model are the preferred input type to this function. If specified, noise_model, noise_params, coupling_map, and basis_gates will try to override anything specified in target, backend, or hardware_model.
"""
def execute_circuits(circuit_input, target=None, backend=None, hardware_model=None, noise_model=DEFAULT, noise_params=DEFAULT, coupling_map=DEFAULT, basis_gates=DEFAULT, method="statevector", optimization_level=0, shots=1024, memory=False, return_circuits_transpiled=False):
    # Resolve circuits
    circuits = []
    if hasattr(circuit_input, "__iter__"):
        for c, circuit in enumerate(circuit_input):
            if isinstance(circuit, QuantumCircuit):
                circuits.append(copy.deepcopy(circuit))
            else:
                raise TypeError(f"Iterable provided for circuits contains non-circuit object(s), first at index {c}: {circuit} (type: {type(circuit)})")
    elif isinstance(circuit_input, QuantumCircuit):
        circuits = [copy.deepcopy(circuit_input)]
    else:
        raise TypeError(f"Invalid type for circuits input: {type(circuit_input)}")

    # Check that the user has appended a measurement to every circuit
    for c, circuit in enumerate(circuits):
        if "measure" not in circuit.count_ops():
            raise ValueError(f"No measurements found in circuit with name {circuit.name} at index {c}; all circuits must have measurements in order to be executed.")

    # Patch to account for backends that do not yet recognize BoxOp's during transpilation
    pm = PassManager([UnBox()])
    for c in range(len(circuits)):
        while "box" in circuits[c].count_ops():
            circuits[c] = pm.run(circuits[c])

    max_num_qubits = max([circuit.num_qubits for circuit in circuits])

    # Resolve noise model
    if noise_model is DEFAULT:
        if hardware_model is not None and "noise_params" in hardware_model:
            # If noise_params are provided but not a noise_model or backend, then construct noise model based on the provided parameters
            noise_model = construct_noise_model_from_hardware_model(hardware_model)
        elif isinstance(noise_params, dict):
            # If noise_params are provided but not a noise_model or backend, then construct noise model based on the provided parameters
            noise_model = construct_noise_model(basis_gates=basis_gates, n_qubits=max_num_qubits, **noise_params)

            backend = "aer_simulator"
        else:
            # Default to no noise model
            noise_model = None
    elif noise_model is not None and not isinstance(noise_model, NoiseModel):
        raise TypeError(f"Invalid type for noise_model input: {type(noise_model)}")

    # Resolve coupling_map
    if coupling_map is DEFAULT:
        if hardware_model is None:
            # # Default to fully-coupled map
            # @TODO - pick a better default (like None) if the backend is a real backend
            coupling_map = "fully_coupled"
        else:
            coupling_map = hardware_model["device_info"].get("coupling_map", None)
    elif coupling_map is not None and not hasattr(coupling_map, "__iter__"):
        raise TypeError(f"Invalid type for coupling_map input: {type(coupling_map)}")

    if coupling_map == "fully_coupled":
        # Create a fully-coupled map
        coupling_map = [list(pair) for pair in itertools.product(range(max_num_qubits), range(max_num_qubits))]

    # Resolve basis_gates
    if basis_gates is DEFAULT:
        if hardware_model is None:
            basis_gates = None
        else:
            basis_gates = list(hardware_model["device_info"].get("basis_gates", None).keys())
    elif basis_gates is not None:
        raise TypeError(f"Invalid type for basis_gates input: {type(basis_gates)}")

    # Resolve backend
    if isinstance(backend, str):
        if backend == "aer_simulator":
            if target is None:
                if basis_gates is None:
                    backend = AerSimulator(method=method, noise_model=noise_model, coupling_map=coupling_map)
                else:
                    backend = AerSimulator(method=method, noise_model=noise_model, basis_gates=basis_gates, coupling_map=coupling_map)
            else:
                if basis_gates is not None:
                    raise ValueError("Cannot specify both target and basis_gates; target should be constructed based on basis gates")
                if coupling_map is not None:
                    raise ValueError("Cannot specify both target and coupling_map; target should be constructed based on coupling map")

                # @TODO - is it fine to specify both target and noise model, given that target includes some gate errors?
                #         at least, what is the expected behavior in such a scenario?
                backend = AerSimulator(method=method, target=target, noise_model=noise_model)
        elif backend.startswith("quantinuum_"):
            device_name = backend.split("_")[1]
            backend = QuantinuumBackend(device_name=device_name)
        elif backend.startswith("qbraid:"):
            provider = QbraidProvider()
            device = backend.split(":")[1]
            device = provider.get_device(device_name)
        else:
            # @TODO - handle this case better, e.g. by checking whether access to the device exists through
            #         Tket or Qbraid, not just IBM

            service = QiskitRuntimeService()
            backend = service.backend(backend)

    if isinstance(backend, AerSimulator):
        _transpile = transpile
        _cost = lambda circuits, shots : None
        _run = lambda circuits, **kwargs : backend.run(circuits, **kwargs).result()
    else:
        if noise_model not in [None, DEFAULT]:
            raise ValueError("Cannot pass noise_model to a non-simulator backend")
        if coupling_map not in [None, DEFAULT]:
            raise ValueError("Cannot pass coupling_map to a non-simulator backend")
        if basis_gates not in [None, DEFAULT]:
            raise ValueError("Cannot pass basis_gates to a non-simulator backend")

        if isinstance(backend, Backend):
            _transpile = transpile

            # @TODO - implement
            _cost = lambda circuits, shots : None

            _run = lambda circuits, **kwargs : backend.run(circuits, **kwargs).result()
        elif isinstance(backend, (QuantinuumBackend)):
            circuits = [qiskit_to_tk(circuit) for circuit in circuits]

            _transpile = lambda circuits, backend=None, coupling_map=None, optimization_level=0, **kwargs : backend.get_compiled_circuits(circuits, optimisation_level=optimization_level, **kwargs)

            device_name = backend._device_name.upper().rstrip("LE")
            _cost = lambda circuits, shots : backend.cost(circuits, n_shots=shots, syntax_checker=device_name+"SC")

            # @TODO - implement a smarter run function that instead uses process_circuits to get a handle and check its status periodically
            _run = lambda circuits, shots, memory=None, **kwargs : backend.run_circuits(circuits, n_shots=shots, **kwargs)
        else:
            raise TypeError(f"backend must be None, 'aer_simulator', the name of a backend, or an instance of AerSimulator, Backend, or QuantinuumBackend not type {type(backend)}")

    # Transpile circuits
    # Method defaults to optimization off to preserve form of benchmarking circuit and full QEC
    # @TODO - non-Qiskit backend instances may require another AerSimulator backend to be used for transpilation
    if target is None:
        circuits_transpiled = _transpile(
            circuits,
            backend=backend,
            optimization_level=optimization_level
        )
    else:
        # @TODO - is it fine to specify both target and backend, given that target has parameters which backend specifies,
        #         and backend is actually constructed with target? at least, what is the expected behavior in such a scenario?
        circuits_transpiled = _transpile(
            circuits,
            target=target,
            backend=backend,
            coupling_map=coupling_map,
            optimization_level=optimization_level
        )

    # Cost circuits (if applicable)
    # @TODO - retrieve user budget/available credits and include in confirmation prompt
    costs = _cost(circuits_transpiled, shots)
    if costs is None:
        circuits_to_run = circuits_transpiled
    else:
        costs_str = "\n"
        for c, (circuit, cost) in enumerate(zip(circuits, costs)):
            name = circuit.name
            costs_str += f"- Circuit {c} (name: '{name}'): {cost}\n"
        cost_confirmation = input(f"Estimated costs are as follows, respectively for each circuit: {costs_str} Enter 'Y' to confirm all jobs, 'N' to reject all jobs, or an input like 'i,j' to confirm jobs i and j or '^i,j' to confirm all jobs except i and j.").upper()
        if cost_confirmation == "Y":
            circuits_to_run = circuits_transpiled
        elif cost_confirmation == "N":
            print("All jobs rejected, returning...")
            return [None]*len(circuits_transpiled)
        else:
            if cost_confirmation.startswith("^"):
                exclude_indices = [int(choice) for choice in cost_confirmation[1:].split(",")]
                circuit_to_run = [circuits[c] for c in range(len(circuits_transpiled)) if c not in exclude_indices]
            else:
                include_indices = [int(choice) for choice in cost_confirmation.split(",")]
                circuit_to_run = [circuits[c] for c in range(len(circuits_transpiled)) if c in include_indices]

    # # Run circuits
    print("would run now")
    # results = []
    # for circuit_to_run in circuits_to_run:
    #     if circuit_to_run is None:
    #         result = None
    #     else:
    #         result = _run([circuit_to_run], shots=shots, memory=memory)
    #
    #     results.append(result)

    if return_circuits_transpiled:
        return results, circuits_transpiled
    else:
        return results

# Basic core experiment function useful for multiprocessing
def _basic_experiment_core(task_id, circuit, noise_model, backend, method, shots):
    print(os.getpid(), "starting")
    result = execute_circuits(circuit, noise_model=noise_model, backend=backend, method=method, shots=shots)
    print(os.getpid(), "stopping")

    return task_id, result

# @TODO - implement more experiments

def circuit_scaling_experiment(circuit_input, noise_model_input, min_n_qubits=1, max_n_qubits=16, min_circuit_length=1, max_circuit_length=16, backend="aer_simulator", method="statevector", shots=1024, with_mp=True, save_dir=None, save_filename=None):
    if isinstance(circuit_input, QuantumCircuit):
        if max_n_qubits != min_n_qubits:
            print("A constant circuit has been provided as the circuit factory, but a non-trivial range of qubit counts has also been provided, so the fixed input will not be scaled in this parameter. If you would like for the number of qubits to be scaled, please provide a callable which takes the number of qubits, n_qubits, as an argument.")
        if max_circuit_length != min_circuit_length:
            print("A constant circuit has been provided as the circuit factory, but a non-trivial range of circuit lengths has also been provided, so the fixed input will not be scaled in this parameter. If you would like for the circuit length to be scaled, please provide a callable which takes the circuit length, circuit_length, as an argument.")

        circuit_factory = lambda n_qubits, circuit_length: circuit_input
    elif callable(circuit_input):
        circuit_factory = circuit_input
    else:
        raise ValueError("Please provide a QuantumCircuit/LogicalCircuit object or a method for constructing QuantumCircuits/LogicalCircuits.")

    if isinstance(noise_model_input, NoiseModel):
        if max_n_qubits != min_n_qubits:
            print("A constant noise model has been provided as the noise model factory, but a non-trivial range of qubit counts has also been provided. The number of qubits will not be scaled; if you would like for the number of qubits to be scaled, please provide a callable which takes the number of qubits, n_qubits, as an argument.")

        noise_model_factory = lambda n_qubits: noise_model_input
    elif callable(noise_model_input):
        noise_model_factory = noise_model_input
    else:
        raise ValueError("Please provide a NoiseModel object or a method for constructing NoiseModels.")

    # Form a dict of dicts with the first layer (n_qubits) initialized to make later access faster and more reliable in parallel
    all_data = dict(zip(range(min_n_qubits, max_n_qubits+1), [{}]*(max_n_qubits+1-min_n_qubits)))

    # Prepare to save progress in the event of program termination
    if save_dir is None:
        save_dir = "./data/"
    if save_filename is None:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_filename = f"circuit_scaling_{date_str}.pkl"
    save_file = open(save_dir + save_filename, "wb")
    def save_progress():
        pickle.dump(all_data, save_file, protocol=5)
        save_file.close()
    atexit.register(save_progress)

    if with_mp:
        exp_inputs_list = [
            (
                (n_qubits, circuit_length),
                circuit_factory(n_qubits=n_qubits, circuit_length=circuit_length),
                noise_model_factory(n_qubits=n_qubits),
                backend, method, shots
            )
            for (n_qubits, circuit_length) in itertools.product(range(min_n_qubits, max_n_qubits+1), range(min_circuit_length, max_circuit_length+1))
        ]

        cpu_count = os.process_cpu_count() or 1

        # batch_size = max(int(np.ceil((max_n_qubits+1-min_n_qubits)*(max_circuit_length+1-min_circuit_length)/cpu_count)), 1)
        print(f"Applying multiprocessing to {len(exp_inputs_list)} samples across {cpu_count} CPUs")

        start = time.perf_counter()

        with Pool(cpu_count) as pool:
            mp_result = pool.map(
                _basic_experiment_core,
                *[list(exp_inputs) for exp_inputs in zip(*exp_inputs_list)],
                # chunksize=batch_size
            )

            # Unzip results
            for (task_id, result) in mp_result:
                all_data[task_id[0]][task_id[1]] = result[0]

        stop = time.perf_counter()
    else:
        start = time.perf_counter()

        for n_qubits in range(min_n_qubits, max_n_qubits+1):
            sub_data = {}

            # We need a new noise model for each qubit count
            noise_model_n = noise_model_factory(n_qubits=n_qubits)

            for circuit_length in range(min_circuit_length, max_circuit_length+1):
                # Construct circuit and benchmark noise
                circuit_nl = circuit_factory(n_qubits=n_qubits, circuit_length=circuit_length)
                result = execute_circuits(circuit_nl, noise_model=noise_model_n, backend=backend, method=method, shots=shots)[0]

                # Save expectation values
                sub_data[circuit_length] = result

                del circuit_nl

            del noise_model_n

            all_data[n_qubits] = sub_data

        stop = time.perf_counter()

    print(f"Completed experiment in {stop-start} seconds")

    # Run save_progress once for good measure and then unregister save_progress so it doesn't clutter our exit routine
    save_progress()
    atexit.unregister(save_progress)

    return all_data

def noise_scaling_experiment(circuit_input, noise_model_input, error_scan_keys, error_scan_val_lists, basis_gates=None, target=None, backend="aer_simulator", method="density_matrix", compute_exact=False, shots=1024, with_mp=False, save_dir=None, save_filename=None):
    if isinstance(circuit_input, QuantumCircuit):
        circuit_input = [circuit_input]
    elif hasattr(circuit_input, "__iter__") and all([isinstance(circuit, QuantumCircuit) for circuit in circuit_input]):
        circuit_input = circuit_input
    elif callable(circuit_input):
        raise NotImplementedError("QuantumCircuit/LogicalCircuit callables are not accepted as inputs, please provide a constant QuantumCircuit/LogicalCircuit object or a list of such.")
    else:
        raise ValueError("Please provide a QuantumCircuit/LogicalCircuit input.")

    if isinstance(noise_model_input, NoiseModel):
        # @TODO - for a more long-term solution, check if the noise_model_input already contains an error listed in error_scan_keys, and decide whether to override or to raise an exception if so
        def noise_model_factory(error_dict):
            # Currently overrides the base NoiseModel with error scan value if the same key is present in both
            updated_error_dict = {
                **noise_model_input.to_dict(),
                **error_dict
            }

            # @TODO - test compatibility between our error_dicts and the specification expected by the NoiseModel.from_dict method
            noise_model = NoiseModel.from_dict(updated_error_dict)

            if basis_gates is not None and len(basis_gates) > 0:
                noise_model.add_basis_gates(basis_gates)

            return noise_model
    elif hasattr(noise_model_input, "__iter__") and all([isinstance(noise_model, NoiseModel) for noise_model in noise_model_input]):
        noise_model_input = noise_model_input
    elif callable(noise_model_input):
        raise NotImplementedError("NoiseModel callables are not accepted as inputs, please provide a constant NoiseModel object or a list of such.")
    else:
        print("No base NoiseModel inputted, using an ideal NoiseModel as a base.")
        def noise_model_factory(error_dict):
            # @TODO - test compatibility between our error_dicts and the specification expected by the NoiseModel.from_dict method
            noise_model = NoiseModel.from_dict(error_dict)

            if basis_gates is not None and len(basis_gates) > 0:
                noise_model.add_basis_gates(basis_gates)

            return noise_model

    num_error_scan_keys = len(error_scan_keys)
    num_error_scan_vals = len(error_scan_val_lists)
    if num_error_scan_vals != num_error_scan_keys:
        raise ValueError(f"error_scan_val_lists has last dimension {num_error_scan_vals}, but error_scan_keys specifies {num_error_scan_keys} keys, which is not equal. Please make sure that error_scan_keys has as many keys as there are values in each list of error_scan_val_lists.")

    # Construct noise models
    error_scan_val_prods = itertools.product(*error_scan_val_lists)
    error_dicts = []
    noise_models = []
    for error_scan_vals in error_scan_val_prods:
        mapping = zip(error_scan_keys, error_scan_vals)
        error_dict = dict(mapping)
        error_dicts.append(error_dict)

        noise_model = noise_model_factory(error_dict)
        noise_models.append(noise_model)

    # Form a list of dicts to make later access faster and more reliable in parallel
    all_data = dict(zip(range(len(circuit_input)), [{}]*len(error_dicts)))

    # Prepare to save progress in the event of program termination
    if save_dir is None:
        save_dir = "./data/"
    if save_filename is None:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_filename = f"noise_scaling_{date_str}.pkl"
    save_file = open(save_dir + save_filename, "wb")
    def save_progress():
        pickle.dump(all_data, save_file, protocol=5)
        save_file.close()
    atexit.register(save_progress)

    if with_mp:
        start = time.perf_counter()

        for c, circuit in enumerate(circuit_input):
            density_matrix_exact = None # Default exact reference
            statevector_exact = None # Alternative exact reference, only used if exact DensityMatrix computation fails

            # @TODO - have better checks in place to proactively avoid exceptions, such as checking qubit counts and memory constraints
            if compute_exact:
                circuit_no_meas = circuit.remove_final_measurements(inplace=False)

                try:
                    # Compute exact density matrix
                    if isinstance(circuit, LogicalCircuit):
                        density_matrix_exact = LogicalDensityMatrix(circuit_no_meas)
                    else:
                        density_matrix_exact = DensityMatrix(circuit_no_meas)
                except:
                    print(f"Failed to compute exact density matrix for circuit input at index {c}, attempting to compute exact statevector...")
                    try:
                        # Compute exact statevector
                        if isinstance(circuit, LogicalCircuit):
                            statevector_exact = LogicalStatevector(circuit_no_meas)
                        else:
                            statevector_exact = Statevector(circuit_no_meas)
                    except Exception as e:
                        # Fail since we don't have any exact reference now
                        raise Exception("Failed to compute exact statevector, exiting...") from e

            # @TODO - determine whether this is a better data structure for this, including a better way to distinguish the data by circuit
            sub_data = {
                "circuit": circuit,
                "density_matrix_exact": density_matrix_exact,
                "statevector_exact": statevector_exact,
                "results": [{ "error_dict": error_dict } for error_dict in error_dicts]
            }

            exp_inputs_list = [
                (
                    nm,
                    circuit,
                    noise_model,
                    backend, method, shots
                )
                for nm, noise_model in enumerate(noise_models)
            ]

            cpu_count = os.process_cpu_count() or 1

            # batch_size = max(int(np.ceil(len(circuit_input)*len(error_dicts)/cpu_count)), 1)
            print(f"Applying multiprocessing to {len(exp_inputs_list)} samples across {cpu_count} CPUs")

            with Pool(cpu_count) as pool:
                mp_result = pool.map(
                    _basic_experiment_core,
                    *[list(exp_inputs) for exp_inputs in zip(*exp_inputs_list)],
                    # chunksize=batch_size
                )

            # Unzip results
            for (task_id, result) in mp_result:
                sub_data["results"][task_id]["result"] = result

            all_data[c] = sub_data

        stop = time.perf_counter()
    else:
        start = time.perf_counter()

        for c, circuit in enumerate(circuit_input):
            density_matrix_exact = None # Default exact reference
            statevector_exact = None # Alternative exact reference, only used if exact DensityMatrix computation fails

            # @TODO - have better checks in place to proactively avoid exceptions, such as checking qubit counts and memory constraints
            if compute_exact:
                circuit_no_meas = circuit.remove_final_measurements(inplace=False)

                try:
                    # Compute exact density matrix
                    if isinstance(circuit_no_meas, LogicalCircuit):
                        density_matrix_exact = LogicalDensityMatrix(circuit_no_meas)
                    else:
                        density_matrix_exact = DensityMatrix(circuit_no_meas)
                except:
                    print(f"Failed to compute exact density matrix for circuit input at index {c}, attempting to compute exact statevector...")
                    try:
                        # Compute exact statevector
                        if isinstance(circuit_no_meas, LogicalCircuit):
                            statevector_exact = LogicalStatevector(circuit_no_meas)
                        else:
                            statevector_exact = Statevector(circuit_no_meas)
                    except Exception as e:
                        # Fail since we don't have any exact reference now
                        raise Exception("Failed to compute exact statevector, exiting...") from e

            # @TODO - determine whether this is a better data structure for this, including a better way to distinguish the data by circuit
            sub_data = {
                "circuit": circuit,
                "density_matrix_exact": density_matrix_exact,
                "statevector_exact": statevector_exact,
                "results": []
            }

            for error_dict, noise_model in zip(error_dicts, noise_models):
                result = execute_circuits(circuit, target=target, backend=backend, noise_model=noise_model, method=method, shots=shots)

                sub_data["results"].append({
                    "error_dict": error_dict,
                    "result": result
                })

            all_data[c] = sub_data

        stop = time.perf_counter()

    print(f"Completed experiment in {stop-start} seconds")

    # Run save_progress once for good measure and then unregister save_progress so it doesn't clutter our exit routine
    save_progress()
    atexit.unregister(save_progress)

    return all_data

def qec_cycle_efficiency_experiment(circuit_input, qecc, constraint_scan_keys, constraint_scan_val_lists, with_mp=False, save_dir=None, save_filename=None, **kwargs):
    if isinstance(circuit_input, LogicalCircuit):
        raise NotImplementedError("LogicalCircuit inputs are not accepted because the original physical circuit(s) are also necessary for this experiment.")
    elif isinstance(circuit_input, QuantumCircuit):
        circuit_input = [circuit_input]
    elif hasattr(circuit_input, "__iter__"):
        if any([isinstance(circuit, LogicalCircuit) for circuit in circuit_input]):
            raise NotImplementedError("LogicalCircuit inputs are not accepted because the original physical circuit(s) are also necessary for this experiment.")
    elif callable(circuit_input):
        raise NotImplementedError("QuantumCircuit/LogicalCircuit callables are not accepted as inputs, please provide a constant QuantumCircuit/LogicalCircuit object or a list of such.")
    else:
        raise ValueError("Please provide a QuantumCircuit/LogicalCircuit input.")

    num_constraint_scan_keys = len(constraint_scan_keys)
    num_constraint_scan_val_lists = len(constraint_scan_val_lists)
    if num_constraint_scan_val_lists != num_constraint_scan_keys:
        raise ValueError(f"constraint_scan_val_lists has first dimension {num_constraint_scan_val_lists}, but constraint_scan_keys specifies {num_constraint_scan_keys} keys, which is not equal. Please make sure that constraint_scan_keys has as many keys as there are lists in constraint_scan_val_lists.")

    # Construct constraint model dictionaries
    constraint_scan_val_prods = itertools.product(*constraint_scan_val_lists)
    constraint_models = []
    for constraint_scan_vals in constraint_scan_val_prods:
        mapping = zip(constraint_scan_keys, constraint_scan_vals)
        constraint_models.append(dict(mapping))

    all_data = []

    # Prepare to save progress in the event of program termination
    if save_dir is None:
        save_dir = "./data/"
    if save_filename is None:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_filename = f"qec_cycle_efficiency_{date_str}.pkl"
    save_file = open(save_dir + save_filename, "wb")
    def save_progress():
        pickle.dump(all_data, save_file, protocol=5)
        save_file.close()
    atexit.register(save_progress)

    start = time.perf_counter()

    for circuit_physical in circuit_input:
        circuit_physical_no_meas = circuit_physical.remove_final_measurements(inplace=False)

        circuit_logical = LogicalCircuit.from_physical_circuit(circuit_physical_no_meas, **qecc)

        # @TODO - replicate code from other experiments which have a fallback to the Statevector if an error occurs
        density_matrix_exact = DensityMatrix(circuit_physical_no_meas)

        # @TODO - determine whether this is a better data structure for this, including a better way to distinguish the data by circuit
        sub_data = {
            "physical_circuit": circuit_physical,
            "logical_circuit": circuit_logical,
            "density_matrix_exact": density_matrix_exact,
            "results": [{ "constraint_model": constraint_model } for constraint_model in constraint_models]
        }

        if with_mp:
            def _qec_cycle_efficiency_experiment_core(cm, _circuit_logical, constraint_model, kwargs):
                print(os.getpid(), "starting")
                qec_cycle_indices = circuit_logical.optimize_qec_cycle_indices(constraint_model=constraint_model)
                _circuit_logical.insert_qec_cycles(qec_cycle_indices=qec_cycle_indices)
                _circuit_logical.measure_all()

                result = execute_circuits(_circuit_logical, **kwargs)[0]
                print(os.getpid(), "stopping")

                return cm, qec_cycle_indices, result

            exp_inputs_list = [
                (
                    cm,
                    copy.deepcopy(circuit_logical),
                    constraint_model,
                    kwargs
                )
                for cm, constraint_model in enumerate(constraint_models)
            ]

            cpu_count = os.process_cpu_count() or 1

            print(f"Applying multiprocessing to {len(exp_inputs_list)} samples across {cpu_count} CPUs")

            with Pool(cpu_count) as pool:
                mp_result = pool.map(
                    _qec_cycle_efficiency_experiment_core,
                    *[list(exp_inputs) for exp_inputs in zip(*exp_inputs_list)],
                    # chunksize=batch_size
                )

            # Unzip results
            for (cm, qec_cycle_indices, result) in mp_result:
                sub_data["results"][cm].update({
                    "qec_cycle_indices": qec_cycle_indices,
                    "result": result
                })
        else:
            for cm, constraint_model in enumerate(constraint_models):
                _circuit_logical = copy.deepcopy(circuit_logical)

                qec_cycle_indices = circuit_logical.optimize_qec_cycle_indices(constraint_model=constraint_model)
                _circuit_logical.insert_qec_cycles(qec_cycle_indices=qec_cycle_indices)

                _circuit_logical.measure_all()

                result = execute_circuits(_circuit_logical, **kwargs)[0]

                sub_data["results"][cm].update({
                    "qec_cycle_indices": qec_cycle_indices,
                    "result": result
                })

        all_data.append(sub_data)

    stop = time.perf_counter()

    print(f"Completed experiment in {stop-start} seconds")

    # Run save_progress once for good measure and then unregister save_progress so it doesn't clutter our exit routine
    save_progress()
    atexit.unregister(save_progress)

    return all_data

def qec_cycle_noise_scaling_experiment(circuit_input, noise_model_input, qecc, constraint_scan_keys, constraint_scan_val_lists, error_scan_keys, error_scan_val_lists, backend="aer_simulator", method="density_matrix", compute_exact=False, shots=1024, with_mp=False, save_dir=None, save_filename=None):
    if isinstance(circuit_input, LogicalCircuit):
        raise NotImplementedError("LogicalCircuit inputs are not accepted because the original physical circuit(s) are also necessary for this experiment.")
    elif isinstance(circuit_input, QuantumCircuit):
        circuit_input = circuit_input
    elif hasattr(circuit_input, "__iter__"):
        raise NotImplementedError("Please provide a single circuit")
    elif callable(circuit_input):
        raise NotImplementedError("QuantumCircuit/LogicalCircuit callables are not accepted as inputs, please provide a constant QuantumCircuit/LogicalCircuit object.")
    else:
        raise ValueError("Please provide a QuantumCircuit/LogicalCircuit input.")

    if isinstance(noise_model_input, NoiseModel):
        noise_model_input = noise_model_input
    elif hasattr(noise_model_input, "__iter__"):
        raise NotImplementedError("Please provide a single noise model")
    elif callable(noise_model_input):
        raise NotImplementedError("NoiseModel callables are not accepted as inputs, please provide a constant NoiseModel object.")
    else:
        raise ValueError("Please provide a NoiseModel object, a method for constructing NoiseModels, or a list of either.")

    num_constraint_scan_keys = len(constraint_scan_keys)
    num_constraint_scan_val_lists = len(constraint_scan_val_lists)
    if num_constraint_scan_val_lists != num_constraint_scan_keys:
        raise ValueError(f"constraint_scan_val_lists has last dimension {num_constraint_scan_val_lists}, but constraint_scan_keys specifies {num_constraint_scan_keys} keys, which is not equal. Please make sure that constraint_scan_keys has as many keys as there are values in each list of constraint_scan_val_lists.")

    if with_mp:
        raise NotImplementedError("with_mp=True specified, but this functionality is not implemented yet for qec_cycle_efficiency_experiment; ignoring.")

    # Construct constraint model dictionaries
    constraint_scan_val_prods = itertools.product(*constraint_scan_val_lists)
    constraint_models = []
    for constraint_scan_vals in constraint_scan_val_prods:
        mapping = zip(constraint_scan_keys, constraint_scan_vals)
        constraint_models.append(dict(mapping))

    all_data = [{}]*len(constraint_models)

    # Prepare to save progress in the event of program termination
    if save_dir is None:
        save_dir = "./data/"
    if save_filename is None:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_filename = f"qec_cycle_noise_scaling_{date_str}.pkl"
    save_file = open(save_dir + save_filename, "wb")
    def save_progress():
        pickle.dump(all_data, save_file, protocol=5)
        save_file.close()
    atexit.register(save_progress)

    for c, constraint_model in enumerate(constraint_models):
        sub_data = {
            "circuit": circuit_input,
            "constraint_model": constraint_model,
            "results_physical": None,
            "results_logical": None,
        }

        # Benchmark physical circuit
        results_physical = noise_scaling_experiment(
            circuit_input=circuit_input,
            noise_model_input=noise_model_input,
            error_scan_keys=error_scan_keys,
            error_scan_val_lists=error_scan_val_lists,
            compute_exact=compute_exact
        )
        all_data[c]["results_physical"] = results_physical

        # Construct LogicalCircuit object
        circuit_logical = LogicalCircuit.from_physical_circuit(circuit_input, **qecc)

        # Apply QEC according to constraint model
        qec_cycle_indices = circuit_logical.optimize_qec_cycle_indices(constraint_model=constraint_model)
        circuit_logical.insert_qec_cycles(qec_cycle_indices=qec_cycle_indices)

        # Benchmark logical circuit
        results_logical = noise_scaling_experiment(
            circuit_input=circuit_logical,
            noise_model_input=noise_model_input,
            error_scan_keys=error_scan_keys,
            error_scan_val_lists=error_scan_val_lists,
            backend=backend, method=method, compute_exact=False, shots=shots
        )
        all_data[c]["results_logical"] = results_logical

    # Run save_progress once for good measure and then unregister save_progress so it doesn't clutter our exit routine
    save_progress()
    atexit.unregister(save_progress)

    return all_data


def qec_scheduling_comparison_experiment(benchmark_circuits, qecc, constraint_model, fixed_intervals=[3, 5, 7], noise_model=None, hardware_model=None,
    backend="aer_simulator", method="density_matrix", shots=1024, compute_exact=True, save_dir=None, save_filename=None):
    
    # Process benchmark circuits
    if isinstance(benchmark_circuits, dict):
        circuit_names = list(benchmark_circuits.keys())
        circuits = list(benchmark_circuits.values())
    elif hasattr(benchmark_circuits, "__iter__"):
        circuits = list(benchmark_circuits)
        circuit_names = [f"circuit_{i}" for i in range(len(circuits))]
    else:
        circuits = [benchmark_circuits]
        circuit_names = ["circuit_0"]
    
    # Remove measurements from physical circuits for exact computation
    circuits_no_meas = []
    for circuit in circuits:
        if hasattr(circuit, 'remove_final_measurements'):
            circuits_no_meas.append(circuit.remove_final_measurements(inplace=False))
        else:
            # Create a copy and manually remove measurements
            circuit_copy = copy.deepcopy(circuit)
            circuit_copy.data = [inst for inst in circuit_copy.data if inst.operation.name != 'measure']
            circuits_no_meas.append(circuit_copy)
    
    # Initialize results storage
    all_data = {
        "circuits": circuit_names,
        "scheduling_methods": {
            "optimized": {},
            "random": {},
            "fixed": {}
        },
        "comparisons": {},
        "metadata": {
            "qecc": qecc,
            "constraint_model": constraint_model,
            "fixed_intervals": fixed_intervals
        }
    }
    
    # Prepare save functionality
    if save_dir is None:
        save_dir = "./data/"
    if save_filename is None:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_filename = f"qec_scheduling_comparison_{date_str}.pkl"
    
    save_path = save_dir + save_filename
    def save_progress():
        with open(save_path, "wb") as save_file:
            pickle.dump(all_data, save_file, protocol=5)
    atexit.register(save_progress)
    
    print(f"QEC scheduling comparison experiment with {len(circuits)} circuits...")
    
    for c_idx, (circuit_name, circuit, circuit_no_meas) in enumerate(zip(circuit_names, circuits, circuits_no_meas)):
        print(f"\nProcessing circuit '{circuit_name}' ({c_idx+1}/{len(circuits)})...")
        
        # Compute exact reference if requested
        exact_state = None
        if compute_exact:
            try:
                if method == "density_matrix":
                    exact_state = DensityMatrix(circuit_no_meas)
                else:
                    exact_state = Statevector(circuit_no_meas)
            except Exception as e:
                print(f"Warning: Could not compute exact state: {e}")
                exact_state = None
        
        circuit_results = {
            "circuit_depth": circuit_no_meas.depth(),
            "circuit_gates": circuit_no_meas.count_ops(),
            "methods": {}
        }
        
        # 1. OPTIMIZED SCHEDULING
        print(f"Testing our optimized QEC insertion...")
            
        lqc_opt = LogicalCircuit.from_physical_circuit(circuit_no_meas, **qecc)
        qec_indices_opt = lqc_opt.optimize_qec_cycle_indices(constraint_model=constraint_model)
        lqc_opt.insert_qec_cycles(qec_cycle_indices=qec_indices_opt)
        lqc_opt.measure_all()
            
        # Execute with noise
        result_opt = execute_circuits(lqc_opt, noise_model=noise_model, hardware_model=hardware_model, backend=backend, method=method, shots=shots)
            
        # Calculate fidelity if exact state available
        fidelity_opt = None
        if exact_state is not None and result_opt[0] is not None:
            try:
                if method == "density_matrix":
                    noisy_state = result_opt[0].data()["density_matrix"]
                else:
                    noisy_state = Statevector(result_opt[0].get_statevector())
                fidelity_opt = state_fidelity(exact_state, noisy_state)
            except:
                counts = result_opt[0].get_counts()
                # Estimate fidelity from counts
                fidelity_opt = _estimate_fidelity_from_counts(counts, exact_state)
            
        circuit_results["methods"]["optimized"] = {
            "qec_indices": qec_indices_opt,
            "num_qec_cycles": len(qec_indices_opt.get(0, [])),
            "result": result_opt[0],
            "fidelity": fidelity_opt
        }
        
        # 2. RANDOM SCHEDULING
        print(f"Testing random QEC insertion...")
        lqc_random = LogicalCircuit.from_physical_circuit(circuit_no_meas, **qecc)
            
        # Generate random QEC insertion points
        circuit_length = len([inst for inst in circuit_no_meas.data if inst.operation.name not in ['barrier', 'measure']])
        if circuit_length > 2:
            # Random number of QEC cycles
            num_qec_random = np.random.randint(1, min(circuit_length//2, 10))
            random_indices = sorted(np.random.choice(range(1, circuit_length), size=min(num_qec_random, circuit_length-1), replace=False))
            qec_indices_random = {0: random_indices}
        else:
            qec_indices_random = {0: []}
            
        lqc_random.insert_qec_cycles(qec_cycle_indices=qec_indices_random)
        lqc_random.measure_all()
            
        result_random = execute_circuits(lqc_random, noise_model=noise_model, hardware_model=hardware_model, backend=backend, method=method, shots=shots)
            
        # Fidelity
        fidelity_random = None
        if exact_state is not None and result_random[0] is not None:
            try:
                if method == "density_matrix":
                    noisy_state = result_random[0].data()["density_matrix"]
                else:
                    noisy_state = Statevector(result_random[0].get_statevector())
                fidelity_random = state_fidelity(exact_state, noisy_state)
            except:
                counts = result_random[0].get_counts()
                fidelity_random = _estimate_fidelity_from_counts(counts, exact_state)
            
        circuit_results["methods"]["random"] = {
            "qec_indices": qec_indices_random,
            "num_qec_cycles": len(qec_indices_random.get(0, [])),
            "result": result_random[0],
            "fidelity": fidelity_random
        }
        
        # 3. FIXED-INTERVAL SCHEDULING
        for interval in fixed_intervals:
            print(f"Testing fixed interval insertion (every {interval} gates)...")
            lqc_fixed = LogicalCircuit.from_physical_circuit(circuit_no_meas, **qecc)
                
            # Generate fixed-interval QEC insertion points
            circuit_length = len([inst for inst in circuit_no_meas.data if inst.operation.name not in ['barrier', 'measure']])
            fixed_indices = list(range(interval, circuit_length, interval))
            qec_indices_fixed = {0: fixed_indices} if fixed_indices else {0: []}
                
            lqc_fixed.insert_qec_cycles(qec_cycle_indices=qec_indices_fixed)
            lqc_fixed.measure_all()
                
            result_fixed = execute_circuits(lqc_fixed, noise_model=noise_model, hardware_model=hardware_model, backend=backend, method=method, shots=shots)
                
            # Fidelity
            fidelity_fixed = None
            if exact_state is not None and result_fixed[0] is not None:
                try:
                    if method == "density_matrix":
                        noisy_state = result_fixed[0].data()["density_matrix"]
                    else:
                        noisy_state = Statevector(result_fixed[0].get_statevector())
                    fidelity_fixed = state_fidelity(exact_state, noisy_state)
                except:
                    counts = result_fixed[0].get_counts()
                    fidelity_fixed = _estimate_fidelity_from_counts(counts, exact_state)
                
            circuit_results["methods"][f"fixed_{interval}"] = {
                "qec_indices": qec_indices_fixed,
                "num_qec_cycles": len(qec_indices_fixed.get(0, [])),
                "result": result_fixed[0],
                "fidelity": fidelity_fixed
            }
        
        all_data["scheduling_methods"][circuit_name] = circuit_results
    
    # Compare
    print("\nComputing comparisons...")
    all_data["comparisons"] = _compute_scheduling_comparisons(all_data["scheduling_methods"])
    
    # Save
    save_progress()
    atexit.unregister(save_progress)
    
    print(f"\nExperiment completed. Results saved to {save_path}")
    return all_data


def _estimate_fidelity_from_counts(counts, exact_state):
    try:
        # Get the most probable state from exact statevector
        if hasattr(exact_state, 'probabilities'):
            probs = exact_state.probabilities()
            target_state = format(np.argmax(probs), f'0{int(np.log2(len(probs)))}b')
        else:
            # Assume |0...0> state if not specified
            num_qubits = len(list(counts.keys())[0])
            target_state = '0' * num_qubits
        
        # Calculate fidelity as probability of measuring the target state
        total_counts = sum(counts.values())
        target_counts = counts.get(target_state, 0)
        return target_counts / total_counts
    except:
        return None


def _compute_scheduling_comparisons(scheduling_results):
    comparisons = {}
    
    for circuit_name, results in scheduling_results.items():
        circuit_comparisons = {}
        methods = results.get("methods", {})
        
        # Get fidelities
        fid_opt = methods.get("optimized", {}).get("fidelity", None)
        fid_random = methods.get("random", {}).get("fidelity", None)
        
        # Compute ratios if fidelities are available
        if fid_opt is not None and fid_random is not None and fid_random > 0:
            circuit_comparisons["opt_vs_random"] = fid_opt / fid_random
        
        # Compare with fixed intervals
        for key in methods:
            if key.startswith("fixed_"):
                fid_fixed = methods[key].get("fidelity", None)
                if fid_opt is not None and fid_fixed is not None and fid_fixed > 0:
                    circuit_comparisons[f"opt_vs_{key}"] = fid_opt / fid_fixed
                if fid_random is not None and fid_fixed is not None and fid_fixed > 0:
                    circuit_comparisons[f"random_vs_{key}"] = fid_random / fid_fixed
        
        # Store absolute fidelities for easy access
        circuit_comparisons["fidelities"] = {
            "optimized": fid_opt,
            "random": fid_random
        }
        for key in methods:
            if key.startswith("fixed_"):
                circuit_comparisons["fidelities"][key] = methods[key].get("fidelity", None)
        
        comparisons[circuit_name] = circuit_comparisons
    
    return comparisons

