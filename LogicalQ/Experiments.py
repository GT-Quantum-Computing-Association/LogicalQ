import time
import copy
import itertools
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor as Pool
import pickle
import atexit
from datetime import datetime

from .Logical import LogicalCircuit
from .NoiseModel import construct_noise_model
from .Benchmarks import *

from .Transpilation.UnBox import UnBox

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.providers import Backend
from qiskit.transpiler import PassManager

# General function to benchmark a circuit using a noise model
def execute_circuits(circuits, backend=None, noise_model=None, noise_params=None, coupling_map=None, basis_gates=None, method="statevector", optimization_level=0, shots=1024):
    # Resolve circuits
    if hasattr(circuits, "__iter__"):
        for c, circuit in enumerate(circuits):
            if not isinstance(circuit, QuantumCircuit):
                raise TypeError(f"Iterable provided for circuits contains non-circuit object(s), first at index {c}: {circuit} (type: {type(circuit)})")
    elif isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    else:
        raise TypeError(f"Invalid type for circuits input: {type(circuits)}")

    # Patch to account for backends that do not yet recognize BoxOp's during transpilation
    pm = PassManager([UnBox()])
    for c in range(len(circuits)):
        while "box" in circuits[c].count_ops():
            circuits[c] = pm.run(circuits[c])

    # Resolve backend
    if backend is None:
        # Resolve noise_model or noise_params
        if noise_model is None:
            if noise_params is not None:
                # If noise_params are provided but not a noise_model or backend, then construct noise model based on the provided parameters
                noise_model = construct_noise_model(basis_gates=basis_gates, n_qubits=circuit.num_qubits, **noise_params)

                backend = "aer_simulator"
            else:
                raise ValueError("One of backend, noise_model, or noise_params must be provided")

    if isinstance(backend, str):
        if backend == "aer_simulator":
            if coupling_map is None:
                # Create a fully-coupled map by default since we don't care about non-fully-coupled hardware modalities
                max_num_qubits = max([circuit.num_qubits for circuit in circuits])
                coupling_map = [list(pair) for pair in itertools.product(range(max_num_qubits), range(max_num_qubits))]

            if basis_gates is not None:
                backend = AerSimulator(method=method, noise_model=noise_model, basis_gates=basis_gates, coupling_map=coupling_map)
            else:
                backend = AerSimulator(method=method, noise_model=noise_model, coupling_map=coupling_map)
        else:
            service = QiskitRuntimeService()
            backend = service.get_backend(backend)
    elif isinstance(backend, (AerSimulator, Backend)):
        # @TODO - handle this case better
        backend = backend
    else:
        raise TypeError(f"backend must be None, 'aer_simulator', the name of a backend, or an instance of AerSimulator or Backend, not type {type(backend)}")

    # Transpile circuit
    # Method defaults to optimization off to preserve form of benchmarking circuit and full QEC
    circuits_transpiled = transpile(circuits, backend=backend, optimization_level=optimization_level)
    result = backend.run(circuits_transpiled, shots=shots).result()

    return result

# Core experiment function useful for multiprocessing
def _experiment_core(task_id, circuit, noise_model, backend, method, shots):
    result = execute_circuits(circuit, noise_model=noise_model, backend=backend, method=method, shots=shots)

    return task_id, result

# @TODO - implement experiments
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

        cpu_count = len(os.sched_getaffinity(0))
        batch_size = max(int(np.ceil((max_n_qubits+1-min_n_qubits)*(max_circuit_length+1-min_circuit_length)/cpu_count)), 1)
        print(f"Applying mulitprocessing to {len(exp_inputs_list)} samples in batches of maximum size {batch_size} across {cpu_count} CPUs")

        start = time.perf_counter()

        with Pool(cpu_count) as pool:
            mp_result = pool.map(
                _experiment_core,
                *[list(exp_inputs) for exp_inputs in zip(*exp_inputs_list)],
                chunksize=batch_size
            )

            # Unzip results
            for (task_id, result) in mp_result:
                all_data[task_id[0]][task_id[1]] = result

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
                result = execute_circuits(circuit_nl, noise_model=noise_model_n, backend=backend, method=method, shots=shots)

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

def noise_scaling_experiment(circuit_input, noise_model_input, error_scan_keys, error_scan_val_lists, noise_qubits=None, basis_gates=None, backend="aer_simulator", method="density_matrix", compute_exact=False, shots=1024, with_mp=False, save_dir=None, save_filename=None):
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
    elif callable(noise_model):
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

    if with_mp:
        raise NotImplementedError("with_mp=True specified, but this functionality is not implemented yet for noise_model_scaling_experiment; ignoring.")

    # Construct noise models
    error_scan_val_prods = itertools.product(*config_scan_val_lists)
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
            if compute_exact:
                # @TODO - have better checks in place to proactively avoid exceptions, such as checking qubit counts and memory constraints
                try:
                    # Compute exact density matrix
                    density_matrix_exact = DensityMatrix(circuit_input)
                except:
                    print(f"Failed to compute exact density matrix for circuit input at index {c}, attempting to compute exact statevector...")
                    try:
                        # Compute exact statevector
                        statevector_exact = Statevector(circuit_input)
                    except:
                        # Fail since we don't have any exact reference now
                        print(f"Failed to compute exact statevector, exiting.")
                        raise

            # @TODO - determine whether this is a better data structure for this, including a better way to distinguish the data by circuit
            sub_data = {
                "circuit": circuit,
                "density_matrix_exact": density_matrix_exact,
                "statevector_exact": statevector_exact,
                "results": [{}]*len(noise_models)
            }

            exp_inputs_list = [
                (
                    nm,
                    circuit,
                    noise_model,
                    backend, method, shots
                )
                for nm, noise_model in enumerate(noise_model)
            ]

            cpu_count = len(os.sched_getaffinity(0))
            batch_size = max(int(np.ceil(len(circuit_input)*len(error_dicts)/cpu_count)), 1)
            print(f"Applying mulitprocessing to {len(exp_inputs_list)} samples in batches of maximum size {batch_size} across {cpu_count} CPUs")

            with Pool(cpu_count) as pool:
                mp_result = pool.map(
                    _experiment_core,
                    *[list(exp_inputs) for exp_inputs in zip(*exp_inputs_list)],
                    chunksize=batch_size
                )

            # Unzip results
            for (task_id, result) in mp_result:
                sub_data["results"][task_id] = result

            all_data[c] = sub_data

        stop = time.perf_counter()
    else:
        start = time.perf_counter()

        for c, circuit in circuit_input:
            density_matrix_exact = None # Default exact reference
            statevector_exact = None # Alternative exact reference, only used if exact DensityMatrix computation fails
            if compute_exact:
                # @TODO - have better checks in place to proactively avoid exceptions, such as checking qubit counts and memory constraints
                try:
                    # Compute exact density matrix
                    density_matrix_exact = DensityMatrix(circuit_input)
                except:
                    print(f"Failed to compute exact density matrix for circuit input at index {c}, attempting to compute exact statevector...")
                    try:
                        # Compute exact statevector
                        statevector_exact = Statevector(circuit_input)
                    except:
                        # Fail since we don't have any exact reference now
                        print(f"Failed to compute exact statevector, exiting.")
                        raise

            # @TODO - determine whether this is a better data structure for this, including a better way to distinguish the data by circuit
            sub_data = {
                "circuit": circuit,
                "density_matrix_exact": density_matrix_exact,
                "statevector_exact": statevector_exact,
                "results": []
            }

            for error_dict, noise_model in zip(error_dict, noise_models):
                result = execute_circuits(circuit_input, noise_model=noise_model, backend=backend, method=method, shots=shots)

                sub_data["results"].append({
                    "error_dict": error_dict,
                    "result": result
                })

            all_data[nm] = sub_data

        stop = time.perf_counter()

    print(f"Completed experiment in {stop-start} seconds")

    # Run save_progress once for good measure and then unregister save_progress so it doesn't clutter our exit routine
    save_progress()
    atexit.unregister(save_progress)

    return all_data

def qec_cycle_efficiency_experiment(circuit_input, noise_model_input, qecc, constraint_scan_keys, constraint_scan_val_lists, backend="aer_simulator", method="density_matrix", shots=1024, with_mp=False, save_dir=None, save_filename=None):
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

    if isinstance(noise_model_input, NoiseModel):
        noise_model_factory = lambda c : noise_model_input
    elif hasattr(noise_model_input, "__iter__"):
        if len(noise_model_input) == len(constraint_models):
            noise_model_callable_list = []
            for noise_model_input_element in noise_model_input:
                if isinstance(noise_model_input_element, NoiseModel):
                    noise_model_callable_list.append(noise_model_input_element)
                else:
                    raise ValueError("List provided for noise_model_input does not match length of constraint_models list input; noise_model_input must either be constant, a callable, or a list.")

            noise_model_factory = lambda c : noise_model_callable_list[c]
        else:
            raise ValueError("List provided for noise_model_input does not match length of constraint_models list - noise_model_input must either be constant, a callable, or a list with length matching that of constraint_models list.")
    elif callable(noise_model_input):
        raise NotImplementedError("NoiseModel callables are not accepted as inputs, please provide a constant NoiseModel object or a list of such.")
    else:
        raise ValueError("Please provide a NoiseModel input.")

    num_constraint_scan_keys = len(constraint_scan_keys)
    num_constraint_scan_val_lists = len(constraint_scan_val_lists)
    if num_constraint_scan_val_lists != num_constraint_scan_keys:
        raise ValueError(f"constraint_scan_val_lists has first dimension {num_constraint_scan_val_lists}, but constraint_scan_keys specifies {num_constraint_scan_keys} keys, which is not equal. Please make sure that constraint_scan_keys has as many keys as there are lists in constraint_scan_val_lists.")

    if with_mp:
        raise NotImplementedError("with_mp=True specified, but this functionality is not implemented yet for qec_cycle_efficiency_experiment; ignoring.")

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

    for circuit_physical in circuit_input:
        density_matrix_exact = None#DensityMatrix(circuit_physical)

        circuit_logical = LogicalCircuit.from_physical_circuit(circuit_physical, **qecc)

        # @TODO - determine whether this is a better data structure for this, including a better way to distinguish the data by circuit
        sub_data = {
            "physical_circuit": circuit_physical,
            "logical_circuit": circuit_logical,
            "density_matrix_exact": density_matrix_exact,
            "results": []
        }

        for constraint_model in constraint_models:
            circuit_logical = LogicalCircuit.from_physical_circuit(circuit_physical, **qecc)
            qec_cycle_indices = circuit_logical.optimize_qec_cycle_indices(constraint_model=constraint_model)
            circuit_logical.insert_qec_cycles(qec_cycle_indices=qec_cycle_indices)

            result = execute_circuits(circuit_logical, noise_model=noise_model_input, backend=backend, method=method, shots=shots)

            sub_data["results"].append({
                "constraint_model": constraint_model,
                "qec_layer_indices": qec_layer_indices,
                "result": result
            })

        all_data.append(sub_data)

    # Run save_progress once for good measure and then unregister save_progress so it doesn't clutter our exit routine
    save_progress()
    atexit.unregister(save_progress)

    return all_data

def qec_cycle_noise_scaling_experiment(circuit_input, noise_model_input, qecc, constraint_scan_keys, constraint_scan_val_lists, error_scan_keys, error_scan_val_lists, backend="aer_simulator", method="density_matrix", compute_exact=False, shots=1024, save_dir=None, save_filename=None):
    if isinstance(circuit_input, LogicalCircuit):
        raise NotImplementedError("LogicalCircuit inputs are not accepted because the original physical circuit(s) are also necessary for this experiment.")
    elif isinstance(circuit_input, QuantumCircuit):
        circuit_input = [circuit_input]
    elif hasattr(circuit_input, "__iter__"):
        raise NotImplementedError("Please provide a single circuit")
    elif callable(circuit_input):
        raise NotImplementedError("QuantumCircuit/LogicalCircuit callables are not accepted as inputs, please provide a constant QuantumCircuit/LogicalCircuit object or a list of such.")
    else:
        raise ValueError("Please provide a QuantumCircuit/LogicalCircuit input.")

    if isinstance(noise_model_input, NoiseModel):
        noise_model_factory = lambda c : noise_model_input
    elif hasattr(noise_model_input, "__iter__"):
        if len(noise_model_input) == len(configs):
            noise_model_callable_list = []
            for noise_model_input_element in noise_model_input:
                if isinstance(noise_model_input_element, NoiseModel):
                    noise_model_callable_list.append(noise_model_input_element)
                else:
                    raise ValueError("List provided for noise_model_input does not match length of configs list input; noise_model_input must either be constant, a callable, or a list.")

            noise_model_factory = lambda c : noise_model_callable_list[c]
        else:
            raise ValueError("List provided for noise_model_input does not match length of configs list - noise_model_input must either be constant, a callable, or a list with length matching that of configs list.")
    elif callable(noise_model_input):
        noise_model_factory = noise_model_input
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
            "results_physical": results_physical,
            "results_logical": results_logical,
        }

        # Benchmark physical circuit
        results_physical = noise_scaling_experiment(
            circuit_input=[circuit_input],
            noise_model_input=noise_model_input,
            error_scan_keys=error_scan_keys,
            error_scan_val_lists=error_scan_val_lists,
        )
        all_data[c]["results_physical"] = results_physical

        # Construct LogicalCircuit object
        circuit_logical = LogicalCircuit.from_physical_circuit(circuit_input, **qecc)

        # Apply QEC according to constraint model
        qec_cycle_indices = circuit_logical.optimize_qec_cycle_indices(constraint_model=constraint_model)
        circuit_logical.insert_qec_cycles(qec_cycle_indices=qec_cycle_indices)

        # Benchmark logical circuit
        results_logical = noise_scaling_experiment(
            circuit_input=[circuit_logical],
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

