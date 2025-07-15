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

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.providers import Backend
from qiskit_aer.noise import NoiseModel

# General function to benchmark a circuit using a noise model
def execute_circuits(circuits, backend=None, noise_model=None, noise_params=None, method="statevector", basis_gates=None, optimization_level=0, shots=1024):
    # Resolve circuits
    if hasattr(circuits, "__iter__"):
        error_out = False
        error_str_list = []
        for c, circuit in enumerate(circuits):
            if not isinstance(circuit, QuantumCircuit):
                error_out = True
                error_str_list.append(f"Index {c}: {circuit} (type: {type(circuit)})")
        if error_out:
            raise TypeError("Iterable provided for circuits contains non-circuit object(s):\n" + "\n".join(error_str_list))
    elif isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    else:
        raise TypeError("")

    # Resolve backend
    if backend is None:
        # Resolve noise_model or noise_params
        if noise_model is None:
            if noise_params is not None:
                # If noise_params are provided but not a noise_model or backend, then construct noise model based on the provided parameters
                noise_model = construct_noise_model(basis_gates=basis_gates, circuit.num_qubits, **noise_params)

                backend = "aer_simulator"
            else:
                raise ValueError("One of backend, noise_model, or noise_params must be provided")

    if isinstance(backend, str):
        if backend == "aer_simulator":
            backend = AerSimulator(method=method, noise_model=noise_model, basis_gates=basis_gates, coupling_map=coupling_map)

            if coupling_map is None:
                # Create a fully-coupled map by default since we don't care about non-fully-coupled hardware modalities
                fully_coupled_map = [list(pair) for pair in itertools.product(range(circuit.num_qubits), range(circuit.num_qubits))]
        else:
            service = QiskitRuntimeService()
            backend = service.get_backend(backend)
    elif isinstance(backend, (AerSimulator, Backend)):
        # @TODO - handle this case better
        backend = backend
    else:
        raise TypeError(f"backend must be None, 'aer_simulator', the name of a backend, or an instance of AerSimulator or Backend, not {type(backend)}")

    # Transpile circuit
    # Method defaults to optimization off to preserve form of benchmarking circuit and full QEC
    circuits_transpiled = transpile(circuits, backend=backend, optimization_level=optimization_level)
    result = backend.run(circuits_transpiled, shots=shots).result()

    return result

# Core experiment function useful for multiprocessing
def _experiment_core(circuit, noise_model, n_qubits, circuit_length, method, shots):
    result = execute_circuits(circuit, noise_model=noise_model, method=method, shots=shots)

    return n_qubits, circuit_length, result

# @TODO - implement experiments
def circuit_scaling_experiment(circuit_input, noise_model_input, min_n_qubits=1, max_n_qubits=50, min_circuit_length=1, max_circuit_length=50, method="statevector", shots=1024, with_mp=True, save_filename=None):
    if isinstance(circuit_input, QuantumCircuit) or isinstance(circuit_input, LogicalCircuit):
        if max_n_qubits != min_n_qubits+1:
            print("A constant circuit has been provided as the circuit factory, but a non-trivial range of qubit counts and/or circuit lengths has also been provided, so the fixed input will not be scaled. If you would like for the number of qubits to be scaled, please provide a callable which takes in as an argument the number of qubits, n_qubits. If you would like for the circuit length to be scaled, please provide a callable which takes in as an argument the circuit length, circuit_length.")

        circuit_factory = lambda n_qubits, circuit_length: circuit_input
    elif callable(circuit_input):
        circuit_factory = circuit_input
    else:
        raise ValueError("Please provide a QuantumCircuit/LogicalCircuit object or a method for constructing QuantumCircuits/LogicalCircuits.")

    if isinstance(noise_model_input, NoiseModel):
        if max_n_qubits != min_n_qubits+1:
            print("A constant noise model has been provided as the noise model factory, but a non-trivial range of qubit counts has also been provided. The number of qubits will not be scaled; if you would like for the number of qubits to be scaled, please provide a callable which takes in as an argument the number of qubits, n_qubits.")

        noise_model_factory = lambda n_qubits: noise_model_input
    elif callable(noise_model_input):
        noise_model_factory = noise_model_input
    else:
        raise ValueError("Please provide a NoiseModel object or a method for constructing NoiseModels.")

    # Form a dict of dicts with the first layer (n_qubits) initialized to make later access faster and more reliable in parallel
    all_data = dict(zip(range(min_n_qubits, max_n_qubits+1), [{}]*(max_n_qubits+1-min_n_qubits)))

    # saving boilerplate
    if save_filename is None:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_filename = f"circuit_scaling_{date_str}.pkl"
    save_file = open(save_filename, "wb")
    def save_progress():
        pickle.dump(all_data, save_file, protocol=5)
    atexit.register(save_progress)

    if with_mp:
        exp_inputs_list = [
            (
                circuit_factory(n_qubits=n_qubits, circuit_length=circuit_length),
                noise_model_factory(n_qubits=n_qubits), n_qubits, circuit_length, method, shots
            )
            for (n_qubits, circuit_length) in itertools.product(range(min_n_qubits, max_n_qubits+1), range(min_circuit_length, max_circuit_length+1))
        ]

        cpu_count = len(os.sched_getaffinity(0))
        batch_size = max(int(np.ceil((max_n_qubits+1-min_n_qubits)*(max_circuit_length+1-min_circuit_length)/cpu_count)), 1)
        print(f"Applying mulitprocessing to {len(exp_inputs_list)} samples in batches of maximum size {batch_size} across {cpu_count} CPUs")

        start = time.perf_counter()

        with Pool(cpu_count) as pool:
            results = pool.map(
                _experiment_core,
                *[list(exp_inputs) for exp_inputs in zip(*exp_inputs_list)],
                chunksize=batch_size
            )

            # Unzip results
            for result in results:
                all_data[result[0]][result[1]] = result[2]

        stop = time.perf_counter()
    else:
        start = time.perf_counter()

        for n_qubits in range(min_n_qubits, max_n_qubits+1):
            sub_data = {}

            # We need a new noise model for each qubit count
            noise_model_n = noise_model_factory(n_qubits=n_qubits)

            for circuit_length in range(min_circuit_length, max_circuit_length+1):
                # Construct circuit and benchmark noise
                circuit_nl = circuit(n_qubits=n_qubits, circuit_length=circuit_length)
                result = execute_circuits(circuit_nl, noise_model=noise_model_n, method=method, shots=shots)

                # Save expectation values
                sub_data[circuit_length] = result

                del circuit_nl

            del noise_model_n

            all_data[n_qubits] = sub_data

        stop = time.perf_counter()

    print(f"Completed experiment in {stop-start} seconds")

    # run save_progress once for good measure and then unregister save_progress so it doesn't clutter our exit routine
    save_progress()
    save_file.close()
    atexit.unregister(save_progress)

    return all_data

def noise_scaling_experiment(circuit_inputs, noise_model_inputs, error_scan_keys, error_scan_val_lists, noise_qubits=None, basis_gates=None, method="density_matrix", compute_exact=False, shots=1024, with_mp=False, save_filename=None):
    if isinstance(circuit_input, QuantumCircuit) or isinstance(circuit_input, LogicalCircuit):
        circuit_factory = lambda : circuit_input
    elif callable(circuit_input):
        raise NotImplementedError("QuantumCircuit/LogicalCircuit callables are not yet accepted as inputs, please provide a constant QuantumCircuit/LogicalCircuit object.")
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
    elif callable(noise_model):
        raise NotImplementedError("NoiseModel callables are not yet accepted as inputs, please provide a constant NoiseModel object.")
    else:
        print("No base NoiseModel inputted, using an ideal NoiseModel as a base.")
        def noise_model_factory(error_dict):
            # @TODO - test compatibility between our error_dicts and the specification expected by the NoiseModel.from_dict method
            noise_model = NoiseModel.from_dict(error_dict)

            if basis_gates is not None and len(basis_gates) > 0:
                noise_model.add_basis_gates(basis_gates)

            return noise_model

    num_error_scan_keys = len(error_scan_keys)
    num_error_scan_vals = np.shape(error_scan_val_lists)[-1]
    if num_error_scan_vals != num_error_scan_keys:
        raise ValueError(f"error_scan_val_lists has last dimension {num_error_scan_vals}, but error_scan_keys specifies {num_error_scan_keys} keys, which is not equal. Please make sure that error_scan_keys has as many keys as there are values in each list of error_scan_val_lists.")

    if with_mp:
        raise NotImplementedError("with_mp=True specified, but this functionality is not implemented yet for noise_scaling_experiment; ignoring.")

    # Construct config dictionaries
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
    all_data = [{}]*len(circuit_inputs)

    # saving boilerplate
    if save_filename is None:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_filename = f"noise_scaling_{date_str}.pkl"
    save_file = open(save_filename, "wb")
    def save_progress():
        pickle.dump(all_data, save_file, protocol=5)
    atexit.register(save_progress)

    if with_mp:
        exp_inputs_list = [
            (
                # circuit_factory(n_qubits=n_qubits, circuit_length=circuit_length),
                # noise_model_factory(n_qubits=n_qubits), n_qubits, circuit_length, method, shots
            )
            # for (n_qubits, circuit_length) in itertools.product(range(min_n_qubits, max_n_qubits+1), range(min_circuit_length, max_circuit_length+1))
        ]

        cpu_count = len(os.sched_getaffinity(0))
        batch_size = max(int(np.ceil((max_n_qubits+1-min_n_qubits)*(max_circuit_length+1-min_circuit_length)/cpu_count)), 1)
        print(f"Applying mulitprocessing to {len(exp_inputs_list)} samples in batches of maximum size {batch_size} across {cpu_count} CPUs")

        start = time.perf_counter()

        with Pool(cpu_count) as pool:
            results = pool.map(
                _experiment_core,
                *[list(exp_inputs) for exp_inputs in zip(*exp_inputs_list)],
                chunksize=batch_size
            )

            sub_data["results"].append({
                "error_dict": error_dict,
                "result": result
            })

            # Unzip results
            for result in results:
                all_data[c] = sub_data

        stop = time.perf_counter()
    else:
        start = time.perf_counter()

        for c, circuit_input in enumerate(circuit_inputs):
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
                "qc": circuit_input,
                "density_matrix_exact": density_matrix_exact,
                "statevector_exact": statevector_exact,
                "results": []
            }

            for error_dict, noise_model in zip(error_dict, noise_models):
                result = execute_circuits(circuit_input, noise_model=noise_model, method=method, shots=shots)

                sub_data["results"].append({
                    "error_dict": error_dict,
                    "result": result
                })

            all_data[c] = sub_data

        stop = time.perf_counter()

    print(f"Completed experiment in {stop-start} seconds")

    # run save_progress once for good measure and then unregister save_progress so it doesn't clutter our exit routine
    save_progress()
    save_file.close()
    atexit.unregister(save_progress)

    return all_data

def qec_cycle_efficiency_experiment(circuit_inputs, noise_model_input, config_scan_keys, config_scan_val_lists, method="density_matrix", shots=1024, with_mp=False, save_filename=None):
    num_config_scan_keys = len(config_scan_keys)
    num_config_scan_vals = np.shape(config_scan_val_lists)[-1]
    if num_config_scan_vals != num_config_scan_keys:
        raise ValueError(f"config_scan_val_lists has last dimension {num_config_scan_vals}, but config_scan_keys specifies {num_config_scan_keys} keys, which is not equal. Please make sure that config_scan_keys has as many keys as there are values in each list of config_scan_val_lists.")

    # Construct config dictionaries
    config_scan_val_prods = itertools.product(*config_scan_val_lists)
    configs = []
    for config_scan_vals in config_scan_val_prods:
        mapping = zip(config_scan_keys, config_scan_vals)
        configs.append(dict(mapping))

    if isinstance(circuit_input, LogicalCircuit):
        circuit_factory = lambda c : copy.deepcopy(circuit_input)
    elif isinstance(circuit_input, QuantumCircuit):
        circuit_factory = lambda c : LogicalCircuit.from_physical_circuit(circuit_input)
    elif hasattr(circuit_input, "__iter__"):
        if len(circuit_input) == len(configs):
            circuit_callable_list = []
            for circuit_input_element in circuit_input:
                if isinstance(circuit_input_element, LogicalCircuit):
                    circuit_callable_list.append(copy.deepcopy(circuit_input_element))
                elif isinstance(circuit_input_element, QuantumCircuit):
                    circuit_callable_list.append(LogicalCircuit.from_physical_circuit(circuit_input_element))
                else:
                    raise ValueError("List provided for circuit_input does not match length of configs list input; circuit_input must either be constant, a callable, or a list.")

            circuit_factory = lambda c : circuit_callable_list[c]
        else:
            raise ValueError("List provided for circuit_input does not match length of configs list - circuit_input must either be constant, a callable, or a list with length matching that of configs list.")
    elif callable(circuit_input):
        # @TODO - make sure this accepts the config c (or is actually just constant)
        circuit_factory = circuit_input
    else:
        raise TypeError("Please provide a QuantumCircuit/LogicalCircuit object, a method for constructing QuantumCircuit/LogicalCircuit, or a list of either.")

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
        # @TODO - make sure this accepts the config c (or is actually just constant)
        noise_model_factory = noise_model_input
    else:
        raise ValueError("Please provide a NoiseModel object, a method for constructing NoiseModels, or a list of either.")

    if with_mp:
        raise NotImplementedError("with_mp=True specified, but this functionality is not implemented yet for qec_cycle_efficiency_experiment; ignoring.")

    all_data = []

    # saving boilerplate
    if save_filename is None:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_filename = f"qec_cycle_efficiency_{date_str}.pkl"
    save_file = open(save_filename, "wb")
    def save_progress():
        pickle.dump(all_data, save_file, protocol=5)
    atexit.register(save_progress)

    for circuit_input in circuit_inputs:
        # Compute exact result
        density_matrix_exact = DensityMatrix(circuit_input)

        # Construct LogicalCircuit from physical circuit
        lqc = LogicalCircuit.from_physical_circuit(circuit_input)

        # @TODO - determine whether this is a better data structure for this, including a better way to distinguish the data by circuit
        sub_data = {
            "qc": circuit_input,
            "lqc": lqc,
            "density_matrix_exact": density_matrix_exact,
            "results": []
        }

        for config in configs:
            qec_layer_idxs = lqc.inject_qec_cycles(**config)

            result = execute_circuits(circuit_input, noise_model=noise_model_input, method=method, shots=shots)

            sub_data["results"].append({
                "config": config,
                "qec_layer_idxs": qec_layer_idxs,
                "result": result
            })

        all_data.append(sub_data)

    # run save_progress once for good measure and then unregister save_progress so it doesn't clutter our exit routine
    save_progress()
    save_file.close()
    atexit.unregister(save_progress)

    return all_data

def qec_cycle_noise_scaling_experiment(circuit_input, noise_model_input, configs, error_scan_keys, error_scan_val_lists, logical_kwargs=None, save_filename=None):
    if isinstance(circuit_input, LogicalCircuit):
        circuit_factory = lambda c : copy.deepcopy(circuit_input)
    elif isinstance(circuit_input, QuantumCircuit):
        circuit_factory = lambda c : LogicalCircuit.from_physical_circuit(circuit_input)
    elif hasattr(circuit_input, "__iter__"):
        if len(circuit_input) == len(configs):
            circuit_callable_list = []
            for circuit_input_element in circuit_input:
                if isinstance(circuit_input_element, LogicalCircuit):
                    circuit_callable_list.append(copy.deepcopy(circuit_input_element))
                elif isinstance(circuit_input_element, QuantumCircuit):
                    circuit_callable_list.append(LogicalCircuit.from_physical_circuit(circuit_input_element))
                else:
                    raise ValueError("List provided for circuit_input does not match length of configs list input; circuit_input must either be constant, a callable, or a list.")

            circuit_factory = lambda c : circuit_callable_list[c]
        else:
            raise ValueError("List provided for circuit_input does not match length of configs list - circuit_input must either be constant, a callable, or a list with length matching that of configs list.")
    elif callable(circuit_input):
        # @TODO - make sure this accepts the config c (or is actually just constant)
        circuit_factory = circuit_input
    else:
        raise TypeError("Please provide a QuantumCircuit/LogicalCircuit object, a method for constructing QuantumCircuit/LogicalCircuit, or a list of either.")

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
        # @TODO - make sure this accepts the config c (or is actually just constant)
        noise_model_factory = noise_model_input
    else:
        raise ValueError("Please provide a NoiseModel object, a method for constructing NoiseModels, or a list of either.")

    if logical_kwargs is None:
        # Default to the Steane code
        logical_kwargs = {
            "label": (7,1,3),
            "stabilizer_tableau": [
                "XXXXIII",
                "IXXIXXI",
                "IIXXIXX",
                "ZZZZIII",
                "IZZIZZI",
                "IIZZIZZ",
            ]
        }

    all_data = []

    # saving boilerplate
    if save_filename is None:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_filename = f"qec_cycle_noise_scaling_{date_str}.pkl"
    save_file = open(save_filename, "wb")
    def save_progress():
        pickle.dump(all_data, save_file, protocol=5)
    atexit.register(save_progress)

    for c, config in enumerate(configs):
        # Build LogicalCircuit with the desired QEC config
        lqc = circuit_factory(c)

        # Build NoiseModel for the desired QEC config
        noise_model = noise_model_factor(c)

        sub_data_results = noise_scaling_experiment(
            circuit_inputs=[lqc],
            noise_model_input=noise_model,
            error_scan_keys=error_scan_keys,
            error_scan_val_lists=error_scan_val_lists,
        )

        sub_data = {
            "config": config,
            "results": sub_data_results
        }

        all_data.append(sub_data)

    # run save_progress once for good measure and then unregister save_progress so it doesn't clutter our exit routine
    save_progress()
    save_file.close()
    atexit.unregister(save_progress)

    return all_data

