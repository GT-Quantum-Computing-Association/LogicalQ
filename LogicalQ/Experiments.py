import os
import time
import copy
import atexit
import pickle
import itertools
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor as Pool

from .Logical import LogicalCircuit, LogicalStatevector, LogicalDensityMatrix
from .Execution import execute_circuits

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix

def _basic_experiment_core(task_id, circuit, noise_model, kwargs):
    """
    Basic core experiment function useful for multiprocessing
    """

    print(os.getpid(), "starting")
    result = execute_circuits(circuit, noise_model=noise_model, **kwargs)
    print(os.getpid(), "stopping")

    return task_id, result

# @TODO - implement more experiments

def circuit_scaling_experiment(circuit_input, noise_model_input=None, min_n_qubits=1, max_n_qubits=16, min_circuit_length=1, max_circuit_length=16, with_mp=True, save_dir=None, save_filename=None, **kwargs):
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

    if noise_model_input is None:
        # Default option which lets users skip the noise model input, especially if their backend is a hardware backend
        noise_model_factory = lambda n_qubits=None, circuit_length=None : None
    elif isinstance(noise_model_input, NoiseModel):
        if max_n_qubits != min_n_qubits:
            print("A constant noise model has been provided as the noise model factory, but a non-trivial range of qubit counts has also been provided. The number of qubits will not be scaled; if you would like for the number of qubits to be scaled, please provide a callable which takes the number of qubits, n_qubits, as an argument.")

        noise_model_factory = lambda n_qubits: noise_model_input
    elif callable(noise_model_input):
        noise_model_factory = noise_model_input
    else:
        raise ValueError("Please provide a NoiseModel object, a method for constructing NoiseModels, or None (default) if backend is a hardware backend.")

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
        circuit_dimensions_list = list(itertools.product(range(min_n_qubits, max_n_qubits+1), range(min_circuit_length, max_circuit_length+1)))
        circuit_list = [circuit_factory(n_qubits=n_qubits, circuit_length=circuit_length) for (n_qubits, circuit_length) in circuit_dimensions_list]
        noise_model_list = [noise_model_factory(n_qubits=n_qubits) for (n_qubits, circuit_length) in circuit_dimensions_list]

        exp_inputs_list = list(zip(
            circuit_dimensions_list,
            circuit_list,
            noise_model_list,
            [kwargs]*len(circuit_dimensions_list)
        ))

        cpu_count = os.process_cpu_count() or 1
        print(f"Applying multiprocessing to {len(exp_inputs_list)} samples across {cpu_count} CPUs")

        start = time.perf_counter()

        with Pool(cpu_count) as pool:
            mp_result = pool.map(
                _basic_experiment_core,
                *[list(exp_inputs) for exp_inputs in zip(*exp_inputs_list)],
            )

            # Unzip results
            circuit_map = dict(zip(circuit_dimensions_list, circuit_list))
            for (task_id, result) in mp_result:
                circuit = circuit_map[task_id]
                all_data[task_id[0]][task_id[1]] = (circuit, result[0])

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
                result = execute_circuits(circuit_nl, **kwargs)[0]

                # Save expectation values
                sub_data[circuit_length] = (circuit_nl, result)

            del noise_model_n

            all_data[n_qubits] = sub_data

        stop = time.perf_counter()

    print(f"Completed experiment in {stop-start} seconds")

    # Run save_progress once for good measure and then unregister save_progress so it doesn't clutter our exit routine
    save_progress()
    atexit.unregister(save_progress)

    return all_data

def noise_scaling_experiment(circuit_input, noise_model_input, error_scan_keys, error_scan_val_lists, basis_gates=None, compute_exact=False, exact_method=None, with_mp=False, save_dir=None, save_filename=None, **kwargs) -> dict:
    """Simulates physical and logical circuits across a range of noise models.

    Args:
        circuit_input: The circuit(s) to simulate. Accepts instances or lists of QuantumCircuit objects.
        noise_model_input: The noise model(s) to scan across. Accepts instances or lists of NoiseModel, or error dictionaries from which a NoiseModel can be constructed.
    
    Returns:
        all_data
        
    Raises:
        :class:`NotImplementedError`: if user attempts to provide noise_scaling_experiment with a circuit factory or noise model factory.
    """
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
                    kwargs
                )
                for nm, noise_model in enumerate(noise_models)
            ]

            cpu_count = os.process_cpu_count() or 1
            print(f"Applying multiprocessing to {len(exp_inputs_list)} samples across {cpu_count} CPUs")

            with Pool(cpu_count) as pool:
                mp_result = pool.map(
                    _basic_experiment_core,
                    *[list(exp_inputs) for exp_inputs in zip(*exp_inputs_list)],
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
                    if exact_method is not None and exact_method != "density_matrix":
                        raise Exception("User did not request exact density matrix computation")

                    # Compute exact density matrix
                    if isinstance(circuit_no_meas, LogicalCircuit):
                        density_matrix_exact = LogicalDensityMatrix(circuit_no_meas)
                    else:
                        density_matrix_exact = DensityMatrix(circuit_no_meas)
                except:
                    if exact_method is not None and exact_method != "statevector":
                        raise Exception("User did not request exact statevector computation") from e

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
                result = execute_circuits(circuit, noise_model=noise_model, **kwargs)

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

def qec_cycle_circuit_scaling_experiment(circuit_input, qecc, constraint_model=None, min_n_qubits=1, max_n_qubits=16, min_circuit_length=1, max_circuit_length=16, with_mp=True, save_dir=None, save_filename=None, **kwargs):
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
        raise NotImplementedError("with_mp=True specified, but this functionality is not implemented yet for qec_cycle_circuit_scaling_experiment; ignoring.")
    else:
        start = time.perf_counter()

        for n_qubits in range(min_n_qubits, max_n_qubits+1):
            for circuit_length in range(min_circuit_length, max_circuit_length+1):
                # Construct physical circuit
                circuit_nl_physical = circuit_factory(n_qubits=n_qubits, circuit_length=circuit_length)
                circuit_nl_physical_no_meas = circuit_nl_physical.remove_final_measurements(inplace=False)

                # Exact physical result
                # @TODO - replicate code from other experiments which have a fallback to the Statevector if an error occurs
                density_matrix_exact = DensityMatrix(circuit_nl_physical_no_meas)

                # Noisy physical result
                result_physical = execute_circuits(circuit_nl_physical, **kwargs)[0]

                # Construct LogicalCircuit
                circuit_nl_logical = LogicalCircuit.from_physical_circuit(circuit_nl_physical, **qecc)

                # Apply QEC according to constraint model
                qec_cycle_indices = circuit_nl_logical.optimize_qec_cycle_indices(constraint_model=constraint_model)
                circuit_nl_logical.insert_qec_cycles(qec_cycle_indices=qec_cycle_indices)

                # Noisy logical result
                result_logical = execute_circuits(circuit_nl_logical, **kwargs)[0]

                # @TODO - determine whether this is a better data structure for this, including a better way to distinguish the data by circuit
                sub_data = {
                    "physical_circuit": circuit_nl_physical,
                    "logical_circuit": circuit_nl_logical,
                    "density_matrix_exact": density_matrix_exact,
                    "constraint_model": constraint_model,
                    "result_physical": result_physical,
                    "result_logical": result_logical
                }

                # Save expectation values
                all_data[n_qubits][circuit_length] = sub_data

        stop = time.perf_counter()

    print(f"Completed experiment in {stop-start} seconds")

    # Run save_progress once for good measure and then unregister save_progress so it doesn't clutter our exit routine
    save_progress()
    atexit.unregister(save_progress)

    return all_data

def qec_cycle_noise_scaling_experiment(circuit_input, noise_model_input, qecc, constraint_scan_keys, constraint_scan_val_lists, error_scan_keys, error_scan_val_lists, compute_exact=False, with_mp=False, save_dir=None, save_filename=None, **kwargs):
    """
    An extension of the `noise_scaling_experiment` method specifically for comparisons with QEC cycles scheduled given constraint models.

    Data format:
    ```
    all_data =
    [ # for each constraint model:
        {
            "circuit_physical": QuantumCircuit,
            "circuit_logical": LogicalCircuit,
            "constraint_model": dict,
            "results_physical":
            [ # for each circuit:
                {
                    "circuit": circuit,
                    "density_matrix_exact": density_matrix_exact,
                    "statevector_exact": statevector_exact,
                    "results":
                    [
                        {
                            "error_dict": error_dict,
                            "result": result
                        },
                    ]
                },
            ],
            "results_logical":
            [
                {
                    "circuit": circuit,
                    "density_matrix_exact": density_matrix_exact,
                    "statevector_exact": statevector_exact,
                    "results":
                    [
                        {
                            "error_dict": error_dict,
                            "result": result
                        },
                    ]
                },
            ],
        },
    ]
    ```
    """

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
        all_data[c]["circuit_physical"] = circuit_input
        all_data[c]["constraint_model"] = constraint_model

        # Benchmark physical circuit
        results_physical = noise_scaling_experiment(
            circuit_input=circuit_input,
            noise_model_input=noise_model_input,
            error_scan_keys=error_scan_keys,
            error_scan_val_lists=error_scan_val_lists,
            compute_exact=compute_exact,
            **kwargs
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
            compute_exact=compute_exact,
            exact_method='statevector',
            **kwargs
        )
        all_data[c]["results_logical"] = results_logical
        
    # Run save_progress once for good measure and then unregister save_progress so it doesn't clutter our exit routine
    save_progress()
    atexit.unregister(save_progress)

    return all_data

