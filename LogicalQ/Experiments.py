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
from .Execution import execute_circuits

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
from qiskit_aer.noise import NoiseModel
from qiskit_addon_utils.slicing import slice_by_depth

from typing import TYPE_CHECKING
from typing import Iterable

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


##############################################
##### QEC cycle scheduling comparisons #####
##############################################

# Scheduling methods which are compared against the optimized schedule; "weinstein" and "abu_nada" are
# fixed-rate policies due to Weinstein (2013, 2014, 2015) and Abu-Nada et al. (2017) respectively, both of
# which perform error correction at a constant interval and differ only in how that interval is chosen
comparison_schedule_types = [
    "no_qec",
    "optimized",
    "random",
    "fixed",
    "weinstein",
    "abu_nada",
]

def create_qec_schedules(circuit, schedule_type="optimized", n_qec_cycles=None, interval=None, constraint_model=None, rng=None) -> dict:
    """Construct QEC cycle indices for a circuit according to the specified scheduling method.

    Args:
        circuit: The circuit for which to construct a schedule; indices refer to positions in its data.
        schedule_type: The scheduling method to use, from `comparison_schedule_types`.
        n_qec_cycles: The number of QEC cycles the schedule should contain. When specified, every method emits exactly this many cycles so that cycle count is not a confounding variable in comparisons.
        interval: The interval at which to insert cycles, used only when `n_qec_cycles` is not specified.
        constraint_model: Constraint model, required only by the optimized method.
        rng: Random generator used by the random method, so that comparisons are reproducible.

    Returns:
        Dictionary mapping logical qubit index to a list of QEC cycle indices.

    Raises:
        :class:`ValueError`: if `schedule_type` is not a recognized scheduling method.
    """

    if rng is None:
        rng = np.random.default_rng()

    # Count non-trivial operations, which are the positions at which a cycle may be inserted
    non_trivial_ops = [inst for inst in circuit.data if inst.operation.name not in ["barrier", "measure"]]
    circuit_length = len(non_trivial_ops)

    if schedule_type == "no_qec":
        # Explicit baseline - the empty schedule, against which every other method must be compared in order
        # to establish whether the optimum is interior
        return {0: []}
    elif schedule_type == "optimized":
        # Handled by the caller via optimize_qec_cycle_indices, which needs the LogicalCircuit itself
        return {}
    elif schedule_type == "random":
        if n_qec_cycles is None:
            n_qec_cycles = max(1, circuit_length//4)

        n_qec_cycles = min(n_qec_cycles, max(0, circuit_length-1))
        if n_qec_cycles <= 0:
            return {0: []}

        indices = sorted([int(i) for i in rng.choice(range(1, circuit_length), size=n_qec_cycles, replace=False)])

        return {0: indices}
    elif schedule_type in ["fixed", "weinstein", "abu_nada"]:
        # All three are fixed-rate policies; when a cycle count is specified, the interval is chosen so that
        # exactly that many cycles are emitted, which is what makes the comparison count-matched
        # @NOTE - at a matched cycle count these three policies coincide, because what distinguishes them is
        #         how each chooses its rate; run with interval instead to compare the policies themselves
        if n_qec_cycles is not None and n_qec_cycles <= 0:
            # A matched count of zero, e.g. when the optimized schedule inserts no cycles, is the empty schedule
            return {0: []}
        elif n_qec_cycles is not None:
            indices = [int(np.round((j+1) * circuit_length/(n_qec_cycles+1))) for j in range(n_qec_cycles)]
            indices = sorted(set([min(max(index, 1), max(circuit_length-1, 1)) for index in indices]))

            return {0: indices}

        if interval is None:
            raise ValueError(f"Scheduling method '{schedule_type}' requires either n_qec_cycles or interval to be specified")

        return {0: [int(i) for i in range(interval, circuit_length, interval)]}
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

def compute_circuit_duration(circuit, hardware_model=None) -> float:
    """Estimate the total duration of a circuit from the gate times of a hardware model.

    Duration is the axis along which schedules with equal cycle counts are compared, because two schedules may place the same number of cycles at very different points and thus accumulate different idle time.

    Args:
        circuit: The circuit whose duration to estimate.
        hardware_model: Hardware model providing gate times, keyed by gate arity and gate name.

    Returns:
        The estimated duration in the time units used by the hardware model, or 0.0 if no gate times exist.
    """

    if hardware_model is None:
        return 0.0

    # Collect gate times, which are specified per qubit specifier, then per gate arity, then per gate name
    gate_times = {}
    for qubit_noise_params in hardware_model.get("noise_params", {}).values():
        for param_n_qubits, n_qubit_data in qubit_noise_params.get("gate_time", {}).items():
            for gate, gate_time in n_qubit_data.items():
                gate_times[(param_n_qubits, gate)] = gate_time

    if len(gate_times) == 0:
        return 0.0

    def instruction_duration(instruction):
        # Recurse through boxes and control flow so that the contents of logical gates and QEC cycles count
        if instruction.operation.name == "barrier":
            return 0.0

        if instruction.is_control_flow():
            duration = 0.0
            for param in instruction.operation.params:
                if isinstance(param, QuantumCircuit):
                    for sub_instruction in param.data:
                        duration += instruction_duration(sub_instruction)

            return duration

        n_qubits = len(instruction.qubits)
        if (n_qubits, instruction.operation.name) in gate_times:
            return gate_times[(n_qubits, instruction.operation.name)]
        elif (n_qubits, "all") in gate_times:
            return gate_times[(n_qubits, "all")]

        return 0.0

    return float(sum([instruction_duration(instruction) for instruction in circuit.data]))

def _non_data_qubits(logical_circuit) -> list:
    """Return the indices of every qubit in a LogicalCircuit which is not a data qubit."""

    non_data_qubits = []
    count = 0
    for qreg in logical_circuit.qregs:
        if qreg in logical_circuit.logical_qregs:
            count += qreg.size
        else:
            non_data_qubits.extend(list(range(count, count+qreg.size)))
            count += qreg.size

    return non_data_qubits

def _exact_logical_state(logical_circuit):
    """Compute the exact state of the data qubits of a noiseless logical circuit.

    The reference is the noiseless output of the logical circuit rather than of the physical circuit, so that algorithmic error is not counted as noise.
    """

    from qiskit.transpiler import PassManager
    from .Transpilation.ClearQEC import ClearQEC
    from .Transpilation.UnBox import UnBox

    circuit = copy.deepcopy(logical_circuit)

    pm_unbox = PassManager([ClearQEC(), UnBox()])
    while "box" in circuit.count_ops():
        circuit = pm_unbox.run(circuit)

    return partial_trace(Statevector(circuit), _non_data_qubits(logical_circuit))

def qec_cycle_schedule_comparison_experiment(circuit_input, qecc, constraint_model, schedule_types=None, n_random_repeats=8, n_repeats=1, fidelity_method="logical_counts", hardware_model=None, seed=None, save_dir=None, save_filename=None, **kwargs) -> dict:
    """Compare QEC cycle scheduling methods at a matched cycle count.

    Every method is given the same number of QEC cycles as the optimized schedule produces, so that any difference in fidelity is attributable to where the cycles are placed rather than to how many there are. The no-QEC baseline is the one exception, since it exists precisely to establish whether performing any error correction at all is beneficial in the operating regime.

    Args:
        circuit_input: Physical circuits to schedule, given as a dictionary keyed by circuit name.
        qecc: The QECC dictionary, splatted into LogicalCircuit construction.
        constraint_model: Constraint model used by the optimized scheduling method.
        schedule_types: Scheduling methods to compare; defaults to `comparison_schedule_types`.
        n_random_repeats: Number of draws over which the random method is averaged.
        n_repeats: Number of repeats for every method, used to estimate run-to-run variation.
        fidelity_method: Passed through to `compute_schedule_fidelity`.
        hardware_model: Hardware model used both for execution and for estimating circuit durations.
        seed: Seed for the random scheduling method, so that comparisons are reproducible.

    Returns:
        Dictionary of per-circuit results, each containing one record per scheduling method.
    """

    from .Analysis import compute_schedule_fidelity

    if schedule_types is None:
        schedule_types = comparison_schedule_types

    if isinstance(circuit_input, QuantumCircuit):
        circuit_input = {circuit_input.name: circuit_input}

    rng = np.random.default_rng(seed)

    all_data = {"circuits": {}, "summary": {}}

    # Prepare to save progress in the event of program termination
    if save_dir is None:
        save_dir = "./data/"
    if save_filename is None:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_filename = f"qec_cycle_schedule_comparison_{date_str}.pkl"
    save_file = open(save_dir + save_filename, "wb")
    def save_progress():
        pickle.dump(all_data, save_file, protocol=5)
        save_file.close()
    atexit.register(save_progress)

    start = time.perf_counter()

    for circuit_name, circuit in circuit_input.items():
        print(f"\nProcessing {circuit_name}...")

        # Remove measurements so that the circuit can be scheduled and an exact reference computed
        circuit_no_meas = circuit.remove_final_measurements(inplace=False)

        circuit_depth = len(slice_by_depth(circuit_no_meas, 1))
        print(f"  Circuit depth: {circuit_depth}")

        # Determine the matched cycle count from the optimized schedule, which is the count every other
        # method is held to
        lqc_reference = LogicalCircuit.from_physical_circuit(circuit_no_meas, **qecc)
        qec_cycle_indices_optimized = lqc_reference.optimize_qec_cycle_indices(constraint_model=constraint_model)
        n_qec_cycles = len(qec_cycle_indices_optimized.get(0, []))
        print(f"  Matched QEC cycle count (from optimized schedule): {n_qec_cycles}")

        circuit_results = {
            "circuit_depth": circuit_depth,
            "n_qec_cycles_matched": n_qec_cycles,
            "methods": {},
        }

        exact_state = None
        if fidelity_method == "density_matrix":
            exact_state = _exact_logical_state(LogicalCircuit.from_physical_circuit(circuit_no_meas, **qecc))
        circuit_results["exact_state"] = exact_state

        for schedule_type in schedule_types:
            print(f"  Running {schedule_type} scheduling...")

            # The random method is averaged over several draws, since a single draw is not a control
            n_draws = n_random_repeats if schedule_type == "random" else n_repeats

            fidelities = []
            durations = []
            n_qec_cycles_actual = []
            for _ in range(n_draws):
                lqc = LogicalCircuit.from_physical_circuit(circuit_no_meas, **qecc)

                if schedule_type == "optimized":
                    qec_cycle_indices = copy.deepcopy(qec_cycle_indices_optimized)
                else:
                    qec_cycle_indices = create_qec_schedules(circuit_no_meas, schedule_type, n_qec_cycles=n_qec_cycles, constraint_model=constraint_model, rng=rng)

                lqc.insert_qec_cycles(qec_cycle_indices=qec_cycle_indices)
                lqc.measure_all()

                durations.append(compute_circuit_duration(lqc, hardware_model=hardware_model))
                n_qec_cycles_actual.append(len(qec_cycle_indices.get(0, [])))

                result = execute_circuits(lqc, hardware_model=hardware_model, save_density_matrix=(fidelity_method == "density_matrix"), **kwargs)[0]

                fidelities.append(compute_schedule_fidelity(result, lqc, exact_state=exact_state, fidelity_method=fidelity_method))

            circuit_results["methods"][schedule_type] = {
                "fidelity": float(np.mean(fidelities)),
                "fidelity_std": float(np.std(fidelities)),
                "fidelities": fidelities,
                "n_repeats": n_draws,
                "num_qec": int(np.mean(n_qec_cycles_actual)),
                "duration": float(np.mean(durations)),
            }

        # Report every method against the no-QEC baseline, which establishes whether the optimum is interior
        if "no_qec" in circuit_results["methods"]:
            fidelity_no_qec = circuit_results["methods"]["no_qec"]["fidelity"]
            if fidelity_no_qec and fidelity_no_qec > 0:
                for method_results in circuit_results["methods"].values():
                    method_results["vs_no_qec"] = method_results["fidelity"]/fidelity_no_qec

        all_data["circuits"][circuit_name] = circuit_results

    stop = time.perf_counter()

    print(f"Completed experiment in {stop-start} seconds")

    save_progress()
    atexit.unregister(save_progress)

    return all_data

def qec_cycle_schedule_noise_scaling_experiment(circuit_input, qecc, constraint_model, hardware_model, noise_scan_key, noise_scan_values, schedule_types=None, save_dir=None, save_filename=None, **kwargs) -> dict:
    """Compare scheduling methods as a function of a noise parameter.

    The constraint model and the schedule are held fixed at the values obtained for the unperturbed hardware model while the noise parameter is scanned, so that the result measures whether a schedule fitted at one operating point continues to perform away from it.

    Args:
        circuit_input: Physical circuits to schedule, given as a dictionary keyed by circuit name.
        qecc: The QECC dictionary, splatted into LogicalCircuit construction.
        constraint_model: Constraint model used by the optimized scheduling method.
        hardware_model: Base hardware model, perturbed at each point of the scan.
        noise_scan_key: Path of keys into the hardware model identifying the parameter to scan.
        noise_scan_values: Values which the scanned parameter takes.
        schedule_types: Scheduling methods to compare; defaults to `comparison_schedule_types`.

    Returns:
        Dictionary mapping each scanned value to the comparison results at that value.
    """

    if schedule_types is None:
        schedule_types = comparison_schedule_types

    all_data = {}

    if save_dir is None:
        save_dir = "./data/"
    if save_filename is None:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_filename = f"qec_cycle_schedule_noise_scaling_{date_str}.pkl"
    save_file = open(save_dir + save_filename, "wb")
    def save_progress():
        pickle.dump(all_data, save_file, protocol=5)
        save_file.close()
    atexit.register(save_progress)

    for noise_scan_value in noise_scan_values:
        print(f"\n=== {'/'.join([str(key) for key in noise_scan_key])} = {noise_scan_value} ===")

        # Perturb a copy of the hardware model, leaving the constraint model and schedule untouched
        hardware_model_scan = copy.deepcopy(hardware_model)
        scan_target = hardware_model_scan
        for key in list(noise_scan_key)[:-1]:
            scan_target = scan_target[key]
        scan_target[list(noise_scan_key)[-1]] = noise_scan_value

        all_data[noise_scan_value] = qec_cycle_schedule_comparison_experiment(circuit_input, qecc, constraint_model, schedule_types=schedule_types, hardware_model=hardware_model_scan, save_dir=save_dir, **kwargs)

    save_progress()
    atexit.unregister(save_progress)

    return all_data

def noise_parameter_comparison_experiment(circuit_input, qecc, constraint_model, hardware_model, noise_parameter_groups, with_qec=True, fidelity_method="logical_counts", save_dir=None, save_filename=None, **kwargs) -> dict:
    """Compare circuit performance across isolated noise channels, with and without QEC.

    Each group applies a single noise channel in isolation so that the contribution of each physical error source can be attributed, rather than only the aggregate effect of the full hardware model.

    Args:
        circuit_input: Physical circuits to evaluate, given as a dictionary keyed by circuit name.
        qecc: The QECC dictionary, splatted into LogicalCircuit construction.
        constraint_model: Constraint model used to schedule QEC cycles when `with_qec` is True.
        hardware_model: Base hardware model whose device_info is reused by every group.
        noise_parameter_groups: Dictionary mapping group name to the noise_params dictionary for that group.
        with_qec: Whether to additionally evaluate a logical circuit with optimized QEC scheduling.
        fidelity_method: Passed through to `compute_schedule_fidelity`.

    Returns:
        Dictionary of per-circuit results, each containing one record per noise parameter group.
    """

    from .Analysis import compute_schedule_fidelity, calculate_state_probability

    if isinstance(circuit_input, QuantumCircuit):
        circuit_input = {circuit_input.name: circuit_input}

    all_data = {"circuits": {}}

    if save_dir is None:
        save_dir = "./data/"
    if save_filename is None:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_filename = f"noise_parameter_comparison_{date_str}.pkl"
    save_file = open(save_dir + save_filename, "wb")
    def save_progress():
        pickle.dump(all_data, save_file, protocol=5)
        save_file.close()
    atexit.register(save_progress)

    for circuit_name, circuit in circuit_input.items():
        print(f"\nProcessing {circuit_name}...")

        circuit_no_meas = circuit.remove_final_measurements(inplace=False)

        circuit_results = {"groups": {}}

        for group_name, noise_params in noise_parameter_groups.items():
            print(f"  Running noise group '{group_name}'...")

            hardware_model_group = {
                "device_info": copy.deepcopy(hardware_model["device_info"]),
                "noise_params": copy.deepcopy(noise_params),
            }

            group_results = {}

            # Physical circuit, i.e., without any error correction
            circuit_physical = copy.deepcopy(circuit)
            if "measure" not in circuit_physical.count_ops():
                circuit_physical.measure_all()

            result_physical = execute_circuits(circuit_physical, hardware_model=hardware_model_group, **kwargs)[0]
            group_results["fidelity_physical"] = calculate_state_probability("0"*circuit_physical.num_qubits, result_physical.get_counts())

            # Logical circuit with optimized QEC scheduling
            if with_qec:
                lqc = LogicalCircuit.from_physical_circuit(circuit_no_meas, **qecc)
                qec_cycle_indices = lqc.optimize_qec_cycle_indices(constraint_model=constraint_model)
                lqc.insert_qec_cycles(qec_cycle_indices=qec_cycle_indices)
                lqc.measure_all()

                exact_state = _exact_logical_state(lqc) if fidelity_method == "density_matrix" else None

                result_logical = execute_circuits(lqc, hardware_model=hardware_model_group, save_density_matrix=(fidelity_method == "density_matrix"), **kwargs)[0]

                group_results["fidelity_logical"] = compute_schedule_fidelity(result_logical, lqc, exact_state=exact_state, fidelity_method=fidelity_method)
                group_results["num_qec"] = len(qec_cycle_indices.get(0, []))

            circuit_results["groups"][group_name] = group_results

        all_data["circuits"][circuit_name] = circuit_results

    save_progress()
    atexit.unregister(save_progress)

    return all_data
