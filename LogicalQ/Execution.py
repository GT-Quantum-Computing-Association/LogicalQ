import time
import copy
import pickle
import itertools
from datetime import datetime

from .Logical import LogicalCircuit
from .NoiseModel import construct_noise_model, construct_noise_model_from_hardware_model

from qiskit import QuantumCircuit

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qiskit import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Decompose
from .Transpilation.UnBox import UnBoxTask
from .Transpilation.DecomposeIfElseOps import DecomposeIfElseOpsTask
from .Transpilation.InsertOps import insert_before_measurement

from qiskit.providers import Backend
from qiskit_ibm_runtime import QiskitRuntimeService
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.qiskit import qiskit_to_tk
from qbraid.runtime.native.device import QbraidDevice

DEFAULT = object()

def execute_circuits(circuit_input, target=None, backend=None, hardware_model=None, noise_model=DEFAULT, noise_params=DEFAULT, coupling_map=DEFAULT, basis_gates=DEFAULT, method="statevector", optimization_level=0, shots=1024, memory=False, save_statevector=False, save_density_matrix=False, return_circuits_transpiled=False):
    """
    General function to execute a circuit with smart handling of parameters, especially for circuits with QEC.

    The parameters target, backend, and hardware_model are the preferred input type to this function. If specified, noise_model, noise_params, coupling_map, and basis_gates will try to override anything specified in target, backend, or hardware_model.
    """

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
        def check_for_measurement(circuit):
            if isinstance(circuit, LogicalCircuit):
                for instruction in circuit.data:
                    if instruction.operation.name == "box" and instruction.operation.label.split(":")[0] == "logical.qec.measure":
                        return True
            else:
                if "measure" in circuit.count_ops():
                    return True

            return False
        
        if not check_for_measurement(circuit):
            pass#raise ValueError(f"No measurements found in circuit with name {circuit.name} at index {c}; all circuits must have measurements in order to be executed.")

    # Save statevector for all circuits if requested
    if save_statevector:
        for i, circuit in enumerate(circuits):
            circuits[i], _ = insert_before_measurement(circuit, "statevector")

    if save_density_matrix:
        for i, circuit in enumerate(circuits):
            circuits[i], _ = insert_before_measurement(circuit, "density_matrix")

    # Patch to account for backends that do not yet recognize BoxOp's during transpilation
    pm = PassManager([UnBoxTask()])
    for c in range(len(circuits)):
        # @TODO - this while loop should be unnecessary with the new DoWhileController-based approach; test first
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
            # Default to fully-coupled map
            coupling_map = None
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
    elif not hasattr(basis_gates, "__iter__") and basis_gates is not None:
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

    # Construct specialized callables for transpilation, costing, and running - the work could be
    # done here, but we're trying to "factor out" common behavior which is present in all cases and
    # only deal with backend-specific patches here - ths makes it easy to modify factored behavior
    # for all use cases and not think about backend-specific code if not necessary
    # @TODO - instead of relying on the transpile function, which is a thin wrapper around
    #         generate_preset_pass_manager with some type-handling, maybe we can create
    #         backend-specific pass managers here and then just run them later with common settings
    if backend is None:
        raise ValueError("Could not resolve backend - make sure to pass one.")
    elif isinstance(backend, AerSimulator):
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
            def _transpile(circuits, backend=None, coupling_map=None, optimization_level=0, **kwargs):
                pm = PassManager([DecomposeIfElseOpsTask(), Decompose()])
                circuits_decomposed = pm.run(circuits)
                tket_circuits_decomposed = [qiskit_to_tk(circuit_decomposed) for circuit_decomposed in circuits_decomposed]
                return backend.get_compiled_circuits(tket_circuits_decomposed, optimisation_level=optimization_level, **kwargs)

            # Do this type check here because pytket's error isn't very easy to understand for users
            if not (
                isinstance(shots, int) or
                (hasattr(shots, "__iter__") and all([isinstance(shot_count, int) for shot_count in shots]))
            ):
                raise TypeError(f"Invalid type for shots input: {type(shots)}; must be int or iterable of ints")

            # @TODO - this part of the code seems buggy, needs fixing
            syntax_checker = backend._device_name.rstrip("LE") + "SC"
            _cost = lambda circuits, shots : [backend.cost(circuit, n_shots=shots, syntax_checker=syntax_checker) for circuit in circuits]

            # @TODO - give the user more control over this callable and its parameters
            # @TODO - add better logging
            def _run(circuits, shots, memory=None, **kwargs):
                # Submit circuits for execution, using process_circuits in order to retrieve handles
                handles = backend.process_circuits(circuits, n_shots=shots, **kwargs)

                # Save handles to a file
                save_dir = "./data/"
                date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_filename = f"quantinuum_handles_{date_str}.pkl"
                save_filepath = save_dir + save_filename
                save_file = open(save_filepath, "wb")
                pickle.dump(handles, save_file, protocol=5)
                save_file.close()

                # Poll handles periodically
                poll_interval_seconds = 10
                max_wait_minutes = 60 * 24 * 7
                max_n_poll_attempts = int(max_wait_minutes * 60 / poll_interval_seconds)
                
                # @TODO - simplify this portion of the code
                n_poll_attempts = 0
                handles_completed = []
                while n_poll_attempts < max_n_poll_attempts and len(handles_completed) < len(handles):
                    for h, handle in enumerate(handles):
                        if h not in handles_completed:
                            handle_status = backend.circuit_status(handle)
                            handle_status_str = str(handle_status.status).lower()
                            
                            print(f"Polling attempt {n_poll_attempts} of {max_n_poll_attempts}:")
                            print(f"- Handle status: {handle_status}")

                            if "completed" in handle_status_str:
                                handles_completed.append(h)
                            else:
                                print(f"- Waiting {poll_interval_seconds} seconds to poll again...")
                                time.sleep(poll_interval_seconds)

                if len(handles_completed) == len(handles):
                    for handle in handles:
                        # Get results
                        results = backend.get_results(handles)

                        # Save results
                        save_dir = "./data/"
                        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        save_filename = f"quantinuum_results_{date_str}.pkl"
                        save_file = open(save_dir + save_filename, "wb")
                        pickle.dump(results, save_file, protocol=5)
                        save_file.close()

                        return results
                else:
                    print(f"Reached maximum number of handle polling attempts, stopping. All handles are stored at '{save_filepath}' for later access.")

                    return None
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

    # Run circuits
    results = []
    for circuit_to_run in circuits_to_run:
        if circuit_to_run is None:
            result = None
        else:
            result = _run([circuit_to_run], shots=shots, memory=memory)

        results.append(result)

    if return_circuits_transpiled:
        return results, circuits_transpiled
    else:
        return results

