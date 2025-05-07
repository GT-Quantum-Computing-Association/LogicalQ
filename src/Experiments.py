import itertools

from Logical import LogicalCircuit
from NoiseModel import construct_noise_model

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

# General function to benchmark a circuit using a noise model
def benchmark_noise(circuit, noise_model=None, noise_params=None, method="statevector", shots=1024, optimization_level=0):
    if noise_model is None:
        if noise_params is not None:
            # If noise_params are provided but not a noise_model, then construct noise model based on the provided parameters
            noise_model = construct_noise_model(circuit.num_qubits, **noise_params)
        else:
            # If a noise_model is not provided at all, then 
            raise ValueError("Either noise_model or noise_params must be provided")
    elif noise_params is not None:
        print("Both noise_model and noise_params were provided, defaulting to use the noise_model and ignoring noise_params. If you would like to use custom noise_params, pass noise_model=None.")

    # Construct noisy simulator based on chosen method using noise model
    noisy_sim = AerSimulator(method=method, noise_model=noise_model)

    # Create a fully-coupled map since we don't care about non-fully-coupled hardware modalities
    fully_coupled_map = itertools.product(range(circuit.num_qubits), range(circuit.num_qubits))
    fully_coupled_map = [list(pair) for pair in fully_coupled_map]

    # Transpile circuit
    # Method defaults to optimization off to preserve form of benchmarking circuit and full QEC
    circuit_transpiled = transpile(circuit, noisy_sim, coupling_map=fully_coupled_map, optimization_level=optimization_level)
    result = noisy_sim.run(circuit_transpiled, shots=shots).result()
    counts = result.get_counts(circuit_transpiled)

    return result, counts

# @TODO - implement experiments
def circuit_scaling_experiment(circuit_factory, noise_model_factory, min_n_qubits=1, max_n_qubits=50, min_circuit_length=1, max_circuit_length=50, method="statevector"):
    if isinstance(circuit_factory, QuantumCircuit) or isinstance(circuit_factory, LogicalCircuit):
        if max_n_qubits != min_n_qubits+1:
            print("A constant circuit has been provided as the circuit factory, but a non-trivial range of qubit counts and/or circuit lengths has also been provided, so the fixed input will not be scaled. If you would like for the number of qubits to be scaled, please provide a callable which takes in as an argument the number of qubits, n_qubits. If you would like for the circuit length to be scaled, please provide a callable which takes in as an argument the circuit length, circuit_length.")
        
        circuit = lambda n_qubits, circuit_length: circuit_factory
    elif callable(circuit_factory):
        circuit = circuit_factory
    else:
        raise ValueError("Please provide a QuantumCircuit/LogicalCircuit object or a method for constructing QuantumCircuits/LogicalCircuits.")
    
    if isinstance(noise_model_factory, NoiseModel):
        if max_n_qubits != min_n_qubits+1:
            print("A constant noise model has been provided as the noise model factory, but a non-trivial range of qubit counts has also been provided. The number of qubits will not be scaled; if you would like for the number of qubits to be scaled, please provide a callable which takes in as an argument the number of qubits, n_qubits.")
        
        noise_model = lambda n_qubits: noise_model_factory
    elif callable(noise_model_factory):
        noise_model = noise_model_factory
    else:
        raise ValueError("Please provide a NoiseModel object or a method for constructing NoiseModels.")
        
    all_data = {}
    for n_qubits in range(min_n_qubits, max_n_qubits+1):
        sub_data = {}
    
        # We need a new noise model for each qubit count
        noise_model_n = noise_model(n_qubits=n_qubits)
    
        for circuit_length in range(min_circuit_length, max_circuit_length+1):
            # Construct circuit and benchmark noise
            circuit_nl = circuit(n_qubits=n_qubits, circuit_length=circuit_length)
            result, counts = benchmark_noise(circuit_nl, noise_model=noise_model_n, method=method)
            
            # Save expectation values
            sub_data[circuit_length] = result, counts
    
        all_data[n_qubits] = sub_data
    
    return all_data
