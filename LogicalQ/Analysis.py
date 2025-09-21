import numpy as np
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity

from LogicalQ.Logical import LogicalCircuit, LogicalStatevector, LogicalDensityMatrix, logical_state_fidelity
from LogicalQ.Utilities import sanitize_save_parameters

"""
    Plot a three-dimensional bar chart comparing qubit count and circuit length to expectation value.
    Parameters:
        - data: dict[n_qubits, dict[circuit_length, (result, counts)]]
        - title: Plot title
        - save: If true, output plot is saved
        - filename: Filename to be saved as, if save is True
        - save_dir: Directory to be saved in, if save is True
        - show: If true, output plot is displayed
    Returns:
        - plt: A matplotlib plot object
"""
def circuit_scaling_bar3d(data, title=None, save=False, filename=None, save_dir=None, show=False):
    if not isinstance(data, dict):
        raise TypeError("Invalid type for data input: must be a dictionary of the form dict[n_qubits, dict[circuit_length, result]].")

    if title == None:
        title = "Circuit scaling bar plot"

    if save:
        filename, save_dir = sanitize_save_parameters(filename, save_dir, default_filename="circuit_scaling_bar3d")

    n_qubits_vals = []
    circuit_length_vals = []
    error_rates = []

    for n_qubits, sub_data in data.items():
        # @TODO - make this work better for data where not every qubit count has the same range of circuit lengths
        for circuit_length, data_tuple in sub_data.items():
            n_qubits_vals.append(n_qubits)
            circuit_length_vals.append(circuit_length)

            circuit, result = data_tuple

            if isinstance(circuit, LogicalCircuit):
                logical_counts = circuit.get_logical_counts(result.get_counts())
                error_rate = 1-logical_counts.get("0", 0)/sum(list(logical_counts.values()))
            elif isinstance(circuit, QuantumCircuit):
                error_rate = 1-calculate_state_probability("0"*n_qubits, result.get_counts())
            else:
                raise ValueError(f"Expected QuantumCircuit or LogicalCircuit, found object {circuit} of type {type(circuit)}.")

            error_rates.append(error_rate)

    ax = plt.figure().add_subplot(projection="3d")

    top = np.array(error_rates)
    bottom = np.zeros_like(top)
    width = depth = 1

    ax.bar3d(n_qubits_vals, circuit_length_vals, bottom, width, depth, top, shade=True)

    ax.set_title(title)
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Circuit length")
    ax.set_zlabel("Error rate $1-P(0)$")

    if save:
        plt.savefig(f"{save_dir}{filename}", dpi=256)

    if show:
        plt.gcf().set_dpi(256)
        plt.show()

    return plt

def noise_model_scaling_bar(all_data, scan_keys=None, separate_plots=False, save=False, filename=None, save_dir=None, show=False):
    # @TODO - sanitize save inputs

    for c, circuit_sub_data in all_data.items():
        qc = circuit_sub_data["circuit"]

        if "density_matrix_exact" in circuit_sub_data and circuit_sub_data["density_matrix_exact"] is not None:
            exact_state = circuit_sub_data["density_matrix_exact"]
        elif "statevector_exact" in circuit_sub_data and circuit_sub_data["statevector_exact"] is not None:
            exact_state = circuit_sub_data["statevector_exact"]
        else:
            raise ValueError("No ideal reference found in data, either 'density_matrix_exact' or 'statevector_exact' are necessary for this analysis function.")

        circuit_results = circuit_sub_data["results"]

        all_keys = list(circuit_results[0]["error_dict"].keys())
        if scan_keys is None:
            # If scan_keys subset is not provided, plot all available
            scan_keys = all_keys
        else:
            # If scan_keys subset contains elements not in all_keys, error out
            invalid_keys = set(scan_keys)-set(all_keys)
            if len(invalid_keys) > 0:
                raise ValueError(f"scan_keys input contains invalid keys not present in experiment data: {invalid_keys}.")

        i = 0
        fig, ax = plt.subplots()

        xdata = []
        ydata = []
        for scan_key in scan_keys:
            for r, results in enumerate(circuit_results):
                error_dict = results["error_dict"]
                result = results["result"][0]
                counts = result.get_counts()

                # Construct a logical state representation object for fidelity computation
                # @TODO - use density matrices instead once LogicalDensityMatrix is fully implemented
                if hasattr(result, "data"):
                    if isinstance(qc, LogicalCircuit):
                        noisy_state = LogicalStatevector.from_counts(counts, qc.n_logical_qubits, qc.label, qc.stabilizer_tableau)
                    elif isinstance(qc, QuantumCircuit):
                        noisy_state = counts_to_statevector(counts)
                    else:
                        raise TypeError(f"Invaild type for circuit at index {c}: {type(qc)}; must be an instance of QuantumCircuit or LogicalCircuit.")

                    fidelity = state_fidelity(exact_state, noisy_state)

                    xdata.append(error_dict[scan_key])
                    ydata.append(fidelity)
                else:
                    raise TypeError(f"Invalid type for data result at index {r}: {type(result)}.")

            if separate_plots:
                plt.bar(xdata, ydata)

                plt.title(f"Circuit {c}: Fidelity vs {scan_key}")

                plt.xlabel(scan_key)
                plt.ylabel("Fidelity")

                # @TODO - format filename with index i
                filename_i = filename
                if save:
                    plt.savefig(f"{save_dir}{filename_i}", dpi=128)
                    i += 1
                if show: plt.show()

        if not separate_plots:
            ax.bar(xdata, ydata)

            title = getattr(qc, "name", f"Circuit {c}")
            ax.set_title(f"{title}: Fidelity vs. noise parameters")

            ax.set_xlabel("Noise parameter value")
            ax.set_ylabel("Fidelity")

            if save: plt.savefig(f"{save_dir}{filename}", dpi=128)
            if show: plt.show()

    return plt

# @TODO - add save functionality
def qec_cycle_efficiency_scatter(all_data, scan_keys=None, plot_metric=None, show=False):
    if plot_metric is None:
        plot_metric = "fidelity"

    for c, circuit_sub_data in enumerate(all_data):
        qc = circuit_sub_data["physical_circuit"]
        lqc = circuit_sub_data["logical_circuit"]
        exact_state = circuit_sub_data["density_matrix_exact"]
        circuit_results = circuit_sub_data["results"]

        all_keys = list(circuit_results[0]["constraint_model"].keys())
        if scan_keys is None:
            # If scan_keys subset is not provided, plot all available
            scan_keys = all_keys
        else:
            # If scan_keys subset contains elements not in all_keys, error out
            invalid_keys = set(scan_keys)-set(all_keys)
            if len(invalid_keys) > 0:
                raise ValueError(f"scan_keys input contains invalid keys not present in experiment data: {invalid_keys}.")

        for scan_key in scan_keys:
            fig, ax = plt.subplots()

            xdata = []
            ydata = []

            for circuit_result in circuit_results:
                constraint_model = circuit_result["constraint_model"]

                result = circuit_result["result"]

                # Construct a LogicalDensityMatrix estimate from experiment counts
                # @TODO - use density matrices instead once LogicalDensityMatrix is fully implemented
                noisy_state = LogicalStatevector.from_counts(result.get_counts(), n_logical_qubits=lqc.n_logical_qubits, label=lqc.label, stabilizer_tableau=lqc.stabilizer_tableau)

                fidelity = logical_state_fidelity(exact_state, noisy_state)

                if plot_metric is None or plot_metric == "fidelity":
                    metric = fidelity
                elif plot_metric == "fidelity_per_qec_cycle":
                    # @TODO - currently assumes one logical qubit, add support for multi-qubit data
                    qec_cycle_indices = circuit_result["qec_cycle_indices"]
                    if qec_cycle_indices:
                        n_qec_cycles = len(list(qec_cycle_indices.values())[0])

                        metric = fidelity/n_qec_cycles
                    else:
                        raise ValueError("No QEC cycles inserted")
                else:
                    raise ValueError(f"Unrecognized input for plot_metric: {plot_metric}; please choose from 'fidelity' or 'fidelity_per_qec_cycle'")

                xdata.append(constraint_model[scan_key])
                ydata.append(metric)

            plt.scatter(xdata, ydata)

            title = getattr(qc, "name", f"Circuit {c}")
            plt.title(f"{title}: Fidelity vs {scan_key}")

            plt.xlabel(scan_key)
            plt.ylabel("Fidelity")

            if show: plt.show()

def counts_to_statevector(counts):
    result_key_0 = list(counts.keys())[0]
    if all([char in ["0", "1"] for char in result_key_0]):
        d = 2**(len(result_key_0))
        fmt_outcome = lambda outcome : bin(outcome)[2:]
    elif result_key_0.startswith("0b"):
        d = 2**len(result_key_0-2)
        fmt_outcome = lambda outcome : bin(outcome)
    elif result_key_0.startswith("0x"):
        d = 16**(len(result_key_0)-2)
        fmt_outcome = lambda outcome : hex(outcome)
    else:
        raise ValueError("Could not resolve result key format")

    outcomes = [fmt_outcome(i) for i in range(d)]

    probabilities = np.array([counts.get(outcome, 0.0) for outcome in outcomes])/np.sum(list(counts.values()))
    amplitudes = np.sqrt(probabilities)

    statevector = Statevector(amplitudes)

    return statevector

def calculate_state_probability(state, counts):
    total_counts = sum(list(counts.values()))

    # @TODO - generalize for superposition states
    state_probability = counts.get(state, 0)/total_counts

    return state_probability

"""
    Computes expectation value from circuit measurement counts.
"""
def calculate_exp_val(counts):
    total_counts = sum(list(counts.values()))

    exp_val = sum([key.count("1") for key in counts])/total_counts

    return exp_val

