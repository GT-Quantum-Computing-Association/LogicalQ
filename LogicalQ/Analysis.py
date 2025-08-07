import time
import numpy as np
from matplotlib import pyplot as plt

from qiskit.quantum_info import state_fidelity

from LogicalQ.Logical import LogicalStatevector
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
    exp_vals = []

    for n_qubits, sub_data in data.items():
        # @TODO - make this work better for data where not every qubit count has the same range of circuit lengths
        for circuit_length, result in sub_data.items():
            n_qubits_vals.append(n_qubits)
            circuit_length_vals.append(circuit_length)

            # @TODO - if the circuit is an instance of LogicalCircuit, use get_logical_counts instead
            exp_val = calculate_exp_val(result.get_counts())
            exp_vals.append(exp_val)

    ax = plt.figure().add_subplot(projection="3d")

    top = np.array(exp_vals)
    bottom = np.zeros_like(top)
    width = depth = 1

    ax.bar3d(n_qubits_vals, circuit_length_vals, bottom, width, depth, top, shade=True)

    ax.set_title(title)
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Circuit length")

    if save:
        plt.savefig(f"{save_dir}{filename}", dpi=500)

    if show:
        plt.gcf().set_dpi(500)
        plt.show()

    return plt

def noise_model_scaling_bar(all_data, scan_keys=None, separate_plots=False, save=False, filename=None, save_dir=None, show=False):
    # @TODO - sanitize save inputs

    for c, circuit_sub_data in all_data.items():
        qc = circuit_sub_data["circuit"]

        # if "density_matrix_exact" in circuit_sub_data and circuit_sub_data["density_matrix_exact"] is not None:
        #     exact_state = circuit_sub_data["density_matrix_exact"]
        # elif "statevector_exact" in circuit_sub_data and circuit_sub_data["statevector_exact"] is not None:
        #     exact_state = circuit_sub_data["statevector_exact"]
        # else:
        #     raise ValueError("No ideal reference found in data, either 'density_matrix_exact' or 'statevector_exact' are necessary for this analysis function.")

        circuit_results = circuit_sub_data["results"]

        # If scan_keys subset is not provided, plot all available
        if scan_keys is None:
            scan_keys = list(circuit_results[0]["error_dict"].keys())

        i = 0
        fig, ax = plt.subplots()

        xdata = []
        ydata = []
        for scan_key in scan_keys:
            for results in circuit_results:
                error_dict = results["error_dict"]
                result = results["result"][0]

                # Construct a logical state representation object for fidelity computation
                if hasattr(result, "data"):
                    noisy_state = LogicalStatevector.from_counts(result.get_counts(), qc.n_logical_qubits, qc.label, qc.stabilizer_tableau)
                    exact_state = noisy_state

                    fidelity = state_fidelity(exact_state, noisy_state)

                    xdata.append(error_dict[scan_key])
                    ydata.append(fidelity)
                else:
                    raise TypeError(f"Invalid type for data result: {type(result)}.")

            if separate_plots:
                plt.bar(xdata, ydata)

                plt.title(f"Circuit {c}: Fidelity vs {scan_key}")

                plt.xlabel(scan_key)
                plt.ylabel("Fidelity")

                # @TODO - format filename with index i
                filename_i = filename
                if save:
                    plt.savefig(f"{save_dir}{filename_i}", dpi=500)
                    i += 1
                if show: plt.show()

        if not separate_plots:
            ax.bar(xdata, ydata)

            ax.set_xlabel("Noise parameter value")
            ax.set_ylabel("Fidelity")
            ax.set_title(f"Circuit {c}: Fidelity vs. noise parameters")

            if save: plt.savefig(f"{save_dir}{filename}", dpi=500)
            if show: plt.show()

    return plt

def qec_cycle_efficiency_bar(all_data, scan_keys=None):
    for entry in all_data:
        qc = entry["qc"]
        exact_dm = entry["density_matrix_exact"]
        results = entry.get("results", entry.get("data", []))

        if scan_keys is not None:
            keys = scan_keys
        else:
            keys = list(results[0]["config"].keys())

        for key in keys:
            x_axis = []
            y_axis = []

            for res in results:
                cfg = res["config"]
                num_QEC = cfg.get("cycles")
                if num_QEC is None or num_QEC == 0:
                    raise ValueError("You didn't experiment with any QEC cycles injected.")

                return_obj = res["result"]

                # Converts the the execute_circuits function return object to a DensityMatrix
                # @TODO change accordingly based on the data structure in qec_cycle_efficiency_experiment
                if isinstance(return_obj, DensityMatrix):
                    noisy_dm = return_obj
                elif isinstance(return_obj, tuple):
                    result_obj = return_obj[0]
                    noisy_dm = DensityMatrix(result_obj.data(0))
                elif hasattr(return_obj, "data"):
                    noisy_dm = DensityMatrix(ret.data(0))
                else:
                    raise TypeError("res['result'] type is not right")

                fidelity = state_fidelity(exact_dm, noisy_dm)
                metric = fidelity / num_QEC

                x_Axis.append(cfg[key])
                y_Axis.append(metric)

            plt.figure()
            plt.bar(x_axis, y_axis)
            plt.xlabel(key)
            plt.ylabel("Fidelity / Cycle Count")
            title = getattr(qc, "name", "Circuit")
            plt.title(f"{title}: Fidelity / Cycle Count vs {key}")
            plt.show()

def calculate_state_probability(state, counts):
    total_counts = sum(list(counts.values()))

    # @TODO - generalize for superposition states
    state_probability = counts[state]/total_counts

    return state_probability

"""
    Computes expectation value from circuit measurement counts.
"""
def calculate_exp_val(counts):
    total_counts = sum(list(counts.values())) 

    exp_val = sum([key.count("1") for key in counts])/total_counts

    return exp_val

