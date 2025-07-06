import time
import numpy as np
from matplotlib import pyplot as plt

from qiskit.visualization import plot_distribution, plot_histogram

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
        raise ValueError("The 'data' parameter should be a dictionary of the form dict[n_qubits, dict[circuit_length, (result, counts)]].")

    if title == None:
        title = "Circuit scaling bar chart"

    if save:
        filename, save_dir = sanitize_save_parameters(filename, save_dir, default_filename="circuit_scaling_bar3d")

    n_qubits_vals = []
    circuit_length_vals = []
    exp_vals = []

    for n_qubits, sub_data in data.items():
        # @TODO - make this work better for data where not every qubit count has the same range of circuit lengths
        for circuit_length, (result, counts) in sub_data.items():
            n_qubits_vals.append(n_qubits)
            circuit_length_vals.append(circuit_length)

            exp_val = calculate_exp_val(counts)
            exp_vals.append(exp_val)

    ax = plt.figure().add_subplot(projection='3d')

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

def noise_model_scaling_bar(all_data, scan_keys=None, separate_plots=False):
    for entry in all_data:
        qc = entry["qc"]

        if "density_matrix_exact" in entry and entry["density_matrix_exact"] is not None:
            ideal_ref = entry["density_matrix_exact"]
        elif "statevector_exact" in entry and entry["statevector_exact"] is not None:
            ideal_ref = entry["statevector_exact"]
        else:
            raise ValueError("No ideal reference found. 'density_matrix_exact' or 'statevector_exact'")

        results = entry["results"]

        if scan_keys is not None:
            keys = scan_keys
        else:
            keys = list(results[0]["error_dict"].keys())

        for key in keys:
            xs = [] #x-Axis
            ys = [] #y-Axis

            for res in results:
                err_dict = res["error_dict"]
                ret = res["result"]

                # Convert ret to an object accepted by state_fidelity
                # Handles different ways return object data is formatted in the experiment function
                # @TODO change accordingly based on the data structure in noise_scaling_experiment
                if isinstance(ret, (DensityMatrix, Statevector)):
                    noisy_state = ret
                elif hasattr(ret, "data"):                               # raw Result object
                    noisy_state = DensityMatrix(ret.data(0))
                elif isinstance(ret, tuple):                             # (Result, counts)
                    result_obj = ret[0]
                    noisy_state = DensityMatrix(result_obj.data(0)) 
                else:
                    raise TypeError("res['result'] type is not right.")

                fidelity = state_fidelity(ideal_ref, noisy_dm)

                xs.append(err_dict[key])
                ys.append(fidelity)

            if separate_plots:                     # one plot per key
                plt.figure(figsize=(max(6, 0.8 * len(xs)), 4))
                plt.bar(xs, ys)
                plt.xlabel(key)
                plt.ylabel("Fidelity")
                title = getattr(qc, "name", "Circuit")
                plt.title(f"{title}: Fidelity vs {key}")
                plt.tight_layout()
                plt.show()


        if not separate_plots:
            title = getattr(qc, "name", "Circuit")
            ax.set_xlabel("Noise-parameter value")
            ax.set_ylabel("Fidelity")
            ax.set_title(f"{title}: Fidelity vs noise parameters")
            ax.legend(title="scan key")
            plt.tight_layout()
            plt.show()


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
            x_Axis = []
            y_Axis = []

            for res in results:
                cfg = res["config"]
                num_QEC = cfg.get("cycles")
                if num_QEC is None or num_QEC == 0:
                    raise ValueError("You didn't experiment with any QEC cycles injected.")

                benchmark_noise_return_obj = res["result"]

                # Converts the the benchmark_noise function return object to a DensityMatrix
                # @TODO change accordingly based on the data structure in qec_cycle_efficiency_experiment
                if isinstance(benchmark_noise_return_obj, DensityMatrix):
                    noisy_dm = benchmark_noise_return_obj
                elif isinstance(benchmark_noise_return_obj, tuple):
                    result_obj = benchmark_noise_return_obj[0]
                    noisy_dm = DensityMatrix(result_obj.data(0))
                elif hasattr(benchmark_noise_return_obj, "data"):
                    noisy_dm = DensityMatrix(ret.data(0))
                else:
                    raise TypeError("res['result'] type is not right")

                fidelity = state_fidelity(exact_dm, noisy_dm)
                metric = fidelity / num_QEC

                x_Axis.append(cfg[key])
                y_Axis.append(metric)


            plt.figure()
            plt.bar(x_Axis, y_Axis)
            plt.xlabel(key)
            plt.ylabel("Fidelity / Cycles")
            title = getattr(qc, "name", "Circuit")
            plt.title(f"{title}: Fidelity / Cycles vs {key}")
            plt.show()


"""
    Computes expectation value from circuit measurement counts.
"""
def calculate_state_probability(state, counts):
    total_counts = sum(list(counts.values()))

    # @TODO - generalize for superposition states
    state_probability = counts[state]/total_counts

    return state_probability

def calculate_exp_val(counts):
    total_counts = sum(list(counts.values())) 

    exp_val = sum([key.count("1") for key in counts])/total_counts

    return exp_val

def sanitize_save_parameters(filename, save_dir, default_filename="plot", default_save_dir="./"):
    if filename == None:
        filename = default_filename + str(int(time.time())) + ".png"
    elif "." not in filename:
        filename += ".png"

    if save_dir == None and filename[:2] != "./" and filename[0] != "/":
        save_dir = default_save_dir
    elif save_dir[-1] != "/":
        save_dir += "/"

