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

"""
    Plot a scatterplot comparing QEC cycle count and fidelity
    Parameters:
        - qec_cycle_counts: An iterable storing the number of QEC cycles run per trial
        - fidelities: An iterable storing the fidelity of each trial as floats. If None, counts are relied on.
        - counts: An iterable storing measurement counts of each trial as dict[output, frequency]. If used, a reference state must be specified. If None, fidelities are relied on.
        - title: Plot title
        - save: If true, output plot is saved
        - filename: Filename to be saved as, if save is True
        - save_dir: Directory to be saved in, if save is True
        - show: If true, output plot is displayed
    Returns:
        - plt: A matplotlib plot object
"""
def qec_cycle_efficiency_plot(qec_cycle_counts, fidelities=None, counts_list=None, reference_state=None, title=None, save=False, filename=None, save_dir=None, show=False):
    if save:
        filename, save_dir = sanitize_save_parameters(filename, save_dir, default_filename="qec_cycle_efficiency_plot")

    if fidelities is None:
        if counts_list is None:
            raise ValueError("One of fidelities or counts and a reference state must be specified.")
        else:
            if reference_state is None:
                raise ValueError("If counts is being used, a reference state must be specified.")
            else:
                fidelities = [calculate_state_probability(reference_state, counts) for counts in counts_list]

    raise NotImplementedError("Circuit length fidelity profiles are not fully implemented!")

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

