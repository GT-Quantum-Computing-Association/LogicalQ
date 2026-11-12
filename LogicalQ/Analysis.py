import numpy as np
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, state_fidelity

from LogicalQ.Logical import LogicalCircuit, LogicalStatevector, LogicalDensityMatrix, logical_state_fidelity
from LogicalQ.Utilities import sanitize_save_parameters

def circuit_scaling_bar3d(data, title=None, save=False, filename=None, save_dir=None, show=False):
    """
    Plot a three-dimensional bar chart comparing qubit count and circuit length to expectation value.

    Parameters
    ----------
        data : dict[n_qubits, dict[circuit_length, (result, counts)]]
        title : str
            Plot title
        save : bool
            If true, output plot is saved
        filename : str
            Filename to be saved as, if save is True
        save_dir : str
            Directory to be saved in, if save is True
        show : str
            If true, output plot is displayed

    Returns
    -------
        plt : matplotlib.pyplot
            A matplotlib plot object
    """

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

def noise_scaling_scatter(all_data, scan_keys=None, separate_plots=False, save=False, filename=None, save_dir=None, show=False):
    """Accepts the results of a noise_scaling_experiment and plots in bar graph / scatter plot format.

    Args:
        all_data: Output of a noise_scaling_experiment() run.

    Returns:
        plt
    """
    
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
            ax.scatter(xdata, ydata)

            title = getattr(qc, "name", f"Circuit {c}")
            ax.set_title(f"{title}: Fidelity vs. noise parameters")

            ax.set_xlabel("Noise parameter value")
            ax.set_ylabel("Fidelity")

            if save: plt.savefig(f"{save_dir}{filename}", dpi=128)
            if show: plt.show()

    return plt

def noise_scaling_Bloch_sphere(all_data, plot_metric=None, save=False, filename=None, save_dir=None, show=False):
    for c, circuit_sub_data in all_data.items():
        qc = circuit_sub_data["circuit"]

        if "density_matrix_exact" in circuit_sub_data and circuit_sub_data["density_matrix_exact"] is not None:
            exact_state = circuit_sub_data["density_matrix_exact"]
        elif "statevector_exact" in circuit_sub_data and circuit_sub_data["statevector_exact"] is not None:
            exact_state = circuit_sub_data["statevector_exact"]
        else:
            raise ValueError("No ideal reference found in data, either 'density_matrix_exact' or 'statevector_exact' are necessary for this analysis function.")

        circuit_results = circuit_sub_data["results"]

        i = 0
        fig, ax = plt.subplots()

        thetas = []
        phis = []
        plot_data = []
        for r, results in enumerate(circuit_results):
            result = results["result"][0]
            counts = result.get_counts()

            # Compute desired metric
            # @TODO - generalize to support multi-qubit data
            # @TODO - use density matrices instead once LogicalDensityMatrix is fully implemented
            if hasattr(result, "data"):
                if isinstance(qc, LogicalCircuit):
                    logical_counts = qc.get_logical_counts(result.get_counts())

                    p0 = logical_counts.get("0", 0)/sum(list(logical_counts.values()))
                    p1 = 1 - p0

                    noisy_state = LogicalStatevector(np.sqrt([p0, p1]), n_logical_qubits=qc.n_logical_qubits, label=qc.label, stabilizer_tableau=qc.stabilizer_tableau)
                elif isinstance(qc, QuantumCircuit):
                    noisy_state = counts_to_statevector(counts)
                else:
                    raise TypeError(f"Invaild type for circuit at index {c}: {type(qc)}; must be an instance of QuantumCircuit or LogicalCircuit.")

                theta = 2 * np.arccos(noisy_state[0])

                # @TODO - compute phi if phase information is available
                phi = 0.0

                fidelity = state_fidelity(exact_state, noisy_state)
                if plot_metric is None or plot_metric == "fidelity":
                    metric = fidelity
                elif plot_metric == "infidelity":
                    metric = 1 - fidelity
                else:
                    raise ValueError(f"Invalid input for plot_metric: '{plot_metric}'; please choose from 'fidelity' and 'infidelity'.")

                thetas.append(theta)
                phis.append(phi)
                plot_data.append(metric)
            else:
                raise TypeError(f"Invalid type for data result at index {r}: {type(result)}.")

        fig = plt.figure(dpi=128)
        ax = fig.add_subplot(projection="3d")
        ax.set_aspect("equal")

        # Plot underlying Bloch sphere
        R = 1
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:30j]
        x = R*np.cos(v) * np.sin(u)
        y = R*np.sin(v) * np.sin(u)
        z = R*np.cos(u)
        ax.plot_surface(x, y, z, cmap=plt.cm.bone)

        # Plot data around the Bloch sphere
        R *= 1.1
        xs = R*np.cos(phis) * np.sin(thetas)
        ys = R*np.sin(phis) * np.sin(thetas)
        zs = R*np.cos(thetas)
        scatter = ax.scatter(xs, ys, zs, c=plot_data, cmap=plt.cm.Oranges, s=30)

        plt.title(f"Circuit {c}: {plot_metric} vs. noisy angle")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.colorbar(scatter)

        plt.show()

        # @TODO - format filename with index i
        filename_i = filename
        if save:
            plt.savefig(f"{save_dir}{filename_i}", dpi=128)
            i += 1
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

def calculate_exp_val(counts):
    """
    Computes expectation value from circuit measurement counts.
    """

    total_counts = sum(list(counts.values()))

    exp_val = sum([key.count("1") for key in counts])/total_counts

    return exp_val


############################################
##### QEC cycle scheduling analysis #####
############################################

# Consistent colors across every scheduling figure so that methods are comparable between plots
schedule_type_colors = {
    "no_qec": "#9E9E9E",
    "optimized": "#2E7D32",
    "random": "#1976D2",
    "fixed": "#D32F2F",
    "weinstein": "#F57C00",
    "abu_nada": "#7B1FA2",
}

def schedule_type_label(schedule_type):
    """Return a human-readable label for a scheduling method."""

    if schedule_type == "no_qec":
        return "No QEC (baseline)"
    elif schedule_type == "abu_nada":
        return "Abu-Nada et al."

    return schedule_type.replace("_", " ").title()

def compute_schedule_fidelity(result, logical_circuit, exact_state=None, fidelity_method="density_matrix"):
    """Score the result of a scheduled circuit against an exact reference state.

    Args:
        result: The result object returned by `execute_circuits`.
        logical_circuit: The LogicalCircuit which produced the result, used to identify the data qubits.
        exact_state: The exact reference state, i.e., the noiseless output of the same logical circuit.
        fidelity_method: Either "density_matrix", which compares saved density matrices on the data qubits and is sensitive to phase, or "logical_counts", which reports the Z-basis probability of the all-zero logical outcome and is not a fidelity.

    Returns:
        The fidelity, or the success probability if `fidelity_method` is "logical_counts".

    Raises:
        :class:`ValueError`: if `fidelity_method` is not recognized, or if a saved density matrix is required but not present in the result.
    """

    if fidelity_method == "logical_counts":
        # @NOTE - this is the Z-basis probability of the all-zero logical outcome, not a state fidelity; it
        #         ignores exact_state entirely and is only meaningful when the ideal output is |0...0>_L
        logical_counts = logical_circuit.get_logical_counts(result.get_counts())

        total_counts = sum(list(logical_counts.values()))
        if total_counts == 0:
            return 0.0

        return logical_counts.get("0"*logical_circuit.n_logical_qubits, 0.0)/total_counts
    elif fidelity_method == "density_matrix":
        if exact_state is None:
            raise ValueError("An exact_state input is required in order to compute a density matrix fidelity")

        data = result.data()
        if "density_matrix" not in data:
            raise ValueError("No saved density matrix found in result; execute with save_density_matrix=True")

        # Trace out every qubit which is not a data qubit, so that the comparison happens on the encoded
        # state rather than on the ancillas, which are left in whatever state the last cycle produced
        non_data_qubits = []
        count = 0
        for qreg in logical_circuit.qregs:
            if qreg in logical_circuit.logical_qregs:
                count += qreg.size
            else:
                non_data_qubits.extend(list(range(count, count+qreg.size)))
                count += qreg.size

        density_matrix_data = partial_trace(DensityMatrix(data["density_matrix"]), non_data_qubits)

        return float(state_fidelity(exact_state, density_matrix_data, validate=False))
    else:
        raise ValueError(f"'{fidelity_method}' is not a valid fidelity method, please choose from 'density_matrix' or 'logical_counts'")

def qec_cycle_schedule_comparison_bar(all_data, title=None, save=False, filename=None, save_dir=None, show=False):
    """Plot infidelity per scheduling method, grouped by circuit, at a matched QEC cycle count."""

    circuit_names = list(all_data["circuits"].keys())

    all_methods = set()
    for circuit_data in all_data["circuits"].values():
        all_methods.update(circuit_data["methods"].keys())
    methods = [schedule_type for schedule_type in schedule_type_colors.keys() if schedule_type in all_methods]

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")

    x = np.arange(len(circuit_names))
    width = 0.8/max(len(methods), 1)

    for i, method in enumerate(methods):
        infidelities = []
        errors = []
        for circuit_name in circuit_names:
            method_results = all_data["circuits"][circuit_name]["methods"].get(method, {})
            infidelities.append(1.0-method_results.get("fidelity", np.nan))
            errors.append(method_results.get("fidelity_std", 0.0))

        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, infidelities, width, yerr=errors, capsize=3, label=schedule_type_label(method), color=schedule_type_colors.get(method, "#757575"), alpha=0.85)

    ax.set_xlabel("Benchmark circuit")
    ax.set_ylabel("Infidelity")
    if title is not None:
        ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(circuit_names, rotation=30, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    if save:
        if save_dir is None: save_dir = "./output/"
        if filename is None: filename = "qec_cycle_schedule_comparison.png"
        plt.savefig(f"{save_dir}{filename}", dpi=256)

    if show: plt.show()

    return fig

def qec_cycle_schedule_duration_scatter(all_data, title=None, save=False, filename=None, save_dir=None, show=False):
    """Plot infidelity against estimated circuit duration at a matched QEC cycle count.

    Because cycle count is held fixed across methods, duration isolates the effect of where the cycles are placed rather than how many of them there are.
    """

    circuit_names = list(all_data["circuits"].keys())

    all_methods = set()
    for circuit_data in all_data["circuits"].values():
        all_methods.update(circuit_data["methods"].keys())
    methods = [schedule_type for schedule_type in schedule_type_colors.keys() if schedule_type in all_methods]

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    for method in methods:
        durations = []
        infidelities = []
        errors = []
        for circuit_name in circuit_names:
            method_results = all_data["circuits"][circuit_name]["methods"].get(method, None)
            if method_results is None:
                continue

            durations.append(method_results["duration"])
            infidelities.append(1.0-method_results["fidelity"])
            errors.append(method_results.get("fidelity_std", 0.0))

        if len(durations) == 0:
            continue

        ax.errorbar(durations, infidelities, yerr=errors, fmt="none", ecolor=schedule_type_colors.get(method, "#757575"), alpha=0.6, capsize=3)
        ax.scatter(durations, infidelities, s=220 if method == "optimized" else 90, marker="*" if method == "optimized" else "o", color=schedule_type_colors.get(method, "#757575"), alpha=0.9, label=schedule_type_label(method))

    ax.set_xlabel("Estimated circuit duration")
    ax.set_ylabel("Infidelity")
    if title is not None:
        ax.set_title(title)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if save:
        if save_dir is None: save_dir = "./output/"
        if filename is None: filename = "qec_cycle_schedule_duration.png"
        plt.savefig(f"{save_dir}{filename}", dpi=256)

    if show: plt.show()

    return fig

def qec_cycle_schedule_noise_scaling_scatter(all_data, circuit_name=None, xlabel=None, title=None, save=False, filename=None, save_dir=None, show=False):
    """Plot infidelity against a scanned noise parameter for each scheduling method."""

    noise_scan_values = sorted(all_data.keys())

    if circuit_name is None:
        circuit_name = list(all_data[noise_scan_values[0]]["circuits"].keys())[0]

    if xlabel is None:
        xlabel = "Noise parameter"

    all_methods = set()
    for results in all_data.values():
        all_methods.update(results["circuits"][circuit_name]["methods"].keys())
    methods = [schedule_type for schedule_type in schedule_type_colors.keys() if schedule_type in all_methods]

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    for method in methods:
        infidelities = []
        errors = []
        for noise_scan_value in noise_scan_values:
            method_results = all_data[noise_scan_value]["circuits"][circuit_name]["methods"].get(method, {})
            infidelities.append(1.0-method_results.get("fidelity", np.nan))
            errors.append(method_results.get("fidelity_std", 0.0))

        ax.errorbar(noise_scan_values, infidelities, yerr=errors, marker="o", capsize=3, color=schedule_type_colors.get(method, "#757575"), label=schedule_type_label(method))

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Infidelity")
    if title is not None:
        ax.set_title(title)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if save:
        if save_dir is None: save_dir = "./output/"
        if filename is None: filename = "qec_cycle_schedule_noise_scaling.png"
        plt.savefig(f"{save_dir}{filename}", dpi=256)

    if show: plt.show()

    return fig

def noise_parameter_comparison_bar(all_data, circuit_name=None, title=None, save=False, filename=None, save_dir=None, show=False):
    """Plot infidelity per isolated noise channel, with and without QEC.

    Isolating one channel per group shows which physical error sources dominate and which of them QEC actually suppresses.
    """

    if circuit_name is None:
        circuit_name = list(all_data["circuits"].keys())[0]

    groups = all_data["circuits"][circuit_name]["groups"]
    group_names = list(groups.keys())

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")

    x = np.arange(len(group_names))
    width = 0.38

    infidelities_physical = [1.0-groups[group_name].get("fidelity_physical", np.nan) for group_name in group_names]
    ax.bar(x - width/2, infidelities_physical, width, label="Without QEC", color="#D32F2F", alpha=0.85)

    if any(["fidelity_logical" in groups[group_name] for group_name in group_names]):
        infidelities_logical = [1.0-groups[group_name].get("fidelity_logical", np.nan) for group_name in group_names]
        ax.bar(x + width/2, infidelities_logical, width, label="With optimized QEC", color="#2E7D32", alpha=0.85)

    ax.set_xlabel("Noise channel")
    ax.set_ylabel("Infidelity")
    if title is not None:
        ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=30, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    if save:
        if save_dir is None: save_dir = "./output/"
        if filename is None: filename = "noise_parameter_comparison.png"
        plt.savefig(f"{save_dir}{filename}", dpi=256)

    if show: plt.show()

    return fig

def qecc_comparison_bar(all_data_by_qecc, title=None, save=False, filename=None, save_dir=None, show=False):
    """Plot infidelity per scheduling method, grouped by QEC code.

    Each group is one code and each bar within it is one scheduling method, so that a single figure carries the whole code comparison; codes are read from the keys of the input, so the figure grows automatically as more codes become supported.
    """

    qecc_names = list(all_data_by_qecc.keys())

    all_methods = set()
    for all_data in all_data_by_qecc.values():
        for circuit_data in all_data["circuits"].values():
            all_methods.update(circuit_data["methods"].keys())
    methods = [schedule_type for schedule_type in schedule_type_colors.keys() if schedule_type in all_methods]

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")

    x = np.arange(len(qecc_names))
    width = 0.8/max(len(methods), 1)

    for i, method in enumerate(methods):
        infidelities = []
        errors = []
        for qecc_name in qecc_names:
            # Average over every circuit evaluated for this code
            circuit_data = all_data_by_qecc[qecc_name]["circuits"]
            method_fidelities = [data["methods"][method]["fidelity"] for data in circuit_data.values() if method in data["methods"]]
            method_errors = [data["methods"][method].get("fidelity_std", 0.0) for data in circuit_data.values() if method in data["methods"]]

            infidelities.append(1.0-np.mean(method_fidelities) if len(method_fidelities) > 0 else np.nan)
            errors.append(np.mean(method_errors) if len(method_errors) > 0 else 0.0)

        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, infidelities, width, yerr=errors, capsize=3, label=schedule_type_label(method), color=schedule_type_colors.get(method, "#757575"), alpha=0.85)

    ax.set_xlabel("Quantum error-correcting code")
    ax.set_ylabel("Infidelity")
    if title is not None:
        ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(qecc_names, rotation=20, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    if save:
        if save_dir is None: save_dir = "./output/"
        if filename is None: filename = "qecc_comparison.png"
        plt.savefig(f"{save_dir}{filename}", dpi=256)

    if show: plt.show()

    return fig

def gadget_cost_differential_scatter(gadget_costs, depth=None, n_qubits=None, title=None, save=False, filename=None, save_dir=None, show=False):
    """Plot ranked gadget cost differentials, colored by sign.

    A positive differential is super-additive, i.e., the gadget is worse than the sum of its parts, and a negative differential is sub-additive; since the sign varies by gadget, coloring by sign is what makes the structure legible.
    """

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    for depth_key, depth_gadgets in gadget_costs.items():
        if depth is not None and depth_key != depth:
            continue

        for n_qubits_key, n_qubits_gadgets in depth_gadgets.items():
            if n_qubits is not None and n_qubits_key != n_qubits:
                continue

            y = np.array(sorted(list(n_qubits_gadgets.values()))[::-1])
            x = np.arange(len(y))

            ax.scatter(x[y >= 0], y[y >= 0], s=12, color="#D32F2F", alpha=0.8, label=f"Super-additive (depth {depth_key}, {n_qubits_key}q)")
            ax.scatter(x[y < 0], y[y < 0], s=12, color="#1976D2", alpha=0.8, label=f"Sub-additive (depth {depth_key}, {n_qubits_key}q)")

    ax.axhline(y=0.0, color="black", linestyle="--", alpha=0.5)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Gadget cost differential")
    if title is not None:
        ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if save:
        if save_dir is None: save_dir = "./output/"
        if filename is None: filename = "gadget_cost_differential.png"
        plt.savefig(f"{save_dir}{filename}", dpi=256)

    if show: plt.show()

    return fig
