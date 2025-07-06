def analyze_qec_cycle_results(all_data, scan_keys=None):
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
