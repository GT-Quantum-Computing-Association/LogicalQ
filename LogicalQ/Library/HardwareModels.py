import numpy as np

from .Gates import RGate, RZGate, RZZGate, ZZGate

# @TODO - construct pre-made noise models for specific hardware
#       - wishlist:
#           - Harvard/MIT/QuEra neutral atom collaboration (e.g. papers by Vuletic, Lukin, Bluvstein, Evered, Levine, Kalinowski, Li)
#           - Caltech neutral atom group (e.g. papers by Endres, Choi, Scholl, Shaw)

# Quantinuum H1-1 (https://docs.quantinuum.com/systems/user_guide/emulator_user_guide/emulators/h1_emulators.html):

hardware_model_Quantinuum_H1_1 = {
    "device_info": {
        "n_qubits": 20,
        "coupling_map": "fully_coupled",
        # "basis_gates": dict(zip(["u", "rz", "rzz", "zz"], [UGate, RZGate, RZZGate, ZZGate])),
        "basis_gates": dict(zip(["r", "rz", "rzz"], [RGate, RZGate, RZZGate])), # @TODO - TEMPORARY WORKAROUND
    },
    "noise_params": {
        "all_qubit": {
            "t1": 60,
            "t2": 4,
            "gate_time": {
                1: {
                    "all": 10E-6,
                },
                2: {
                    "all": 100E-6,
                },
            },
            "readout_error": {
                1: {
                    ("0","1"): 1.22E-3,
                    ("1","0"): 3.43E-3,
                },
            },
            "depolarizing_error": {
                1: {
                    "all": (1-0.747) * 1.80E-5,
                },
                2: {
                    "all": (1-0.421) * 9.73E-4,
                },
            },
            "amplitude_damping_error": {
                1: {
                    "all": (0.747) * 1.80E-5,
                },
                # 2: {
                #     "all": (0.421) * 9.73E-4, # @TODO - not supported by Qiskit
                # },
            },
            "dephasing_error": {
                1: {
                    # @TODO - verify angle convention (i.e. theta vs. theta/2)
                    "id": np.sin(0.122 * 10E-6**2)**2, # Using a single-qubit gate time approximation
                }
            }
        },
    },
}

# Quantinuum H2-1 (https://docs.quantinuum.com/systems/user_guide/emulator_user_guide/emulators/h2_emulators.html):
hardware_model_Quantinuum_H2_1 = {
    "device_info": {
        "n_qubits": 56,
        "coupling_map": "fully_coupled",
        # "basis_gates": dict(zip(["u", "rz", "rzz", "zz"], [UGate, RZGate, RZZGate, ZZGate])),
        "basis_gates": dict(zip(["r", "rz", "rzz"], [RGate, RZGate, RZZGate])), # @TODO - TEMPORARY WORKAROUND
    },
    "noise_params": {
        "all_qubit": {
            "t1": 60,
            "t2": 4,
            "gate_time": {
                1: {
                    "all": 10E-6,
                },
                2: {
                    "all": 100E-6,
                },
            },
            "readout_error": {
                1: {
                    ("0","1"): 6E-4,
                    ("1","0"): 1.39E-3,
                },
            },
            "depolarizing_error": {
                1: {
                    "all": (1-0.54) * 1.89E-5,
                },
                2: {
                    "all": (1-0.255) * 1.05E-3,
                },
            },
            "amplitude_damping_error": {
                1: {
                    "all": (0.54) * 1.89E-5,
                },
                # 2: {
                #     "all": (0.255) * 1.05E-3, # @TODO - not supported by Qiskit
                # },
            },
            "dephasing_error": {
                1: {
                    # @TODO - verify angle convention (i.e. theta vs. theta/2)
                    "id": np.sin(0.0028 * 10E-6 + 0.43 * 10E-6**2)**2, # Using a single-qubit gate time approximation
                }
            }
        },
    },
}

# Quantinuum H2-2 (https://docs.quantinuum.com/systems/user_guide/emulator_user_guide/emulators/h2_emulators.html):
hardware_model_Quantinuum_H2_2 = {
    "device_info": {
        "n_qubits": 56,
        "coupling_map": "fully_coupled",
        # "basis_gates": dict(zip(["u", "rz", "rzz", "zz"], [UGate, RZGate, RZZGate, ZZGate])),
        "basis_gates": dict(zip(["r", "rz", "rzz"], [RGate, RZGate, RZZGate])), # @TODO - TEMPORARY WORKAROUND
    },
    "noise_params": {
        "all_qubit": {
            "t1": 60,
            "t2": 4,
            "gate_time": {
                1: {
                    "all": 10E-6,
                },
                2: {
                    "all": 100E-6,
                },
            },
            "readout_error": {
                1: {
                    ("0","1"): 9E-4,
                    ("1","0"): 1.8E-3,
                },
            },
            "depolarizing_error": {
                1: {
                    "all": (1-0.32) * 7.3E-5,
                },
                2: {
                    "all": (1-0.59) * 1.29E-3,
                },
            },
            "amplitude_damping_error": {
                1: {
                    "all": (0.32) * 7.3E-5,
                },
                # 2: {
                #     "all": (0.59) * 1.29E-3, # @TODO - not supported by Qiskit
                # },
            },
            "dephasing_error": {
                1: {
                    # @TODO - verify angle convention (i.e. theta vs. theta/2)
                    "id": np.sin(0.0028 * 10E-6 + 0.43 * 10E-6**2)**2, # Using a single-qubit gate time approximation
                }
            }
        },
    },
}

hardware_models_Quantinuum = {
    "H1-1": hardware_model_Quantinuum_H1_1,
    "H2-1": hardware_model_Quantinuum_H2_1,
    "H2-2": hardware_model_Quantinuum_H2_2,
}

