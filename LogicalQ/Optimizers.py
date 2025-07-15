from nlopt import opt

from qiskit_aer import AerSimulator
from qiskit.providers import Backend
from qiskit_aer.noise import NoiseModel

def optimize_costs(backend=None, noise_model=None, noise_params=None):
    if backend is not None:
        if isinstance(backend, (AerSimulator, Backend)):
            noise_model = NoiseModel.from_backend(backend) # @TODO - check syntax
        else:
            raise TypeError(f"Backend must be an instance of AerSimulator, BackendV1, BackendV2, or IBMBackend, not {type(backend)}")
    elif noise_model is not None:
        if not isinstance(noise_model, NoiseModel):
            raise TypeError(f"noise_model must be an instance of NoiseModel, not {type(noise_model)}")
    elif noise_params is not None:
        if isinstance(noise_params, dict):
            noise_model = construct_noise_model(noise_params)
        else:
            raise TypeError(f"noise_params must be a dict, not {type(noise_params)}")

